"""Offline racing-line generator + runtime loader for TORCS tracks.

Pipeline (run once per track, offline):
  1. Parse the track XML -> ordered segments (straights + arcs, incl. spirals).
  2. Integrate the centerline (x, y, heading, curvature) in the track frame.
  3. Relax a racing line inside the track corridor (minimum-curvature smoothing).
  4. Build a physically realizable speed profile (corner cap + accel/brake passes).
  5. Save everything as an .npz keyed by arc-length s.

At runtime the env loads the .npz and queries it by `distFromStart` (= s).
This module depends only on numpy + the stdlib (no torch, no TORCS).

The geometry comes from the XML on purpose: the obvious in-sim source
(`tTrackSeg`) is a TORCS C++ runtime struct and is NOT reachable from the
Python SCR/UDP client. The XML encodes the identical segment geometry.
"""

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


def _load_track_xml_root(xml_path):
    """Parse a TORCS track XML, tolerating its external/undefined entities.

    Track XMLs pull in `&default-surfaces;` etc. via a DOCTYPE that references
    files we don't need. Strip the DOCTYPE and any non-standard entity refs so
    the stdlib parser can read the geometry we care about.
    """
    with open(xml_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    text = re.sub(r"<!DOCTYPE.*?\]>", "", text, flags=re.DOTALL)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)
    text = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)[\w.-]+;", "", text)
    return ET.fromstring(text)


# --------------------------------------------------------------------------- #
# 1. XML parsing
# --------------------------------------------------------------------------- #
def _seg_attr(section, name, default=None):
    """Read a direct-child <attnum>/<attstr name=... val=...> of a segment.

    Only direct children are inspected so the nested Left/Right Side widths do
    not shadow the segment's own type/lg/arc/radius values.
    """
    for tag in ("attnum", "attstr"):
        for node in section.findall(tag):
            if node.get("name") == name:
                value = node.get("val")
                if tag == "attnum":
                    return float(value)
                return value
    return default


def parse_track_xml(xml_path):
    """Return (width, segments). Each segment is a dict:
        {'type': 'str'|'lft'|'rgt', 'length': m, 'grade': slope}            straights
        {'type': 'lft'|'rgt', 'arc': rad, 'r0': m, 'r1': m, 'grade': slope}  arcs
    Arc r0/r1 differ when the XML gives an 'end radius' (spiral / clothoid).
    'grade' is the longitudinal slope as a fraction (XML 'grade' % / 100,
    +uphill / -downhill); 0.0 when the segment declares none.
    """
    root = _load_track_xml_root(xml_path)

    main_track = None
    for section in root.iter("section"):
        if section.get("name") == "Main Track":
            main_track = section
            break
    if main_track is None:
        raise ValueError(f"No 'Main Track' section in {xml_path}")

    width = None
    for node in main_track.findall("attnum"):
        if node.get("name") == "width":
            width = float(node.get("val"))
            break
    if width is None:
        width = 12.0  # TORCS default road width fallback

    track_segments = None
    for section in main_track.iter("section"):
        if section.get("name") == "Track Segments":
            track_segments = section
            break
    if track_segments is None:
        raise ValueError(f"No 'Track Segments' section in {xml_path}")

    segments = []
    for seg in track_segments.findall("section"):
        seg_type = _seg_attr(seg, "type")
        if seg_type is None:
            continue  # not a real track segment (defensive)
        grade_pct = _seg_attr(seg, "grade")          # longitudinal slope, percent
        grade = float(grade_pct) / 100.0 if grade_pct is not None else 0.0
        if seg_type == "str":
            length = _seg_attr(seg, "lg")
            if length is None or length <= 0:
                continue
            segments.append({"type": "str", "length": float(length), "grade": grade})
        elif seg_type in ("lft", "rgt"):
            arc_deg = _seg_attr(seg, "arc")
            radius = _seg_attr(seg, "radius")
            if arc_deg is None or radius is None:
                continue
            end_radius = _seg_attr(seg, "end radius", radius)
            segments.append(
                {
                    "type": seg_type,
                    "arc": math.radians(float(arc_deg)),
                    "r0": float(radius),
                    "r1": float(end_radius),
                    "grade": grade,
                }
            )
        # any other type (e.g. pit-only) is ignored
    if not segments:
        raise ValueError(f"No usable segments parsed from {xml_path}")
    return width, segments


# --------------------------------------------------------------------------- #
# 2. Centerline integration
# --------------------------------------------------------------------------- #
def integrate_centerline(segments, ds=2.0):
    """Walk the segment list and return uniformly-sampled centerline arrays.

    Returns dict with s, x, y, psi (unwrapped heading, rad), kappa, length.
    Starts at (0,0) heading 0 in the track-local frame; absolute orientation is
    irrelevant for relaxation and for the (s, lateral) runtime frame.
    """
    s_raw = [0.0]
    x_raw = [0.0]
    y_raw = [0.0]
    psi_raw = [0.0]
    kappa_raw = [0.0]
    grade_raw = [0.0]      # longitudinal slope at each node (+up / -down)
    z_raw = [0.0]          # integrated elevation (m), relative to the start

    x = y = psi = s = z = 0.0
    for seg in segments:
        grade = float(seg.get("grade", 0.0))
        if seg["type"] == "str":
            length = seg["length"]
            n = max(1, int(round(length / ds)))
            step = length / n
            for _ in range(n):
                x += step * math.cos(psi)
                y += step * math.sin(psi)
                z += step * grade           # rise = run * slope (small-angle)
                s += step
                s_raw.append(s); x_raw.append(x); y_raw.append(y)
                psi_raw.append(psi); kappa_raw.append(0.0)
                grade_raw.append(grade); z_raw.append(z)
        else:
            sign = 1.0 if seg["type"] == "lft" else -1.0
            arc = seg["arc"]
            r0, r1 = seg["r0"], seg["r1"]
            mean_r = 0.5 * (r0 + r1)
            arc_len = mean_r * arc
            n = max(2, int(round(arc_len / ds)))
            dtheta = arc / n
            for i in range(n):
                t = (i + 0.5) / n
                r = r0 + (r1 - r0) * t
                psi_mid = psi + sign * dtheta / 2.0
                step = r * dtheta
                x += step * math.cos(psi_mid)
                y += step * math.sin(psi_mid)
                z += step * grade
                psi += sign * dtheta
                s += step
                s_raw.append(s); x_raw.append(x); y_raw.append(y)
                psi_raw.append(psi); kappa_raw.append(sign / r)
                grade_raw.append(grade); z_raw.append(z)

    length = s
    s_raw = np.asarray(s_raw)
    x_raw = np.asarray(x_raw); y_raw = np.asarray(y_raw)
    # Close the loop in xy: integrating in 2D leaves a small gap because track
    # grade (ignored here) shortens the horizontal projection. Distribute that
    # gap linearly along s so the centerline closes cleanly, which keeps the
    # periodic relaxation free of a seam kink at the start/finish node. s itself
    # is left untouched so it still matches distFromStart (sum of segment lengths).
    gap_x = x_raw[-1] - x_raw[0]
    gap_y = y_raw[-1] - y_raw[0]
    x_raw = x_raw - gap_x * (s_raw / length)
    y_raw = y_raw - gap_y * (s_raw / length)
    # Resample to a uniform arc-length grid (drop the closing duplicate at s=L).
    s_grid = np.arange(0.0, length, ds)
    x = np.interp(s_grid, s_raw, x_raw)
    y = np.interp(s_grid, s_raw, y_raw)
    psi = np.interp(s_grid, s_raw, np.asarray(psi_raw))
    kappa = np.interp(s_grid, s_raw, np.asarray(kappa_raw))
    grade = np.interp(s_grid, s_raw, np.asarray(grade_raw))
    z = np.interp(s_grid, s_raw, np.asarray(z_raw))
    return {
        "s": s_grid, "x": x, "y": y, "psi": psi, "kappa": kappa,
        "grade": grade, "z": z,
        "length": length, "ds": ds,
    }


# --------------------------------------------------------------------------- #
# 3. Racing-line relaxation
# --------------------------------------------------------------------------- #
def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _periodic_smooth(values, window=5):
    """Moving average over a closed loop (odd window)."""
    if window < 3:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-(window // 2):], values, values[: window // 2]])
    return np.convolve(padded, kernel, mode="valid")


def relax_racing_line(center, width, margin=1.5, iters=800, relax=0.2):
    """Minimum-curvature racing line inside the corridor (periodic / closed loop).

    Returns offset n (signed, +left), racing-line P (N,2), psi_rl, kappa_rl.
    """
    cx, cy, psi_c = center["x"], center["y"], center["psi"]
    ds = center["ds"]
    C = np.stack([cx, cy], axis=1)
    normal = np.stack([-np.sin(psi_c), np.cos(psi_c)], axis=1)  # left normal
    n_max = max(0.5, width / 2.0 - margin)

    P = C.copy()
    for _ in range(iters):
        mid = 0.5 * (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0))
        P = P + relax * (mid - P)
        n_off = np.sum((P - C) * normal, axis=1)
        n_off = np.clip(n_off, -n_max, n_max)
        P = C + n_off[:, None] * normal

    n_off = np.sum((P - C) * normal, axis=1)
    # Derive racing-line heading analytically from the lateral-offset gradient
    # instead of differentiating xy. Heading closes mod 2*pi even when the xy
    # path has a small seam gap from ignoring track grade, so this is robust.
    dn = (np.roll(n_off, -1) - np.roll(n_off, 1)) / (2.0 * ds)
    denom = 1.0 - center["kappa"] * n_off          # parallel-offset Jacobian
    denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
    psi_rl = psi_c + np.arctan(dn / denom)
    dpsi = _wrap_pi(np.roll(psi_rl, -1) - np.roll(psi_rl, 1))
    kappa_rl = dpsi / (2.0 * ds)
    # Finite-differencing a corridor-clamped offset gives single-node curvature
    # spikes; smooth so the speed profile follows real corners, not artifacts.
    kappa_rl = _periodic_smooth(kappa_rl, window=5)
    return n_off, P, psi_rl, kappa_rl


# --------------------------------------------------------------------------- #
# 4. Speed profile
# --------------------------------------------------------------------------- #
def build_speed_profile(kappa_rl, ds, mu=1.1, a_accel=5.0, a_brake=8.0,
                        v_top=55.0, grip=0.70, g=9.81, grade=None):
    """Forward/backward speed profile (m/s) over the closed loop.

    When `grade` (per-node slope, +up/-down) is supplied the longitudinal
    accel/brake limits become grade-aware: gravity ADDS to braking and SUBTRACTS
    from acceleration uphill, and the reverse downhill. This bakes the hill
    physics into the offline target so the runtime no longer has to guess from
    the noisy speedZ sensor -- a downhill corner gets a lower, earlier-braked
    entry speed automatically.
    """
    a_lat = grip * mu * g
    v = np.sqrt(a_lat / np.maximum(np.abs(kappa_rl), 1e-5))
    v = np.minimum(v, v_top)
    n = len(v)
    if grade is None:
        grade = np.zeros(n)
    grade = np.asarray(grade, dtype=np.float64)
    # Per-node longitudinal limits incl. the gravity component g*slope.
    a_brake_arr = np.clip(a_brake + g * grade, 1.0, None)   # uphill brakes harder
    a_accel_arr = np.clip(a_accel - g * grade, 0.5, None)   # uphill accelerates less
    # Two wrap passes converge the periodic boundary.
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * a_brake_arr[i] * ds))
    for _ in range(2):
        for i in range(n):
            j = (i - 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * a_accel_arr[i] * ds))
    return np.minimum(v, v_top)


# --------------------------------------------------------------------------- #
# Generator entry point
# --------------------------------------------------------------------------- #
def generate(xml_path, out_path, ds=2.0, margin=1.5, iters=800, relax=0.2,
             mu=1.1, a_accel=5.0, a_brake=8.0, v_top=55.0, grip=0.70,
             s_offset=0.0, plot=False):
    width, segments = parse_track_xml(xml_path)
    center = integrate_centerline(segments, ds=ds)

    # Loop-closure check: the integrated centerline must return near its start.
    gap = math.hypot(center["x"][-1] - center["x"][0],
                     center["y"][-1] - center["y"][0])
    gap_pct = 100.0 * gap / max(center["length"], 1.0)
    n_off, P, psi_rl, kappa_rl = relax_racing_line(
        center, width, margin=margin, iters=iters, relax=relax)
    vmax = build_speed_profile(
        kappa_rl, ds, mu=mu, a_accel=a_accel, a_brake=a_brake, v_top=v_top,
        grip=grip, grade=center["grade"])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(
        out_path,
        s=center["s"], x=P[:, 0], y=P[:, 1],
        psi_center=center["psi"], psi_rl=psi_rl,
        offset=n_off, kappa_center=center["kappa"], kappa_rl=kappa_rl,
        grade=center["grade"], z=center["z"],
        vmax=vmax, length=np.float64(center["length"]),
        width=np.float64(width), ds=np.float64(ds),
        s_offset=np.float64(s_offset),
    )

    print(f"Track XML        : {xml_path}")
    print(f"Segments         : {len(segments)}")
    print(f"Track length     : {center['length']:.1f} m  (nodes={len(center['s'])}, ds={ds} m)")
    print(f"Track width      : {width:.1f} m  (corridor +/-{width/2 - margin:.2f} m)")
    closure_ok = gap_pct < 2.0  # grade (ignored in 2D) leaves a small benign gap
    print(f"Loop-closure gap : {gap:.2f} m ({gap_pct:.2f}% of length)  "
          f"({'OK' if closure_ok else 'CHECK arc signs!'})")
    print(f"Racing offset    : min={n_off.min():.2f} max={n_off.max():.2f} m")
    grade = center["grade"]
    print(f"Grade            : min={grade.min()*100:.1f}% max={grade.max()*100:.1f}%  "
          f"elev range={center['z'].max()-center['z'].min():.1f} m")
    print(f"Speed target     : min={vmax.min():.1f} max={vmax.max():.1f} m/s "
          f"({vmax.min()*3.6:.0f}-{vmax.max()*3.6:.0f} km/h)")
    print(f"Saved            : {out_path}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 9))
            half = width / 2.0
            nx = -np.sin(center["psi"]); ny = np.cos(center["psi"])
            plt.plot(center["x"] + half * nx, center["y"] + half * ny, "k:", lw=0.5)
            plt.plot(center["x"] - half * nx, center["y"] - half * ny, "k:", lw=0.5)
            plt.plot(center["x"], center["y"], "b-", lw=0.6, label="centerline")
            plt.plot(P[:, 0], P[:, 1], "r-", lw=1.2, label="racing line")
            plt.axis("equal"); plt.legend(); plt.title(os.path.basename(xml_path))
            plt.savefig(out_path.replace(".npz", ".png"), dpi=120)
            print(f"Plot             : {out_path.replace('.npz', '.png')}")
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"(plot skipped: {exc})")
    return out_path


# --------------------------------------------------------------------------- #
# Runtime loader
# --------------------------------------------------------------------------- #
class RacingLine:
    """Loads a generated .npz and answers (s -> targets) queries by nearest node."""

    V_REF = 80.0  # m/s, fixed normalization reference (above any car top speed)

    def __init__(self, data):
        self.s = np.asarray(data["s"], dtype=np.float64)
        self.x = np.asarray(data["x"], dtype=np.float64)
        self.y = np.asarray(data["y"], dtype=np.float64)
        self.psi_center = np.asarray(data["psi_center"], dtype=np.float64)
        self.psi_rl = np.asarray(data["psi_rl"], dtype=np.float64)
        self.offset = np.asarray(data["offset"], dtype=np.float64)
        self.kappa_rl = np.asarray(data["kappa_rl"], dtype=np.float64)
        self.vmax = np.asarray(data["vmax"], dtype=np.float64)
        # grade (slope, +up/-down) is optional so pre-grade .npz files still load.
        if "grade" in data:
            self.grade = np.asarray(data["grade"], dtype=np.float64)
        else:
            self.grade = np.zeros(len(self.s), dtype=np.float64)
        self.has_grade = "grade" in data
        self.length = float(data["length"])
        self.width = float(data["width"])
        self.ds = float(data["ds"])
        self.s_offset = float(data["s_offset"]) if "s_offset" in data else 0.0
        self.n = len(self.s)
        self.half_width = self.width / 2.0

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def _index(self, s):
        u = (s + self.s_offset) % self.length
        return int(round(u / self.ds)) % self.n

    def query(self, s):
        i = self._index(s)
        # heading offset of the racing line vs centerline tangent (wrapped small)
        dpsi = math.atan2(math.sin(self.psi_rl[i] - self.psi_center[i]),
                          math.cos(self.psi_rl[i] - self.psi_center[i]))
        return {
            "offset": float(self.offset[i]),       # m, +left of centerline
            "psi_center": float(self.psi_center[i]),
            "psi_rl": float(self.psi_rl[i]),
            "dpsi": float(dpsi),                    # psi_rl - psi_center, wrapped
            "kappa_rl": float(self.kappa_rl[i]),
            "vmax": float(self.vmax[i]),            # m/s
        }

    def preview(self, s, distances):
        """List of (kappa_rl, vmax) at s + each look-ahead distance (m)."""
        out = []
        for d in distances:
            i = self._index(s + d)
            out.append((float(self.kappa_rl[i]), float(self.vmax[i])))
        return out

    def min_vmax_ahead(self, s, horizon):
        """Lowest target speed (m/s) anywhere in [s, s+horizon].

        Lets the agent brake for a corner that is still far away: a 70 m
        preview cannot cover the ~200 m needed to scrub off top-speed before a
        hairpin, so expose the worst upcoming speed directly.
        """
        steps = max(1, int(round(horizon / self.ds)))
        start = self._index(s)
        idx = (start + np.arange(steps + 1)) % self.n
        return float(self.vmax[idx].min())

    def safe_speed(self, s, a_brake, horizon):
        """Highest speed at s from which the car can STILL brake down to every
        upcoming vmax inside the horizon, decelerating at a_brake (m/s^2).

        This is the per-distance braking envelope

            v_safe(s) = min_j  sqrt( vmax(s_j)^2 + 2 * a_brake * d_j )

        over nodes j in [s, s+horizon], with d_j = (s_j - s) the distance to
        each node. Unlike min_vmax_ahead (which only returns the worst speed and
        then lets the caller assume the FULL horizon to reach it), this accounts
        for HOW FAR each slow point actually is, so a near hairpin forces a low
        speed now while a distant one does not.

        Pass a CONSERVATIVE a_brake (below the value the speed profile was built
        with) to leave real margin for controller lag, downhill grade and tyre
        wear -- the profile alone has zero margin where vmax was set by its own
        braking pass.
        """
        steps = max(1, int(round(horizon / self.ds)))
        start = self._index(s)
        idx = (start + np.arange(steps + 1)) % self.n
        d = np.arange(steps + 1, dtype=np.float64) * self.ds
        return float(np.sqrt(self.vmax[idx] ** 2 + 2.0 * a_brake * d).min())

    def grade_at(self, s):
        """Longitudinal slope at s (+uphill / -downhill), from the track XML."""
        return float(self.grade[self._index(s)])

    def min_grade_ahead(self, s, horizon):
        """Steepest DOWNHILL (most negative slope) in [s, s+horizon]. Used to
        brake earlier when a descent is coming up."""
        steps = max(1, int(round(horizon / self.ds)))
        start = self._index(s)
        idx = (start + np.arange(steps + 1)) % self.n
        return float(self.grade[idx].min())

    def max_abs_kappa_ahead(self, s, horizon):
        """Largest |curvature| anywhere in [s, s+horizon]. Near zero => the road
        is straight for the whole window (no corner being set up)."""
        steps = max(1, int(round(horizon / self.ds)))
        start = self._index(s)
        idx = (start + np.arange(steps + 1)) % self.n
        return float(np.abs(self.kappa_rl[idx]).max())


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_xml = os.path.normpath(
        os.path.join(here, "..", "torcs", "tracks", "road", "corkscrew", "corkscrew.xml"))
    parser = argparse.ArgumentParser(description="Generate a TORCS racing line .npz.")
    parser.add_argument("--track-xml", default=default_xml)
    parser.add_argument("--out", default=os.path.join(here, "lines", "corkscrew.npz"))
    parser.add_argument("--ds", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--relax", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=1.1)
    parser.add_argument("--a-accel", type=float, default=5.0)
    parser.add_argument("--a-brake", type=float, default=8.0)
    parser.add_argument("--v-top", type=float, default=55.0)
    parser.add_argument("--grip", type=float, default=0.70)
    parser.add_argument("--s-offset", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    generate(
        args.track_xml, args.out, ds=args.ds, margin=args.margin, iters=args.iters,
        relax=args.relax, mu=args.mu, a_accel=args.a_accel, a_brake=args.a_brake,
        v_top=args.v_top, grip=args.grip, s_offset=args.s_offset, plot=args.plot,
    )


if __name__ == "__main__":
    main()
