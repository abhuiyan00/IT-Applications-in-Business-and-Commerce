"""Offline telemetry analyzer — turn logged data into targeted, data-driven fixes.

Reads the per-step telemetry CSV(s) written by TorcsEnv and, like a race engineer
between sessions:
  * calibrates real grip from MEASURED lateral-g (v^2*|kappa|), not an assumed mu;
  * clusters off-track events by location -> the corners that actually cost laps;
  * audits speed deficit / overspeed / wheelspin by location and by curvature;
  * (optional) writes a data-derived per-node speed-scale (corner memory) built
    from the crash clusters, which the env loads next run.

Pure stdlib csv + numpy (no pandas/sklearn dependency).

Usage (from gym_torcs/):
  python analyze_telemetry.py                       # newest telemetry log
  python analyze_telemetry.py --logs "logs/telemetry_*.csv"
  python analyze_telemetry.py --write-corner-memory # emit data-driven corner speeds
"""

import argparse
import csv
import glob
import math
import os

import numpy as np

from racing_line import RacingLine


def load_telemetry(paths):
    """Load one or more telemetry CSVs into a dict of numpy arrays (by column)."""
    rows = []
    header = None
    for path in paths:
        with open(path, "r", newline="") as handle:
            reader = csv.reader(handle)
            file_header = next(reader, None)
            if file_header is None:
                continue
            if header is None:
                header = file_header
            for row in reader:
                if len(row) == len(header):
                    rows.append(row)
    if not rows:
        raise SystemExit("No telemetry rows found.")

    cols = {name: [] for name in header}
    for row in rows:
        for name, value in zip(header, row):
            cols[name].append(value)

    def to_float(name):
        out = np.empty(len(rows), dtype=np.float64)
        for i, v in enumerate(cols[name]):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                out[i] = np.nan
        return out

    data = {}
    for name in header:
        if name in ("termination_reason",):
            data[name] = np.array(cols[name], dtype=object)
        else:
            data[name] = to_float(name)
    return data, header


def cluster_1d(values, gap=30.0):
    """Cluster sorted scalar positions; merge points within `gap` metres. Returns
    list of (center, count, lo, hi)."""
    values = np.sort(values[~np.isnan(values)])
    if values.size == 0:
        return []
    clusters = []
    start = prev = values[0]
    members = [values[0]]
    for v in values[1:]:
        if v - prev <= gap:
            members.append(v)
        else:
            clusters.append((float(np.mean(members)), len(members), start, prev))
            start = v
            members = [v]
        prev = v
    clusters.append((float(np.mean(members)), len(members), start, prev))
    return clusters


def report(data, line, mu, g):
    n = len(data["s"])
    episodes = int(np.nanmax(data["episode"])) if n else 0
    print(f"\n=== TELEMETRY OVERVIEW ===")
    print(f"rows={n}  episodes={episodes}  "
          f"dist={np.nansum(np.diff(np.clip(data['distRaced'],0,None)).clip(0)):.0f} m")

    speed_ms = data["speedX"] / 3.6
    on_track = data["off_track"] < 0.5

    # --- grip calibration from measured lateral-g ---
    # a_lat = v^2*|kappa_rl| is only a valid proxy for the car's REAL lateral
    # acceleration when the car is tracking the line (kappa_rl ~= path curvature)
    # and actually in a corner at speed. Filter to those rows so the grip estimate
    # is not polluted by off-line / straight-line / crawling samples.
    half_w = line.half_width if line is not None else 1.0
    on_line = on_track & (np.abs(data["e_y"]) < 0.3 * half_w) \
        & (np.abs(data["kappa_rl"]) > 0.004) & (data["speedX"] > 30.0)
    lat = data["lat_accel"][on_line]
    lat = lat[~np.isnan(lat)]
    print(f"\n=== GRIP CALIBRATION (measured a_lat = v^2*|kappa|, on-line corners) ===")
    if lat.size >= 20:
        a_lat_p95 = float(np.percentile(lat, 95))
        a_lat_max = float(lat.max())
        mu_eff = a_lat_p95 / g
        grip_used = a_lat_p95 / (mu * g)
        print(f"samples={lat.size}  a_lat p95={a_lat_p95:.1f}  max={a_lat_max:.1f} "
              f"m/s^2  => mu_eff={mu_eff:.2f}")
        print(f"suggested racing_line --grip {grip_used:.2f}  (raise if the car never "
              f"reaches vmax in corners; lower if it spins/offs at the apex)")
    else:
        print(f"  too few on-line corner samples ({lat.size}) to calibrate grip yet")

    # --- off-track hotspots (the corners that cost laps) ---
    crash = data["crash_s"]
    crash = crash[~np.isnan(crash)]
    print(f"\n=== OFF-TRACK HOTSPOTS (clustered crash_s) ===")
    if crash.size:
        for center, count, lo, hi in sorted(
                cluster_1d(crash), key=lambda c: -c[1]):
            q = line.query(center) if line else {"vmax": float("nan")}
            print(f"  s~{center:6.0f} m  x{count:3d} crashes  "
                  f"(range {lo:.0f}-{hi:.0f})  vmax_here={q['vmax']*3.6:.0f} km/h")
    else:
        print("  none recorded")

    # --- speed audit by location ---
    print(f"\n=== SPEED AUDIT BY LOCATION (50 m bins, on-track) ===")
    s = data["s"]
    length = line.length if line else float(np.nanmax(s) + 1)
    nbins = max(1, int(length // 50))
    binned = (s / 50).astype(int) % nbins
    deficit = (data["vmax"] - speed_ms)            # +ve = under target
    overspeed = (speed_ms - data["vmax"]).clip(0)  # +ve = over target (risk)
    worst = []
    for b in range(nbins):
        m = on_track & (binned == b)
        if m.sum() < 5:
            continue
        worst.append((b * 50,
                      float(np.nanmean(deficit[m])),
                      float(np.nanmean(overspeed[m])),
                      float(np.nanmean(data["slip"][m])),
                      int((data["off_track"][m] > 0.5).sum())))
    worst.sort(key=lambda r: -r[1])
    print("  most UNDER target (slow - gearing/power/over-cautious):")
    for s0, d, ov, sl, off in worst[:6]:
        print(f"    s~{s0:5.0f} m  deficit {d*3.6:5.1f} km/h  slip {sl:5.1f}")
    worst.sort(key=lambda r: -r[2])
    print("  most OVER target (overspeed - crash risk):")
    for s0, d, ov, sl, off in worst[:6]:
        if ov <= 0.01:
            break
        print(f"    s~{s0:5.0f} m  overspeed {ov*3.6:5.1f} km/h  offs {off}")

    # --- curvature clusters ---
    print(f"\n=== CURVATURE CLUSTERS (by |kappa|, on-track) ===")
    ak = np.abs(data["kappa_rl"])
    edges = [0.0, 0.004, 0.01, 0.02, 0.04, 1.0]
    names = ["straight", "gentle", "medium", "tight", "hairpin"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = on_track & (ak >= lo) & (ak < hi)
        if m.sum() < 5:
            continue
        print(f"  {nm:8s} |k| {lo:.3f}-{hi:.3f}: "
              f"speed {np.nanmean(speed_ms[m])*3.6:5.1f} km/h  "
              f"deficit {np.nanmean(deficit[m])*3.6:5.1f}  "
              f"slip {np.nanmean(data['slip'][m]):5.1f}  "
              f"offs {int((data['off_track'][m]>0.5).sum())}")


def write_corner_memory(data, line, out_path, floor=0.80, decay=0.90,
                        window=45.0):
    """Build a data-derived per-node speed-scale from ALL crash clusters across the
    runs (robust offline version of the env's online corner memory) and save it
    where the env auto-loads it. Scale is <=1 (a safe ceiling); the env reclaims
    speed online when corners are taken cleanly."""
    n = line.n
    scale = np.ones(n, dtype=np.float64)
    crash = data["crash_s"]
    crash = crash[~np.isnan(crash)]
    ds = line.ds
    window_nodes = max(1, int(round(window / ds)))
    for s_crash in crash:
        apex = line._index(s_crash)
        for k in range(window_nodes + 1):
            idx = (apex - k) % n
            weight = 1.0 - k / (window_nodes + 1)
            factor = 1.0 - (1.0 - decay) * weight
            scale[idx] = max(floor, scale[idx] * factor)
    np.save(out_path, scale)
    print(f"\nWrote data-derived corner memory: {out_path}  "
          f"(min={scale.min():.3f} mean={scale.mean():.3f}, from {crash.size} crashes)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Analyze TORCS telemetry logs.")
    ap.add_argument("--logs", default=None,
                    help="Glob for telemetry CSVs (default: newest in logs/).")
    ap.add_argument("--track-npz", default=os.path.join(here, "lines", "corkscrew.npz"))
    ap.add_argument("--mu", type=float, default=1.1)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--write-corner-memory", action="store_true")
    args = ap.parse_args()

    if args.logs:
        paths = sorted(glob.glob(args.logs))
    else:
        all_logs = sorted(glob.glob(os.path.join(here, "logs", "telemetry_*.csv")))
        paths = all_logs[-1:] if all_logs else []
    if not paths:
        raise SystemExit("No telemetry logs found (run training with telemetry on).")
    print("Analyzing:", ", ".join(os.path.basename(p) for p in paths))

    data, _ = load_telemetry(paths)
    line = RacingLine.load(args.track_npz) if os.path.exists(args.track_npz) else None
    report(data, line, args.mu, args.g)

    if args.write_corner_memory and line is not None:
        base, _ = os.path.splitext(args.track_npz)
        write_corner_memory(data, line, base + "_corner_memory.npy")


if __name__ == "__main__":
    main()
