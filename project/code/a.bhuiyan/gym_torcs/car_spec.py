"""Data-driven car model for TORCS — every number comes from the car's own XML.

No hardcoded driveline. We parse the configured car (resolved from practice.xml ->
scr_server.xml -> car XML) and expose the real physics:

  * gear ratios + final drive + rear-tyre radius        -> rpm <-> speed
  * engine torque curve (rpm -> N.m) + rev limiter       -> wheel tractive force
  * optimal_gear(speed) = the gear that maximises wheel force right now

So gear selection follows the engine's actual power map (like a real ECU's shift
map), and it self-adapts to whatever car is configured. Pure numpy + stdlib.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from racing_line import _load_track_xml_root  # shared DOCTYPE/entity-tolerant XML loader


def _attnum(section, name, default=None):
    for node in section.findall("attnum"):
        if node.get("name") == name:
            return float(node.get("val"))
    return default


def _attstr(section, name, default=None):
    for node in section.findall("attstr"):
        if node.get("name") == name:
            return node.get("val")
    return default


def _find_section(root, name):
    for sec in root.iter("section"):
        if sec.get("name") == name:
            return sec
    return None


def resolve_configured_car(torcs_root, raceman="practice"):
    """Return (car_name, car_xml_path) for the driver configured in the raceman.

    practice.xml Drivers -> idx + module (e.g. scr_server) ; then
    <module>.xml Robots/index/<idx>/'car name'. Falls back to scanning if any
    step is missing. Returns (None, None) only if nothing can be resolved.
    """
    try:
        raceman_path = os.path.join(torcs_root, "config", "raceman", f"{raceman}.xml")
        root = _load_track_xml_root(raceman_path)
        drivers = _find_section(root, "Drivers")
        idx, module = 0, "scr_server"
        for sec in drivers.findall("section"):
            module = _attstr(sec, "module", module)
            idx = int(_attnum(sec, "idx", idx))
            break
        robot_path = os.path.join(torcs_root, "drivers", module, f"{module}.xml")
        rroot = _load_track_xml_root(robot_path)
        index = _find_section(rroot, "index")
        car_name = None
        for sec in index.findall("section"):
            if sec.get("name") == str(idx):
                car_name = _attstr(sec, "car name")
                break
        if car_name:
            car_xml = os.path.join(torcs_root, "cars", car_name, f"{car_name}.xml")
            if os.path.exists(car_xml):
                return car_name, car_xml
    except Exception:
        pass
    return None, None


class CarSpec:
    """Driveline + engine model parsed from a TORCS car XML."""

    # Upshift at this fraction of the rev limiter (real-life: shift BEFORE the
    # rev cut so the engine never bounces off the limiter). Per-gear speed
    # thresholds fall out of this automatically via the gear ratios.
    DEFAULT_SHIFT_FRAC = 0.93

    def __init__(self, name, ratios, final_drive, tyre_radius,
                 torque_rpm, torque_nm, rev_limiter, tickover, gear_eff=None,
                 shift_frac=None):
        self.name = name
        self.shift_frac = float(shift_frac) if shift_frac is not None else self.DEFAULT_SHIFT_FRAC
        self.ratios = list(ratios)               # forward gear ratios, gear 1..N
        self.n_gears = len(self.ratios)
        self.final_drive = float(final_drive)
        self.tyre_radius = float(tyre_radius)     # m
        self.tyre_circ = 2.0 * np.pi * self.tyre_radius
        self.torque_rpm = np.asarray(torque_rpm, dtype=np.float64)
        self.torque_nm = np.asarray(torque_nm, dtype=np.float64)
        self.rev_limiter = float(rev_limiter)
        self.tickover = float(tickover)
        self.gear_eff = list(gear_eff) if gear_eff else [1.0] * self.n_gears
        # engine rpm per (m/s) per unit gear ratio: wheel_rev/s * 60 * final * ratio
        self._rpm_k = 60.0 * self.final_drive / self.tyre_circ

    # ---- kinematics ----
    def rpm_at(self, speed_mps, gear):
        """Engine rpm at a road speed in a given gear (1..N)."""
        gear = max(1, min(self.n_gears, int(gear)))
        return abs(speed_mps) * self._rpm_k * self.ratios[gear - 1]

    def speed_at(self, rpm, gear):
        gear = max(1, min(self.n_gears, int(gear)))
        return rpm / (self._rpm_k * self.ratios[gear - 1])

    def engine_torque(self, rpm):
        """Engine torque (N.m) at rpm, linearly interpolated from the XML curve."""
        return float(np.interp(rpm, self.torque_rpm, self.torque_nm))

    def wheel_force(self, speed_mps, gear):
        """Tractive force at the wheels (N) in a gear at this speed. Zero past the
        limiter (gear unusable). This is the quantity an optimal shift maximises."""
        gear = max(1, min(self.n_gears, int(gear)))
        rpm = self.rpm_at(speed_mps, gear)
        if rpm > self.rev_limiter:
            return 0.0
        tq = self.engine_torque(rpm)
        ratio = self.ratios[gear - 1] * self.final_drive * self.gear_eff[gear - 1]
        return tq * ratio / self.tyre_radius

    # ---- the shift map ----
    def shift_ceiling(self):
        return self.shift_frac * self.rev_limiter

    def optimal_gear(self, speed_mps, current_gear=1):
        """Best gear at this speed: the LOWEST gear that keeps rpm under the shift
        ceiling (= shift_frac * limiter). Lowest feasible gear = most wheel force
        (max acceleration), and the ceiling forces an upshift BEFORE the rev cut
        so the engine never pins the limiter (the gear-1 stuck bug). Under braking
        the same rule downshifts as speed falls, keeping revs high for corner exit.
        All thresholds derive from the car's own limiter + ratios -- no magic
        numbers. Returns top gear if even it would over-rev (beyond top speed)."""
        speed_mps = abs(speed_mps)
        ceiling = self.shift_ceiling()
        best_gear, best_force = None, -1.0
        for g in range(1, self.n_gears + 1):
            if self.rpm_at(speed_mps, g) > ceiling:
                continue  # would over-rev -> need a taller gear
            f = self.wheel_force(speed_mps, g)
            if f > best_force:
                best_force, best_gear = f, g
        if best_gear is None:        # even top gear past the ceiling -> hold top
            return self.n_gears
        return best_gear

    def upshift_speeds_kmh(self):
        """Speed (km/h) at which each gear hits the shift ceiling -> the per-gear
        upshift thresholds implied by the ceiling. For inspection/telemetry."""
        ceiling = self.shift_ceiling()
        return [round(ceiling / (self._rpm_k * r) * 3.6, 1) for r in self.ratios]

    @classmethod
    def from_xml(cls, car_xml, name=None):
        root = _load_track_xml_root(car_xml)
        if name is None:
            name = os.path.splitext(os.path.basename(car_xml))[0]

        # Engine: limiter, tickover, torque curve.
        engine = _find_section(root, "Engine")
        rev_limiter = _attnum(engine, "revs limiter", 9000.0)
        tickover = _attnum(engine, "tickover", 1000.0)
        data_points = _find_section(engine, "data points")
        rpms, tqs = [], []
        for pt in data_points.findall("section"):
            r = _attnum(pt, "rpm")
            t = _attnum(pt, "Tq")
            if r is not None and t is not None:
                rpms.append(r); tqs.append(t)
        order = np.argsort(rpms)
        rpms = list(np.asarray(rpms)[order]); tqs = list(np.asarray(tqs)[order])

        # Gearbox: forward ratios (ratio > 0), in section-name order 1..N.
        gears_sec = None
        gbox = _find_section(root, "Gearbox")
        if gbox is not None:
            gears_sec = _find_section(gbox, "gears")
        ratios, effs = [], []
        if gears_sec is not None:
            numbered = []
            for g in gears_sec.findall("section"):
                ratio = _attnum(g, "ratio")
                if ratio is None or ratio <= 0:
                    continue  # skip reverse / neutral
                try:
                    order_key = int(g.get("name"))
                except (TypeError, ValueError):
                    order_key = len(numbered)
                numbered.append((order_key, ratio, _attnum(g, "efficiency", 1.0)))
            numbered.sort(key=lambda x: x[0])
            ratios = [r for _, r, _ in numbered]
            effs = [e for _, _, e in numbered]

        # Final drive: the differential ratio.
        final_drive = 4.5
        for diff_name in ("Rear Differential", "Central Differential", "Front Differential"):
            diff = _find_section(root, diff_name)
            if diff is not None:
                fd = _attnum(diff, "ratio")
                if fd:
                    final_drive = fd
                    break

        # Rear tyre radius = rim radius + sidewall (height-width ratio * width).
        tyre_radius = 0.33
        rear = _find_section(root, "Rear Right Wheel") or _find_section(root, "Rear Left Wheel")
        if rear is not None:
            rim_in = _attnum(rear, "rim diameter", 13.0)
            width_mm = _attnum(rear, "tire width", 300.0)
            hw_ratio = _attnum(rear, "tire height-width ratio", 0.5)
            tyre_radius = (rim_in * 0.0254) / 2.0 + (width_mm / 1000.0) * hw_ratio

        # Fail loud if the XML did not yield a usable model -- callers
        # (load_configured) catch this and fall back to the rpm/speed gearbox,
        # instead of silently building a 0-gear car that selects gear 0.
        if not ratios:
            raise ValueError(f"No forward gear ratios parsed from {car_xml}")
        if len(rpms) < 2:
            raise ValueError(f"No engine torque curve parsed from {car_xml}")
        if not (tyre_radius > 0 and final_drive > 0 and rev_limiter > 0):
            raise ValueError(f"Bad driveline scalars parsed from {car_xml}")

        return cls(name, ratios, final_drive, tyre_radius, rpms, tqs,
                   rev_limiter, tickover, effs)

    @classmethod
    def load_configured(cls, torcs_root, raceman="practice"):
        """Resolve + load the car configured for the given raceman. None on failure."""
        name, path = resolve_configured_car(torcs_root, raceman)
        if path is None:
            return None
        try:
            return cls.from_xml(path, name)
        except Exception:
            return None

    def summary(self):
        return (f"CarSpec {self.name}: {self.n_gears} gears {self.ratios}, "
                f"final {self.final_drive}, tyre r={self.tyre_radius:.3f} m, "
                f"limiter {self.rev_limiter:.0f}, idle {self.tickover:.0f}, "
                f"shift@{self.shift_ceiling():.0f}rpm, "
                f"upshift km/h {self.upshift_speeds_kmh()[:-1]}")
