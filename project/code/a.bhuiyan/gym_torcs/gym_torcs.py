import numpy as np
try:
    from gym import spaces
except ModuleNotFoundError:
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.shape = shape if shape is not None else self.low.shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _Spaces:
        Box = _Box

    spaces = _Spaces()
# from os import path
import snakeoil3_gym as snakeoil3
import copy
import csv
import math
import collections as col
import os
import subprocess
import time

from racing_line import RacingLine


def summarize_track_sensors(track_values):
    track = np.asarray(track_values, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return {
            "left_mean": 0.0,
            "center_mean": 0.0,
            "right_mean": 0.0,
            "curve_hint": 0.0,
            "center_bias": 0.0,
        }

    third = max(1, track.size // 3)
    left = track[:third]
    center = track[third: track.size - third]
    right = track[track.size - third:]

    left_mean = float(left.mean()) if left.size else 0.0
    center_mean = float(center.mean()) if center.size else float(track.mean())
    right_mean = float(right.mean()) if right.size else 0.0
    curve_hint = float(np.clip((right_mean - left_mean) / 200.0, -1.0, 1.0))
    center_bias = float(np.clip((center_mean - 0.5 * (left_mean + right_mean)) / 200.0, -1.0, 1.0))

    return {
        "left_mean": left_mean,
        "center_mean": center_mean,
        "right_mean": right_mean,
        "curve_hint": curve_hint,
        "center_bias": center_bias,
    }


def track_heading_hint(track_values):
    track = np.asarray(track_values, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return 0.0

    sensor_angles = np.deg2rad(np.linspace(-45.0, 45.0, track.size, dtype=np.float32))
    x = track * np.cos(sensor_angles)
    y = track * np.sin(sensor_angles)
    heading = float(np.arctan2(y.sum(), max(x.sum(), 1e-6)))

    # Ignore tiny asymmetries on straights so they do not create a constant steer bias.
    straight_confidence = float(np.clip(abs(y.sum()) / max(track.sum(), 1e-6), 0.0, 1.0))
    if straight_confidence < 0.03:
        return 0.0

    return float(np.clip(heading / np.deg2rad(45.0), -1.0, 1.0))


def curve_proximity_factor(track_values):
    track = np.asarray(track_values, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return 0.0

    center_distance = float(track[track.size // 2])
    return float(np.clip((120.0 - center_distance) / 90.0, 0.0, 1.0))


class TorcsEnv:
    terminal_judge_start = 150  # End obviously bad episodes earlier during steering-only training.
    termination_limit_progress = 5  # [km/h], treat low progress as a failed episode sooner.
    default_speed = 50
    steer_smoothing = 0.0
    steer_rate_limit = 1.0
    # --- racing-line mode (active only when a racing_line_path is loaded) ---
    rl_progress_gain = 0.10         # reward for speed projected along the line
    rl_cross_track_gain = 0.50      # penalty per metre off the racing line
    # Speed shaping is ASYMMETRIC on purpose: a symmetric (vmax-speed)^2 penalty
    # made "creep slowly" the global minimum (huge flat negative reward the agent
    # could not climb out of). Now going fast is rewarded by the progress term and
    # only OVERSPEED (faster than the corner allows) is punished quadratically;
    # being under target gets a mild linear nudge. Net: "go fast, learn to brake
    # from the overspeed penalty + off-track crashes."
    rl_overspeed_gain = 0.05        # quadratic penalty per (speed-vmax)^2 over the line speed
    rl_underspeed_gain = 0.01       # mild LINEAR nudge per (vmax-speed) when below target
    rl_clear_speed_gain = 0.08      # bonus per (m/s) when the track ahead is clear (no corner)
    rl_speed_gain = 0.0025          # (legacy symmetric speed penalty; unused in racing reward)
    rl_heading_gain = 0.10          # penalty on |heading error| vs the line  [rad^-1]
    rl_edge_threshold = 0.70        # |trackPos| beyond which proactive edge penalty starts
    rl_edge_penalty_gain = 4.0      # penalty per unit |trackPos| over the threshold
    rl_steer_smooth_gain = 0.05     # penalty on per-step steer change (anti-jerk)
    rl_pedal_smooth_gain = 0.02     # penalty on per-step accel/brake change (anti-jerk)
    rl_steer_rate_limit = 0.15      # max |steer| change per step in racing mode
    rl_steer_speed_falloff = 0.004  # steering authority lost per km/h over base speed
    rl_steer_base_speed = 40.0      # km/h below which full steering is allowed
    rl_steer_min_authority = 0.45   # floor on steering authority at high speed
    rl_brake_horizon = 200.0        # m look-ahead window for the worst upcoming speed
    # --- baseline path-follower (residual RL) ---
    # A deterministic controller derives steer/accel/brake from the precomputed
    # racing line every step; the PPO policy only adds a bounded RESIDUAL on top.
    # This reproduces the known-good regime (hug the line, ~0 heading error, hold
    # the target speed) from step 1 -> kills exploration randomness, escapes the
    # slow do-nothing local optimum, and keeps the car ON the gas (speed climbs
    # toward vmax instead of coasting down). Disable with baseline_assist=False.
    baseline_assist = True
    baseline_steer_kpsi = 2.5       # steer per rad of heading error vs the line
    baseline_steer_ky = 0.35        # steer per (e_y / half_width) cross-track error
    baseline_steer_kff = 6.0        # curvature feed-forward (steer per kappa_rl)
    baseline_accel_band = 6.0       # m/s under target mapping to full throttle
    baseline_brake_band = 5.0       # m/s over target mapping to full brake
    baseline_brake_decel = 6.0      # m/s^2 assumed decel for look-ahead safe speed
    baseline_speed_margin = 1.0     # m/s deadband around the target speed
    baseline_lookahead_gain = 4.0   # braking horizon (m) per m/s of current speed
    baseline_lookahead_min = 60.0   # m floor on the speed-scaled look-ahead
    baseline_lookahead_max = 250.0  # m cap on the speed-scaled look-ahead
    residual_steer_scale = 0.5      # agent steer authority added to the baseline
    residual_accel_scale = 0.5      # agent accel authority added to the baseline
    residual_brake_scale = 0.5      # agent brake authority added to the baseline
    # Straight-line calm: on a clear straight the baseline zeroes tiny steer
    # corrections (deadband) and drops the curvature feed-forward, so the car
    # stops micro-jerking when it has nothing to correct.
    straight_kappa_threshold = 0.004  # |kappa| below this (1/m) ahead == "straight"
    straight_steer_deadband = 0.02    # |steer| below this on a straight -> 0
    # Corner shaping: brake EARLY for the worst speed in the look-ahead (slow
    # entry) and add throttle when the corner is opening up ahead (fast exit).
    corner_exit_accel_gain = 4.0      # extra throttle per (kappa_now - kappa_ahead)
    corner_exit_preview = 30.0        # m ahead used to detect an opening corner
    # Smooth return to mid-lane: once the road is clear for the whole settle
    # preview (no corner being set up), pull the steering's TARGET offset toward
    # center so the car stops hugging the corner-exit edge and lines up straight
    # and stable before the next braking zone. Gated on a clear look-ahead, so it
    # never flattens a real corner setup (which always has a corner inside the
    # preview). Steering only -- the speed profile / reward still use the true line.
    center_settle_gain = 0.5          # fraction of target offset pulled to center
    center_settle_preview = 60.0      # m of clear road required before settling
    # Hill logic from the SCR z-speed: climbing bleeds speed (add throttle, the
    # car can also carry more), descending adds speed (trim the target, brake
    # earlier). slope ~= speedZ / speedX.
    hill_accel_gain = 1.5             # throttle added per unit upslope
    hill_downhill_speed_trim = 0.6    # target-speed fraction shaved per unit downslope
    # Smooth gearbox: hysteresis (separate up/down speeds) + a minimum hold time
    # stop the box hunting at band edges (the GUI "jerk"). Uses all six gears.
    # Shift points from car1-trb1's REAL driveline (gear ratios
    # 3.0/1.9/1.4/1.1/0.9/0.77, final drive 4.5, rear tyre r=0.328 m, torque
    # peak 8000 rpm, rev limiter 9150). The torque curve rises all the way to
    # 8000 and the gear steps are wide, so the wheel-force crossover (the truly
    # optimal upshift) sits at each gear's limiter speed. The OLD bands shifted
    # at ~6750 rpm -> short-shifted out of the power band and gutted corner-exit
    # acceleration ("gear ratio not good"). Upshift near redline instead;
    # downshifts land ~8000 rpm in the lower gear with a wide hysteresis gap so
    # the box never hunts. (2->1 at 51 keeps the ~46 km/h hairpins in 1st for a
    # punchy exit.)
    # --- rpm-based dynamic gearbox (fraction-of-redline, car-agnostic) ---
    # Shift on engine rpm, NOT a per-car speed table. Thresholds are FRACTIONS of
    # the rev limiter so they self-calibrate to any car; the actual limiter is also
    # auto-learned at runtime (max rpm ever seen) in case the default is wrong.
    # Configured car = car1-ow1 (scr_server idx 0): 7 gears
    # (ratios 3.9/2.9/2.3/1.87/1.68/1.54/1.46, final 4.5, rear r=0.315 m),
    # limiter 18700, tickover 5000, torque PLATEAU 340-360 N.m from 9k-18k rpm,
    # peak power ~18000. So: upshift ~0.95*limiter (rev it out to peak power),
    # downshift below ~0.64*limiter (stay on the torque plateau -> strong corner
    # exit + "keep revs high on braking"). Hysteresis is safe: the widest ratio
    # step is 1->2 = 1.345, and 0.64*1.345 = 0.861 < 0.95, so after a downshift
    # the engine lands below the upshift point -> no hunting. A hold timer adds
    # extra anti-hunt margin. NOTE the OLD bug: thresholds were tuned for a
    # 9150-limiter car, so gear_down_rpm=4600 sat BELOW this car's 5000 idle ->
    # downshift could never fire -> the box got stuck in a tall top gear at low
    # rpm with no torque ("stuck at 6 / not speed efficient").
    gear_use_rpm = True               # False -> fall back to the speed bands below
    gear_max = 7                      # forward gears on car1-ow1
    gear_redline_rpm = 18700.0        # rev limiter (auto-raised if a higher rpm is seen)
    gear_up_frac = 0.95               # upshift at this fraction of redline (peak power)
    gear_down_frac = 0.64             # downshift below this fraction (torque plateau)
    gear_min_speed = 5.0              # km/h below which gear is forced to 1
    gear_hold_steps = 6               # min steps between shifts (anti-hunt)
    # Legacy speed-band fallback (gear_use_rpm=False or rpm missing), recomputed for
    # car1-ow1 from the same up/down fractions so it agrees with the rpm logic.
    gear_up_speeds = (120.0, 162.0, 204.0, 251.0, 279.0, 305.0)   # km/h, UP into 2..7
    gear_down_speeds = (109.0, 137.0, 169.0, 188.0, 205.0, 216.0)  # km/h, DOWN into 1..6
    # Self-learning corner memory: a per-node speed-scale field along the line,
    # decayed (with an interpolated kernel) around every spot the car goes
    # off-track so the baseline enters repeat offenders slower. It is now a
    # TEMP (in-RAM) field during a run -- committed to disk ONLY when the trainer
    # saves a new best checkpoint (commit_corner_memory()), so a bad run's panic
    # slowdowns are never baked in permanently. With the braking-envelope fix
    # above, genuine crashes are rare, so the gains are a light touch (high
    # floor, fast recovery) instead of the old sledgehammer that left every
    # corner crawling (floor 0.55 -> the "cornering too slow" symptom).
    corner_memory_enabled = True
    corner_memory_decay = 0.93        # multiply scale by this at the crash apex
    corner_memory_window = 45.0       # m upstream of the crash the decay spreads over
    corner_memory_floor = 0.80        # never scale a corner below this fraction
    corner_memory_recover = 0.010     # per-crash nudge of all nodes back toward 1.0
    # Two-way + generalising corner memory. The field used to only ever go DOWN
    # (slow a corner after a crash); now a corner that is taken CLEANLY at/near
    # its current target speed is nudged UP toward corner_memory_cap, so the
    # baseline reclaims speed where it has proven safe. Every up/down nudge also
    # bleeds a fraction (corner_memory_cluster) onto all nodes of SIMILAR
    # curvature, so one good/bad corner teaches its look-alikes -> the policy
    # converges faster across the repeated corner families on a lap.
    corner_memory_cap = 1.12          # never scale a corner above this fraction
    corner_memory_boost = 0.015       # up-nudge rate toward cap on a clean pass
    corner_memory_cluster = 0.25      # share of a nudge applied to similar-kappa nodes
    corner_kappa_min = 0.004          # |kappa| (1/m) above which a node counts as a corner
    corner_clean_margin = 0.60        # |trackPos| below this = a clean pass
    corner_clean_speed_frac = 0.85    # only boost if speed >= this * scaled target
    progress_reward_scale = 20.0
    track_pos_penalty_gain = 1.0
    track_pos_nonlinear_gain = 0.6
    track_pos_edge_threshold = 0.3
    track_pos_edge_penalty_gain = 0.8
    # Off-track is no longer an instant kill: a car that clips a corner but is
    # pointing the right way gets a few frames to steer back before the episode
    # (and the visible TORCS grid restart) ends. Only a DEEP excursion or running
    # off for too long terminates.
    off_track_grace_steps = 20      # consecutive off-track frames tolerated before ending
    off_track_hard_limit = 1.5      # |trackPos| past which recovery is hopeless -> end now
    off_track_step_penalty = 1.0    # per-frame penalty while off track (vs the terminal hit)
    angle_penalty_gain = 1.0
    steer_smoothness_penalty_gain = 0.01
    steer_magnitude_penalty = 0.01
    angle_improvement_gain = 0.3
    track_pos_improvement_gain = 0.3
    damage_penalty = 3.0
    off_track_penalty = 25.0
    launch_speed_threshold = 30.0
    launch_min_accel = 0.35
    brake_enable_speed = 20.0
    # High-speed launch warmup: each episode, spool the car up with full throttle
    # (agent-free, not added to the rollout) until it reaches launch_warmup_target
    # km/h, so PPO starts every episode already moving fast instead of from a
    # standstill. Only active in throttle/racing-line mode.
    launch_warmup_enabled = True
    launch_warmup_target = 90.0     # km/h to reach before handing control to agent
    launch_warmup_max_steps = 150   # safety cap on warmup steps
    server_wait_loops = 5
    reset_connect_attempts = 3
    client_create_attempts = 5
    client_create_retry_delay = 1.0
    reconnect_each_episode = True

    initial_reset = True
    reset_observation_attempts = 50


    def __init__(
        self,
        vision=False,
        throttle=False,
        gear_change=False,
        port=3001,
        auto_start=None,
        torcs_command=None,
        kill_on_shutdown=False,
        debug=False,
        debug_interval=25,
        racing_line_path=None,
        launch_warmup_enabled=None,
        launch_warmup_target=None,
        baseline_assist=None,
        residual_scale=None,
        telemetry=False,
    ):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.port = port
        if auto_start is None:
            auto_start = (os.name != "nt")
        self.auto_start = auto_start
        self.kill_on_shutdown = kill_on_shutdown
        self.torcs_command = self._resolve_torcs_command(torcs_command)
        self.debug = debug
        self.debug_interval = max(1, int(debug_interval))

        self.racing_line = None
        self._racingline_path = racing_line_path
        if racing_line_path:
            self.racing_line = RacingLine.load(racing_line_path)

        if launch_warmup_enabled is not None:
            self.launch_warmup_enabled = bool(launch_warmup_enabled)
        if launch_warmup_target is not None:
            self.launch_warmup_target = float(launch_warmup_target)
        if baseline_assist is not None:
            self.baseline_assist = bool(baseline_assist)
        if residual_scale is not None:
            self.residual_steer_scale = float(residual_scale)
            self.residual_accel_scale = float(residual_scale)
            self.residual_brake_scale = float(residual_scale)

        self.initial_run = True
        self.client = None
        self.pending_reset = False
        self.last_applied_steer = 0.0
        self.last_applied_accel = 0.0
        self.last_applied_brake = 0.0
        self.previous_reward_steer = 0.0
        self._off_track_steps = 0
        self._last_gear = 1
        self._gear_hold = 0
        self._rpm_redline = float(self.gear_redline_rpm)  # auto-learned at runtime

        # Data-driven car model: driveline + engine torque map parsed from the
        # configured car's own XML (resolved practice.xml -> scr_server.xml -> car).
        # Gears are then chosen by the real wheel-force shift map, not constants.
        self.car_spec = None
        try:
            from car_spec import CarSpec
            torcs_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "torcs"))
            self.car_spec = CarSpec.load_configured(torcs_root)
            if self.car_spec is not None:
                self.gear_max = self.car_spec.n_gears
                self.gear_redline_rpm = self.car_spec.rev_limiter
                self._rpm_redline = self.car_spec.rev_limiter
                print(self.car_spec.summary())
        except Exception as exc:  # pragma: no cover
            print(f"(car spec load failed, using fallback gearbox: {exc})")

        # Per-step telemetry recorder: dumps a wide row every step so runs can be
        # clustered + analysed offline (analyze_telemetry.py). Lazy-opened.
        self.telemetry = bool(telemetry)
        self._telemetry_writer = None
        self._telemetry_file = None
        self._telemetry_path = None
        self._episode_idx = 0

        # Self-learning corner-speed memory (loaded from / saved next to the
        # line). Temp in-RAM during a run; persisted only on a best checkpoint.
        self.corner_scale = None
        self._corner_memory_path = None
        self._corner_memory_dirty = False
        self._kappa_bin = None
        self._kappa_bin_members = {}
        if self.racing_line is not None and self.corner_memory_enabled:
            self._load_corner_memory()
            self._build_kappa_clusters()

        if self.auto_start:
            self.reset_torcs()

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(
            p=self.port,
            vision=self.vision,
            auto_relaunch=False,
            torcs_command=self.torcs_command,
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            action_dim = 3
            if gear_change:
                action_dim += 1
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        # Snapshot the previously-applied controls before they are overwritten,
        # so the racing-line reward can penalise per-step input jerk.
        prev_applied_steer = self.last_applied_steer
        prev_applied_accel = self.last_applied_accel
        prev_applied_brake = self.last_applied_brake

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        action_torcs['meta'] = False

        # Baseline path-follower (residual RL): derive a deterministic
        # steer/accel/brake from the racing line at the CURRENT (pre-step) obs,
        # then add the agent's bounded residual. None unless a line is loaded
        # and assist is on -> falls back to pure-agent control.
        baseline = None
        if self.racing_line is not None and self.baseline_assist:
            baseline = self._baseline_action(client.S.d)

        # Steering (speed-scaled authority + rate limit in racing-line mode)
        if baseline is not None:
            desired_steer = baseline['steer'] + self.residual_steer_scale * this_action['steer']
        else:
            desired_steer = this_action['steer']
        action_torcs['steer'] = self._apply_steer_constraints(
            desired_steer, float(client.S.d['speedX']))

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            if baseline is not None:
                accel = baseline['accel'] + self.residual_accel_scale * this_action['accel']
                brake = baseline['brake'] + self.residual_brake_scale * this_action['brake']
            else:
                accel = this_action['accel']
                brake = this_action['brake']
            accel = float(np.clip(accel, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))

            action_torcs['accel'] = accel
            if client.S.d['speedX'] < self.brake_enable_speed:
                action_torcs['brake'] = 0.0
            else:
                action_torcs['brake'] = brake

            if client.S.d['speedX'] < self.launch_speed_threshold:
                action_torcs['accel'] = max(action_torcs['accel'], self.launch_min_accel)
                action_torcs['brake'] = 0.0

            # Store the EXECUTED pedals (baseline + residual, post-clip) so the
            # next step's jerk penalty and the prevApplied* state use what really
            # ran, not the agent's raw residual.
            self.last_applied_accel = float(action_torcs['accel'])
            self.last_applied_brake = float(action_torcs['brake'])

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
            if client.S.d['speedX'] < self.launch_speed_threshold:
                action_torcs['gear'] = 1
        else:
            #  Automatic speed-based gear change. With throttle enabled the agent
            #  controls accel/brake, so the car must shift up or it redlines in
            #  first gear and never reaches racing-line speeds.
            if self.throttle:
                action_torcs['gear'] = self._gear_select(
                    float(client.S.d['speedX']), float(client.S.d.get('rpm', 0.0)))
            else:
                action_torcs['gear'] = 1

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input(max_wait_loops=self.server_wait_loops)

        # Get the current full-observation from torcs
        obs = client.S.d
        obs['prevAppliedSteer'] = self.last_applied_steer
        obs['prevAppliedAccel'] = self.last_applied_accel
        obs['prevAppliedBrake'] = self.last_applied_brake

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        track = np.array(obs['track'], dtype=np.float32)
        speed = float(obs['speedX'])
        angle = float(obs['angle'])
        track_pos = float(obs['trackPos'])
        speed_delta = speed - float(obs_pre['speedX'])
        prev_angle = float(obs_pre['angle'])
        prev_track_pos = float(obs_pre['trackPos'])
        heading_hint = track_heading_hint(track)
        prev_heading_hint = track_heading_hint(obs_pre.get('track', []))
        curve_proximity = curve_proximity_factor(track)

        progress = speed * np.cos(angle)
        rl_metrics = None
        if self.racing_line is not None:
            reward, rl_metrics = self._racingline_reward(
                obs,
                applied_steer=float(action_torcs['steer']),
                applied_accel=float(action_torcs.get('accel', 0.0)),
                applied_brake=float(action_torcs.get('brake', 0.0)),
                prev_steer=prev_applied_steer,
                prev_accel=prev_applied_accel,
                prev_brake=prev_applied_brake,
            )
        else:
            reward = progress / self.progress_reward_scale
            abs_track_pos = abs(track_pos)
            abs_prev_track_pos = abs(prev_track_pos)
            edge_excess = max(0.0, abs_track_pos - self.track_pos_edge_threshold)
            reward -= self.track_pos_penalty_gain * abs_track_pos
            reward -= self.track_pos_nonlinear_gain * (abs_track_pos ** 1.5)
            reward -= self.track_pos_edge_penalty_gain * edge_excess
            reward -= self.angle_penalty_gain * abs(angle)
            reward += self.angle_improvement_gain * (abs(prev_angle) - abs(angle))
            reward += self.track_pos_improvement_gain * (abs_prev_track_pos - abs_track_pos)
            reward -= (
                self.steer_smoothness_penalty_gain
                * abs(float(action_torcs['steer']) - self.previous_reward_steer)
            )
            reward -= self.steer_magnitude_penalty * abs(float(action_torcs['steer']))
            self.previous_reward_steer = float(action_torcs['steer'])

        self.last_u = np.array(u, copy=True)

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward -= self.damage_penalty

        # Termination judgement #########################
        # Termination judgement #########################
        episode_terminate = False
        termination_reason = None
        crash_distance = None

        # Off-track handling with a recovery grace. A brief clip is penalised per
        # frame but the car keeps driving so it can steer back; only a DEEP
        # excursion (|trackPos| past the hard limit) or staying off for longer
        # than the grace window ends the episode + triggers the grid restart.
        if track.min() < 0:
            self._off_track_steps += 1
            reward -= self.off_track_step_penalty
            hard_off = abs(track_pos) > self.off_track_hard_limit
            if hard_off or self._off_track_steps > self.off_track_grace_steps:
                reward -= self.off_track_penalty
                episode_terminate = True
                termination_reason = "off_track"
                client.R.d['meta'] = True
                crash_distance = float(obs.get('distFromStart', 0.0))
                # Self-learning: remember this corner was taken too hot so the
                # baseline enters it slower next time (interpolated decay).
                self._register_offtrack(crash_distance)
        else:
            self._off_track_steps = 0
            # Two-way corner memory: a corner cleared cleanly at speed earns a
            # small speed-scale boost (and teaches its curvature look-alikes).
            if self.racing_line is not None and self.corner_scale is not None:
                self._register_clean_pass(
                    float(obs.get('distFromStart', 0.0)), track_pos, speed / 3.6)

        if self.terminal_judge_start < self.time_step and not episode_terminate:
            if progress < self.termination_limit_progress:
                reward -= 5.0
                episode_terminate = True
                termination_reason = "low_progress"
                client.R.d['meta'] = True

        if np.cos(angle) < 0 and not episode_terminate:  # Episode is terminated if the agent runs backward
            reward = -10.0
            episode_terminate = True
            termination_reason = "backward"
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()
            self.pending_reset = True

        if client.R.d['meta'] is True and termination_reason is None:
            termination_reason = "meta_reset"

        self._maybe_debug_step(
            raw_action=u,
            applied_action=copy.deepcopy(action_torcs),
            obs_pre=obs_pre,
            obs=obs,
            reward=reward,
            heading_hint=heading_hint,
            prev_heading_hint=prev_heading_hint,
            curve_proximity=curve_proximity,
            heading_error=angle,
            heading_error_improvement=speed_delta,
        )

        self._telemetry_write(
            raw_action=u,
            applied=action_torcs,
            obs=obs,
            baseline=baseline,
            reward=reward,
            off_track=bool(track.min() < 0),
            termination_reason=termination_reason,
            crash_s=crash_distance,
        )

        self.time_step += 1

        info = {
            "termination_reason": termination_reason,
            "crash_s": crash_distance,
        }
        if rl_metrics is not None:
            info.update(rl_metrics)

        return self.get_obs(), reward, client.R.d['meta'], info

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0
        self._episode_idx += 1

        if self.client is not None and self.initial_reset is not True:
            if self.client.so is not None and not self.pending_reset:
                self.client.R.d['meta'] = True
                self.client.respond_to_server()
                self.pending_reset = True
                time.sleep(0.2)

            if self.reconnect_each_episode and self.client.so is not None:
                self.client.shutdown()
                time.sleep(0.5)

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True and self.auto_start:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        if (
            self.client is None
            or self.client.so is None
            or self.reconnect_each_episode
            or (relaunch is True and self.auto_start)
        ):
            # Modify here if you use multiple tracks in the environment
            self.client = self._create_client()

        client = self.client
        client.R = snakeoil3.DriverAction()
        client.R.d['meta'] = False
        obs = self._wait_for_valid_reset_state(client)

        self.last_u = None
        self.last_applied_steer = 0.0
        self.last_applied_accel = 0.0
        self.last_applied_brake = 0.0
        self.previous_reward_steer = 0.0
        self._off_track_steps = 0
        self._last_gear = 1
        self._gear_hold = 0

        # Spool up to launch speed before the agent takes over (no-op unless in
        # throttle/racing-line mode). Seeds last_applied_* with the final inputs.
        obs = self._launch_warmup(client)
        obs['prevAppliedSteer'] = self.last_applied_steer
        obs['prevAppliedAccel'] = self.last_applied_accel
        obs['prevAppliedBrake'] = self.last_applied_brake
        self.observation = self.make_observaton(obs)

        self.initial_reset = False
        self.pending_reset = False
        return self.get_obs()

    def end(self):
        if self._telemetry_file is not None:
            try:
                self._telemetry_file.flush()
                self._telemetry_file.close()
            except Exception:  # pragma: no cover
                pass
            self._telemetry_file = None
        if self.kill_on_shutdown:
            self._stop_torcs_processes()

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        if not self.auto_start:
            return
        self._stop_torcs_processes()
        time.sleep(0.5)
        launch_command = self.torcs_command
        if launch_command is None:
            if os.name == "nt":
                raise RuntimeError(
                    "Automatic start on Windows requires torcs_command or TORCS_COMMAND/TORCS_PATH."
                )
            launch_command = ['torcs', '-nofuel', '-nodamage', '-nolaptime']
            if self.vision is True:
                launch_command.append('-vision')
        subprocess.Popen(launch_command)
        time.sleep(0.5)
        if os.name != "nt":
            autostart_path = os.path.join(os.path.dirname(__file__), 'autostart.sh')
            subprocess.Popen(['sh', autostart_path])
            time.sleep(0.5)

    def _resolve_torcs_command(self, torcs_command):
        if torcs_command:
            return self._normalize_command(torcs_command)

        env_command = os.environ.get("TORCS_COMMAND")
        if env_command:
            return self._normalize_command(env_command)

        torcs_path = os.environ.get("TORCS_PATH")
        if torcs_path:
            return self._normalize_command([torcs_path])

        return None

    def _normalize_command(self, command):
        if isinstance(command, str):
            return [command]
        return list(command)

    def _stop_torcs_processes(self):
        if os.name == "nt":
            subprocess.run(
                ['taskkill', '/IM', 'wtorcs.exe', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            subprocess.run(
                ['taskkill', '/IM', 'torcs.exe', '/F'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        subprocess.run(
            ['pkill', 'torcs'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _wait_for_valid_reset_state(self, client):
        last_error = None
        for _ in range(self.reset_connect_attempts):
            latest_obs = client.S.d
            try:
                for _ in range(self.reset_observation_attempts):
                    client.get_servers_input(max_wait_loops=self.server_wait_loops)
                    latest_obs = client.S.d
                    track = np.array(latest_obs.get('track', []), dtype=np.float32)
                    angle = float(latest_obs.get('angle', 0.0))
                    if track.size == 0:
                        time.sleep(0.02)
                        continue
                    if track.min() >= 0 and np.cos(angle) > 0:
                        return latest_obs
                    time.sleep(0.02)
                return latest_obs
            except (TimeoutError, ConnectionError) as error:
                last_error = error
                if client.so is not None:
                    client.shutdown()
                client = self._create_client()
                self.client = client
                client.R = snakeoil3.DriverAction()
                client.R.d['meta'] = False
        raise ConnectionError("Could not obtain a valid reset state from TORCS.") from last_error

    def _create_client(self):
        last_error = None
        for attempt_idx in range(self.client_create_attempts):
            try:
                client = snakeoil3.Client(
                    p=self.port,
                    vision=self.vision,
                    auto_relaunch=False,
                    torcs_command=self.torcs_command,
                )  # Open new UDP in vtorcs
                client.MAX_STEPS = np.inf
                return client
            except ConnectionError as error:
                last_error = error
                if attempt_idx < self.client_create_attempts - 1:
                    wait_seconds = self.client_create_retry_delay * (attempt_idx + 1)
                    print(
                        f"TORCS server on port {self.port} not ready yet. "
                        f"Retrying in {wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
        raise last_error

    def _launch_warmup(self, client):
        """Spool the car up to launch_warmup_target km/h with full throttle before
        the agent takes over, so every episode begins at speed. Agent-free and
        NOT added to the rollout. Uses a gentle proportional centering controller
        to hold the start straight; bails early if it leaves the track. Returns
        the latest raw obs dict (client.S.d)."""
        if not (self.launch_warmup_enabled and self.throttle):
            return client.S.d
        target = self.launch_warmup_target
        for _ in range(self.launch_warmup_max_steps):
            speed = float(client.S.d.get('speedX', 0.0))
            if speed >= target:
                break
            track = np.array(client.S.d.get('track', []), dtype=np.float32)
            if track.size and track.min() < 0:
                break  # already off track -> let reward/termination handle it
            angle = float(client.S.d.get('angle', 0.0))
            track_pos = float(client.S.d.get('trackPos', 0.0))
            # Hold center: steer to null heading + lateral offset (small authority).
            steer = float(np.clip(angle * 3.0 - track_pos * 0.3, -0.4, 0.4))
            client.R.d['steer'] = steer
            client.R.d['accel'] = 1.0
            client.R.d['brake'] = 0.0
            # Use the live rpm gearbox during warmup so the gear/limiter state is
            # already correct when the agent takes over (no wrong-car seed).
            client.R.d['gear'] = self._gear_select(
                speed, float(client.S.d.get('rpm', 0.0)))
            client.R.d['meta'] = False
            client.respond_to_server()
            client.get_servers_input(max_wait_loops=self.server_wait_loops)
        # Seed the "previously applied" controls so the first agent step's rate
        # limit and jerk penalty are computed against the warmup's final inputs.
        self.last_applied_steer = float(client.R.d.get('steer', 0.0))
        self.last_applied_accel = float(client.R.d.get('accel', 0.0))
        self.last_applied_brake = float(client.R.d.get('brake', 0.0))
        self.previous_reward_steer = self.last_applied_steer
        # _last_gear is already correct from the live _gear_select calls in the
        # warmup loop; just clear the hold so the first agent step can shift freely.
        self._gear_hold = 0
        return client.S.d

    def _gear_for_speed(self, speed):
        """Coarse speed->gear from the fallback bands (used for the debug
        target_gear readout). Generic over gear count."""
        gear = 1
        for threshold in self.gear_up_speeds:
            if speed >= threshold:
                gear += 1
            else:
                break
        return min(gear, self.gear_max)

    def _baseline_action(self, raw_obs):
        """Deterministic path-follower from the racing line (residual-RL base).

        Steer = heading-error + cross-track + curvature feed-forward toward the
        line. Pedals = P-controller onto a speed target that is the local vmax
        capped by what can still be scrubbed off for the worst corner inside a
        speed-scaled look-ahead (faster -> looks further). Returns a dict of
        steer/accel/brake plus the tracking errors for debugging. Signs follow
        the proven launch-warmup convention (+steer = left, +trackPos = left)."""
        line = self.racing_line
        s = float(raw_obs.get('distFromStart', 0.0))
        q = line.query(s)
        speed_ms = float(raw_obs.get('speedX', 0.0)) / 3.6
        angle = float(raw_obs.get('angle', 0.0))
        track_pos = float(raw_obs.get('trackPos', 0.0))

        # Smooth return to mid-lane: when the road is clear for the whole settle
        # preview (genuinely between corners), pull the steering target offset
        # toward center so the car settles to the middle lane on exits instead of
        # hugging the edge it apexed on. A real corner setup keeps a corner inside
        # the preview -> gate stays closed -> the setup line is untouched.
        target_offset = q['offset']
        if line.max_abs_kappa_ahead(s, self.center_settle_preview) < self.straight_kappa_threshold:
            target_offset *= (1.0 - self.center_settle_gain)

        # Tracking errors vs the line (+e_y = car is LEFT of the line).
        e_y = track_pos * line.half_width - target_offset
        e_psi = math.atan2(
            math.sin(angle - q['dpsi']), math.cos(angle - q['dpsi']))

        # Curvature ahead: feed-forward, plus entry/exit detection (kappa now vs
        # a short look-ahead). |kappa| small both here and ahead -> a straight.
        kappa_now = q['kappa_rl']
        kappa_ahead = line.query(s + self.corner_exit_preview)['kappa_rl']
        is_straight = (abs(kappa_now) < self.straight_kappa_threshold
                       and abs(kappa_ahead) < self.straight_kappa_threshold)

        steer = (
            self.baseline_steer_kpsi * e_psi
            - self.baseline_steer_ky * (e_y / line.half_width)
        )
        if not is_straight:
            steer += self.baseline_steer_kff * kappa_now
        steer = float(np.clip(steer, -1.0, 1.0))
        # Straight-line calm: kill tiny twitch when there is nothing to correct.
        if is_straight and abs(steer) < self.straight_steer_deadband:
            steer = 0.0

        # Speed target: local cap, limited by a proper per-distance braking
        # envelope so the car is already slow enough on ENTRY (slow-in). The
        # look-ahead grows with speed ("sensors reach further/faster").
        horizon = float(np.clip(
            speed_ms * self.baseline_lookahead_gain,
            self.baseline_lookahead_min,
            self.baseline_lookahead_max,
        ))
        # baseline_brake_decel is deliberately BELOW the speed profile's build
        # decel (a_brake=8) -> a real braking margin. The OLD formula
        #   sqrt(min_vmax_ahead^2 + 2*decel*horizon)
        # assumed the full horizon to reach the worst corner, so v_safe was
        # always >= vmax and the car never actually braked early: it arrived hot
        # at the corkscrew hairpin, overshot the left apex and was flung wide
        # right into the following right-hander (the "touches the right line"
        # crash that killed every lap). safe_speed() brakes for the corner at
        # its true distance instead.
        v_safe = line.safe_speed(s, self.baseline_brake_decel, horizon)
        v_target = min(q['vmax'], v_safe)

        # Hill logic from the OFFLINE grade profile (track XML), not the noisy
        # speedZ sensor: slope = +uphill / -downhill at the car's current s. The
        # speed PROFILE already folds grade into vmax/braking, so the runtime no
        # longer trims v_target for the local descent (that would double-count).
        # We still (a) brake a touch earlier when a STEEP descent looms just ahead
        # (grade lag/controller margin) and (b) add throttle uphill below. Falls
        # back to speedZ for legacy lines generated before grade existed.
        if line.has_grade:
            slope = line.grade_at(s)
            slope_ahead = line.min_grade_ahead(s, horizon)
            if slope_ahead < -0.04:  # notable descent within the braking horizon
                v_target *= max(0.6, 1.0 + self.hill_downhill_speed_trim * slope_ahead)
        else:
            speed_z = float(raw_obs.get('speedZ', 0.0))
            slope = speed_z / max(abs(float(raw_obs.get('speedX', 0.0))), 1.0)
            if slope < 0.0:
                v_target *= max(0.4, 1.0 + self.hill_downhill_speed_trim * slope)

        # Self-learning corner memory: enter known-bad corners slower.
        if self.corner_scale is not None:
            v_target *= float(self.corner_scale[line._index(s)])

        err = v_target - speed_ms
        if err > self.baseline_speed_margin:
            accel = float(np.clip(err / self.baseline_accel_band, 0.0, 1.0))
            brake = 0.0
        elif err < -self.baseline_speed_margin:
            accel = 0.0
            brake = float(np.clip(-err / self.baseline_brake_band, 0.0, 1.0))
        else:
            accel = 0.0
            brake = 0.0

        # Fast exit: corner opening ahead (curvature dropping) -> get on the gas
        # early instead of waiting for the speed error to grow.
        if not is_straight and abs(kappa_now) - abs(kappa_ahead) > 0.0 and brake == 0.0:
            accel = float(np.clip(
                accel + self.corner_exit_accel_gain * (abs(kappa_now) - abs(kappa_ahead)),
                0.0, 1.0))

        # Uphill: add throttle to fight gravity (and the car can carry it).
        if slope > 0.0 and brake == 0.0:
            accel = float(np.clip(accel + self.hill_accel_gain * slope, 0.0, 1.0))

        return {
            'steer': steer, 'accel': accel, 'brake': brake,
            'e_y': e_y, 'e_psi': e_psi, 'v_target': v_target,
            'slope': slope, 'is_straight': is_straight,
        }

    def _gear_select(self, speed, rpm):
        """Dynamic gearbox, 3 tiers (most data-driven first):
          1. car_spec wheel-force shift map -- the gear giving the most tractive
             force right now, straight from the engine torque curve + ratios. This
             revs each gear to the power peak and DOWNSHIFTS under braking to keep
             revs high for corner exit, with no hand-picked rpm constants.
          2. rpm fraction-of-(auto-learned)-limiter -- if no car_spec.
          3. legacy speed bands -- last resort.
        A ±1-per-shift clamp + a hold timer prevent hunting in every tier."""
        gmax = self.gear_max
        gear = int(self._last_gear) if self._last_gear else 1
        gear = max(1, min(gmax, gear))
        # Crawl / standstill -> first gear, clear the hold so launch is responsive.
        if speed < self.gear_min_speed:
            self._last_gear = 1
            self._gear_hold = 0
            return 1
        if self._gear_hold > 0:
            self._gear_hold -= 1
            self._last_gear = gear
            return gear
        new_gear = gear
        if self.car_spec is not None:
            target = self.car_spec.optimal_gear(speed / 3.6, gear)
            if target > gear:
                new_gear = gear + 1     # upshift one ratio at a time (speed rises slowly)
            elif target < gear:
                new_gear = target       # downshift STRAIGHT to target (fast braking 3->1)
        elif self.gear_use_rpm and rpm > 0.0:
            if rpm > self._rpm_redline:
                self._rpm_redline = rpm  # learn the true limiter from observed peak
            up_rpm = self.gear_up_frac * self._rpm_redline
            down_rpm = self.gear_down_frac * self._rpm_redline
            if gear < gmax and rpm >= up_rpm:
                new_gear = gear + 1
            elif gear > 1 and rpm <= down_rpm:
                new_gear = gear - 1
        else:
            if gear < gmax and gear - 1 < len(self.gear_up_speeds) \
                    and speed >= self.gear_up_speeds[gear - 1]:
                new_gear = gear + 1
            elif gear > 1 and gear - 2 < len(self.gear_down_speeds) \
                    and speed <= self.gear_down_speeds[gear - 2]:
                new_gear = gear - 1
        if new_gear != gear:
            self._gear_hold = self.gear_hold_steps
        self._last_gear = new_gear
        return new_gear

    def _corner_memory_file(self):
        if not self._racingline_path:
            return None
        base, _ = os.path.splitext(self._racingline_path)
        return base + "_corner_memory.npy"

    def _load_corner_memory(self):
        """Load (or initialise) the per-node speed-scale field for this line."""
        path = self._corner_memory_file()
        self._corner_memory_path = path
        n = self.racing_line.n
        if path and os.path.exists(path):
            try:
                scale = np.load(path)
                if scale.shape == (n,):
                    # Clamp into [floor, cap]: an older file may hold troughs from a
                    # previous regime that would keep corners crawling below today's
                    # floor, or (with two-way memory) values above the cap. Clamping
                    # on load relaxes stale over/under-penalties while keeping valid
                    # relative scaling between them.
                    raw = scale.astype(np.float64)
                    self.corner_scale = np.clip(
                        raw, self.corner_memory_floor, self.corner_memory_cap)
                    print(f"Loaded corner memory: {path} "
                          f"(min={self.corner_scale.min():.3f}, "
                          f"mean={self.corner_scale.mean():.3f})")
                    if not np.array_equal(self.corner_scale, raw):
                        self._corner_memory_dirty = True  # clamp to be saved
                    return
            except Exception as exc:  # pragma: no cover
                print(f"(corner memory reset: {exc})")
        self.corner_scale = np.ones(n, dtype=np.float64)

    def _build_kappa_clusters(self):
        """Group line nodes into curvature buckets so a speed-scale nudge on one
        corner can generalise to look-alikes (same |kappa| family). Straight nodes
        (|kappa| below the corner threshold) all share bucket 0 and are excluded
        from generalisation -- only genuine corners teach each other."""
        if self.racing_line is None:
            return
        kappa = np.abs(self.racing_line.kappa_rl)
        bucket = 0.005  # 1/m curvature resolution for "similar" corners
        bins = np.where(
            kappa < self.corner_kappa_min, 0,
            (np.round(kappa / bucket).astype(int)))
        self._kappa_bin = bins
        self._kappa_bin_members = {
            int(b): np.where(bins == b)[0]
            for b in np.unique(bins) if b != 0
        }

    def _nudge_corner_scale(self, idx, target, rate):
        """Move corner_scale[idx] a fraction `rate` toward `target`, and bleed a
        smaller share onto all nodes of similar curvature (the cluster). Clamped
        to [floor, cap]. Marks the temp field dirty for the next best-checkpoint
        commit."""
        cs = self.corner_scale
        if cs is None:
            return
        cs[idx] += rate * (target - cs[idx])
        if self._kappa_bin is not None and self.corner_memory_cluster > 0.0:
            b = int(self._kappa_bin[idx])
            members = self._kappa_bin_members.get(b)
            if members is not None and members.size > 1:
                cluster_rate = self.corner_memory_cluster * rate
                cs[members] += cluster_rate * (target - cs[members])
        np.clip(cs, self.corner_memory_floor, self.corner_memory_cap, out=cs)
        self._corner_memory_dirty = True

    def _register_clean_pass(self, s, track_pos, speed_ms):
        """Reward a corner taken CLEANLY at speed: nudge its speed-scale UP toward
        the cap (and similar corners with it). Only fires on real corner nodes
        the car is hugging the line through (small |trackPos|) while actually
        carrying close to the current scaled target speed -- proof the corner can
        be taken faster. This is the UP half of the two-way memory; crashes still
        pull it back down via _register_offtrack."""
        if self.corner_scale is None:
            return
        line = self.racing_line
        idx = line._index(s)
        if abs(line.kappa_rl[idx]) < self.corner_kappa_min:
            return  # straight -> nothing to reclaim
        if abs(track_pos) > self.corner_clean_margin:
            return  # not hugging the line -> not a clean pass
        scaled_target = float(line.vmax[idx]) * float(self.corner_scale[idx])
        if speed_ms < self.corner_clean_speed_frac * scaled_target:
            return  # crawling through -> no evidence it can go faster
        self._nudge_corner_scale(idx, self.corner_memory_cap, self.corner_memory_boost)

    def _register_offtrack(self, s_crash):
        """Decay the TEMP (in-RAM) speed-scale around a crash with an interpolated
        triangular kernel UPSTREAM of the apex (the entry speed is what was too
        high). A tiny global recovery nudge lets over-penalised corners heal. The
        field is NOT written to disk here -- the trainer calls
        commit_corner_memory() only when the run earns a new best checkpoint, so
        a bad run cannot poison the persisted memory."""
        if self.corner_scale is None:
            return
        line = self.racing_line
        ds = line.ds
        window_nodes = max(1, int(round(self.corner_memory_window / ds)))
        crash_idx = line._index(s_crash)
        # Gentle global recovery toward 1.0 so stale penalties relax over time.
        self.corner_scale += self.corner_memory_recover * (1.0 - self.corner_scale)
        for k in range(window_nodes + 1):
            idx = (crash_idx - k) % line.n
            weight = 1.0 - (k / (window_nodes + 1))   # triangular: strongest at apex
            factor = 1.0 - (1.0 - self.corner_memory_decay) * weight
            self.corner_scale[idx] = max(
                self.corner_memory_floor, self.corner_scale[idx] * factor)
        # Generalise the slow-down: bleed the apex's decay onto similar corners so
        # the car enters the whole family more cautiously after one crash.
        self._nudge_corner_scale(crash_idx, self.corner_memory_floor,
                                 1.0 - self.corner_memory_decay)
        self._corner_memory_dirty = True

    def commit_corner_memory(self):
        """Persist the temp corner-speed field to disk. Called by the trainer
        when a new best checkpoint is saved (i.e. the run is GOOD), so the
        learned slow-downs that accompanied a good policy survive, while a bad
        run's panic braking is discarded on process exit. No-op if nothing
        changed or no line is loaded."""
        if (self.corner_scale is None or not self._corner_memory_dirty
                or not self._corner_memory_path):
            return
        try:
            np.save(self._corner_memory_path, self.corner_scale)
            self._corner_memory_dirty = False
        except Exception as exc:  # pragma: no cover
            print(f"(corner memory save failed: {exc})")

    def _apply_steer_constraints(self, desired_steer, speed_kmh):
        """Clip steering and (in racing mode) scale authority by speed + rate-limit.

        Updates and returns the final applied steer. The speed-scaled authority
        prevents high-speed spin-outs; the rate limit keeps inputs smooth.
        """
        steer = float(np.clip(desired_steer, -1.0, 1.0))
        if self.racing_line is not None:
            authority = 1.0 - self.rl_steer_speed_falloff * max(0.0, speed_kmh - self.rl_steer_base_speed)
            authority = float(np.clip(authority, self.rl_steer_min_authority, 1.0))
            steer = float(np.clip(steer, -authority, authority))
            max_delta = self.rl_steer_rate_limit
            steer = float(np.clip(
                steer,
                self.last_applied_steer - max_delta,
                self.last_applied_steer + max_delta,
            ))
        self.last_applied_steer = steer
        return steer

    def _racingline_reward(self, obs, applied_steer=0.0, applied_accel=0.0,
                           applied_brake=0.0, prev_steer=0.0, prev_accel=0.0,
                           prev_brake=0.0):
        """Dense physics reward: progress along the line, minus offset / speed /
        heading errors relative to the precomputed racing line, plus proactive
        edge-clearance and input-smoothness penalties."""
        line = self.racing_line
        q = line.query(float(obs.get('distFromStart', 0.0)))
        speed_ms = float(obs['speedX']) / 3.6
        angle = float(obs['angle'])
        track_pos = float(obs['trackPos'])

        cross_track_error = track_pos * line.half_width - q['offset']
        heading_error = math.atan2(
            math.sin(angle - q['dpsi']), math.cos(angle - q['dpsi']))
        speed_delta = q['vmax'] - speed_ms          # +ve = below target speed
        overspeed = max(0.0, speed_ms - q['vmax'])  # +ve = faster than corner allows
        underspeed = max(0.0, speed_delta)

        reward = self.rl_progress_gain * speed_ms * math.cos(heading_error)
        reward -= self.rl_cross_track_gain * abs(cross_track_error)
        # Asymmetric speed shaping: quadratic punish for exceeding vmax (corner
        # danger -> teaches braking), mild linear nudge when under target. Going
        # fast is otherwise rewarded by the progress term above.
        reward -= self.rl_overspeed_gain * overspeed ** 2
        reward -= self.rl_underspeed_gain * underspeed
        reward -= self.rl_heading_gain * abs(heading_error)

        # "Go faster when the road is clear": when no corner is within the
        # forward look-ahead (long centre beam) AND the car is tracking the line
        # well (small offset + heading error), reward extra speed. This pushes the
        # car off its straight-line plateau without fighting the overspeed brake
        # into corners. Gated by stability so it never rewards charging off-line.
        clear_factor = 1.0 - curve_proximity_factor(obs.get('track', []))
        stable = float(np.clip(
            1.0 - abs(cross_track_error) / line.half_width - abs(heading_error), 0.0, 1.0))
        reward += self.rl_clear_speed_gain * speed_ms * clear_factor * stable

        # Proactive corner clearance: punish drifting toward the edge *before*
        # the terminal off-track event, so the policy lifts/turns in time.
        edge_excess = max(0.0, abs(track_pos) - self.rl_edge_threshold)
        reward -= self.rl_edge_penalty_gain * edge_excess

        # Smoothness: discourage steering/pedal jerk for a stable line.
        reward -= self.rl_steer_smooth_gain * abs(applied_steer - prev_steer)
        reward -= self.rl_pedal_smooth_gain * (
            abs(applied_accel - prev_accel) + abs(applied_brake - prev_brake))

        metrics = {
            "e_y": cross_track_error,
            "e_psi": heading_error,
            "speed_delta": speed_delta,
            "vmax": q['vmax'],
            "offset_target": q['offset'],
            "edge_excess": edge_excess,
            "clear_factor": clear_factor,
        }
        return reward, metrics

    def agent_to_torcs(self, u):
        # Final applied steer is set later by _apply_steer_constraints (which
        # needs the previous step's value for rate limiting), so don't overwrite
        # last_applied_steer here.
        steer = float(np.clip(u[0], -1.0, 1.0))

        torcs_action = {'steer': steer}

        if self.throttle is True:  # throttle action is enabled
            accel = float(np.clip(max(0.0, float(u[1])), 0.0, 1.0))
            brake = float(np.clip(max(0.0, float(u[2])), 0.0, 1.0))
            self.last_applied_accel = accel
            self.last_applied_brake = brake
            torcs_action.update({
                'accel': accel,
                'brake': brake,
            })

        if self.gear_change is True: # gear change action is enabled
            gear_index = 3 if self.throttle is True else 1
            gear_signal = float(u[gear_index])
            gear = int(np.clip(np.round(((gear_signal + 1.0) / 2.0) * 5.0) + 1, 1, 6))
            torcs_action.update({'gear': gear})

        return torcs_action

    def _maybe_debug_step(
        self,
        raw_action,
        applied_action,
        obs_pre,
        obs,
        reward,
        heading_hint,
        prev_heading_hint,
        curve_proximity,
        heading_error,
        heading_error_improvement,
    ):
        if not self.debug:
            return
        if self.time_step % self.debug_interval != 0:
            return

        steer_raw = float(raw_action[0]) if len(raw_action) > 0 else 0.0
        accel_raw = float(raw_action[1]) if len(raw_action) > 1 else 0.0
        brake_raw = float(raw_action[2]) if len(raw_action) > 2 else 0.0
        gear_raw = float(raw_action[3]) if len(raw_action) > 3 else 0.0
        track_values = np.asarray(obs.get('track', []), dtype=np.float32).reshape(-1)
        track_summary = summarize_track_sensors(track_values)
        track_raw = ", ".join(f"{value:.1f}" for value in track_values.tolist())
        target_gear = self._gear_for_speed(float(obs.get('speedX', 0.0)))
        rpm = float(obs.get('rpm', 0.0))
        grade = 0.0
        if self.racing_line is not None and self.racing_line.has_grade:
            grade = self.racing_line.grade_at(float(obs.get('distFromStart', 0.0)))
        edge_delta = 0.0
        if track_values.size >= 2:
            edge_delta = float(track_values[-1] - track_values[0])

        print(
            "DEBUG "
            f"step={self.time_step:4d} "
            f"speed={float(obs.get('speedX', 0.0)):6.2f} "
            f"angle={float(obs.get('angle', 0.0)): .3f} "
            f"trackPos={float(obs.get('trackPos', 0.0)): .3f} "
            f"hint={heading_hint: .3f} "
            f"prox={curve_proximity: .3f} "
            f"prev_hint={prev_heading_hint: .3f} "
            f"err={heading_error: .3f} "
            f"err_delta={heading_error_improvement: .3f} "
            f"raw_steer={steer_raw: .3f} "
            f"applied_steer={float(applied_action.get('steer', 0.0)): .3f} "
            f"raw_accel={accel_raw: .3f} "
            f"applied_accel={float(applied_action.get('accel', 0.0)): .3f} "
            f"raw_brake={brake_raw: .3f} "
            f"applied_brake={float(applied_action.get('brake', 0.0)): .3f} "
            f"gear_signal={gear_raw: .3f} "
            f"gear={int(applied_action.get('gear', obs.get('gear', 0)))} "
            f"target_gear={target_gear} "
            f"rpm={rpm: 6.0f} "
            f"grade={grade*100: .1f}% "
            f"edge_delta={edge_delta: .2f} "
            f"reward={float(reward): .3f}"
        )
        print(
            "DEBUG_TRACK "
            f"left={track_summary['left_mean']:.1f} "
            f"center={track_summary['center_mean']:.1f} "
            f"right={track_summary['right_mean']:.1f} "
            f"raw=[{track_raw}]"
        )


    TELEMETRY_COLUMNS = [
        "wall_time", "episode", "step", "s", "distRaced",
        "speedX", "speedY", "speedZ", "rpm", "gear",
        "applied_steer", "applied_accel", "applied_brake",
        "raw_steer", "raw_accel", "raw_brake",
        "base_steer", "base_accel", "base_brake",
        "trackPos", "angle", "damage", "z",
        "e_y", "e_psi", "vmax", "v_target", "speed_delta",
        "grade", "kappa_rl", "corner_scale", "slope", "is_straight",
        "wheelSpin_fl", "wheelSpin_fr", "wheelSpin_rl", "wheelSpin_rr", "slip",
        "lat_accel", "reward", "off_track", "termination_reason", "crash_s",
    ] + [f"track{i}" for i in range(19)]

    def _telemetry_init(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._telemetry_path = os.path.join(
            log_dir, f"telemetry_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self._telemetry_file = open(self._telemetry_path, "w", newline="")
        self._telemetry_writer = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow(self.TELEMETRY_COLUMNS)
        print(f"Telemetry log: {self._telemetry_path}")

    def _telemetry_write(self, *, raw_action, applied, obs, baseline, reward,
                         off_track, termination_reason, crash_s):
        """Append one wide telemetry row. Pulls raw + derived signals so the
        offline analyzer can cluster by location / curvature / grade / slip."""
        if not self.telemetry:
            return
        if self._telemetry_writer is None:
            self._telemetry_init()

        def _f(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        s = _f(obs.get("distFromStart", 0.0))
        speed = _f(obs.get("speedX", 0.0))
        spin = list(obs.get("wheelSpinVel", [0, 0, 0, 0])) + [0, 0, 0, 0]
        spin = [_f(x) for x in spin[:4]]
        front = 0.5 * (spin[0] + spin[1])
        rear = 0.5 * (spin[2] + spin[3])
        slip = rear - front  # driven-wheel overspeed (RWD wheelspin indicator)

        # Line-relative + curvature/grade context (None-safe).
        e_y = e_psi = vmax = grade = kappa = corner_scale = 0.0
        lat_accel = 0.0
        if self.racing_line is not None:
            line = self.racing_line
            q = line.query(s)
            vmax = q["vmax"]; kappa = q["kappa_rl"]
            track_pos = _f(obs.get("trackPos", 0.0))
            angle = _f(obs.get("angle", 0.0))
            e_y = track_pos * line.half_width - q["offset"]
            e_psi = math.atan2(math.sin(angle - q["dpsi"]), math.cos(angle - q["dpsi"]))
            grade = line.grade_at(s) if line.has_grade else 0.0
            lat_accel = (speed / 3.6) ** 2 * abs(kappa)  # measured lateral accel m/s^2
            if self.corner_scale is not None:
                corner_scale = float(self.corner_scale[line._index(s)])

        b = baseline or {}
        raw = list(raw_action) if raw_action is not None else []
        row = [
            f"{time.time():.3f}", self._episode_idx, self.time_step,
            f"{s:.1f}", _f(obs.get("distRaced", 0.0)),
            f"{speed:.2f}", _f(obs.get("speedY", 0.0)), _f(obs.get("speedZ", 0.0)),
            _f(obs.get("rpm", 0.0)), int(_f(applied.get("gear", obs.get("gear", 0)))),
            f"{_f(applied.get('steer')): .4f}", f"{_f(applied.get('accel')):.4f}",
            f"{_f(applied.get('brake')):.4f}",
            f"{(raw[0] if len(raw) > 0 else 0.0): .4f}",
            f"{(raw[1] if len(raw) > 1 else 0.0):.4f}",
            f"{(raw[2] if len(raw) > 2 else 0.0):.4f}",
            f"{_f(b.get('steer')): .4f}", f"{_f(b.get('accel')):.4f}",
            f"{_f(b.get('brake')):.4f}",
            f"{_f(obs.get('trackPos', 0.0)): .4f}", f"{_f(obs.get('angle', 0.0)): .4f}",
            _f(obs.get("damage", 0.0)), f"{_f(obs.get('z', 0.0)):.3f}",
            f"{e_y: .3f}", f"{e_psi: .4f}", f"{vmax:.2f}",
            f"{_f(b.get('v_target')):.2f}", f"{(vmax - speed / 3.6): .2f}",
            f"{grade*100: .2f}", f"{kappa: .5f}", f"{corner_scale:.4f}",
            f"{_f(b.get('slope')): .4f}", int(bool(b.get("is_straight", False))),
            f"{spin[0]:.2f}", f"{spin[1]:.2f}", f"{spin[2]:.2f}", f"{spin[3]:.2f}",
            f"{slip: .2f}", f"{lat_accel:.2f}", f"{reward: .3f}",
            int(bool(off_track)),
            termination_reason if termination_reason else "",
            f"{crash_s:.1f}" if crash_s is not None else "",
        ]
        row += [f"{_f(v):.1f}" for v in list(obs.get("track", []))[:19]]
        self._telemetry_writer.writerow(row)
        if self.time_step % 50 == 0:
            self._telemetry_file.flush()

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
