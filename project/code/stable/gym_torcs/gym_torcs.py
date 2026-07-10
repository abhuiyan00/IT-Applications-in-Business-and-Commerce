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
import collections as col
import os
import subprocess
import time


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


def curve_proximity_factor(track_values):
    track = np.asarray(track_values, dtype=np.float32).reshape(-1)
    if track.size == 0:
        return 0.0

    center_distance = float(track[track.size // 2])
    return float(np.clip((120.0 - center_distance) / 90.0, 0.0, 1.0))

class TorcsEnv:
    terminal_judge_start = 150  # End obviously bad episodes earlier during steering-only training.
    termination_limit_progress = 5  # [km/h], treat low progress as a failed episode sooner.
    default_speed = 600
    other_speed = 50
    steer_smoothing = 0.0
    steer_rate_limit = 1.0
    progress_reward_scale = 10.0
    track_pos_penalty_gain = 1.0
    track_pos_nonlinear_gain = 0.6
    track_pos_edge_threshold = 0.3
    track_pos_edge_penalty_gain = 0.8
    angle_penalty_gain = 1.0
    steer_smoothness_penalty_gain = 0.01
    steer_magnitude_penalty = 0.01
    angle_improvement_gain = 0.3
    track_pos_improvement_gain = 0.3
    damage_penalty = 200.0
    off_track_penalty = 10000.0
    #launch_speed_threshold = 30.0
    target_speed = 360
    corner_speed = 120
    launch_min_accel = 0.35
    brake_enable_speed = 80.0
    max_accel = 1.0
    max_brake = 1.0
    pedal_overlap_threshold = 0.2
    accel_overlap_limit = 0.12
    brake_overlap_limit = 0.12
    server_wait_loops = 5
    reset_connect_attempts = 3
    client_create_attempts = 5
    client_create_retry_delay = 1.0
    reconnect_each_episode = True

    initial_reset = True
    reset_observation_attempts = 50
    off_track_track_pos_limit = 1.05


    def __init__(
        self,
        vision=False,
        throttle=True,
        gear_change=False,
        port=3001,
        auto_start=None,
        torcs_command=None,
        kill_on_shutdown=False,
        debug=False,
        debug_interval=25,
        terminate_on_off_track=True,
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
        self.terminate_on_off_track = terminate_on_off_track

        self.initial_run = True
        self.client = None
        self.pending_reset = False
        self.last_applied_steer = 0.0
        self.previous_reward_steer = 0.0
        self.last_applied_accel = 0.0
        self.last_applied_brake = 0.0

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

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        action_torcs['meta'] = False

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            self.target_speed = self.default_speed
            if client.S.d['speedX'] < self.target_speed - (client.R.d['steer']*50):
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
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

            #if client.S.d['speedX'] < self.launch_speed_threshold:
            #    action_torcs['accel'] = max(action_torcs['accel'], self.launch_min_accel)

            # Safety clamp while steer+pedals are still being learned.
            #if abs(float(client.S.d.get('angle', 0.0))) > 0.35 or abs(float(client.S.d.get('trackPos', 0.0))) > 0.7:
            #    action_torcs['accel'] = min(action_torcs['accel'], 0.5)
            if client.S.d['speedX'] < self.brake_enable_speed:
                action_torcs['brake'] = 0.0

            self.last_applied_accel = float(action_torcs['accel'])
            self.last_applied_brake = float(action_torcs['brake'])

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
            #if client.S.d['speedX'] < self.launch_speed_threshold:
            #    action_torcs['gear'] = 1
        else:
            action_torcs['gear'] = self._gear_for_speed(float(client.S.d['speedX']))

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

        curve_proximity = curve_proximity_factor(track)

        prev_angle = float(obs_pre['angle'])
        prev_track_pos = float(obs_pre['trackPos'])

        abs_track_pos = abs(track_pos)
        abs_prev_track_pos = abs(prev_track_pos)

        accel_value = float(action_torcs.get('accel', 0.0))
        brake_value = float(action_torcs.get('brake', 0.0))
        steer_value = float(action_torcs.get('steer', 0.0))

        # F1-class speed target. Keep this aligned with speedX normalization so
        # the policy still sees useful speed differences above 150 km/h.
        

        # Actual progress along the track, not just raw speed.
        progress = speed * np.cos(angle)
        prev_progress = float(obs_pre['speedX']) * np.cos(float(obs_pre['angle']))
        progress_delta = progress - prev_progress

        reward = 0.0

        # Main reward: driving forward.
        reward += progress / self.progress_reward_scale

        # Bonus for reaching the target speed.
        # reward += 2.0 * min(speed / self.target_speed, 1.0)


        speed_mps = max(speed / 3.6, 1e-3)

        center_idx = track.size // 2
        lookahead_span = max(1, track.size // 4)
        sector = track[
            max(0, center_idx - lookahead_span):
            min(track.size, center_idx + lookahead_span + 1)
        ]
        lookahead_distance = np.max(sector) if sector.size else 0.0

        # Seconds until the visible track edge at the current speed.
        time_to_edge = lookahead_distance / speed_mps


        # =====================================================
        # Drift / slip penalty
        # =====================================================

        lateral_speed = abs(float(obs['speedY']))

        drift_excess = max(0.0, lateral_speed - 5.0)

        speed_factor = np.clip(speed / 200.0, 0.0, 1.0)

        drift_penalty = (
            0.3
            * drift_excess
            * speed_factor
        )

        reward -= drift_penalty
        # danger:
        # 0.0 -> safe
        # 1.0 -> very little time to the corner / track edge
        danger = np.clip((2.4 - time_to_edge) / 2.4, 0.0, 1.0)

        # ============================================================
        # Brake reward ONLY when danger is high
        # ============================================================

        reward += 1.5 * danger * brake_value

        safe_zone = danger < 0.2

        # Penalty for driving below the target speed.
        if safe_zone:
            low_speed_penalty =  max(0.0, self.target_speed - speed) / self.target_speed
            accel_penalty = 50.0 * max(0.0, 1.0 - accel_value)
            reward -= low_speed_penalty
            reward -= accel_penalty

            if self.debug and self.time_step % self.debug_interval == 0:
                print(
                    f"Penalty low_speed={low_speed_penalty:.3f} "
                    f"accel={accel_penalty:.3f}"
                )
        else:
            
            reward -= 2.0 * max(0.0, self.corner_speed - speed) / self.corner_speed

            overspeed = max(0.0, speed - self.corner_speed) / self.corner_speed

            reward -= 1.5 * (overspeed ** 1.5)
            reward -= 3.0 * drift_penalty

        # Penalty for braking when the car is too slow.
        if speed < self.target_speed:
            reward -= 0.8 * brake_value

        # Braking is mostly allowed near corners / bad positioning.
        brake_need = np.clip(
            1.8 * abs(angle) + 1.4 * max(0.0, abs_track_pos - 0.25),
            0.0,
            1.0
        )

        reward += 0.05 * brake_value * brake_need
        reward -= 0.25 * brake_value * (1.0 - brake_need)

        # Small acceleration bonus, but only when the car is stable.
        stable = np.clip(1.0 - 2.0 * abs(angle) - 1.2 * abs_track_pos, 0.0, 1.0)

        if speed < self.target_speed:
            reward += 0.25 * accel_value * stable
        else:
            reward += 0.05 * accel_value * stable

        # Bonus for improving progress.
        reward += 0.10 * max(progress_delta, 0.0)

        # Penalties for bad position and angle.
        edge_excess = max(0.0, abs_track_pos - self.track_pos_edge_threshold)

        reward -= self.track_pos_penalty_gain * abs_track_pos
        reward -= self.track_pos_nonlinear_gain * (abs_track_pos ** 1.5)
        reward -= self.track_pos_edge_penalty_gain * edge_excess

        reward -= self.angle_penalty_gain * abs(angle)

        # Bonus for improving the racing line.
        reward += self.angle_improvement_gain * (abs(prev_angle) - abs(angle))
        reward += self.track_pos_improvement_gain * (abs_prev_track_pos - abs_track_pos)

        # Penalties for nervous steering.
        reward -= (
            self.steer_smoothness_penalty_gain
            * abs(steer_value - self.previous_reward_steer)
        )
        reward -= self.steer_magnitude_penalty * abs(steer_value)

        self.previous_reward_steer = steer_value

        # Strong penalty for accelerating and braking at the same time.
        reward -= 0.50 * accel_value * brake_value

        self.last_u = np.array(u, copy=True)

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward -= self.damage_penalty

        # Termination judgement #########################
        episode_terminate = False
        termination_reason = None

        center_track = float(track[track.size // 2]) if track.size else -1.0
        track_min = float(track.min()) if track.size else -1.0
        # Range sensors can report -1 near edges/corners while the car is still
        # recoverable. trackPos is the authoritative lateral off-track signal.
        if abs(track_pos) > self.off_track_track_pos_limit:
            reward -= self.off_track_penalty
            termination_reason = "off_track"
            if self.terminate_on_off_track:
                episode_terminate = True
                client.R.d['meta'] = True

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
            curve_proximity=curve_proximity,
            heading_error=angle,
        )

        self.time_step += 1

        info = {
            "termination_reason": termination_reason,
            "applied_steer": float(action_torcs.get('steer', 0.0)),
            "applied_accel": float(action_torcs.get('accel', 0.0)),
            "applied_brake": float(action_torcs.get('brake', 0.0)),
            "track_min": track_min,
            "track_center": center_track,
            "track_pos": track_pos,
            "angle": angle,
            "speed_x": speed,
        }

        return self.get_obs(), reward, client.R.d['meta'], info

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

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
        obs['prevAppliedSteer'] = 0.0
        obs['prevAppliedAccel'] = 0.0
        obs['prevAppliedBrake'] = 0.0
        self.observation = self.make_observaton(obs)

        self.last_u = None
        self.last_applied_steer = 0.0
        self.last_applied_accel = 0.0
        self.last_applied_brake = 0.0
        self.previous_reward_steer = 0.0

        self.initial_reset = False
        self.pending_reset = False
        return self.get_obs()

    def end(self):
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

    def _gear_for_speed(self, speed):
        if speed < 30:
            return 1
        if speed < 60:
            return 2
        if speed < 95:
            return 3
        if speed < 140:
            return 4
        if speed < 185:
            return 5
        return 6 

    def agent_to_torcs(self, u):
        steer = float(np.clip(u[0], -1.0, 1.0))
        self.last_applied_steer = steer

        torcs_action = {'steer': steer}

        if self.throttle is True:  # transitional version: steer + accel + brake with safety gates
            accel_signal = float(np.clip(u[1], -1.0, 1.0))
            brake_signal = float(np.clip(u[2], -1.0, 1.0))
            accel = float(np.clip(max(0.0, accel_signal) * self.max_accel, 0.0, self.max_accel))
            brake = float(np.clip(max(0.0, brake_signal) * self.max_brake, 0.0, self.max_brake))
            speed = float(self.client.S.d.get('speedX', 0.0)) if self.client is not None else 0.0

            if speed < self.brake_enable_speed:
                brake = 0.0

            if accel > 0.0 and brake > 0.0:
                if accel >= brake:
                    brake = 0.0
                else:
                    accel = 0.0

            self.last_applied_accel = accel
            self.last_applied_brake = brake
            torcs_action.update({
                'accel': accel,
                'brake': brake,
            })

        if self.gear_change is True:
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
        curve_proximity,
        heading_error,
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
        edge_delta = 0.0
        if track_values.size >= 2:
            edge_delta = float(track_values[-1] - track_values[0])

        print(
            "DEBUG "
            f"step={self.time_step:4d} "
            f"speed={float(obs.get('speedX', 0.0)):6.2f} "
            f"angle={float(obs.get('angle', 0.0)): .3f} "
            f"trackPos={float(obs.get('trackPos', 0.0)): .3f} "
            f"prox={curve_proximity: .3f} "
            f"err={heading_error: .3f} "
            f"raw_steer={steer_raw: .3f} "
            f"applied_steer={float(applied_action.get('steer', 0.0)): .3f} "
            f"raw_accel={accel_raw: .3f} "
            f"applied_accel={float(applied_action.get('accel', 0.0)): .3f} "
            f"raw_brake={brake_raw: .3f} "
            f"applied_brake={float(applied_action.get('brake', 0.0)): .3f} "
            f"gear_signal={gear_raw: .3f} "
            f"gear={int(applied_action.get('gear', obs.get('gear', 0)))} "
            f"target_gear={target_gear} "
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
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.other_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.other_speed,
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
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.other_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.other_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
