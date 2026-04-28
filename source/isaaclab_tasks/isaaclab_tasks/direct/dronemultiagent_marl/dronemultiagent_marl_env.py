# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import numpy as np

from isaaclab.envs.mdp.observations import joint_pos
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import UsdFileCfg, PreviewSurfaceCfg
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.markers import CUBOID_MARKER_CFG
from isaaclab_tasks.manager_based.navigation.mdp import rewards  
from .dronemultiagent_marl_env_cfg import DronemultiagentMarlEnvCfg
        

class DronemultiagentMarlEnv(DirectMARLEnv):
    cfg: DronemultiagentMarlEnvCfg

    def __init__(self, cfg: DronemultiagentMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_phase_settings()
        self._print_phase_banner()
        # Boat movement parameters 
        self._platform_dx = torch.zeros(self.num_envs, device=self.device)
        self._platform_dy = torch.zeros(self.num_envs, device=self.device)
        self._platform_dz = torch.zeros(self.num_envs, device=self.device)
        self._platform_roll = torch.zeros(self.num_envs, device=self.device)
        self._platform_pitch = torch.zeros(self.num_envs, device=self.device)



        # Adding the wind forces to the drone
        self.wind_force = torch.zeros((self.num_envs, 1, 3), device=self.device)  # shape: [envs, bodies, vec3]
        self.wind_timer = torch.zeros(self.num_envs, device=self.device)  # How long current wind lasts
        self.wind_cooldown = torch.zeros(self.num_envs, device=self.device)  # Delay before wind changes
        self.wind_direction = torch.nn.functional.normalize(torch.randn(self.num_envs, 2, device=self.device), dim=1)  # XY wind
        self.wind_strength = torch.empty(self.num_envs, device=self.device).uniform_(self.cfg.lower_wind_scale, self.cfg.upper_wind_scale)  # m/s² force range
        # For wind gust control
        self.wind_gust_timer = torch.zeros(self.num_envs, device=self.device)         # seconds remaining of gust
        self.wind_gust_cooldown = torch.zeros(self.num_envs, device=self.device)      # cooldown before next gust
        self.active_wind_force = torch.zeros((self.num_envs, 1, 3), device=self.device)  # actual force applied

        # Toggle
        self.RealisticWindYesOrNo = getattr(self.cfg, "RealisticWindYesOrNo", True)

        # Wind model parameters (put these in cfg if you want)
        self.wind_vmin = getattr(self.cfg, "wind_speed_min", self.cfg.lower_wind_scale)
        self.wind_vmax = getattr(self.cfg, "wind_speed_max", self.cfg.upper_wind_scale)

        # Option A: max turning rate (rad/s)
        self.wind_max_yaw_rate = math.radians(getattr(self.cfg, "wind_max_yaw_rate_deg", 25.0))

        # Option B: cone half-angle (rad) + how often we choose a new target
        self.wind_cone_half_angle = math.radians(getattr(self.cfg, "wind_cone_half_angle_deg", 35.0))
        self.wind_update_interval = getattr(self.cfg, "wind_update_interval_s", 2.0)

        # Smoothing time-constants (both A & B can use these)
        self.wind_dir_tau = getattr(self.cfg, "wind_direction_tau_s", 0.7)
        self.wind_speed_tau = getattr(self.cfg, "wind_speed_tau_s", 0.9)

        # Persistent wind state (2D direction stored as yaw)
        self._wind_yaw = torch.empty(self.num_envs, device=self.device).uniform_(-math.pi, math.pi)
        self._wind_speed = torch.empty(self.num_envs, device=self.device).uniform_(self.wind_vmin, self.wind_vmax)

        # For option B target-hold behavior
        self._wind_target_yaw = self._wind_yaw.clone()
        self._wind_target_speed = self._wind_speed.clone()
        self._wind_time_to_update = torch.empty(self.num_envs, device=self.device).uniform_(0.0, self.wind_update_interval)

        # Global time for noise
        self._wind_time = torch.tensor(0.0, device=self.device)

        # Per-env offsets/seeds so all envs don't get identical noise
        self._wind_noise_seed = torch.randint(
            low=0, high=2**31-1, size=(self.num_envs,), device=self.device, dtype=torch.int64
        )
        self._wind_noise_offset = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 1000.0)
        self._magnet_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Magnetic capture condition active
        
        # Magnet condition tracking with the second counter 
        self._magnet_condition_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._magnet_required_steps = int(self.cfg.magnet_time_threshold_in_seconds / self.step_dt)  # e.g., 3s / 0.05 = 60 steps

        # === UR10 Arm Initialization ===
        self.num_arm_dofs = self._Ur10Arm.num_joints
        self.arm_dof_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.arm_prev_targets = torch.zeros_like(self.arm_dof_targets)
        self.arm_curr_targets = torch.zeros_like(self.arm_dof_targets)
    

        # Get actuated joint indices (if needed)
        self.actuated_dof_indices = [
            self._Ur10Arm.joint_names.index(joint_name)
            for joint_name in self.cfg.UR10_CFG.actuators["_Ur10Arm"].joint_names_expr
            if joint_name in self._Ur10Arm.joint_names  # Ensure it's valid
        ]

        # Joint limits
        joint_limits = self._Ur10Arm.root_physx_view.get_dof_limits().to(self.device)
        self.arm_dof_lower_limits = joint_limits[..., 0]
        self.arm_dof_upper_limits = joint_limits[..., 1]
        
        # Unit tensors (optional, useful for directional reward calculations)
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Resolve important UR10 body indices once
        self.ee_idx = self._get_single_body_index("ee_link")
        self.wrist_1_idx = self._get_single_body_index("wrist_1_link")
        self.wrist_2_idx = self._get_single_body_index("wrist_2_link")
        self.wrist_3_idx = self._get_single_body_index("wrist_3_link")

        self.arm_curr_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)
        self.arm_prev_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)
        
        # Added from the pre_physics_step function
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        # === Add wind marker config ===
        self.wind_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/WindMarkers",
            markers={
                "wind_arrow": UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    #scale=(0.5, 0.1, 0.1),
                    scale=(0.25, 0.05, 1.5),
                    # visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.8)),
                )
            }
        )

        self.wind_markers = VisualizationMarkers(self.wind_marker_cfg)

        self._step_count = 0

        # --------------------------------------------------------------------
        # PLATFORM MOTION
        # --------------------------------------------------------------------
        self._platform_motion_enabled = getattr(self.cfg, "enable_platform_motion", False)

        # Base UR10 root state around which motion is applied (pos+quat+linvel+angvel)
        self._ur10_root_state_base = self._Ur10Arm.data.default_root_state.clone()  # [num_envs, 13]

        # Global time accumulator for the platform motion
        self._platform_time = torch.tensor(0.0, device=self.device)

        if self._platform_motion_enabled:
            # Random phases per env and per axis (surge, sway, heave, roll, pitch)
            if getattr(self.cfg, "platform_random_phase", True):
                self._platform_phase = torch.empty((self.num_envs, 5), device=self.device).uniform_(0.0, 2.0 * math.pi)
            else:
                self._platform_phase = torch.zeros((self.num_envs, 5), device=self.device)

        # Debug marker: arrow (we reuse arrow_x.usd like your wind marker)
        self.platform_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/PlatformMotion",
            markers={
                "platform_arrow": UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.15, 0.03, 0.6),  # base scale; we'll modulate via position/meaning not true scaling
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                )
            }
        )
        self.platform_markers = VisualizationMarkers(self.platform_marker_cfg)



        ###### Code added here for logging and success/failure tracking ##########
        # add a episode level success tracker 
        self._episode_success_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # add a winning condition
        self._winning_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # add a episode level failure tracker
        self._episode_failure_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ########################################################


        self._actions = {}
        self._thrust = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._moment = torch.zeros((self.num_envs, 1, 3), device=self.device)

        # #Use live UR10 end-effector position as drone's target
        # ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :]  # shape [num_envs, 1, 3]
        # self._desired_pos_w = ee_pos.squeeze(1)  # Save as [num_envs, 3]
        
        # --------------------------------------------------------------------
        # GOAL BUFFERS (PACE-ready)
        # --------------------------------------------------------------------
        # Drone goal in world coordinates
        self._drone_goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # Arm goal pose in world coordinates
        self._arm_goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._arm_goal_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._arm_goal_quat_w[:, 0] = 1.0  # identity quaternion (w, x, y, z)

        # Temporary compatibility alias used by legacy reward/obs/state code.
        # For now, the drone goal remains the old "desired position".
        self._desired_pos_w = self._drone_goal_pos_w

        # Initialize goals once so they are valid immediately after construction.
        self._update_phase_goals()

        if torch.rand(1).item() < 0.01:
            print(f"[DEBUG] drone_goal_pos_w avg Z: {self._drone_goal_pos_w[:, 2].mean():.3f}")

        # PACE reward progress buffers
        self._prev_drone_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._prev_arm_goal_distance = torch.zeros(self.num_envs, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Drone-centric
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "smooth_landing",
                "proximity",
                "time_shaping",
                "alignment_reward",
                "magnet_reward",

                # UR10-centric
                "orientation_reward",
                "wrist_height_reward", 
                "arm_go_safe",
                "arm_hold_still",
                "arm_near_jitter",

                # Shared penalty 
                "died_penalty",
            ]
        }

        # Add after self._episode_sums
        self._success_status = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        # Get specific body indices
        self._body_id = self._DroneRobot.find_bodies("body")[0]
        self._robot_mass = (self._DroneRobot.root_physx_view.get_masses()[0].sum())  # scale to 3x size (volume scales with the cube of length)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # Add drone, arm, and ground plane to the scene
        self._DroneRobot = Articulation(self.cfg.Drone_CFG)
        self.scene.articulations["Drone_CFG"] = self._DroneRobot
        self._Ur10Arm = Articulation(self.cfg.UR10_CFG)
        self.scene.articulations["UR10"] = self._Ur10Arm

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Optionally disable collisions on the ground plane (visual only)
        # --- NEW: disable ground collisions for boat scenario ---
        if getattr(self.cfg, "disable_ground_collisions", False):
            self._disable_ground_collisions("/World/ground")


        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_phase_settings(self) -> None:
        """Resolve curriculum/phase settings from cfg into runtime attributes."""

        self._pace = int(getattr(self.cfg, "PACE", 0))
        pace_key = str(self._pace)

        pace_name_map = getattr(self.cfg, "PACE_NAME_MAP", {})
        pace_desc_map = getattr(self.cfg, "PACE_PRESET_DESCRIPTIONS", {})

        self._pace_name = pace_name_map.get(pace_key, f"PACE_{self._pace}")
        self._pace_description = pace_desc_map.get(pace_key, "No description available.")

        # ------------------------------------------------------------------
        # Default behavior from cfg
        # ------------------------------------------------------------------
        self._use_separated_training_boxes = bool(getattr(self.cfg, "USE_SEPARATED_TRAINING_BOXES", False))
        self._use_shared_goal_logic = bool(getattr(self.cfg, "USE_SHARED_GOAL_LOGIC", True))
        self._use_moving_goals = bool(getattr(self.cfg, "USE_MOVING_GOALS", False))
        self._use_drone_goal = bool(getattr(self.cfg, "USE_DRONE_GOAL", True))
        self._use_arm_goal = bool(getattr(self.cfg, "USE_ARM_GOAL", False))
        self._use_magnet_logic = bool(getattr(self.cfg, "USE_MAGNET_LOGIC", True))
        self._use_shared_success_condition = bool(getattr(self.cfg, "USE_SHARED_SUCCESS_CONDITION", True))

        self._drone_goal_mode = getattr(self.cfg, "DRONE_GOAL_MODE", "ee_tracking")
        self._arm_goal_mode = getattr(self.cfg, "ARM_GOAL_MODE", "none")

        # ------------------------------------------------------------------
        # Phase-specific overrides
        # ------------------------------------------------------------------
        if self._pace == 0:
            self._use_separated_training_boxes = False
            self._use_shared_goal_logic = True
            self._use_moving_goals = False
            self._use_drone_goal = True
            self._use_arm_goal = False
            self._use_magnet_logic = True
            self._use_shared_success_condition = True
            self._drone_goal_mode = "ee_tracking"
            self._arm_goal_mode = "none"

        elif self._pace == 1:
            self._use_separated_training_boxes = True
            self._use_shared_goal_logic = False
            self._use_moving_goals = False
            self._use_drone_goal = True
            self._use_arm_goal = True
            self._use_magnet_logic = False
            self._use_shared_success_condition = False
            self._drone_goal_mode = "static_world"
            self._arm_goal_mode = "arm_sphere_pose"
            # PACE 1 should be clean/static: no boat/platform motion yet.
            self._platform_motion_enabled = False

        # Disturbances
        self._wind_enabled = bool(getattr(self.cfg, "enable_wind", True))
        self._wind_gusts_enabled = bool(getattr(self.cfg, "enable_wind_gusts", True))
        if self._pace == 1:
            self._platform_motion_enabled = False
        else:
            self._platform_motion_enabled = bool(getattr(self.cfg, "enable_platform_motion", False))
       
        # Observation toggles
        self._include_wind_in_obs = bool(getattr(self.cfg, "INCLUDE_WIND_IN_OBS", True))
        self._include_cross_agent_info_in_obs = bool(getattr(self.cfg, "INCLUDE_CROSS_AGENT_INFO_IN_OBS", True))
        self._include_goal_in_obs = bool(getattr(self.cfg, "INCLUDE_GOAL_IN_OBS", True))
        self._include_goal_orientation_in_obs = bool(getattr(self.cfg, "INCLUDE_GOAL_ORIENTATION_IN_OBS", False))

    def _print_phase_banner(self) -> None:
        if not getattr(self.cfg, "PRINT_PACE_COMMENTS", False):
            return

        print("=" * 80)
        print(f"[PACE] Active phase: {self._pace}")
        print(f"[PACE] Name: {self._pace_name}")
        print(f"[PACE] Description: {self._pace_description}")
        print(f"[PACE] Separated boxes: {self._use_separated_training_boxes}")
        print(f"[PACE] Shared goal logic: {self._use_shared_goal_logic}")
        print(f"[PACE] Drone goal mode: {self._drone_goal_mode}")
        print(f"[PACE] Arm goal mode: {self._arm_goal_mode}")
        print(f"[PACE] Wind enabled: {self._wind_enabled}")
        print(f"[PACE] Wind gusts enabled: {self._wind_gusts_enabled}")
        print(f"[PACE] Platform motion enabled: {self._platform_motion_enabled}")
        print("=" * 80)

    def _get_single_body_index(self, body_name: str) -> int:
        """Resolve a body name to a single plain Python int index."""
        body_indices = self._Ur10Arm.find_bodies(body_name)
        if len(body_indices) == 0:
            raise RuntimeError(f"Could not find body '{body_name}' on UR10!")

        idx = body_indices[0]

        while isinstance(idx, (list, tuple)):
            if len(idx) == 0:
                raise RuntimeError(f"Body '{body_name}' resolved to an empty index container.")
            idx = idx[0]

        if hasattr(idx, "item"):
            idx = idx.item()

        return int(idx)

    def _update_phase_goals(self) -> None:
        """
        Update goal buffers according to the currently active PACE phase settings.
        """

        ee_idx = self.ee_idx

        # ----------------------------
        # Drone goal update
        # ----------------------------
        if self._drone_goal_mode == "ee_tracking":
            ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_idx, :]
            if ee_pos.ndim == 3:
                ee_pos = ee_pos.squeeze(1)

            self._drone_goal_pos_w.copy_(ee_pos)

        elif self._drone_goal_mode == "static_world":
            # Static world goals are assigned during reset and stay unchanged during the episode.
            pass

        elif self._drone_goal_mode == "moving_world":
            # Placeholder for future PACE phases.
            pass

        else:
            raise ValueError(f"Unsupported drone goal mode: {self._drone_goal_mode}")

        # Keep legacy compatibility alias synchronized
        self._desired_pos_w = self._drone_goal_pos_w

        # ----------------------------
        # Arm goal update
        # ----------------------------
        if self._arm_goal_mode == "none":
            arm_goal_pos = self._Ur10Arm.data.body_pos_w[:, ee_idx, :]
            arm_goal_quat = self._Ur10Arm.data.body_quat_w[:, ee_idx, :]

            if arm_goal_pos.ndim == 3:
                arm_goal_pos = arm_goal_pos.squeeze(1)
            if arm_goal_quat.ndim == 3:
                arm_goal_quat = arm_goal_quat.squeeze(1)

            self._arm_goal_pos_w.copy_(arm_goal_pos)
            self._arm_goal_quat_w.copy_(arm_goal_quat)

        elif self._arm_goal_mode == "arm_sphere_pose":
            # Static arm pose goals are assigned during reset and stay unchanged during the episode.
            pass

        else:
            raise ValueError(f"Unsupported arm goal mode: {self._arm_goal_mode}")

        if getattr(self.cfg, "PRINT_PACE_GOAL_UPDATES", False) and torch.rand(1).item() < 0.005:
            print(
                f"[PACE][GOALS] drone_goal_mean={self._drone_goal_pos_w.mean(dim=0).tolist()} "
                f"arm_goal_mean={self._arm_goal_pos_w.mean(dim=0).tolist()}"
            )

    def _reset_phase_goals(self, env_ids: torch.Tensor) -> None:
        """
        Reset goal buffers for the selected environments according to the active PACE phase.
        """

        ee_idx = self.ee_idx

        # ----------------------------
        # PACE 0: legacy shared behavior
        # ----------------------------
        if self._pace == 0:
            ee_pos = self._Ur10Arm.data.body_pos_w[env_ids, ee_idx, :]
            ee_quat = self._Ur10Arm.data.body_quat_w[env_ids, ee_idx, :]

            if ee_pos.ndim == 3:
                ee_pos = ee_pos.squeeze(1)
            if ee_quat.ndim == 3:
                ee_quat = ee_quat.squeeze(1)

            self._drone_goal_pos_w[env_ids] = ee_pos
            self._arm_goal_pos_w[env_ids] = ee_pos
            self._arm_goal_quat_w[env_ids] = ee_quat

        # ----------------------------
        # PACE 1: separated static goals
        # ----------------------------
        elif self._pace == 1:
            drone_goal = self._sample_drone_static_goal(env_ids)
            arm_goal_pos, arm_goal_quat = self._sample_arm_static_goal(env_ids)

            self._drone_goal_pos_w[env_ids] = drone_goal
            self._arm_goal_pos_w[env_ids] = arm_goal_pos
            self._arm_goal_quat_w[env_ids] = arm_goal_quat

        else:
            # Fallback for phases not yet implemented
            ee_pos = self._Ur10Arm.data.body_pos_w[env_ids, ee_idx, :]
            ee_quat = self._Ur10Arm.data.body_quat_w[env_ids, ee_idx, :]

            if ee_pos.ndim == 3:
                ee_pos = ee_pos.squeeze(1)
            if ee_quat.ndim == 3:
                ee_quat = ee_quat.squeeze(1)

            self._drone_goal_pos_w[env_ids] = ee_pos
            self._arm_goal_pos_w[env_ids] = ee_pos
            self._arm_goal_quat_w[env_ids] = ee_quat

        # Keep legacy compatibility alias synchronized
        self._desired_pos_w = self._drone_goal_pos_w

        # Reset progress tracking after goals have been assigned
        drone_pos = self._DroneRobot.data.root_pos_w[env_ids, :3]
        ee_pos = self._Ur10Arm.data.body_pos_w[env_ids, self.ee_idx, :]
        if ee_pos.ndim == 3:
            ee_pos = ee_pos.squeeze(1)

        self._prev_drone_goal_distance[env_ids] = torch.linalg.norm(
            self._drone_goal_pos_w[env_ids] - drone_pos, dim=1
        )
        self._prev_arm_goal_distance[env_ids] = torch.linalg.norm(
            self._arm_goal_pos_w[env_ids] - ee_pos, dim=1
        )

        if getattr(self.cfg, "PRINT_PACE_ON_RESET", False):
            print(
                f"[PACE][RESET] phase={self._pace} "
                f"envs={len(env_ids)} "
                f"drone_goal_mean={self._drone_goal_pos_w[env_ids].mean(dim=0).tolist()} "
                f"arm_goal_mean={self._arm_goal_pos_w[env_ids].mean(dim=0).tolist()}"
            )

    def _get_drone_out_of_bounds(self) -> torch.Tensor:
        """Check drone boundary violation in local environment coordinates."""

        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        local_pos = drone_pos - self._terrain.env_origins

        if self._use_separated_training_boxes:
            x_min = self.cfg.drone_side_x_center - self.cfg.side_half_width
            x_max = self.cfg.drone_side_x_center + self.cfg.side_half_width
            y_min = self.cfg.drone_box_y_min
            y_max = self.cfg.drone_box_y_max
            z_min = self.cfg.drone_box_z_min
            z_max = self.cfg.drone_box_z_max
        else:
            x_min, x_max = self.cfg.shared_box_x_min, self.cfg.shared_box_x_max
            y_min, y_max = self.cfg.shared_box_y_min, self.cfg.shared_box_y_max
            z_min, z_max = self.cfg.shared_box_z_min, self.cfg.shared_box_z_max

        x_oob = (local_pos[:, 0] < x_min) | (local_pos[:, 0] > x_max)
        y_oob = (local_pos[:, 1] < y_min) | (local_pos[:, 1] > y_max)
        z_oob_local = (local_pos[:, 2] < z_min) | (local_pos[:, 2] > z_max)

        # Emergency world-floor kill.
        # This catches drones falling through the visual/disabled ground plane.
        z_oob_world = drone_pos[:, 2] < getattr(self.cfg, "drone_world_z_kill", 0.05)

        return x_oob | y_oob | z_oob_local | z_oob_world

    def _get_arm_out_of_bounds(self) -> torch.Tensor:
        """
        For now:
        - In PACE 0 → arm never dies independently
        - Future: enforce workspace limits or stability checks
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _sample_drone_static_goal(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample static drone goals on the drone side of each environment."""
        n = len(env_ids)
        device = self.device
        env_origins = self._terrain.env_origins[env_ids]

        x = torch.empty(n, device=device).uniform_(
            self.cfg.drone_side_x_center - self.cfg.side_half_width,
            self.cfg.drone_side_x_center + self.cfg.side_half_width,
        )
        y = torch.empty(n, device=device).uniform_(
            self.cfg.drone_box_y_min + self.cfg.reset_spawn_margin_xy,
            self.cfg.drone_box_y_max - self.cfg.reset_spawn_margin_xy,
        )
        z = torch.empty(n, device=device).uniform_(
            self.cfg.drone_box_z_min + self.cfg.reset_spawn_margin_z,
            self.cfg.drone_box_z_max - self.cfg.reset_spawn_margin_z,
        )

        goal = torch.zeros((n, 3), device=device)
        goal[:, 0] = env_origins[:, 0] + x
        goal[:, 1] = env_origins[:, 1] + y
        goal[:, 2] = z

        return goal

    def _sample_arm_static_goal(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample static arm goals on the arm side of each environment.
        For now, orientation is identity quaternion.
        """
        n = len(env_ids)
        device = self.device
        env_origins = self._terrain.env_origins[env_ids]

        x = torch.empty(n, device=device).uniform_(
            self.cfg.arm_side_x_center - self.cfg.side_half_width,
            self.cfg.arm_side_x_center + self.cfg.side_half_width,
        )
        y = torch.empty(n, device=device).uniform_(
            -self.cfg.side_half_width,
            self.cfg.side_half_width,
        )
        z = torch.empty(n, device=device).uniform_(
            self.cfg.arm_goal_min_height,
            self.cfg.arm_goal_max_height,
        )

        goal_pos = torch.zeros((n, 3), device=device)
        goal_pos[:, 0] = env_origins[:, 0] + x
        goal_pos[:, 1] = env_origins[:, 1] + y
        goal_pos[:, 2] = z

        goal_quat = torch.zeros((n, 4), device=device)
        goal_quat[:, 0] = 1.0  # identity quaternion

        return goal_pos, goal_quat
    
    def _compute_drone_reward_pace_1(
        self,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        drone_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        PACE 1 drone reward:
        - main objective: reduce distance to static goal
        - terminate/fail if out of bounds
        - only punish angular velocity when it becomes excessive
        """

        distance_to_goal = torch.linalg.norm(self._drone_goal_pos_w - drone_pos, dim=1)

        # Main reward: closer to goal = better
        #distance_reward = (1.0 - torch.tanh(distance_to_goal / 0.8)) * self.cfg.distance_to_goal_reward_scale * self.step_dt
        distance_reward = -distance_to_goal
        
        # Progress reward: only rewards actually moving closer
        progress = self._prev_drone_goal_distance - distance_to_goal
        progress_reward = progress.clamp(-0.05, 0.05) * 50.0
        self._prev_drone_goal_distance = distance_to_goal.detach()

        # Sparse goal bonus
        goal_reached = distance_to_goal < 0.25
        goal_bonus = goal_reached.float() * 100.0 * self.step_dt

        # Death penalty
        died = self._get_drone_out_of_bounds()
        died_penalty = died.float() * self.cfg.died_penalty

        # Only punish angular velocity if it is excessive.
        # This avoids teaching the drone to stay rigid forever.
        excessive_ang_vel = torch.clamp(ang_vel - 8.0, min=0.0)
        excessive_ang_vel_penalty = -0.02 * excessive_ang_vel * self.step_dt

        rewards = {
            "lin_vel": torch.zeros_like(distance_to_goal),
            "ang_vel": excessive_ang_vel_penalty,
            "distance_to_goal": distance_reward,
            "smooth_landing": torch.zeros_like(distance_to_goal),
            "proximity": goal_bonus,
            "time_shaping": progress_reward,
            "alignment_reward": torch.zeros_like(distance_to_goal),
            "magnet_reward": torch.zeros_like(distance_to_goal),
            "died_penalty": died_penalty,
        }

        for k, v in rewards.items():
            if k in self._episode_sums:
                self._episode_sums[k] += v

        if torch.rand(1).item() < 0.002:
            print(
                f"[DRONE DEBUG] "
                f"dist_mean={distance_to_goal.mean().item():.3f}, "
                f"goal_rel_mean={(self._drone_goal_pos_w - drone_pos).mean(dim=0).tolist()}, "
                f"reward_mean={distance_reward.mean().item():.3f}, "
                f"thrust_mean={self._actions['_DroneRobot'][:, 0].mean().item():.3f}"
            )        


        return (
            rewards["distance_to_goal"]
            + rewards["time_shaping"]
            + rewards["proximity"]
            + rewards["ang_vel"]
            + rewards["died_penalty"]
        )

    def _compute_arm_reward_pace_1(self) -> torch.Tensor:
        """
        PACE 1 arm reward:
        - reward moving EE closer to static arm goal
        - reward being near the goal
        - softly discourage jitter
        """

        ee_pos = self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :]
        if ee_pos.ndim == 3:
            ee_pos = ee_pos.squeeze(1)

        arm_distance = torch.linalg.norm(self._arm_goal_pos_w - ee_pos, dim=1)
        arm_distance_mapped = 1.0 - torch.tanh(arm_distance / 0.4)

        progress = self._prev_arm_goal_distance - arm_distance
        progress_reward = progress.clamp(-0.05, 0.05) * 150.0
        self._prev_arm_goal_distance = arm_distance.detach()

        arm_qd = self._Ur10Arm.data.joint_vel
        arm_motion = torch.sum(arm_qd * arm_qd, dim=1)

        is_close = arm_distance < 0.10
        reach_bonus = is_close.float() * 50.0 * self.step_dt

        rewards = {
            "orientation_reward": torch.zeros_like(arm_distance),
            "wrist_height_reward": torch.zeros_like(arm_distance),
            "arm_go_safe": arm_distance_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "arm_hold_still": -0.02 * arm_motion.clamp(max=20.0) * self.step_dt,
            "arm_near_jitter": progress_reward + reach_bonus,
        }

        for k, v in rewards.items():
            if k in self._episode_sums:
                self._episode_sums[k] += v

        return (
            rewards["orientation_reward"]
            + rewards["wrist_height_reward"]
            + rewards["arm_go_safe"]
            + rewards["arm_hold_still"]
            + rewards["arm_near_jitter"]
        )
    
    # I will try to disable the ground collisions. 
    def _disable_ground_collisions(self, prim_path: str = "/World/ground"):
        """Disable collisions for the ground prim and its descendants (visual-only ground)."""
        try:
            from pxr import Usd, UsdPhysics
            stage = sim_utils.get_current_stage()

            root_prim = stage.GetPrimAtPath(prim_path)
            if not root_prim.IsValid():
                print(f"[WARN] Ground prim not found at {prim_path}; cannot disable collisions.")
                return

            for prim in Usd.PrimRange(root_prim):
                if not prim.IsValid():
                    continue

                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.CollisionAPI(prim)
                else:
                    api = UsdPhysics.CollisionAPI.Apply(prim)

                api.GetCollisionEnabledAttr().Set(False)

            print(f"[INFO] Disabled collisions for ground prim subtree: {prim_path}")

        except Exception as e:
            print(f"[WARN] Failed to disable ground collisions on {prim_path}: {e}")
    
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Clamp and store actions
        self._actions["_Ur10Arm"] = actions["_Ur10Arm"].clone().clamp(-1.0, 1.0)
        self._actions["_DroneRobot"] = actions["_DroneRobot"].clone().clamp(-1.0, 1.0)


        # -----------------------------------------------------------
        # Apply platform motion (boat-like) to UR10 base 
        # -----------------------------------------------------------
        if self._platform_motion_enabled:
            dt = self.step_dt
            self._platform_time = self._platform_time + dt

            f_hz = getattr(self.cfg, "platform_motion_frequency_hz", 0.20)
            omega = 2.0 * math.pi * f_hz

            # amplitudes
            Ax = getattr(self.cfg, "platform_surge_amplitude", 0.0)
            Ay = getattr(self.cfg, "platform_sway_amplitude", 0.0)
            Az = getattr(self.cfg, "platform_heave_amplitude", 0.0)

            roll_amp = math.radians(getattr(self.cfg, "platform_roll_amplitude_deg", 0.0))
            pitch_amp = math.radians(getattr(self.cfg, "platform_pitch_amplitude_deg", 0.0))

            # phases per env
            phase = getattr(self, "_platform_phase", None)
            if phase is None:
                phase = torch.zeros((self.num_envs, 5), device=self.device)

            t = self._platform_time

            # sinusoidal displacements
            dx = Ax * torch.sin(omega * t + phase[:, 0])
            dy = Ay * torch.sin(omega * t + phase[:, 1])
            dz = Az * torch.sin(omega * t + phase[:, 2])

            roll = roll_amp * torch.sin(omega * t + phase[:, 3])
            pitch = pitch_amp * torch.sin(omega * t + phase[:, 4])

            # Store for debug visualization
            self._platform_dx[:] = dx
            self._platform_dy[:] = dy
            self._platform_dz[:] = dz
            self._platform_roll[:] = roll
            self._platform_pitch[:] = pitch



            # velocities (derivatives)
            vx = Ax * omega * torch.cos(omega * t + phase[:, 0])
            vy = Ay * omega * torch.cos(omega * t + phase[:, 1])
            vz = Az * omega * torch.cos(omega * t + phase[:, 2])

            roll_dot = roll_amp * omega * torch.cos(omega * t + phase[:, 3])
            pitch_dot = pitch_amp * omega * torch.cos(omega * t + phase[:, 4])

            # build new UR10 root pose around base root state
            base = self._ur10_root_state_base  # [N, 13]
            ur10_root = base.clone()

            ur10_root[:, 0] = base[:, 0] + dx
            ur10_root[:, 1] = base[:, 1] + dy
            ur10_root[:, 2] = base[:, 2] + dz

            # orientation: base_quat ⊗ delta_quat(roll,pitch)
            base_q = base[:, 3:7]
            dq = self._quat_from_roll_pitch(roll, pitch)
            ur10_root[:, 3:7] = self._quat_mul(base_q, dq)

            # root velocities: linear + angular (approx)
            root_vel = torch.zeros((self.num_envs, 6), device=self.device, dtype=ur10_root.dtype)
            root_vel[:, 0] = vx
            root_vel[:, 1] = vy
            root_vel[:, 2] = vz
            root_vel[:, 3] = roll_dot
            root_vel[:, 4] = pitch_dot
            root_vel[:, 5] = 0.0

            # write to sim for all envs
            env_ids_all = self._DroneRobot._ALL_INDICES
            self._Ur10Arm.write_root_pose_to_sim(ur10_root[:, :7], env_ids_all)
            self._Ur10Arm.write_root_velocity_to_sim(root_vel, env_ids_all)

        # # Update desired_pos_w (target for drone) to UR10 end-effector position
        # ee_indices = self._Ur10Arm.find_bodies("ee_link")
        # if len(ee_indices) == 0:
        #     raise RuntimeError("Could not find 'ee_link' on UR10!")
        
        # # Always fetch the current ee_link position each step
        # ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :]  # shape [num_envs, 3]

        # # Try a fixed position for the goal 
        # self._desired_pos_w = ee_pos.squeeze(1)  # Update the dynamic goal position
        
        # Update phase-dependent goal buffers
        self._update_phase_goals()
    
    #____________________________________________________________________________#
    #____________________________________________________________________________#
    #_______________________THESE_ARE_HELPER_FUNCTIONS___________________________#
    #____________________________________________________________________________#
    #____________________________________________________________________________#

    def _wrap_pi(self, a: torch.Tensor) -> torch.Tensor:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _angle_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # shortest signed difference b-a in [-pi, pi]
        return self._wrap_pi(b - a)

    def _exp_smooth_alpha(self, dt: float, tau: float) -> float:
        # stable smoothing factor (0..1)
        if tau <= 1e-6:
            return 1.0
        return float(1.0 - math.exp(-dt / tau))

    def _hash_u01(self, i: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        # splitmix64-style with signed int64 constants
        c1 = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
        c2 = -4658895280553007687  # 0xBF58476D1CE4E5B9 as int64
        c3 = -7723592293110705685  # 0x94D049BB133111EB as int64

        x = i.to(torch.int64) ^ (seed.to(torch.int64) * c1)
        x = (x ^ (x >> 30)) * c2
        x = (x ^ (x >> 27)) * c3
        x = x ^ (x >> 31)

        return (x & 0xFFFFFFFF).to(torch.float32) / 2**32

    def _value_noise_1d(self, t: torch.Tensor, seed: torch.Tensor, freq: float) -> torch.Tensor:
        # t: [N] float, seed: [N] int64
        x = t * freq
        i0 = torch.floor(x).to(torch.int64)
        f = (x - i0.to(x.dtype)).clamp(0.0, 1.0)
        # smoothstep
        u = f * f * (3.0 - 2.0 * f)
        v0 = self._hash_u01(i0, seed)
        v1 = self._hash_u01(i0 + 1, seed)
        v = v0 * (1.0 - u) + v1 * u
        return v * 2.0 - 1.0  # [-1, 1]

    def _fbm_1d(self, t: torch.Tensor, seed: torch.Tensor, base_freq: float, octaves: int = 3) -> torch.Tensor:
        # fractal noise sum for richer motion
        out = torch.zeros_like(t)
        amp = 1.0
        freq = base_freq
        norm = 0.0
        for k in range(octaves):
            out = out + amp * self._value_noise_1d(t, seed + k * 1013, freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        return out / max(norm, 1e-6)

    def _update_wind_realistic(self, dt: float):
        # Noise-driven TURN RATES + integrate yaw (prevents big flips automatically)
        self._wind_time += dt
        t = self._wind_time + self._wind_noise_offset  # [N]

        # yaw rate noise (slow)
        yaw_rate_n = self._fbm_1d(t, self._wind_noise_seed, base_freq=0.08, octaves=3)  # [-1,1]
        yaw_rate = yaw_rate_n * self.wind_max_yaw_rate  # rad/s

        # integrate yaw
        self._wind_yaw = self._wrap_pi(self._wind_yaw + yaw_rate * dt)

        # speed noise (even slower)
        speed_n = self._fbm_1d(t + 17.3, self._wind_noise_seed, base_freq=0.04, octaves=3)  # [-1,1]
        target_speed = self.wind_vmin + (speed_n + 1.0) * 0.5 * (self.wind_vmax - self.wind_vmin)

        # smooth speed a bit (optional, but looks nicer)
        a_v = self._exp_smooth_alpha(dt, self.wind_speed_tau)
        self._wind_speed = self._wind_speed + a_v * (target_speed - self._wind_speed)

    def _update_wind_cone(self, dt: float):
        # “change every few seconds” but stay within cone around previous direction
        self._wind_time += dt
        self._wind_time_to_update -= dt

        needs = self._wind_time_to_update <= 0.0
        if needs.any():
            # new target within cone: delta in [-cone, +cone]
            t = (self._wind_time + self._wind_noise_offset)[needs]
            seed = self._wind_noise_seed[needs]

            delta_n = self._fbm_1d(t, seed, base_freq=0.12, octaves=2)  # [-1,1]
            delta = delta_n * self.wind_cone_half_angle

            self._wind_target_yaw[needs] = self._wrap_pi(self._wind_yaw[needs] + delta)

            speed_n = self._fbm_1d(t + 33.7, seed, base_freq=0.07, octaves=2)
            self._wind_target_speed[needs] = self.wind_vmin + (speed_n + 1.0) * 0.5 * (self.wind_vmax - self.wind_vmin)

            # reset timer
            self._wind_time_to_update[needs] = self.wind_update_interval

        # smooth toward targets
        a_dir = self._exp_smooth_alpha(dt, self.wind_dir_tau)
        a_spd = self._exp_smooth_alpha(dt, self.wind_speed_tau)

        dtheta = self._angle_diff(self._wind_yaw, self._wind_target_yaw)
        self._wind_yaw = self._wrap_pi(self._wind_yaw + a_dir * dtheta)
        self._wind_speed = self._wind_speed + a_spd * (self._wind_target_speed - self._wind_speed)

    def _update_wind(self, dt: float):
        if self.RealisticWindYesOrNo:
            self._update_wind_realistic(dt)
        else:
            self._update_wind_cone(dt)

        # write back to your existing buffers (direction + strength)
        self.wind_direction[:, 0] = torch.cos(self._wind_yaw)
        self.wind_direction[:, 1] = torch.sin(self._wind_yaw)
        self.wind_strength[:] = self._wind_speed  # keep your naming; it's your force scale

    def _apply_action(self) -> None:
        """
        Multi-agent action application with single-agent wind/gust logic
        and magnet (winning) behavior. Assumes the following buffers exist:
        - self.wind_direction [N,2], self.wind_strength [N]
        - self.wind_force [N,1,3], self.active_wind_force [N,1,3]
        - self.wind_timer [N], self.wind_cooldown [N]
        - self.wind_gust_timer [N], self.wind_gust_cooldown [N]
        - self._winning_condition [N] (bool), self._magnet_active [N] (bool)
        - self._thrust [N,1,3], self._moment [N,1,3], self._body_id (int or [1])
        """

        # === 1) Apply per-agent actions ===
        # Drone (writes self._thrust and self._moment)
        self._apply_drone_action(self._actions["_DroneRobot"])

        # UR10 arm (your original smoothing + saturation)
        ur10_action = self._actions["_Ur10Arm"]
        scaled_targets = scale(ur10_action, self.arm_dof_lower_limits, self.arm_dof_upper_limits)

        self.arm_curr_targets = (
            self.cfg.act_moving_average * scaled_targets
            + (1.0 - self.cfg.act_moving_average) * self.arm_prev_targets
        )
        self.arm_curr_targets = saturate(self.arm_curr_targets, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        self._Ur10Arm.set_joint_position_target(self.arm_curr_targets)
        self.arm_prev_targets = self.arm_curr_targets.clone()

        dt = self.step_dt
        device = self.device

        # -----------------------------------------------------------
        # Wind / gusts controlled by PACE/config flags
        # -----------------------------------------------------------
        if self._wind_enabled:
            self._update_wind(dt)

            self.wind_force[:, 0, 0] = self.wind_direction[:, 0] * self.wind_strength
            self.wind_force[:, 0, 1] = self.wind_direction[:, 1] * self.wind_strength
            self.wind_force[:, 0, 2] = 0.0
        else:
            self.wind_force.zero_()

        if self._wind_gusts_enabled:
            self.wind_gust_timer -= dt
            self.wind_gust_cooldown -= dt

            end_gust = self.wind_gust_timer <= 0
            if end_gust.any():
                self.active_wind_force[end_gust] = 0.0

            can_gust = self.wind_gust_cooldown <= 0
            trigger_gust = can_gust & (torch.rand(self.num_envs, device=device) < 0.02)

            suppress_after_win = getattr(self.cfg, "suppress_gusts_on_win", True)
            eligible = (~self._winning_condition) if suppress_after_win else torch.ones_like(self._winning_condition)

            if trigger_gust.any():
                tg = trigger_gust & eligible
                if tg.any():
                    gust_dirs = torch.nn.functional.normalize(torch.randn_like(self.active_wind_force), dim=-1)
                    gust_mags = torch.empty((self.num_envs, 1, 1), device=device).uniform_(0.1, 0.3)
                    self.active_wind_force[tg] = gust_dirs[tg] * gust_mags[tg]
                    self.wind_gust_timer[tg] = torch.randint(15, 40, (tg.sum(),), device=device) * dt
                    self.wind_gust_cooldown[tg] = torch.randint(100, 300, (tg.sum(),), device=device) * dt
        else:
            self.wind_gust_timer.zero_()
            self.wind_gust_cooldown.zero_()
            self.active_wind_force.zero_()


        # apply combined wind + thrust/torque to ALL envs (no "magnetized" split)
        combined_forces  = self._thrust + (self.wind_force + self.active_wind_force)
        combined_torques = self._moment
        self._DroneRobot.set_external_force_and_torque(
            forces=combined_forces,
            torques=combined_torques,
            body_ids=self._body_id
        )

            # NOTE: no magnet_offset, no PD attach, no special forces after "win".

    def _apply_drone_action(self, action: torch.Tensor) -> None:
        """
        Apply the drone action to the environment.
        """
        if not hasattr(self, "_thrust"):
            print("[ERROR] _thrust not initialized!")
        
        if not hasattr(self, "_moment"):
            print("[ERROR] _moment not initialized!")

        # Convert action to thrust and moment
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (action[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * action[:, 1:4]

        # # Apply force and torque
        # self._DroneRobot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _apply_ur10_action(self, action: torch.Tensor) -> None:
        """
        Apply the UR10 action to the environment.
        """
        # Convert normalized actions to joint position targets
        joint_targets = action * torch.tensor(
            [2.0, 2.0, 2.0, 3.14, 3.14, 3.14], device=self.device
        )
        self._Ur10Arm.set_joint_position_target(joint_targets)
     
    # def _compute_drone_obs(self) -> torch.Tensor:
    #     return torch.cat((
    #         self._DroneRobot.data.root_pos_w[:, 0, :],
    #         self._DroneRobot.data.root_lin_vel_w[:, 0, :],
    #         self._DroneRobot.data.root_ang_vel_w[:, 0, :],
    #         self._DroneRobot.data.root_quat_w[:, 0, :],
    #         self.actions["_DroneRobot"],
    #         self._desired_pos_w,
    #         self.wind_force[:, 0, :].squeeze(1),  # shape [num_envs, 3]
    #     ), dim=-1)

    def _compute_drone_obs(self) -> torch.Tensor:
        wind_obs = self.wind_force[:, 0, :].squeeze(1) if self._include_wind_in_obs else torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        return torch.cat((
            self._DroneRobot.data.root_pos_w[:, 0, :],
            self._DroneRobot.data.root_lin_vel_w[:, 0, :],
            self._DroneRobot.data.root_ang_vel_w[:, 0, :],
            self._DroneRobot.data.root_quat_w[:, 0, :],
            self.actions["_DroneRobot"],
            self._drone_goal_pos_w,
            wind_obs,
        ), dim=-1)


    # def _compute_ur10_obs(self) -> torch.Tensor:
    #     return torch.cat((
    #         self._Ur10Arm.data.joint_pos,
    #         self._Ur10Arm.data.joint_vel,
    #         self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :],
    #         self.actions["_Ur10Arm"],
    #         self.wind_force[:, 0, :].squeeze(1),  # shape [num_envs, 3]
    #     ), dim=-1)

    def _compute_ur10_obs(self) -> torch.Tensor:
        wind_obs = self.wind_force[:, 0, :].squeeze(1) if self._include_wind_in_obs else torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        ee_pos = self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :]
        if ee_pos.ndim == 3:
            ee_pos = ee_pos.squeeze(1)

        return torch.cat((
            self._Ur10Arm.data.joint_pos,
            self._Ur10Arm.data.joint_vel,
            ee_pos,
            self.actions["_Ur10Arm"],
            wind_obs,
        ), dim=-1)

    # def _get_observations(self) -> dict[str, torch.Tensor]:
        
        
    #     # Get the wind as part of the observation
    #     wind_forces = self.wind_force[:, 0, :].squeeze(1)  # shape [num_envs, 3]
        
    #     observations = {

    #         # === UR10 Arm Observations ===
    #         # Joint positions (scaled)
    #         # Joint velocities (scaled)
    #         # End-effector position (world space)
    #         # End-effector orientation (world space)
    #         # End-effector linear velocity (world space)
    #         # End-effector angular velocity (world space)
    #         # Previous actions
    #         # Drone position (world space)
            
    #         "_Ur10Arm": torch.cat(
    #             (
    #                 unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
    #                 self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
    #                 self.ee_pos,
    #                 self.ee_quat,
    #                 self.ee_lin_vel,
    #                 self.ee_ang_vel,
    #                 self.actions["_Ur10Arm"],
                    
    #                 #self._desired_pos_w, 
    #                 # Probably also add the position of the drone as well
    #                 self._DroneRobot.data.root_pos_w,

    #                 # add the wind forces as well
    #                 wind_forces, # (3)
    #             ),
    #             dim=-1,
    #         ),
    #         "_DroneRobot": torch.cat(
    #             (
    #                 # Drone position (3)
    #                 self._DroneRobot.data.root_pos_w,
    #                 # Drone orientation (quat, 4)
    #                 self._DroneRobot.data.root_quat_w,
    #                 # Drone linear velocity (3)
    #                 self._DroneRobot.data.root_lin_vel_w,
    #                 # Drone angular velocity (3)
    #                 self._DroneRobot.data.root_ang_vel_w,
    #                 # Previously applied actions
    #                 self.actions["_DroneRobot"],
    #                 # Goal again put as _desired_pos_w as this is the ee_pos
    #                 self._desired_pos_w,

    #                 # add the wind information as well
    #                 wind_forces, # (3)
    #             ),
    #             dim=-1,
    #         ),
    #     }
    #     return observations

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Wind observation
        if self._include_wind_in_obs:
            wind_forces = self.wind_force[:, 0, :].squeeze(1)
        else:
            wind_forces = torch.zeros((self.num_envs, 3), device=self.device)

        # Arm goal info.
        if self._include_goal_in_obs:
            arm_goal_rel_obs = self._arm_goal_pos_w - self.ee_pos
        else:
            arm_goal_rel_obs = torch.zeros((self.num_envs, 3), device=self.device)

        drone_pos_w = self._DroneRobot.data.root_pos_w
        drone_local_pos = drone_pos_w - self._terrain.env_origins

        # Goal info: relative vector from drone to goal.
        # This tells the policy directly which direction the red dot is.
        if self._include_goal_in_obs:
            drone_goal_obs = self._drone_goal_pos_w - drone_pos_w
        else:
            drone_goal_obs = torch.zeros((self.num_envs, 3), device=self.device)
        
        observations = {
            "_Ur10Arm": torch.cat(
                (
                    unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
                    self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
                    self.ee_pos,
                    self.ee_quat,
                    self.ee_lin_vel,
                    self.ee_ang_vel,
                    self.actions["_Ur10Arm"],
                    arm_goal_rel_obs,
                    wind_forces,
                ),
                dim=-1,
            ),
            "_DroneRobot": torch.cat(
                (
                    drone_local_pos,
                    self._DroneRobot.data.root_quat_w,
                    self._DroneRobot.data.root_lin_vel_w,
                    self._DroneRobot.data.root_ang_vel_w,
                    self.actions["_DroneRobot"],
                    drone_goal_obs,
                    wind_forces,
                ),
                dim=-1,
            ),
        }
        return observations 
        
    # def _get_states(self) -> torch.Tensor:
    #     states = torch.cat(
    #         (
    #             # === UR10 ===
    #             unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
    #             self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
    #             self.ee_pos,
    #             self.ee_quat,
    #             self.ee_lin_vel,
    #             self.ee_ang_vel,
    #             self.actions["_Ur10Arm"],
    #             # === Drone ===
    #             self._DroneRobot.data.root_pos_w,
    #             self._DroneRobot.data.root_quat_w,
    #             self._DroneRobot.data.root_lin_vel_w,
    #             self._DroneRobot.data.root_ang_vel_w,
    #             self.actions["_DroneRobot"],
    #             # === Goal ===
    #             self._desired_pos_w, # this is the ee_pos (Note to self: Change the naming of this variable to ee_pos as its easier to understand :D )
    #             # === Wind ===
    #             self.wind_force[:, 0, :].squeeze(1),  # shape [num_envs, 3]
    #         ),
    #         dim=-1,
    #     )
    #     return states

    def _get_states(self) -> torch.Tensor:
        wind_state = self.wind_force[:, 0, :].squeeze(1) if self._include_wind_in_obs else torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        states = torch.cat(
            (
                # === UR10 ===
                unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
                self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
                self.ee_pos,
                self.ee_quat,
                self.ee_lin_vel,
                self.ee_ang_vel,
                self.actions["_Ur10Arm"],

                # === Drone ===
                self._DroneRobot.data.root_pos_w,
                self._DroneRobot.data.root_quat_w,
                self._DroneRobot.data.root_lin_vel_w,
                self._DroneRobot.data.root_ang_vel_w,
                self.actions["_DroneRobot"],

                # === Goal ===
                self._drone_goal_pos_w,

                # === Wind ===
                wind_state,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        Phase-aware multi-agent rewards.
        """

        # ----------------------------
        # Shared signals
        # ----------------------------
        lin_vel = torch.sum(torch.square(self._DroneRobot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._DroneRobot.data.root_ang_vel_b), dim=1)

        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        drone_quat = self._DroneRobot.data.root_quat_w

        ee_idx = self.ee_idx
        ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_idx, :]
        ee_quat = self._Ur10Arm.data.body_quat_w[:, ee_idx, :]

        if ee_pos.ndim == 3:
            ee_pos = ee_pos.squeeze(1)
        if ee_quat.ndim == 3:
            ee_quat = ee_quat.squeeze(1)

        # ============================================================
        # PACE 1: separated static-goal training
        # ============================================================
        if self._pace == 1:
            drone_total_reward = self._compute_drone_reward_pace_1(
                lin_vel=lin_vel,
                ang_vel=ang_vel,
                drone_pos=drone_pos,
            )

            arm_total_reward = self._compute_arm_reward_pace_1()

            return {
                "_Ur10Arm": arm_total_reward,
                "_DroneRobot": drone_total_reward,
            }

        # ============================================================
        # PACE 0: legacy shared landing behavior
        # ============================================================
        distance_to_goal = torch.linalg.norm(self._drone_goal_pos_w - self._DroneRobot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        is_close = distance_to_goal < 0.25
        is_slow = lin_vel < 10
        smooth_landing = (is_close & is_slow).float()
        proximity = (distance_to_goal < 0.25).float()

        drone_up = quat_apply(
            drone_quat,
            torch.tensor([0.0, 0.0, 1.0], device=drone_quat.device, dtype=drone_quat.dtype).expand(self.num_envs, 3)
        )

        local_x = torch.tensor([1, 0, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_y = torch.tensor([0, 1, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_z = torch.tensor([0, 0, 1], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)

        world_x = quat_apply(ee_quat, local_x)
        world_y = quat_apply(ee_quat, local_y)
        world_z = quat_apply(ee_quat, local_z)
        self._ee_local_axes_in_world = [world_x, world_y, world_z]

        ee_up = world_x

        alignment = torch.sum(drone_up * ee_up, dim=1)
        aligned_enough = alignment > self.cfg.alignment_threshold
        in_approach_zone = distance_to_goal < self.cfg.approach_zone
        alignment_reward = (alignment * self.cfg.alignment_reward * self.step_dt) * in_approach_zone.float()

        magnet_condition_raw = (
            (distance_to_goal < self.cfg.magnet_condition_distance)
            & (lin_vel < self.cfg.magnet_condition_max_speed)
            & aligned_enough
        )

        self._magnet_condition_counter = torch.where(
            magnet_condition_raw,
            self._magnet_condition_counter + 1,
            torch.zeros_like(self._magnet_condition_counter)
        )

        magnet_condition = self._magnet_condition_counter >= self._magnet_required_steps
        magnet_reward = magnet_condition.float() * self.cfg.magnet_reward * self.step_dt
        self._winning_condition |= magnet_condition

        time_shaping = (1.0 - (self.episode_length_buf / self.max_episode_length))

        z_alignment = ee_up[:, 2]
        self._ee_alignment = z_alignment
        orientation_reward = z_alignment * self.cfg.orientation_reward_scale

        near = in_approach_zone.float()
        far = 1.0 - near

        safe_thr = getattr(self.cfg, "safe_z_alignment_threshold", 0.90)
        safe = (z_alignment > safe_thr).float()

        arm_qd = self._Ur10Arm.data.joint_vel
        arm_motion = torch.sum(arm_qd * arm_qd, dim=1)

        died = self._get_drone_out_of_bounds()
        died_penalty = died.float() * self.cfg.died_penalty

        z_threshold = self.cfg.wrist_height_penalty_scale
        w1 = self._Ur10Arm.data.body_pos_w[:, self.wrist_1_idx, 2]
        w2 = self._Ur10Arm.data.body_pos_w[:, self.wrist_2_idx, 2]
        w3 = self._Ur10Arm.data.body_pos_w[:, self.wrist_3_idx, 2]

        w1_above = (w1 > z_threshold).float()
        w2_above = (w2 > z_threshold).float()
        w3_above = (w3 > z_threshold).float()

        wrist_height_score = (w1_above + w2_above + w3_above) / 3.0
        wrist_reward = (wrist_height_score * self.cfg.wrist_height_reward_scale * self.step_dt).squeeze(-1)

        if torch.rand(1).item() < 0.05:
            print(f"[DEBUG] dist: {distance_to_goal.mean():.3f}, vel: {lin_vel.mean():.3f}, ang_vel: {ang_vel.mean():.3f}")
            print(f"[DEBUG] drone Z: {self._DroneRobot.data.root_pos_w[:, 2].mean():.3f}")
            print(f"[DEBUG] ee_link Z: {self._Ur10Arm.data.body_pos_w[:, self.ee_idx, 2].mean():.3f}")
            print(f"[DEBUG] Orientation reward mean: {(orientation_reward * self.step_dt).mean():.3f}")
            print(f"[DEBUG] wind_enabled={self._wind_enabled}, gusts={self._wind_gusts_enabled}, wind_mean={self.wind_force.mean():.4f}")
        
        self._episode_success_flags |= (is_close & is_slow)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "smooth_landing": smooth_landing * self.cfg.smooth_landing_bonus * self.step_dt,
            "proximity": proximity * self.cfg.proximity_bonus * self.step_dt,
            "time_shaping": time_shaping * self.cfg.time_bonus_scale * self.step_dt,
            "alignment_reward": alignment_reward,
            "magnet_reward": magnet_reward,
            "orientation_reward": orientation_reward * self.step_dt,
            "wrist_height_reward": wrist_reward,
            "arm_go_safe": (
                getattr(self.cfg, "arm_go_safe_scale", 1.0)
                * far * (1.0 - safe) * z_alignment * self.step_dt
            ),
            "arm_hold_still": (
                -getattr(self.cfg, "arm_hold_still_scale", 0.5)
                * far * safe * arm_motion * self.step_dt
            ),
            "arm_near_jitter": (
                -getattr(self.cfg, "arm_near_jitter_scale", 0.05)
                * near * arm_motion * self.step_dt
            ),
            "died_penalty": died_penalty,
        }

        for k, v in rewards.items():
            self._episode_sums[k] += v

        drone_total_reward = (
            rewards["lin_vel"]
            + rewards["ang_vel"]
            + rewards["distance_to_goal"]
            + rewards["smooth_landing"]
            + rewards["proximity"]
            + rewards["time_shaping"]
            + rewards["alignment_reward"]
            + rewards["magnet_reward"]
            + rewards["died_penalty"]
        )

        ur10_help_distance = rewards["distance_to_goal"] * near

        ur10_total_reward = (
            rewards["orientation_reward"]
            + rewards["wrist_height_reward"]
            + ur10_help_distance
            + rewards["arm_go_safe"]
            + rewards["arm_hold_still"]
            + rewards["arm_near_jitter"]
        )

        return {"_Ur10Arm": ur10_total_reward, "_DroneRobot": drone_total_reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Phase-aware termination logic.

        Important:
        DirectMARLEnv resets cloned environments at the env level.
        So if the drone dies, both agents in that env must receive the same
        termination signal, otherwise the env may not reset.
        """

        # --- Boundary checks ---
        drone_oob = self._get_drone_out_of_bounds()
        arm_oob = self._get_arm_out_of_bounds()

        # --- Timeout ---
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # If either important actor causes an env-level failure, reset the whole env.
        # For PACE 1, drone_oob is the main reset trigger.
        env_terminated = drone_oob | arm_oob

        terminated = {
            "_DroneRobot": env_terminated,
            "_Ur10Arm": env_terminated,
        }

        time_outs = {
            "_DroneRobot": time_out,
            "_Ur10Arm": time_out,
        }

        return terminated, time_outs
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Normalize env_ids to a 1D tensor of indices
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._DroneRobot._ALL_INDICES

        device = self.device

        def _get_flag(name_list, default=False):
            """Try several attribute names; if none found, return a bool tensor (default)."""
            for nm in name_list:
                if hasattr(self, nm):
                    buf = getattr(self, nm)
                    # Some wrappers store as float {0,1}; coerce to bool
                    if buf.dtype != torch.bool:
                        return (buf != 0)
                    return buf
            return torch.zeros(self.num_envs, dtype=torch.bool, device=device) if default is False \
                else torch.ones(self.num_envs, dtype=torch.bool, device=device)

        terminated_flags = _get_flag(
            ["reset_terminated", "terminated_buf", "done_buf", "resets_terminated"], default=False
        )
        timeout_flags = _get_flag(
            ["reset_time_outs", "time_out_buf", "timeout_buf", "resets_time_outs"], default=False
        )

        # -----------------------------
        # Logging (episode reward sums + metrics)
        # -----------------------------
        # final_distance_to_goal = torch.linalg.norm(
        #     self._desired_pos_w[env_ids] - self._DroneRobot.data.root_pos_w[env_ids], dim=1
        # ).mean()
        final_distance_to_goal = torch.linalg.norm(
            self._drone_goal_pos_w[env_ids] - self._DroneRobot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Per-episode termination counts (robust to missing flags)
        extras["Episode_Termination/died"] = torch.count_nonzero(terminated_flags[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(timeout_flags[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # -----------------------------
        # Finalize episode outcome at reset
        # -----------------------------
        # Ensure status buffers exist
        if not hasattr(self, "_success_status"):
            self._success_status = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        if not hasattr(self, "_episode_success_flags"):
            self._episode_success_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        if not hasattr(self, "_winning_condition"):
            self._winning_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        # Reset status for selected envs
        self._success_status[env_ids] = 0

        # Mark successes
        success_env_ids = env_ids[self._episode_success_flags[env_ids]]
        self._success_status[success_env_ids] = 1

        magnet_env_ids = env_ids[self._winning_condition[env_ids]]
        self._success_status[magnet_env_ids] = 2  # magnet success > landing

        # Crashes / timeouts (use compat flags)
        crash_env_ids = env_ids[terminated_flags[env_ids]]
        self._success_status[crash_env_ids] = -1

        timeout_env_ids = env_ids[timeout_flags[env_ids]]
        timeout_failed_env_ids = timeout_env_ids[~self._episode_success_flags[timeout_env_ids]]
        self._success_status[timeout_failed_env_ids] = -2

        # Clear per-episode flags for these envs
        self._episode_success_flags[env_ids] = False
        self._winning_condition[env_ids] = False

        # Aggregate counts for logs
        success_count = torch.sum(self._success_status[env_ids] == 1).item()
        magnet_was_success_count = torch.sum(self._success_status[env_ids] == 2).item()
        crash_count = torch.sum(self._success_status[env_ids] == -1).item()
        timeout_count = torch.sum(self._success_status[env_ids] == -2).item()

        self.extras["log"]["Episode_Success/success"] = success_count
        self.extras["log"]["Episode_Success/magnet"] = magnet_was_success_count
        self.extras["log"]["Episode_Success/crash"] = crash_count
        self.extras["log"]["Episode_Success/timeout"] = timeout_count

        # -----------------------------
        # Reset actors
        # -----------------------------
        self._DroneRobot.reset(env_ids)
        self._Ur10Arm.reset(env_ids)

        # Also reset magnet-active flags if you maintain one
        if hasattr(self, "_magnet_active"):
            self._magnet_active[env_ids] = False

        # Base class reset (handles buffers like episode_length_buf, etc.)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid spikes
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # -----------------------------
        # Zero actions per agent (MARL-safe)
        # -----------------------------
        if isinstance(self._actions, dict):
            for k in self._actions.keys():
                self._actions[k][env_ids] = 0.0
        else:
            # Fallback if wrapper stacks into a single tensor
            self._actions[env_ids] = 0.0



        #____________________________________________________________________________
        #____________________________________________________________________________
        #____________________________________________________________________________
        #Reset the wind buffers as well
        # reset wind state for these envs
        self._wind_yaw[env_ids] = torch.empty_like(self._wind_yaw[env_ids]).uniform_(-math.pi, math.pi)
        self._wind_speed[env_ids] = torch.empty_like(self._wind_speed[env_ids]).uniform_(self.wind_vmin, self.wind_vmax)
        self._wind_target_yaw[env_ids] = self._wind_yaw[env_ids]
        self._wind_target_speed[env_ids] = self._wind_speed[env_ids]
        self._wind_time_to_update[env_ids] = torch.empty_like(self._wind_time_to_update[env_ids]).uniform_(0.0, self.wind_update_interval)

        # keep the old buffers consistent
        self.wind_direction[env_ids, 0] = torch.cos(self._wind_yaw[env_ids])
        self.wind_direction[env_ids, 1] = torch.sin(self._wind_yaw[env_ids])
        self.wind_strength[env_ids] = self._wind_speed[env_ids]
        self.wind_force[env_ids] = 0.0

        # gust state reset
        self.wind_gust_timer[env_ids] = 0.0
        self.wind_gust_cooldown[env_ids] = 0.0
        self.active_wind_force[env_ids] = 0.0

        #____________________________________________________________________________
        #____________________________________________________________________________
        #____________________________________________________________________________

        # -----------------------------
        # Randomize initial states
        # -----------------------------
        # Drone
        # joint_pos = self._DroneRobot.data.default_joint_pos[env_ids]
        # joint_vel = self._DroneRobot.data.default_joint_vel[env_ids]
        # default_root_state = self._DroneRobot.data.default_root_state[env_ids]

        joint_pos = self._DroneRobot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._DroneRobot.data.default_joint_vel[env_ids].clone()
        drone_default_root_state = self._DroneRobot.data.default_root_state[env_ids].clone()


        # drone_default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # drone_default_root_state[:, 0] += torch.zeros(len(env_ids), device=device).uniform_(-0.5, 0.5)
        # drone_default_root_state[:, 1] += torch.zeros(len(env_ids), device=device).uniform_(-0.5, 0.5)
        # drone_default_root_state[:, 2] += torch.zeros(len(env_ids), device=device).uniform_(0.0, 0.5)

        drone_default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        if self._use_separated_training_boxes:
            n = len(env_ids)

            drone_default_root_state[:, 0] += torch.empty(n, device=device).uniform_(
                self.cfg.drone_side_x_center - self.cfg.side_half_width,
                self.cfg.drone_side_x_center + self.cfg.side_half_width,
            )
            drone_default_root_state[:, 1] += torch.empty(n, device=device).uniform_(
                self.cfg.drone_box_y_min + self.cfg.reset_spawn_margin_xy,
                self.cfg.drone_box_y_max - self.cfg.reset_spawn_margin_xy,
            )
            drone_default_root_state[:, 2] += torch.empty(n, device=device).uniform_(
                self.cfg.drone_box_z_min + self.cfg.reset_spawn_margin_z,
                self.cfg.drone_box_z_max - self.cfg.reset_spawn_margin_z,
            )
        else:
            drone_default_root_state[:, 0] += torch.empty(len(env_ids), device=device).uniform_(-0.5, 0.5)
            drone_default_root_state[:, 1] += torch.empty(len(env_ids), device=device).uniform_(-0.5, 0.5)
            drone_default_root_state[:, 2] += torch.empty(len(env_ids), device=device).uniform_(0.0, 0.5)

        self._DroneRobot.write_root_pose_to_sim(drone_default_root_state[:, :7], env_ids)
        self._DroneRobot.write_root_velocity_to_sim(drone_default_root_state[:, 7:], env_ids)
        self._DroneRobot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # --- UR10 root state (single source of truth) ---
        ur10_root_state = self._Ur10Arm.data.default_root_state[env_ids].clone()
        # ur10_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # ur10_root_state[:, 0] += torch.zeros(len(env_ids), device=device).uniform_(-0.2, 0.2)
        # ur10_root_state[:, 1] += torch.zeros(len(env_ids), device=device).uniform_(-0.2, 0.2)

        ur10_root_state[:, :3] += self._terrain.env_origins[env_ids]

        if self._use_separated_training_boxes:
            n = len(env_ids)

            ur10_root_state[:, 0] += torch.empty(n, device=device).uniform_(
                self.cfg.arm_side_x_center - 0.20,
                self.cfg.arm_side_x_center + 0.20,
            )
            ur10_root_state[:, 1] += torch.empty(n, device=device).uniform_(-0.20, 0.20)
        else:
            ur10_root_state[:, 0] += torch.empty(len(env_ids), device=device).uniform_(-0.2, 0.2)
            ur10_root_state[:, 1] += torch.empty(len(env_ids), device=device).uniform_(-0.2, 0.2)

        # Deterministic joint reset
        joint_pos = self._Ur10Arm.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        # Store the reset root state as the reference for platform motion
        self._ur10_root_state_base[env_ids] = ur10_root_state


        self._Ur10Arm.write_root_pose_to_sim(ur10_root_state[:, :7], env_ids)
        self._Ur10Arm.write_root_velocity_to_sim(ur10_root_state[:, 7:], env_ids)
        self._Ur10Arm.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Keep controller buffers aligned
        self.arm_prev_targets[env_ids] = joint_pos
        self.arm_curr_targets[env_ids] = joint_pos

        # Reset phase-dependent goals after actors have been placed
        self._reset_phase_goals(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/toggle debug markers."""
        if debug_vis:
            # --- Goal marker ---
            # if not hasattr(self, "goal_pos_visualizer"):
            #     marker_cfg = CUBOID_MARKER_CFG.copy()
            #     marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
            #     marker_cfg.prim_path = "/Visuals/Command/goal_position"
            #     self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # self.goal_pos_visualizer.set_visibility(True)

            # --- Drone goal marker ---
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/drone_goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            # --- Arm goal marker ---
            if not hasattr(self, "arm_goal_visualizer"):
                arm_marker_cfg = CUBOID_MARKER_CFG.copy()
                arm_marker_cfg.markers["cuboid"].size = (0.07, 0.07, 0.07)
                arm_marker_cfg.prim_path = "/Visuals/Command/arm_goal_position"
                self.arm_goal_visualizer = VisualizationMarkers(arm_marker_cfg)
            self.arm_goal_visualizer.set_visibility(True)

            # --- End-effector frame marker ---
            if not hasattr(self, "ee_frame_visualizer"):
                frame_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/EndEffector/frame",
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.05, 0.05, 0.05),
                        )
                    },
                )
                self.ee_frame_visualizer = VisualizationMarkers(frame_marker_cfg)
            self.ee_frame_visualizer.set_visibility(True)

            # --- Wind markers ---
            if hasattr(self, "wind_markers"):
                self.wind_markers.set_visibility(True)

            # --- Platform motion debug marker (arrow above UR10 base) ---
            show_platform = getattr(self.cfg, "platform_motion_debug_vis", True)
            if show_platform:
                if not hasattr(self, "platform_markers"):
                    self.platform_marker_cfg = VisualizationMarkersCfg(
                        prim_path="/World/Visuals/PlatformMotion",
                        markers={
                            "platform_arrow": sim_utils.UsdFileCfg(
                                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                                scale=(0.15, 0.03, 0.6),
                                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                            )
                        },
                    )
                    self.platform_markers = VisualizationMarkers(self.platform_marker_cfg)
                self.platform_markers.set_visibility(True)
            else:
                if hasattr(self, "platform_markers"):
                    self.platform_markers.set_visibility(False)

        else:
            # turn everything off
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "arm_goal_visualizer"):
                self.arm_goal_visualizer.set_visibility(False)
            if hasattr(self, "ee_frame_visualizer"):
                self.ee_frame_visualizer.set_visibility(False)
            if hasattr(self, "wind_markers"):
                self.wind_markers.set_visibility(False)
            if hasattr(self, "platform_markers"):
                self.platform_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug markers each frame."""
        # # --- Goal marker ---
        # if hasattr(self, "goal_pos_visualizer"):
        #     #self.goal_pos_visualizer.visualize(self._desired_pos_w)
        #     self.goal_pos_visualizer.visualize(self._drone_goal_pos_w)
        
        # --- Drone goal marker ---
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._drone_goal_pos_w)

        # --- Arm goal marker ---
        if hasattr(self, "arm_goal_visualizer"):
            self.arm_goal_visualizer.visualize(self._arm_goal_pos_w)


        # --- Existing success/failure print (optional; can be noisy) ---
        status = self._success_status.cpu().numpy()
        print(
            f"[STEP {self._step_count}] Success: {(status == 1).sum()} | "
            f"Magnet Success: {(status == 2).sum()} | Failure: {(status == -1).sum()} | Timeout: {(status == -2).sum()}"
        )

        # --- Wind arrow visualization ---
        if hasattr(self, "wind_markers"):
            drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
            active_vecs = self.active_wind_force[:, 0, :]     # [N, 3]
            constant_vecs = self.wind_force[:, 0, :]          # [N, 3]
            wind_vecs = active_vecs + constant_vecs           # [N, 3]

            wind_dirs = torch.nn.functional.normalize(wind_vecs, dim=1)
            yaw_angles = torch.atan2(wind_dirs[:, 1], wind_dirs[:, 0])
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
            arrow_orients = quat_from_angle_axis(yaw_angles, z_axis)

            wind_mags = torch.norm(wind_vecs, dim=-1)
            normed = wind_mags.clamp(0.0, 1.0)
            colors = torch.zeros((self.num_envs, 3), device=self.device)

            low_mask = normed < 0.5
            t_low = normed[low_mask] * 2.0
            colors[low_mask] = (
                (1.0 - t_low).unsqueeze(-1) * torch.tensor([0.2, 0.6, 1.0], device=self.device)
                + t_low.unsqueeze(-1) * torch.tensor([1.0, 1.0, 0.0], device=self.device)
            )

            high_mask = ~low_mask
            t_high = (normed[high_mask] - 0.5) * 2.0
            colors[high_mask] = (
                (1.0 - t_high).unsqueeze(-1) * torch.tensor([1.0, 1.0, 0.0], device=self.device)
                + t_high.unsqueeze(-1) * torch.tensor([1.0, 0.0, 0.0], device=self.device)
            )

            self.wind_markers.visualize(drone_pos, arrow_orients, colors)

        # --- End-effector frame marker ---
        if hasattr(self, "ee_frame_visualizer"):
            ee_indices = self._Ur10Arm.find_bodies("ee_link")
            if len(ee_indices) > 0:
                ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :].squeeze(1)
                ee_quat = self._Ur10Arm.data.body_quat_w[:, ee_indices[0], :].squeeze(1)
                self.ee_frame_visualizer.visualize(ee_pos, ee_quat)

        # --- Platform motion marker (arrow above UR10 base) ---
        if hasattr(self, "platform_markers") and getattr(self.cfg, "platform_motion_debug_vis", True):
            ur10_pos = self._Ur10Arm.data.root_pos_w[:, :3]
            arrow_pos = ur10_pos + torch.tensor([0.0, 0.0, 0.30], device=self.device)

            # Point arrow "up": arrow_x rotated +90° around Y
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, -1)
            up_orient = quat_from_angle_axis(
                torch.full((self.num_envs,), math.pi / 2, device=self.device),
                y_axis,
            )

            # Color encodes normalized heave magnitude |dz|/Az
            Az = float(getattr(self.cfg, "platform_heave_amplitude", 0.0))
            dz = getattr(self, "_platform_dz", torch.zeros(self.num_envs, device=self.device))
            norm = torch.abs(dz) / max(Az, 1e-6)
            norm = norm.clamp(0.0, 1.0)

            colors = torch.zeros((self.num_envs, 3), device=self.device)

            # green -> yellow -> red
            low = norm < 0.5
            t = (norm[low] * 2.0).unsqueeze(-1)
            colors[low] = (1.0 - t) * torch.tensor([0.2, 1.0, 0.2], device=self.device) + t * torch.tensor(
                [1.0, 1.0, 0.0], device=self.device
            )

            high = ~low
            t2 = ((norm[high] - 0.5) * 2.0).unsqueeze(-1)
            colors[high] = (1.0 - t2) * torch.tensor([1.0, 1.0, 0.0], device=self.device) + t2 * torch.tensor(
                [1.0, 0.2, 0.2], device=self.device
            )

            self.platform_markers.visualize(arrow_pos, up_orient, colors)

        
            

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions q = q1 ⊗ q2, with quats in (w, x, y, z)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    @staticmethod
    def _quat_from_roll_pitch(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        """Quaternion from roll (x) and pitch (y), zero yaw. Output (w, x, y, z)."""
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        # q = q_pitch ⊗ q_roll (or roll then pitch; for small angles it won't matter much)
        # roll about x: (cr, sr, 0, 0)
        # pitch about y: (cp, 0, sp, 0)
        w = cp * cr
        x = cp * sr
        y = sp * cr
        z = -sp * sr
        return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def saturate(x, low, high):
    return torch.max(torch.min(x, high), low)

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

#yeah
