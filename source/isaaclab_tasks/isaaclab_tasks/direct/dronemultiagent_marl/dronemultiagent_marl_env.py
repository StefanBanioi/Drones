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

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.markers import CUBOID_MARKER_CFG  
from .dronemultiagent_marl_env_cfg import DronemultiagentMarlEnvCfg
        

class DronemultiagentMarlEnv(DirectMARLEnv):
    cfg: DronemultiagentMarlEnvCfg

    def __init__(self, cfg: DronemultiagentMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._magnet_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Magnetic capture condition active
        
        # Magnet condition tracking with the second counter 
        self._magnet_condition_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._magnet_required_steps = int(self.cfg.magnet_time_threshold_in_seconds / self.step_dt)  # e.g., 3s / 0.05 = 60 steps
        self._actions = {}

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

        # === Goal setup (optional, based on your reward function) ===
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Base local goal position
        base_local_goal = torch.tensor([1.0, -1.0, 1.0], device=self.device)  # (3,)

        # Add randomness per environment (e.g., ±0.2 meters)
        random_offset = torch.empty((self.num_envs, 3), device=self.device).uniform_(-0.4, 0.4)

        # Compute world-space goal per environment
        goal_pos_w = self._terrain.env_origins + base_local_goal + random_offset  # (num_envs, 3)

        # Assign goal positions
        self.goal_pos[:] = goal_pos_w



        # Marker for visualization
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

         # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_penalty",
                "ang_vel_penalty",
                "distance_reward",
                "distance_reward_ur10",
                "smooth_landing_bonus",
                "proximity_bonus",
                "time_shaping_reward",
                "orientation_reward",
                "died_penalty",
                "wrist_height_reward", 
            ]
        }

        # Get specific body indices
        self._body_id = self._DroneRobot.find_bodies("body")[0]
        self._robot_mass = (self._DroneRobot.root_physx_view.get_masses()[0].sum())  # scale to 3x size (volume scales with the cube of length)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    

        # Unit tensors (optional, useful for directional reward calculations)
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        ee_indices = self._Ur10Arm.find_bodies("ee_link")
        self.ee_idx = ee_indices[0]

        if len(ee_indices) == 0:
            raise RuntimeError("Could not find 'ee_link' on UR10!")

        # Use live UR10 end-effector position as drone's target
        ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :]  # shape [num_envs, 1, 3]
        self._desired_pos_w = ee_pos.squeeze(1)  # Save as [num_envs, 3]

        # Try a fixed position for the goal 
        #self._desired_pos_w = self.goal_pos


        
        if torch.rand(1).item() < 0.01:
            print(f"[DEBUG] desired_pos_w avg Z: {self._desired_pos_w[:, 2].mean():.3f}")


        self.arm_curr_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)
        self.arm_prev_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)

        self._thrust = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._moment = torch.zeros((self.num_envs, 1, 3), device=self.device)
        
        # Added from the pre_physics_step function
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
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
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Clamp and store actions
        self._actions["_Ur10Arm"] = actions["_Ur10Arm"].clone().clamp(-1.0, 1.0)
        self._actions["_DroneRobot"] = actions["_DroneRobot"].clone().clamp(-1.0, 1.0)

                # Update desired_pos_w (target for drone) to UR10 end-effector position
        ee_indices = self._Ur10Arm.find_bodies("ee_link")
        if len(ee_indices) == 0:
            raise RuntimeError("Could not find 'ee_link' on UR10!")
        
        # Always fetch the current ee_link position each step
        ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :]  # shape [num_envs, 3]

        # Try a fixed position for the goal 
        #self._desired_pos_w = self.goal_pos
        self._desired_pos_w = ee_pos.squeeze(1)  # Update the dynamic goal position

    def _apply_action(self) -> None:
        # === Drone ===
        self._apply_drone_action(self._actions["_DroneRobot"])

        # === UR10 ===
        ur10_action = self._actions["_Ur10Arm"]
        scaled_targets = scale(
            ur10_action,
            self.arm_dof_lower_limits,
            self.arm_dof_upper_limits
        )

        # Moving average smoothing
        self.arm_curr_targets = (
            self.cfg.act_moving_average * scaled_targets
            + (1.0 - self.cfg.act_moving_average) * self.arm_prev_targets
        )

        # Saturate just in case
        self.arm_curr_targets = saturate(
            self.arm_curr_targets,
            self.arm_dof_lower_limits,
            self.arm_dof_upper_limits
        )

        # Apply to robot
        self._Ur10Arm.set_joint_position_target(self.arm_curr_targets)

        # Update buffer
        self.arm_prev_targets = self.arm_curr_targets.clone()

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

        # Apply force and torque
        self._DroneRobot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _apply_ur10_action(self, action: torch.Tensor) -> None:
        """
        Apply the UR10 action to the environment.
        """
        # Convert normalized actions to joint position targets
        joint_targets = action * torch.tensor(
            [2.0, 2.0, 2.0, 3.14, 3.14, 3.14], device=self.device
        )
        self._Ur10Arm.set_joint_position_target(joint_targets)
     
    def _compute_drone_obs(self) -> torch.Tensor:
        return torch.cat((
            self._DroneRobot.data.root_pos_w[:, 0, :],
            self._DroneRobot.data.root_lin_vel_w[:, 0, :],
            self._DroneRobot.data.root_ang_vel_w[:, 0, :],
            self._DroneRobot.data.root_quat_w[:, 0, :],
            self.actions["_DroneRobot"],
            self._desired_pos_w
        ), dim=-1)

    def _compute_ur10_obs(self) -> torch.Tensor:
        return torch.cat((
            self._Ur10Arm.data.joint_pos,
            self._Ur10Arm.data.joint_vel,
            self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :],
            self.actions["_Ur10Arm"],
        ), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {

            # === UR10 Arm Observations ===
            # Joint positions (scaled)
            # Joint velocities (scaled)
            # End-effector position (world space)
            # End-effector orientation (world space)
            # End-effector linear velocity (world space)
            # End-effector angular velocity (world space)
            # Previous actions
            # Drone position (world space)
            
            "_Ur10Arm": torch.cat(
                (
                    unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
                    self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
                    self.ee_pos,
                    self.ee_quat,
                    self.ee_lin_vel,
                    self.ee_ang_vel,
                    self.actions["_Ur10Arm"],
                    
                    #self._desired_pos_w, 
                    # Probably also add the position of the drone as well
                    self._DroneRobot.data.root_pos_w,
                ),
                dim=-1,
            ),
            "_DroneRobot": torch.cat(
                (
                    # Drone position (3)
                    self._DroneRobot.data.root_pos_w,
                    # Drone orientation (quat, 4)
                    self._DroneRobot.data.root_quat_w,
                    # Drone linear velocity (3)
                    self._DroneRobot.data.root_lin_vel_w,
                    # Drone angular velocity (3)
                    self._DroneRobot.data.root_ang_vel_w,
                    # Previously applied actions
                    self.actions["_DroneRobot"],
                    # Goal again put as _desired_pos_w as this is the ee_pos
                    self._desired_pos_w,
                ),
                dim=-1,
            ),
        }
        return observations
        
    def _get_states(self) -> torch.Tensor:
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
                self._desired_pos_w, # this is the ee_pos (Note to self: Change the naming of this variable to ee_pos as its easier to understand :D )
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """
        Multi-agent rewards aligned with the latest single-agent logic.
        Returns per-actor rewards: {"_Ur10Arm": ur10_reward, "_DroneRobot": drone_reward}
        """

        # ----------------------------
        # Shared signals (computed once)
        # ----------------------------
        # Drone dynamics
        lin_vel = torch.sum(torch.square(self._DroneRobot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._DroneRobot.data.root_ang_vel_b), dim=1)

        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        drone_quat = self._DroneRobot.data.root_quat_w

        # UR10 EE pose
        ee_idx = self._Ur10Arm.find_bodies("ee_link")[0]
        ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_idx, :]
        ee_quat = self._Ur10Arm.data.body_quat_w[:, ee_idx, :]  # [N, 4]

        # Goal distance (drone -> desired pos)  (same as single-agent)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._DroneRobot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # Smooth landing & proximity (same thresholds as single-agent)
        is_close = distance_to_goal < 0.25
        is_slow = lin_vel < 10
        smooth_landing = (is_close & is_slow).float()
        proximity = (distance_to_goal < 0.25).float()

        # Alignment between drone up and EE up-reference (same as single-agent)
        drone_up = quat_apply(
            drone_quat,
            torch.tensor([0.0, 0.0, 1.0], device=drone_quat.device, dtype=drone_quat.dtype).expand(self.num_envs, 3)
        )

        # Try all three local axes to visualize; we’ll use X as up-reference (like your single-agent)
        local_x = torch.tensor([1, 0, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_y = torch.tensor([0, 1, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_z = torch.tensor([0, 0, 1], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)

        world_x = quat_apply(ee_quat, local_x)
        world_y = quat_apply(ee_quat, local_y)
        world_z = quat_apply(ee_quat, local_z)
        self._ee_local_axes_in_world = [world_x, world_y, world_z]

        # Use X as "up" reference as in the single-agent version
        ee_up = world_x

        # Alignment (cosine similarity) and gating by approach zone
        alignment = torch.sum(drone_up * ee_up, dim=1)
        aligned_enough = alignment > self.cfg.alignment_threshold
        in_approach_zone = distance_to_goal < self.cfg.approach_zone
        alignment_reward = (alignment * self.cfg.alignment_reward * self.step_dt) * in_approach_zone.float()

        # Magnet condition counters (same logic as single-agent)
        magnet_condition_raw = (
            (distance_to_goal < self.cfg.magnet_condition_distance)
            & (lin_vel < self.cfg.magnet_condition_max_speed)
            & aligned_enough
        )

        prev_magnet_counter = self._magnet_condition_counter.clone()
        self._magnet_condition_counter = torch.where(
            magnet_condition_raw,
            self._magnet_condition_counter + 1,
            torch.zeros_like(self._magnet_condition_counter)
        )
        magnet_condition = self._magnet_condition_counter >= self._magnet_required_steps
        magnet_reward = magnet_condition.float() * self.cfg.magnet_reward * self.step_dt
        self._winning_condition |= magnet_condition  # keep your success flag behavior

        # Time shaping (same direction/sign as single-agent dict term)
        time_shaping = (1.0 - (self.episode_length_buf / self.max_episode_length))

        # Orientation reward for UR10: EE "up" wrt world Z (same as single-agent)
        z_alignment = ee_up[:, 2]                    # [-1, 1], higher is better
        self._ee_alignment = z_alignment             # for debugging
        orientation_reward = z_alignment * self.cfg.orientation_reward_scale  # scale now; dt later in dict

        # Bounds / death penalty (same as single-agent added code)
        env_origins = self._terrain.env_origins  # (num_envs, 3)
        local_pos = drone_pos - env_origins
        x_oob = torch.logical_or(local_pos[:, 0] < -2.0, local_pos[:, 0] > 2.0)
        y_oob = torch.logical_or(local_pos[:, 1] < -2.0, local_pos[:, 1] > 2.0)
        died_sideways = torch.logical_or(x_oob, y_oob)
        z_oob = torch.logical_or(drone_pos[:, 2] < 0.1, drone_pos[:, 2] > 2.0)
        died = torch.logical_or(z_oob, died_sideways)
        died_penalty = died.float() * self.cfg.died_penalty

        # Wrist joint elevation reward (same as single-agent, with your scales)
        z_threshold = self.cfg.wrist_height_penalty_scale
        w1 = self._Ur10Arm.data.body_pos_w[:, self._Ur10Arm.find_bodies("wrist_1_link")[0], 2]
        w2 = self._Ur10Arm.data.body_pos_w[:, self._Ur10Arm.find_bodies("wrist_2_link")[0], 2]
        w3 = self._Ur10Arm.data.body_pos_w[:, self._Ur10Arm.find_bodies("wrist_3_link")[0], 2]

        w1_above = (w1 > z_threshold).float()
        w2_above = (w2 > z_threshold).float()
        w3_above = (w3 > z_threshold).float()

        wrist_height_score = (w1_above + w2_above + w3_above) / 3.0
        wrist_above = w1_above + w2_above + w3_above
        wrist_below = 3.0 - wrist_above

        wrist_reward = (
            wrist_above * self.cfg.wrist_height_reward_scale
            - wrist_below * self.cfg.wrist_height_penalty_scale
        ) * self.step_dt

        # (Optionally) compress to the averaged form as in your single-agent end:
        wrist_reward = (wrist_height_score * self.cfg.wrist_height_reward_scale * self.step_dt).squeeze(-1)

        # Episode diagnostics similar to single-agent
        if torch.rand(1).item() < 0.05:
            print(f"[DEBUG] dist: {distance_to_goal.mean():.3f}, vel: {lin_vel.mean():.3f}, ang_vel: {ang_vel.mean():.3f}")
            print(f"[DEBUG] drone Z: {self._DroneRobot.data.root_pos_w[:, 2].mean():.3f}")
            print(f"[DEBUG] ee_link Z: {ee_pos[:, 2].mean():.3f}")
            print(f"[DEBUG] Orientation reward mean: {(orientation_reward * self.step_dt).mean():.3f}")

        # Track "landing condition" like single-agent
        self._episode_success_flags |= (is_close & is_slow)

        # ----------------------------
        # Compose per-term reward dict (for logging parity)
        # ----------------------------
        rewards = {
            # Drone-centric
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "smooth_landing": smooth_landing * self.cfg.smooth_landing_bonus * self.step_dt,
            "proximity": proximity * self.cfg.proximity_bonus * self.step_dt,
            "time_shaping": time_shaping * self.cfg.time_bonus_scale * self.step_dt,
            "alignment_reward": alignment_reward,
            "magnet_reward": magnet_reward,

            # UR10-centric
            "orientation_reward": orientation_reward * self.step_dt,
            "wrist_height_reward": wrist_reward,

            # Shared penalty
            "died_penalty": died_penalty,
        }

        # Log episode sums
        for k, v in rewards.items():
            self._episode_sums[k] += v

        # ----------------------------
        # Split into multi-agent totals
        # ----------------------------
        # Drone gets: motion shaping + approach + alignment/magnet + time + penalties
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

        # UR10 gets: orientation + helping by being close (distance shaping) + wrist pose
        # (Optionally also give it a slice of proximity/smooth_landing if you want cooperative coupling)
        ur10_total_reward = (
            rewards["orientation_reward"]
            + rewards["wrist_height_reward"]
            + rewards["distance_to_goal"]   # encourages the arm to “meet” the drone
        )

        return {"_Ur10Arm": ur10_total_reward, "_DroneRobot": drone_total_reward}


    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Get drone positions and environment origins
        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        env_origins = self._terrain.env_origins  # shape: (num_envs, 3)

        # Compute position relative to the environment origin
        local_pos = drone_pos - env_origins  # (num_envs, 3)

        # Apply local box bounds (e.g., within [-2, 2] in x and y)
        x_out_of_bounds = torch.logical_or(local_pos[:, 0] < -2.0, local_pos[:, 0] > 2.0)
        y_out_of_bounds = torch.logical_or(local_pos[:, 1] < -2.0, local_pos[:, 1] > 2.0)
        died_sideways = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        z_out_of_bounds = torch.logical_or(drone_pos[:, 2] < 0.3, drone_pos[:, 2] > 2.0)
        died = torch.logical_or(z_out_of_bounds, died_sideways)


        terminated = {
            "_DroneRobot": died,
            "_Ur10Arm": died,
        }
        time_outs = {
            "_DroneRobot": time_out,
            "_Ur10Arm": time_out,
        }

        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._DroneRobot._ALL_INDICES

         # === Randomize goal position for the selected envs ===
        base_local_goal = torch.tensor([1.0, -1.0, 1.0], device=self.device)  # (3,)
        random_offset = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.4, 0.4)
        goal_pos_w = self._terrain.env_origins[env_ids] + base_local_goal + random_offset
        self.goal_pos[env_ids] = goal_pos_w
        self._desired_pos_w[env_ids] = goal_pos_w  # if you're using this elsewhere in reward computation

        # Reset the articulation and rigid body attributes
        super()._reset_idx(env_ids)
    
        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._DroneRobot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Reset the robotDrone 
        self._DroneRobot.reset(env_ids)
        # Reset the UR10 arm
        self._Ur10Arm.reset(env_ids)
       
        # -----------------------------
        # Randomize robotDrone initial position
        # -----------------------------
        joint_pos = self._DroneRobot.data.default_joint_pos[env_ids]
        joint_vel = self._DroneRobot.data.default_joint_vel[env_ids]
        default_root_state = self._DroneRobot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.zeros(len(env_ids)).uniform_(-0.5, 0.5).to(default_root_state.device)  # X
        default_root_state[:, 1] += torch.zeros(len(env_ids)).uniform_(-0.5, 0.5).to(default_root_state.device)  # Y
        default_root_state[:, 2] += torch.zeros(len(env_ids)).uniform_(0.0, 0.5).to(default_root_state.device)   # Z

        self._DroneRobot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._DroneRobot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._DroneRobot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -----------------------------
        # Randomize UR10 initial position
        # -----------------------------
        joint_pos = self._Ur10Arm.data.default_joint_pos[env_ids]
        joint_vel = self._Ur10Arm.data.default_joint_vel[env_ids]
        default_root_state = self._Ur10Arm.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.zeros(len(env_ids)).uniform_(-0.2, 0.2).to(default_root_state.device)  # X
        default_root_state[:, 1] += torch.zeros(len(env_ids)).uniform_(-0.2, 0.2).to(default_root_state.device)  # Y

        self._Ur10Arm.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._Ur10Arm.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._Ur10Arm.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    
        #obs = self._get_observations()
        # print(f"[RESET DEBUG] DroneRobot observation shape: {obs['_DroneRobot'].shape}")
        # print(f"[RESET DEBUG] UR10 Arm observation shape: {obs['_Ur10Arm'].shape}")

    def _set_debug_vis_impl(self, debug_vis: bool):
            # create markers if necessary for the first tome
            if debug_vis:
                if not hasattr(self, "goal_pos_visualizer"):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                    # -- goal pose
                    marker_cfg.prim_path = "/Visuals/Command/goal_position"
                    self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                    
                # Add frame marker for end effector
                if not hasattr(self, "ee_frame_visualizer"):
                    frame_marker_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/EndEffector/frame",
                        markers={
                            "frame": sim_utils.UsdFileCfg(
                                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                scale=(0.05, 0.05, 0.05),
                            )
                        }
                    )
                    self.ee_frame_visualizer = VisualizationMarkers(frame_marker_cfg)
                    
                # set their visibility to true
                self.goal_pos_visualizer.set_visibility(True)
                if hasattr(self, "ee_frame_visualizer"):
                    self.ee_frame_visualizer.set_visibility(True)
            else:
                if hasattr(self, "goal_pos_visualizer"):
                    self.goal_pos_visualizer.set_visibility(False)
                if hasattr(self, "ee_frame_visualizer"):
                    self.ee_frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        # Update end effector frame marker
        if hasattr(self, "ee_frame_visualizer"):
            ee_indices = self._Ur10Arm.find_bodies("ee_link")
            if len(ee_indices) > 0:
                ee_pos = self._Ur10Arm.data.body_pos_w[:, ee_indices[0], :].squeeze(1)  # Remove extra dimension
                ee_quat = self._Ur10Arm.data.body_quat_w[:, ee_indices[0], :].squeeze(1)  # Remove extra dimension
                self.ee_frame_visualizer.visualize(ee_pos, ee_quat)
            
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
