# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import UsdFileCfg, PreviewSurfaceCfg
from isaaclab.utils.math import quat_from_angle_axis



from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip



from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .arm_drone_communication_env_cfg import ArmDroneCommunicationEnvCfg


class ArmDroneCommunicationEnv(DirectRLEnv):
    cfg: ArmDroneCommunicationEnvCfg

    def __init__(self, cfg: ArmDroneCommunicationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

        self._magnet_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Magnetic capture condition active
        
        # Magnet condition tracking with the second counter 
        self._magnet_condition_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._magnet_required_steps = int(self.cfg.magnet_time_threshold_in_seconds / self.step_dt)  # e.g., 3s / 0.05 = 60 steps

        
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

        # add a episode level success tracker 
        self._episode_success_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # add a winning condition
        self._winning_condition = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # add a episode level failure tracker
        self._episode_failure_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "smooth_landing",
                "proximity",
                "time_shaping",
                "orientation_reward",
                "magnet_reward",
                "alignment_reward",
                
                
                # === Added code today 15/05/2025 ===
                "died_penalty",
                # === End of added code ===
                "wrist_height_reward",
            ]
        }
        # Add after self._episode_sums
        self._success_status = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        # Get specific body indices
        # self._body_id = self._DroneRobot.find_bodies("MainBody")[0]
        self._body_id = self._DroneRobot.find_bodies("body")[0]
        self._robot_mass = (self._DroneRobot.root_physx_view.get_masses()[0].sum()) 
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._DroneRobot = Articulation(self.cfg.robotDrone)
        self.scene.articulations["robotDrone"] = self._DroneRobot
        self._finalUr10 = Articulation(self.cfg.UR10_CFG)
        self.scene.articulations["UR10"] = self._finalUr10
        

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Clamp actions to [-1, 1]
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._step_count += 1

        # ----------------------------------------------------------------------
        # Drone thrust and body torques
        # ----------------------------------------------------------------------
        # Scale thrust relative to weight (maps action[0] in [-1,1] → [0, thrust_to_weight * weight])
        thrust_cmd = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # Safety clamp to avoid crazy impulses
        thrust_cmd = torch.clamp(thrust_cmd, 0.0, 2.5 * self._robot_weight)
        self._thrust[:, 0, 2] = thrust_cmd

        # Scale angular moments (actions[1:4] are roll/pitch/yaw)
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]
        # Clamp torques to something reasonable for ~3kg craft
        max_moment = 0.8 * self._robot_weight   # ≈ 23 N·m if mass ≈ 3kg
        self._moment = torch.clamp(self._moment, -max_moment, max_moment)

        # TEMP sanity injection: add a small roll torque at the very start to check if body rotates
        if self._step_count < 50:
            self._moment[:, 0, 0] += 0.5  # 0.5 N·m roll kick for debugging

        # ----------------------------------------------------------------------
        # UR10 joint targets (scaled from actions[4:])
        # ----------------------------------------------------------------------
        joint_target_pos = self._actions[:, 4:] * torch.tensor(
            [
                1.5708,   # shoulder_pan_joint: 90°
                -0.7854,  # shoulder_lift_joint: -45°
                -0.7854,  # elbow_joint: -45°
                0.0000,   # wrist_1_joint
                1.5708,   # wrist_2_joint: 90°
                0.0000    # wrist_3_joint
            ],
            device=self.device
        )
        self._finalUr10.set_joint_position_target(joint_target_pos)

        # ----------------------------------------------------------------------
        # Update drone goal position (UR10 ee_link)
        # ----------------------------------------------------------------------
        ee_indices = self._finalUr10.find_bodies("ee_link")
        if len(ee_indices) == 0:
            raise RuntimeError("Could not find 'ee_link' on UR10!")

        ee_pos = self._finalUr10.data.body_pos_w[:, ee_indices[0], :]  # [num_envs, 3]
        self._desired_pos_w = ee_pos.squeeze(1)  # dynamic goal position

    def _apply_action(self):

        dt = self.step_dt
        self.wind_timer -= dt
        self.wind_cooldown -= dt

        # === Persistent wind ===
        self.wind_timer -= dt
        self.wind_cooldown -= dt

        # Update base wind if needed
        needs_new_wind = (self.wind_timer <= 0) & (self.wind_cooldown <= 0)
        if needs_new_wind.any():
            new_dirs = torch.nn.functional.normalize(torch.randn((self.num_envs, 2), device=self.device), dim=1)
            new_strengths = torch.empty(self.num_envs, device=self.device).uniform_(
                self.cfg.lower_wind_scale, self.cfg.upper_wind_scale)

            self.wind_direction[needs_new_wind] = new_dirs[needs_new_wind]
            self.wind_strength[needs_new_wind] = new_strengths[needs_new_wind]

            self.wind_timer[needs_new_wind] = torch.randint(50, 150, (needs_new_wind.sum(),), device=self.device) * dt  # ~0.5–1.5s wind duration
            self.wind_cooldown[needs_new_wind] = torch.randint(100, 300, (needs_new_wind.sum(),), device=self.device) * dt # ~1–3s pause before update

        # Update steady wind force
        self.wind_force[:, 0, 0] = self.wind_direction[:, 0] * self.wind_strength
        self.wind_force[:, 0, 1] = self.wind_direction[:, 1] * self.wind_strength
        self.wind_force[:, 0, 2] = 0.0

        # === Gusts ===
        self.wind_gust_timer -= dt
        self.wind_gust_cooldown -= dt

        # End gusts
        gust_end = self.wind_gust_timer <= 0
        self.active_wind_force[gust_end] = 0.0

        # Trigger new gusts
        can_gust = self.wind_gust_cooldown <= 0
        start_gust = torch.rand(self.num_envs, device=self.device) < 0.02
        trigger_gust = can_gust & start_gust

        
        not_magnetized = ~self._magnet_active
        if not_magnetized.any():
            env_ids = torch.nonzero(not_magnetized, as_tuple=False).squeeze(-1)
            if trigger_gust.any():
                gust_dirs = torch.nn.functional.normalize(torch.randn_like(self.active_wind_force), dim=-1)
                gust_mags = torch.empty((self.num_envs, 1, 1), device=self.device).uniform_(0.1, 0.3)  # stronger range
                self.active_wind_force[trigger_gust] = gust_dirs[trigger_gust] * gust_mags[trigger_gust]
                self.wind_gust_timer[trigger_gust] = torch.randint(15, 40, (trigger_gust.sum(),), device=self.device) * dt  # ~0.15–0.4s gust
                self.wind_gust_cooldown[trigger_gust] = torch.randint(100, 300, (trigger_gust.sum(),), device=self.device) * dt  # ~1–3s before next gust   
            
            combined_wind_force = self.wind_force[env_ids] + self.active_wind_force[env_ids]  # shape: [num_envs, 1, 3]
            combined_forces = self._thrust + combined_wind_force
            combined_torques = self._moment



            # Apply combined force and torque to the drone
            self._DroneRobot.set_external_force_and_torque(
                forces=combined_forces,
                torques=combined_torques,
                body_ids=self._body_id
            )
       

        # If magnet is active, override position/force to simulate attachment
        magnetized = self._winning_condition  # shape: [num_envs]
        if magnetized.any():
            attached = torch.nonzero(magnetized).squeeze(-1)  # indices of magnetized drones
            #print(f"[DEBUG] Magnetized drones: {attached.tolist()}")

    def _get_observations(self) -> dict:


        # Get the desired position in world space
        desired_pos_b, _ = subtract_frame_transforms(
            self._DroneRobot.data.root_state_w[:, :3], self._DroneRobot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # UR10 joint state 
        joint_pos = self._finalUr10.data.joint_pos
        joint_vel = self._finalUr10.data.joint_vel

        # Get the wind as part of the observation
        wind_forces = self.wind_force[:, 0, :].squeeze(1)  # shape [num_envs, 3]

        

        obs = torch.cat(
            [
                self._DroneRobot.data.root_lin_vel_b,        #(3,)
                self._DroneRobot.data.root_ang_vel_b,        #(3,)
                self._DroneRobot.data.projected_gravity_b,   #(3,)
                desired_pos_b,                          #(3,)

                # add the joint state of the UR10 arm
                joint_pos,                              #(6,)
                joint_vel,                              #(6,)

                # add the wind forces
                wind_forces,                            #(3,)

            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations #this step is neccesary as its the model input

    def _get_rewards(self) -> torch.Tensor:
        # Velocity penalties/rewards — from the drone
        lin_vel = torch.sum(torch.square(self._DroneRobot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._DroneRobot.data.root_ang_vel_b), dim=1)
        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        ee_pos = self._finalUr10.data.body_pos_w[:, self._finalUr10.find_bodies("ee_link")[0], :]
            # [N, 4] quaternions
        drone_quat = self._DroneRobot.data.root_quat_w
        ee_quat = self._finalUr10.data.body_quat_w[:, self._finalUr10.find_bodies("ee_link")[0], :]
        # Distance from drone to robot end-effector (goal)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._DroneRobot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # --- Smooth landing reward (close + slow) ---
        is_close = distance_to_goal < 0.25
        is_slow = lin_vel < 10
        smooth_landing = (is_close & is_slow).float() 

        
        # Drone's "up" vector (assume Z-axis in drone local frame)
        drone_up = quat_apply(drone_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))

        # Arm's X-axis (used as "up" reference)
        ee_up = quat_apply(ee_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
        
        # Require that the drone is close AND moving faster than X m/s
        proximity = (distance_to_goal < 0.25) #& (lin_vel > 5.0)).float()
        
        # Compute alignment (dot product should be close to 1 if aligned)
        # I want to also make sure that the drone only gets rewarded with the alignment only when the drone is close. 
        # This is to avoid the drone getting rewarded/ punnished for being aligned with the arm even if the arm is not correctly pointing.
        # This is done by checking the approach zone and the alignment condition
        #checking_for_alignment = (distance_to_goal < self.cfg.approach_zone)

        alignment = torch.sum(drone_up * ee_up, dim=1)  # cosine similarity [-1, 1]
        aligned_enough = alignment > self.cfg.alignment_threshold          # ~23° alignment cone (arccos(0.92) ≈ 23°)


        # Condition for the magnet to catch the drone
        # magnet_condition_distance = 0.1  # Distance at which the magnet can catch the drone
        # magnet_condition_max_speed = 5       # Speed at which the magnet can catch the drone
        # Add magnetic condition (you can tweak thresholds)
        
        #magnet_condition = (distance_to_goal < self.cfg.magnet_condition_distance) & (lin_vel < self.cfg.magnet_condition_max_speed) & aligned_enough # ~23° alignment cone

        # --- MAGNET CONDITION RAW --- used to update the magnet condition counter
        magnet_condition_raw = (
            (distance_to_goal < self.cfg.magnet_condition_distance)
            & (lin_vel < self.cfg.magnet_condition_max_speed)
            & aligned_enough
        )

        # --- STORE OLD COUNTER FOR DEBUG ---
        prev_magnet_counter = self._magnet_condition_counter.clone()

        # --- UPDATE MAGNET COUNTER ---
        self._magnet_condition_counter = torch.where(
            magnet_condition_raw,
            self._magnet_condition_counter + 1,
            torch.zeros_like(self._magnet_condition_counter)
        )

        # --- DEBUG PRINT: Show which counters reset ---
        # if torch.rand(1).item() < 0.01:
        #     reset_mask = ~magnet_condition_raw
        #     num_reset = reset_mask.sum().item()
        #     reset_indices = torch.nonzero(reset_mask).squeeze().tolist()

        #     print(f"[DEBUG] Magnet counters BEFORE update: {prev_magnet_counter}")
        #     print(f"[DEBUG] Magnet condition raw mask: {magnet_condition_raw}")
        #     print(f"[DEBUG] Magnet counters AFTER update: {self._magnet_condition_counter}")
        #     print(f"[DEBUG] Counters reset this step: {num_reset}")
        #     print(f"[DEBUG] Reset indices (envs): {reset_indices}")

        # --- FINAL MAGNET CONDITION ---
        magnet_condition = self._magnet_condition_counter >= self._magnet_required_steps

        # --- MAGNET REWARD ---
        magnet_reward = magnet_condition.float() * self.cfg.magnet_reward * self.step_dt

        # --- WINNING CONDITION ---
        self._winning_condition |= magnet_condition

        # --- Time-based shaping (inverse of time taken) ---
        time_shaping = (1.0 - (self.episode_length_buf / self.max_episode_length)) 

    
        # --- Orientation reward: keep UR10 ee_link pointing up ---
        # UR10 X-axis should align with world Z-axis [0, 0, 1]

        
        # Try all three local axes to see which one we actually want pointing up
        local_x = torch.tensor([1, 0, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_y = torch.tensor([0, 1, 0], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        local_z = torch.tensor([0, 0, 1], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)
        
        # Transform all three axes to world space
        world_x = quat_apply(ee_quat, local_x)
        world_y = quat_apply(ee_quat, local_y)
        world_z = quat_apply(ee_quat, local_z)
        
        # Store these for debug visualization
        self._ee_local_axes_in_world = [world_x, world_y, world_z]
        
        # Use X-axis for alignment as that seems to be the one that should point upward
        #up_vector = world_x

        # === Added code today 15/05/2025 ===
        #up_vector = world_z
        #up_vector = world_y
        up_vector = world_x
        # === End of added code ===
        
        # Measure alignment with world Z [0,0,1] - this gives a value between -1 and 1
        z_alignment = up_vector[:, 2]
  
        # Log the actual alignment values for debugging
        self._ee_alignment = z_alignment
        # Scale the reward
        orientation_reward = z_alignment * self.cfg.orientation_reward_scale
        
        # Log the actual alignment values for debugging
        self._ee_alignment = z_alignment

        # === Added code today 15/05/2025 ===

        # Get drone positions and environment origins
        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        env_origins = self._terrain.env_origins  # shape: (num_envs, 3)

        # Compute position relative to the environment origin
        local_pos = drone_pos - env_origins  # (num_envs, 3)

        # Apply local box bounds (e.g., within [-2, 2] in x and y)
        x_out_of_bounds = torch.logical_or(local_pos[:, 0] < -2.0, local_pos[:, 0] > 2.0)
        y_out_of_bounds = torch.logical_or(local_pos[:, 1] < -2.0, local_pos[:, 1] > 2.0)
        died_sideways = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        z_out_of_bounds = torch.logical_or(drone_pos[:, 2] < 0.1, drone_pos[:, 2] > 2.0)
        died = torch.logical_or(z_out_of_bounds, died_sideways)

        # Penalize if the drone is dead
        died_penalty = died.float() * self.cfg.died_penalty 

        # === End of added code ===

        # === Added code today 15/05/2025 ===
        if torch.rand(1).item() < 0.05:
            print(f"[DEBUG] dist: {distance_to_goal.mean():.3f}, vel: {lin_vel.mean():.3f}, ang_vel: {ang_vel.mean():.3f}")
            print(f"[DEBUG] drone Z: {self._DroneRobot.data.root_pos_w[:, 2].mean():.3f}")
            print(f"[DEBUG] ee_link Z: {self._finalUr10.data.body_pos_w[:, self._finalUr10.find_bodies('ee_link')[0], 2].mean():.3f}")
            print(f"[DEBUG] Orientation reward mean: {orientation_reward.mean():.3f}")
        # === End of added code ===

        # === Wrist joint elevation reward ===
        z_threshold = 1.1  # Minimum height in world Z

        # Get Z positions of the wrist links
        wrist_1_z = self._finalUr10.data.body_pos_w[:, self._finalUr10.find_bodies("wrist_1_link")[0], 2]
        wrist_2_z = self._finalUr10.data.body_pos_w[:, self._finalUr10.find_bodies("wrist_2_link")[0], 2]
        wrist_3_z = self._finalUr10.data.body_pos_w[:, self._finalUr10.find_bodies("wrist_3_link")[0], 2]

        # Check if each wrist is above the threshold
        wrist_1_above = (wrist_1_z > z_threshold).float()
        wrist_2_above = (wrist_2_z > z_threshold).float()
        wrist_3_above = (wrist_3_z > z_threshold).float()

        # Compute the average score
        wrist_height_score = (wrist_1_above + wrist_2_above + wrist_3_above) / 3.0

        # Count how many wrists are above/below threshold
        wrist_above = (wrist_1_z > z_threshold).float() + (wrist_2_z > z_threshold).float() + (wrist_3_z > z_threshold).float()
        wrist_below = 3.0 - wrist_above  # max is 3

        # Reward for each wrist above, penalty for each wrist below
        wrist_reward = (
            wrist_above * self.cfg.wrist_height_reward_scale
            - wrist_below * self.cfg.wrist_height_penalty_scale
        ) * self.step_dt

        # Scale the reward
        wrist_reward = (wrist_height_score * self.cfg.wrist_height_reward_scale * self.step_dt).squeeze(-1)

        # Compute mask for being in approach zone
        in_approach_zone = distance_to_goal < self.cfg.approach_zone  

        # Only reward alignment if the drone is in the approach zone
        alignment_reward = (
            alignment * self.cfg.alignment_reward * self.step_dt
        ) * in_approach_zone.float()
        # If not in approach zone, set alignment reward to 0

        # Magnet reward
        #magnet_reward = magnet_condition.float() * self.cfg.magnet_reward * self.step_dt

        # --- Update the success status ---

        # --- Smooth landing reward (close + slow) ---
        # is_close = distance_to_goal < 0.25
        # is_slow = lin_vel < 10
        # Track whether each env has met landing condition (but don't mark it as successful yet)
        self._episode_success_flags |= (is_close & is_slow)

        # track if the magnet condition is met
        #self._winning_condition |= magnet_condition 

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "smooth_landing": smooth_landing * self.cfg.smooth_landing_bonus * self.step_dt,
            "proximity": proximity * self.cfg.proximity_bonus * self.step_dt,
            "time_shaping": time_shaping * self.cfg.time_bonus_scale * self.step_dt,
            "orientation_reward": orientation_reward * self.step_dt, # this is for the ur10 arm
            "magnet_reward": magnet_reward,
            "alignment_reward": alignment_reward, # this is for the drone alignment with the arm's end-effector



            # === Added code today 15/05/2025 ===
            "died_penalty": died_penalty ,
            # === End of added code ===   
            "wrist_height_reward": wrist_reward           
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # === Added code today 15/05/2025 ===
        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        env_origins = self._terrain.env_origins  # shape: (num_envs, 3)

        # Compute position relative to the environment origin
        local_pos = drone_pos - env_origins  # (num_envs, 3)

        # Basically a 4*4*2 box around the origin of the environment
        # Apply local box bounds (e.g., within [-2, 2] in x and y)
        x_out_of_bounds = torch.logical_or(local_pos[:, 0] < -2.5, local_pos[:, 0] > 2.5)
        y_out_of_bounds = torch.logical_or(local_pos[:, 1] < -2.5, local_pos[:, 1] > 2.5)
        died_sideways = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        z_out_of_bounds = torch.logical_or(drone_pos[:, 2] < 0.3, drone_pos[:, 2] > 4.0)
        died = torch.logical_or(z_out_of_bounds, died_sideways)

        # === End of added code ===

        # Comment this out if you want to use the new died condition
        # died = torch.logical_or(self._DroneRobot.data.root_pos_w[:, 2] < 0.1, self._DroneRobot.data.root_pos_w[:, 2] > 2.0)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._DroneRobot._ALL_INDICES

        # === Added code today 15/05/2025 ===
        # === Randomize goal position for the selected envs ===
        base_local_goal = torch.tensor([1.0, -1.0, 1.0], device=self.device)  # (3,)
        random_offset = torch.empty((len(env_ids), 3), device=self.device).uniform_(-0.4, 0.4)
        goal_pos_w = self._terrain.env_origins[env_ids] + base_local_goal + random_offset
        # self.goal_pos[env_ids] = goal_pos_w
        # self._desired_pos_w[env_ids] = goal_pos_w  # if you're using this elsewhere in reward computation
        # === End of added code ===

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
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # === Finalize episode outcome at reset ===
        # Reset status for selected envs
        self._success_status[env_ids] = 0

        # Mark environments that successfully landed during the episode
        success_env_ids = env_ids[self._episode_success_flags[env_ids]]
        self._success_status[success_env_ids] = 1

        # Mark environments that got close enough that the magnet condition was met
        magnet_env_ids = env_ids[self._winning_condition[env_ids]]
        self._success_status[magnet_env_ids] = 2  # Treat magnet condition as a superior success condition

        # Crashed environments (terminated)
        crash_env_ids = env_ids[self.reset_terminated[env_ids]]
        self._success_status[crash_env_ids] = -1

        # Timed out environments that never landed = failure (-2)
        timeout_env_ids = env_ids[self.reset_time_outs[env_ids]]
        timeout_failed_env_ids = timeout_env_ids[~self._episode_success_flags[timeout_env_ids]]
        self._success_status[timeout_failed_env_ids] = -2

        # Reset the flags so next episode can track success again
        self._episode_success_flags[env_ids] = False
        self._winning_condition[env_ids] = False


        # === Log total success/failure counts ===
        success_count = torch.sum(self._success_status[env_ids] == 1).item()
        magnet_was_success_count = torch.sum(self._success_status[env_ids] == 2).item()
        crash_count = torch.sum(self._success_status[env_ids] == -1).item()
        timeout_count = torch.sum(self._success_status[env_ids] == -2).item()

        self.extras["log"]["Episode_Success/success"] = success_count
        self.extras["log"]["Episode_Success/magnet"] = magnet_was_success_count
        self.extras["log"]["Episode_Success/crash"] = crash_count
        self.extras["log"]["Episode_Success/timeout"] = timeout_count

        # Reset the robotDrone 
        self._DroneRobot.reset(env_ids)
        # Reset the UR10 arm
        self._finalUr10.reset(env_ids)
        # Reset the magnet
        self._winning_condition[env_ids] = False

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

       
        # -----------------------------
        # Randomize robotDrone initial position
        # -----------------------------
        joint_pos = self._DroneRobot.data.default_joint_pos[env_ids]
        joint_vel = self._DroneRobot.data.default_joint_vel[env_ids]
        default_root_state = self._DroneRobot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.zeros(len(env_ids)).uniform_(-1.5, 1.5).to(default_root_state.device)  # X
        default_root_state[:, 1] += torch.zeros(len(env_ids)).uniform_(-1.5, 1.5).to(default_root_state.device)  # Y
        default_root_state[:, 2] += torch.zeros(len(env_ids)).uniform_(0.0, 0.5).to(default_root_state.device)   # Z

        self._DroneRobot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._DroneRobot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._DroneRobot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -----------------------------
        # Randomize UR10 initial position
        # -----------------------------
        joint_pos = self._finalUr10.data.default_joint_pos[env_ids]
        joint_vel = self._finalUr10.data.default_joint_vel[env_ids]
        default_root_state = self._finalUr10.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.zeros(len(env_ids)).uniform_(-0.2, 0.2).to(default_root_state.device)  # X
        default_root_state[:, 1] += torch.zeros(len(env_ids)).uniform_(-0.2, 0.2).to(default_root_state.device)  # Y

        self._finalUr10.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._finalUr10.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._finalUr10.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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
        # === Existing success/failure print ===
        status = self._success_status.cpu().numpy()
        print(f"[STEP {self._step_count}] Success: {(status == 1).sum()} | Magnet Success: {(status == 2).sum()} | Failure: {(status == -1).sum()} | Timeout: {(status == -2).sum()}")

        # === Wind arrow visualization ===
        drone_pos = self._DroneRobot.data.root_pos_w[:, :3]
        active_vecs = self.active_wind_force[:, 0, :]  # [N, 3]
        constant_wind_vecs = self.wind_force[:, 0, :]  # [N, 3]
        wind_vecs = active_vecs + constant_wind_vecs  # [N, 3]
        # Normalize direction for orientation
        wind_dirs = torch.nn.functional.normalize(wind_vecs, dim=1)
        arrow_length = 0.4
        arrow_tip = drone_pos + wind_dirs * arrow_length

        # Orientation (yaw around Z-axis)
        yaw_angles = torch.atan2(wind_dirs[:, 1], wind_dirs[:, 0])
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        arrow_orientations = quat_from_angle_axis(yaw_angles, z_axis)

        # === Compute wind color ===
        # Norm of each force vector (N)
        wind_mags = torch.norm(wind_vecs, dim=-1)

        # Normalize for coloring: 0 N → blue, 1.0+ N → red
        normed = wind_mags.clamp(0.0, 1.0)

        # Map to RGB: blue → yellow → red
        # We'll use a 3-point linear interpolation
        # 0.0  → light blue:  (0.2, 0.6, 1.0)
        # 0.5  → yellow:      (1.0, 1.0, 0.0)
        # 1.0+ → red:         (1.0, 0.0, 0.0)
        colors = torch.zeros((self.num_envs, 3), device=self.device)

        # Between blue and yellow
        low_mask = normed < 0.5
        t_low = normed[low_mask] * 2.0  # map to [0, 1]
        colors[low_mask] = (
            (1.0 - t_low).unsqueeze(-1) * torch.tensor([0.2, 0.6, 1.0], device=self.device)
            + t_low.unsqueeze(-1) * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        )

        # Between yellow and red
        high_mask = normed >= 0.5
        t_high = (normed[high_mask] - 0.5) * 2.0  # map to [0, 1]
        colors[high_mask] = (
            (1.0 - t_high).unsqueeze(-1) * torch.tensor([1.0, 1.0, 0.0], device=self.device)
            + t_high.unsqueeze(-1) * torch.tensor([1.0, 0.0, 0.0], device=self.device)
        )

        # === Visualize ===
        self.wind_markers.visualize(
            drone_pos, arrow_orientations, colors
        ) 
        # Update end effector frame marker
        if hasattr(self, "ee_frame_visualizer"):
            ee_indices = self._finalUr10.find_bodies("ee_link")
            if len(ee_indices) > 0:
                ee_pos = self._finalUr10.data.body_pos_w[:, ee_indices[0], :].squeeze(1)  # Remove extra dimension
                ee_quat = self._finalUr10.data.body_quat_w[:, ee_indices[0], :].squeeze(1)  # Remove extra dimension
                self.ee_frame_visualizer.visualize(ee_pos, ee_quat)
                