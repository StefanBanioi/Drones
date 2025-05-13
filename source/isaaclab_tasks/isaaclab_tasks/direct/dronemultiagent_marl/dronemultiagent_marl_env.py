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

        # === Drone Initialization ===
        # Usually drones don't use DOF targets in the same way, but you might want to store desired thrusts or velocities
        # self.num_drone_actions = self.cfg.action_spaces["_DroneRobot"]
        # self.drone_actions = torch.zeros((self.num_envs, self.num_drone_actions), dtype=torch.float, device=self.device)

        # === Goal setup (optional, based on your reward function) ===
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:] = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # Example: 1m above base

        # Marker for visualization
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

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
            ]
        }
        # Get specific body indices
        self._body_id = self._DroneRobot.find_bodies("body")[0]
        self._robot_mass = (self._DroneRobot.root_physx_view.get_masses()[0].sum()) *27.0 # scale to 3x size
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

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

        self.arm_curr_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)
        self.arm_prev_targets = torch.zeros_like(self._Ur10Arm.data.joint_pos)






        self._thrust = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._moment = torch.zeros((self.num_envs, 1, 3), device=self.device)
        




        # Added from the pre_physics_step function
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

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

        # Update goal position based on UR10 end-effector
        # self._desired_pos_w = self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :]

        # # Cache ee kinematic states
        # self.ee_pos = self._Ur10Arm.data.body_pos_w[:, self.ee_idx, :] # [num_envs, 3]
        # self.ee_quat = self._Ur10Arm.data.body_quat_w[:, self.ee_idx, :] # [num_envs, 4]
        # self.ee_lin_vel = self._Ur10Arm.data.body_lin_vel_w[:, self.ee_idx, :] # [num_envs, 3]
        # self.ee_ang_vel = self._Ur10Arm.data.body_ang_vel_w[:, self.ee_idx, :] # [num_envs, 3]



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
            "_Ur10Arm": torch.cat(
                (
                    unscale(self._Ur10Arm.data.joint_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits),
                    self.cfg.vel_obs_scale * self._Ur10Arm.data.joint_vel,
                    self.ee_pos,
                    self.ee_quat,
                    self.ee_lin_vel,
                    self.ee_ang_vel,
                    self.actions["_Ur10Arm"],
                    self.goal_pos,  # could also be `self.ee_pos` depending on goal logic
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
                    # Goal
                    self.goal_pos,
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
                self.goal_pos,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Velocity penalties/rewards — from the drone
        lin_vel = torch.sum(torch.square(self._DroneRobot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._DroneRobot.data.root_ang_vel_b), dim=1)
        
        # Distance from drone to robot end-effector (goal)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._DroneRobot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # --- Smooth landing reward (close + slow) ---
        is_close = distance_to_goal < 0.3
        is_slow = lin_vel < 0.1
        smooth_landing = (is_close & is_slow).float() 
        # --- Bonus for being very close to the target ---
        proximity = (distance_to_goal < 0.1).float() 

        # --- Time-based shaping (inverse of time taken) ---
        time_shaping = (1.0 - (self.episode_length_buf / self.max_episode_length)) 

        # --- Orientation reward: keep UR10 ee_link pointing up ---
        # UR10 X-axis should align with world Z-axis [0, 0, 1]
        # Assuming ee_link's orientation is available in quaternion
        ee_quat = self._Ur10Arm.data.body_quat_w[:, self._Ur10Arm.find_bodies("ee_link")[0], :]  # [N, 4]
        
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
        up_vector = world_x
        
        # Measure alignment with world Z [0,0,1] - this gives a value between -1 and 1
        z_alignment = up_vector[:, 2]
        
        # Reward positive alignment AND penalize negative alignment
        # When alignment is positive (pointing up): reward proportionally
        # When alignment is negative (pointing down): penalize proportionally
        orientation_reward = z_alignment * self.cfg.orientation_reward_scale
        
        # Log the actual alignment values for debugging
        self._ee_alignment = z_alignment

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "smooth_landing": smooth_landing * self.cfg.smooth_landing_bonus * self.step_dt,
            "proximity": proximity * self.cfg.proximity_bonus * self.step_dt,
            "time_shaping": time_shaping * self.cfg.time_bonus_scale * self.step_dt,
            "orientation_reward": orientation_reward * self.step_dt,
        }
        drone_total_reward = (
            rewards["lin_vel"] +
            rewards["ang_vel"] +
            rewards["distance_to_goal"] +
            rewards["smooth_landing"] +
            rewards["proximity"] +
            rewards["time_shaping"] 
        )
        ur10_reward = (
            rewards["orientation_reward"] + 
            rewards["smooth_landing"] +
            rewards["proximity"] +
            rewards["distance_to_goal"]     
        ) * self.step_dt

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return {"_Ur10Arm": ur10_reward, "_DroneRobot": drone_total_reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._DroneRobot.data.root_pos_w[:, 2] < 0.1, self._DroneRobot.data.root_pos_w[:, 2] > 2.0)

        terminated = {
            "_DroneRobot": died,
            "_Ur10Arm": died,  # or a custom condition for the UR10 if needed
        }
        time_outs = {
            "_DroneRobot": time_out,
            "_Ur10Arm": time_out,
        }

        return terminated, time_outs


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._DroneRobot._ALL_INDICES

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
        # extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
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


        obs = self._get_observations()
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
                
                # Print alignment value for debugging
                # if hasattr(self, "_ee_alignment"):
                #     alignment = self._ee_alignment[0].item()  # Get the first environment's alignment
                #     # Only print every 100 steps to avoid flooding the console
                #     if self.step_count % 100 == 0:
                #         print(f"EE alignment with world up: {alignment:.4f} - " +
                #               f"{'GOOD (pointing up)' if alignment > 0.7 else 'POOR (not pointing up)'}")

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


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.sum(
        torch.square(pole_pos + pendulum_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)

    total_reward = {
        "cart": rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel,
        "pendulum": rew_alive + rew_termination + rew_pendulum_pos + rew_pendulum_vel,
    }
    return total_reward