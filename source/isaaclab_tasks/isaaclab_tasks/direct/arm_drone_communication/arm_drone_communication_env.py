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


from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .arm_drone_communication_env_cfg import ArmDroneCommunicationEnvCfg


class ArmDroneCommunicationEnv(DirectRLEnv):
    cfg: ArmDroneCommunicationEnvCfg

    def __init__(self, cfg: ArmDroneCommunicationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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
            ]
        }
        # Get specific body indices
        self._body_id = self._DroneRobot.find_bodies("body")[0]
        self._robot_mass = self._DroneRobot.root_physx_view.get_masses()[0].sum()
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
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Drone thrust and moment (new)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]

        # UR10 joint targets
        joint_targets = self._actions[:, 4:] * torch.tensor(
            [2.0, 2.0, 2.0, 3.14, 3.14, 3.14], device=self.device
        )
        self._finalUr10.set_joint_position_target(joint_targets)

        # Update desired_pos_w (target for drone) to UR10 end-effector position
        ee_indices = self._finalUr10.find_bodies("ee_link")
        if len(ee_indices) == 0:
            raise RuntimeError("Could not find 'ee_link' on UR10!")
        
        # Always fetch the current ee_link position each step
        ee_pos = self._finalUr10.data.body_pos_w[:, ee_indices[0], :]  # shape [num_envs, 3]
        self._desired_pos_w = ee_pos.squeeze(1)  # Update the dynamic goal position

    def _apply_action(self):
        self._DroneRobot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:


        # Get the desired position in world space
        desired_pos_b, _ = subtract_frame_transforms(
            self._DroneRobot.data.root_state_w[:, :3], self._DroneRobot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # UR10 joint state 
        joint_pos = self._finalUr10.data.joint_pos
        joint_vel = self._finalUr10.data.joint_vel

        obs = torch.cat(
            [
                self._DroneRobot.data.root_lin_vel_b,        #(3,)
                self._DroneRobot.data.root_ang_vel_b,        #(3,)
                self._DroneRobot.data.projected_gravity_b,   #(3,)
                desired_pos_b,                          #(3,)

                # add the joint state of the UR10 arm
                joint_pos,                              #(6,)
                joint_vel,                              #(6,)


            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations #this step is neccesary as its the model input

    def _get_rewards(self) -> torch.Tensor:
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
        # UR10 Z-axis should align with world Z-axis [0, 0, 1]
        # Assuming ee_link's orientation is available in quaternion
        ee_quat = self._finalUr10.data.body_quat_w[:, self._finalUr10.find_bodies("ee_link")[0], :]  # [N, 4]
        vec = torch.tensor([0, 0, 1], device=ee_quat.device, dtype=ee_quat.dtype).expand(ee_quat.shape[0], 3)  # [N, 3]
        up_vector = quat_apply(ee_quat, vec)  # [N, 3]
        z_alignment = up_vector[:, 2]
        orientation_reward = torch.clamp(z_alignment, 0.0, 1.0) * self.cfg.orientation_reward_scale


        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "smooth_landing": smooth_landing * self.cfg.smooth_landing_bonus * self.step_dt,
            "proximity": proximity * self.cfg.proximity_bonus * self.step_dt,
            "time_shaping": time_shaping * self.cfg.time_bonus_scale * self.step_dt,
            "orientation_reward": orientation_reward * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._DroneRobot.data.root_pos_w[:, 2] < 0.1, self._DroneRobot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._DroneRobot._ALL_INDICES

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

        # Reset the robotDrone 
        self._DroneRobot.reset(env_ids)
        # Reset the UR10 arm
        self._finalUr10.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0


        # This maked the desired pos random which I do not want currently so just comment it out ;D
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        
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
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
