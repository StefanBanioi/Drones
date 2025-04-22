# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher
# Add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates a UR10 robot arm and a drone.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")

# Append AppLauncher CLI args
# This is a custom script, so we need to add the AppLauncher arguments to the parser
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG
from isaaclab.sim import SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.envs import mdp


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene with only the UR10 robot."""
    # Create a specific origin for the UR10 robot and a drone
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = define_origins(num_origins=2, spacing=2.5)

    # Origin with UR10
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[0])
    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    )
    cfg.func("/World/Origin2/Table", cfg, translation=(0.0, 0.0, 1.03))
    # -- Robot
    ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    ur10 = Articulation(cfg=ur10_cfg)

    # -- Drone Setup
    prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[1])
    drone_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Origin3/Drone")
    drone_cfg.init_state.pos = (0.0, 0.0, 1.0)
    drone_cfg.spawn.func("/World/Origin3/Drone", drone_cfg.spawn, translation=drone_cfg.init_state.pos)
    drone = Articulation(cfg=drone_cfg)

    # return the scene information with only the UR10 robot
    scene_entities = {
        "ur10": ur10,
        "drone": drone,
    }
    return scene_entities, origins

# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
#     """Runs the simulation loop."""
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     # Simulate physics
#     while simulation_app.is_running():
#         # reset
#         if count % 200 == 0:
#             # reset counters
#             sim_time = 0.0
#             count = 0
#             # reset the scene entities
#             for index, robot in enumerate(entities.values()):
#                 # root state
#                 root_state = robot.data.default_root_state.clone()
#                 root_state[:, :3] += origins[index]
#                 robot.write_root_pose_to_sim(root_state[:, :7])
#                 robot.write_root_velocity_to_sim(root_state[:, 7:])
#                 # set joint positions
#                 joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#                 robot.write_joint_state_to_sim(joint_pos, joint_vel)
#                 # clear internal buffers
#                 robot.reset()
#             print("[INFO]: Resetting robots state...")
#         # apply random actions to the robots
#         for robot in entities.values():
#             # generate random joint positions
#             joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
#             joint_pos_target = joint_pos_target.clamp_(
#                 robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
#             )
#             # apply action to the robot
#             robot.set_joint_position_target(joint_pos_target)
#             # write data to sim
#             robot.write_data_to_sim()
#         # perform step
#         sim.step()
#         # update sim-time
#         sim_time += sim_dt
#         count += 1
#         # update buffers
#         for robot in entities.values():
#             robot.update(sim_dt)

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop for UR10 and drone."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Precompute for drone hover
    drone = entities["drone"]
    prop_body_ids = drone.find_bodies("m.*_prop")[0]
    drone_mass = drone.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    while simulation_app.is_running():
        if count % 100 == 0:
            sim_time = 0.0
            count = 0
            for index, (name, robot) in enumerate(entities.items()):
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
            print("[INFO]: Resetting robots state...")
            

        # -- UR10 random motion
        ur10 = entities["ur10"]
        joint_pos_target = ur10.data.default_joint_pos #+ torch.randn_like(ur10.data.joint_pos) * 0.1
        joint_pos_target = joint_pos_target.clamp_(
            ur10.data.soft_joint_pos_limits[..., 0], ur10.data.soft_joint_pos_limits[..., 1]
        )
        ur10.set_joint_position_target(joint_pos_target)
        ur10.write_data_to_sim()

        # # -- Drone hovering
        # forces = torch.zeros(drone.num_instances, 4, 3, device=sim.device)
        # torques = torch.zeros_like(forces)
        # forces[..., 2] = drone_mass * gravity / 4.0
        # drone.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        # drone.write_data_to_sim()
        
        
        # -- Drone hovering with slight random motion
        forces = torch.zeros(drone.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)

        # Base hover thrust
        base_thrust = drone_mass * gravity / 4.0

        # Add small random noise for hovering variation
        noise = torch.randn_like(forces[..., 2]) * 0.005 * base_thrust  # 0.5% noise
        forces[..., 2] = base_thrust + noise

        # Optional: Add a bit of random torque (small wobbles)
        torques += torch.randn_like(torques) * 0.001  # Tiny wobble torque

        drone.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)

        drone.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        for robot in entities.values():
            robot.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design the scene with the UR10 robot and drone
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation app
    simulation_app.close()