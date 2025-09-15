# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate the Avular Vertex One quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/vertex_one.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate the Vertex One quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import VERTEX_ONE_CFG  # <-- our new config


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.0, 1.0, 1.5], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = VERTEX_ONE_CFG.replace(prim_path="/World/VertexOne")
    robot_cfg.spawn.func("/World/VertexOne", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # create handle
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Fetch mass and gravity
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # For applying forces: use the root body (whole quad as one rigid body)
    root_body_ids = robot.find_bodies(".*")[0]  # matches the root

    print("[INFO]: Vertex One setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            sim_time = 0.0
            count = 0
            # reset state
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            print(">>>>>>>> Reset!")

        # Apply upward force to hover
        forces = torch.zeros(robot.num_instances, 1, 3, device=sim.device)
        torques = torch.zeros_like(forces)

        # Lift force = mass * g
        forces[..., 2] = robot_mass * gravity

        # (Optional) Small roll torque just to show it moves
        # torques[..., 0] = 0.2

        # Apply to root body
        robot.set_external_force_and_torque(forces, torques, body_ids=root_body_ids)

        robot.write_data_to_sim()
        sim.step()

        # update counters
        sim_time += sim_dt
        count += 1

        # refresh buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
