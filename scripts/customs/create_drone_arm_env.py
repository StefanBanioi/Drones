# create_drone_arm_env.py

from __future__ import annotations

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

from dataclasses import MISSING
from typing import Optional

# Import math stuff
import torch
import math
import os 
import numpy as np

# Isaac Lab
from isaaclab.utils.configclass import configclass 
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import UR10_CFG
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import GroundPlaneCfg
from isaaclab.managers import action_manager, observation_manager , reward_manager, termination_manager, command_manager
from isaaclab.managers import curriculum_manager, scene_entity_cfg, event_manager ,EventTermCfg ,SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# ============
# Assets Setup
# ============

CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
        scale= (3.0, 3.0, 3.0), # Scale the quadcopter to 3x its original size
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

@configclass
class ActionsCfg(action_manager.ActionManager):
    """Actions configuration for the drone arm environment."""
    # Define actions for the drone and arm

    joint_velocities: action_manager.ActionCfg = action_manager.ActionCfg()
    


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

# ==================
# Define Origins
# ==================

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


# ==================
# Scene Configuration
# ==================

def DroneArmSceneCfg() -> tuple[dict, list[list[float]]]:
    """Designs the scene with the UR10 robot and drone."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Origins
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

    # return the scene information with only the UR10 robot and drone
    scene_entities = {
        "ur10": ur10,
        "drone": drone,
    }
    return scene_entities, origins

# ====================
# Environment Config
# ====================

class DroneArmEnvCfg(ManagerBasedRLEnvCfg):
    scene: DroneArmSceneCfg = DroneArmSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg(
        episode_end=EventTerm(func="mdp.reset_scene_to_default", mode="reset")
    )
    rewards: RewardsCfg = RewardsCfg()
    termination: TerminationCfg = TerminationCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

# ================
# Main Environment
# ================

class DroneArmEnv(ManagerBasedRLEnv):
    cfg: DroneArmEnvCfg

    def __init__(self, cfg: Optional[DroneArmEnvCfg] = None, **kwargs):
        super().__init__(cfg, **kwargs)

    def _post_reset(self):
        # Any post-reset logic for drone/arm, sensors, etc.
        pass

# =========
# Entrypoint
# =========

def main():
    parser = argparse.ArgumentParser(description="Training of drone and arm")

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Create environment
    env = DroneArmEnv(DroneArmEnvCfg())

    # Print something useful (optional)
    print("DroneArmEnv launched with:", env.num_envs, "environments")

    # Optional: Headless loop for debug
    while simulation_app.is_running():
        env.step()

if __name__ == "__main__":
    main()
