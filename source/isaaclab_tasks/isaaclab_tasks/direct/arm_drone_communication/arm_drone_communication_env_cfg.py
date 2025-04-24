# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
#from isaaclab_tasks.direct.arm_drone_communication.arm_drone_communication_env import ArmDroneCommunicationEnv
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class ArmDroneCommunicationEnvCfg(DirectRLEnvCfg):
    # env
    #episode_length_s = 4 # seconds
    
    # going to try a little test where the episode_length is less than 4 seconds (try to converge faster)
    episode_length_s = 2 # seconds

    decimation = 2
    #action_space = 4 # this means we have 4 actions output. (for only the drone)
    action_space = 10 # this means we have 10 actions output. (for the drone 4 + the arm 6)
    #observation_space = 12 # this means we have 12 observations (for the drone)
    observation_space = 24 # this means we have 24 observations (for the drone 12 + the arm 12)
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # robotDrone
    robotDrone: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    UR10_CFG = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR10",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,   # # Set to True to disable gravity for the UR10 arm
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
    """Configuration of UR-10 arm using implicit actuator models."""


    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
