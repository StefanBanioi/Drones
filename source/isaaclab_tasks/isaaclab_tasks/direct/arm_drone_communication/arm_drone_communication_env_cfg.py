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
    
    episode_length_s = 4 # second
    decimation = 2  #(120Hz simulation, 60Hz inputs)
    action_space = 10 # this means we have 10 actions output. (for the drone 4 + the arm 6)
    
    # The following line is used when we also want to include wind in the environment
    observation_space = 27 # this means we have 27 observations (for the drone 12 + the arm 12 + 3 for the wind)
    # The following line is used when we do not want to include wind in the environment
    #observation_space = 24 # this means we have 24 observations (for the drone 12 + the arm 12)
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

    robotDrone: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9 
    moment_scale = 0.01

    UR10_CFG = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR10",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #disable_gravity=False,   # # Set to True to disable gravity for the UR10 arm
                disable_gravity=True,   # Set to True to disable gravity for the UR10 arm
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # (These parameters makes the Ur10 robot look up instead of laying down on it's side)
            joint_pos={
                "shoulder_pan_joint": 1.5708,
                "shoulder_lift_joint": -0.7854,
                "elbow_joint": -0.7854,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.5708,
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

    # The following reward scales are used for the ArmDroneCommunicationEnv
    # Reward scales are used to tune the rewards for the environment
    # rewards

    #++++++++++++++++++++++++++++++++++++++ Reward Scales ++++++++++++++++++++++++++++++++++++++++++++++++
    # Drone related rewards
    distance_to_goal_reward_scale = 300.0        
    smooth_landing_bonus = 180                   
    proximity_bonus = 250.0                      
    alignment_reward = 25              
    time_bonus_scale = 5.0  

    # punishments    
    lin_vel_reward_scale = 0.0                   
    ang_vel_reward_scale = -0.1  

    # Arm related rewards
    orientation_reward_scale = 25                
    wrist_height_reward_scale = 0 #75            
    wrist_height_penalty_scale = 0 #-75          
                          
    # Overall environment Goal rewards
    magnet_reward = 1000 
    died_penalty = -100.0

    # Condition for the magnet to catch the drone
    magnet_condition_distance = 0.1      # Distance at which the magnet can catch the drone
    magnet_condition_max_speed = 5       # Speed at which the magnet can catch the drone
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
                            

    # +++++++++++++++++++++++++++++ Wind Configuration For Experiments ++++++++++++++++++++++++++++++++++

    # Wind scale for the environment
    # Uncomment the desired wind scale for the environment
    # It can also be changed after the environment is TRAINED for testing purposes

    # === Wind scale for no wind ===
    # lower_wind_scale = 0.0
    # upper_wind_scale = 0.0 

    # === Wind scale for light wind ===
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.2

    # === Wind scale for medium wind ===
    # lower_wind_scale = 0.3
    # upper_wind_scale = 0.45 

    # === Wind scale for high wind ===
    # lower_wind_scale = 0.5
    # upper_wind_scale = 0.6

    # === Wind scale for overall performance === 
    # This has wind speeds ranging from low to high
    lower_wind_scale = 0.1
    upper_wind_scale = 0.6

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
