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
    observation_space = 27 # this means we have 27 observations (for the drone 12 + the arm 12 + 3 for the wind)
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
    moment_scale = 0.01  #3.0 # scale the moment of inertia

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
            # joint_pos={
            #     "shoulder_pan_joint": 0.0,
            #     "shoulder_lift_joint": -1.712,
            #     "elbow_joint": 1.712,
            #     "wrist_1_joint": 0.0,
            #     "wrist_2_joint": 0.0,
            #     "wrist_3_joint": 0.0,
            # },
            #This is the new initial state for the arm (it makes it look up instead of laying down)
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


    # reward scales

    # distance_to_goal_reward_scale = 1   # Reward approaching robot EE
    # smooth_landing_bonus = 1            # Bonus when drone is both slow and close
    # ee_height_bonus = 0.5      
    # alive_bonus =1   
    # died_penalty = -2               # Penalty for going out of bounds
    # lin_vel_reward_scale = -0.01           # Penalize high linear velocity (drone)
    # orientation_reward_scale = 1        # Encourage robot EE to face upwards


    # distance_to_goal_reward_scale = 125.0   # Reward approaching robot EE
    # smooth_landing_bonus = 180.0            # Bonus when drone is both slow and close
    # proximity_bonus = 250.0                 # Bonus when drone is very close
    # time_bonus_scale = 1.0                  # Encourage early task completion
    # orientation_reward_scale = 50.0         # Encourage robot EE to face upwards
    # orientation_penalty_scale = -50.0       # Penalize robot EE facing downwards
    # wrist_height_reward_scale = 75          # Encourage wrists to be at a certain height
    # wrist_height_penalty_scale = -25        # Penalize wrists being too low
    # interception_reward = 15.0              # Reward for intercepting the drone
    # # punishments    
    # lin_vel_reward_scale = -0.05           # Penalize high linear velocity (drone)
    # ang_vel_reward_scale = -0.01           # Penalize angular velocity (drone)
    # unstable_penalty = -2.0                # Penalty when drone is unstable
    # time_penalty = -0.01                   # Per-step penalty to encourage speed
    # died_penalty = -100.0                 # Penalty for going out of bounds



    # Testing a more aggressive catching 
    distance_to_goal_reward_scale = 300.0        # ⬆️ prioritize approach
    smooth_landing_bonus = 180                   # ⛔ remove slow-landing bias
    proximity_bonus = 250.0                      # ✅ but modify in code to require drone speed > X
    time_bonus_scale = 5.0                       # ⬆️ push fast intercepts
    orientation_reward_scale = 25                # 👍 keep
    wrist_height_reward_scale = 0 #75            # maybe keep (depends on your arm catching height)
    wrist_height_penalty_scale = 0 #-75          # maybe keep
    interception_reward = 15.0                   # keep (but modify in code to require drone speed > X)

    alignment_reward = 25                         # keep (but modify in code to require drone speed > X)
    magnet_reward = 1000 

    # punishments    
    lin_vel_reward_scale = 0.0                   # 🟡 temporarily disable velocity penalty — we want movement!
    ang_vel_reward_scale = -0.1                  # keep
    unstable_penalty = -2.0                      # keep
    time_penalty = -0.02                         # ⬆️ to push efficiency
    died_penalty = -100.0                        # keep

    
    # wind scale
    lower_wind_scale = 0.1
    upper_wind_scale = 0.6

    # Condition for the magnet to catch the drone
    magnet_condition_distance = 0.1  # Distance at which the magnet can catch the drone
    magnet_condition_max_speed = 5       # Speed at which the magnet can catch the drone















    # #reward scales
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0 
    # smooth_landing_bonus = 10.0  # This is a bonus for smooth landing 
    # proximity_bonus = 25.0       # Strong bonus when drone is really close 
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 1.0  # Reward for pointing ee_link up

    #test 2
    # Same reward scales but keep everything at 1.0 except for the orientation reward scale
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 1.0
    # smooth_landing_bonus = 1.0  # Tune as needed 
    # proximity_bonus = 1.0       # Strong bonus when drone is really close
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 10  # Reward for pointing ee_link up

    #test 3
    # Abusrd value for the orientation reward scale
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 1.0
    # smooth_landing_bonus = 1.0  # Tune as needed
    # proximity_bonus = 1.0       # Strong bonus when drone is really close
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 1000  # Reward for pointing ee_link up
    # This is a test to see if the orientation reward even matters or works

    #test 5
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0
    # smooth_landing_bonus = 10.0  # Tune as needed (was 2 now 10)
    # proximity_bonus = 25.0       # Strong bonus when drone is really close 
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 1.0  # Reward for pointing ee_link up


    
    #test 6
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0
    # smooth_landing_bonus = 10.0  # Tune as needed (was 2 now 10)
    # proximity_bonus = 15.0       # Strong bonus when drone is really close 
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 25.0  # Reward for pointing ee_link up


    # #test 7
    # #with changes suggested by chatgpt
    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0
    # smooth_landing_bonus = 10.0  # Tune as needed (was 2 now 10)
    # proximity_bonus = 15.0       # Strong bonus when drone is really close
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 100.0 # Reward for pointing ee_link up
    # upright_bonus_scale = 50.0 

    # #test 8

    # lin_vel_reward_scale = -0.05
    # ang_vel_reward_scale = -0.01
    # distance_to_goal_reward_scale = 15.0
    # smooth_landing_bonus = 10.0  # Tune as needed (was 2 now 10)
    # proximity_bonus = 25.0       # Strong bonus when drone is really close 
    # time_bonus_scale = 1.0      # Scales with how fast it finishes
    # orientation_reward_scale = 10 # Reward for pointing ee_link up
    # upright_bonus_scale = 20.0