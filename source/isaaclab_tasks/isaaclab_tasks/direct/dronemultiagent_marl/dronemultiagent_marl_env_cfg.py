# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
#from isaaclab_tasks.direct.arm_drone_communication.arm_drone_communication_env import ArmDroneCommunicationEnv
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

@configclass
class DronemultiagentMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 6.0

    # multi-agent specification and spaces definition
    possible_agents = ["_DroneRobot", "_Ur10Arm"]
    action_spaces = {"_DroneRobot": 4, "_Ur10Arm": 6}
    observation_spaces = {"_DroneRobot": 20, "_Ur10Arm": 34}
    state_space = 51  # sum of global state dims
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

    # terrain
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

    # robot(s)

    # UR10 arm
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
            "_Ur10Arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )
    """Configuration of UR-10 arm using implicit actuator models."""

    # Drone
    Drone_CFG: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01 #* 3.0  # Scale the moment of inertia by 3.0

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.0335,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4090, env_spacing=4.0, replicate_physics=True)

    # scales and constants
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    
    distance_to_goal_reward_scale = 125.0   # Reward approaching robot EE
    smooth_landing_bonus = 180.0            # Bonus when drone is both slow and close
    proximity_bonus = 250.0                 # Bonus when drone is very close
    time_bonus_scale = 1.0                  # Encourage early task completion
    orientation_reward_scale = 125.0        # Encourage robot EE to face upwards
    wrist_height_reward_scale = 180         # Encourage wrists to be at a certain height
    wrist_height_penalty_scale = -180       # Penalize wrists being too low

    # punishments    
    lin_vel_reward_scale = -0.05           # Penalize high linear velocity (drone)
    ang_vel_reward_scale = -0.10           # Penalize angular velocity (drone)
    unstable_penalty = -2.0                # Penalty when drone is unstable
    time_penalty = -0.01                   # Per-step penalty to encourage speed
    angular_vel_threshold = 55            # Threshold for defining "unstable"
    died_penalty = -500.0                 # Penalty for going out of bounds
