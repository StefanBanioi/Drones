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
    episode_length_s = 2.0 #was 6 seconds then 3 seconds now 2

    # multi-agent specification and spaces definition
    possible_agents = ["_DroneRobot", "_Ur10Arm"]
    action_spaces = {"_DroneRobot": 4, "_Ur10Arm": 6}
    observation_spaces = {"_DroneRobot": 23, "_Ur10Arm": 37}
    state_space = 54  # sum of global state dims
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
    moment_scale = 0.01 

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4090, env_spacing=4.0, replicate_physics=True)

    # scales and constants
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    
    disable_ground_collisions = True
    """
    If True, disables collisions for the terrain plane (/World/ground).
    This is useful for open-water / boat scenarios, where the manipulator should not
    physically collide with a 'ground' surface. The plane remains as a visual reference.
    """

    #------------------------------------------------------------
    # PLATFORM MOVENT PARAMETERS
    #------------------------------------------------------------

    enable_platform_motion = True
    """
    If True, applies a smooth, kinematic motion to the UR10 *base* each simulation step.
    This approximates boat/deck motion (heave/sway/surge + optional pitch/roll) without
    requiring a full floating-boat physics model.
    """

    # --- Translational components (meters) ---
    platform_surge_amplitude = 0.0 #was 0.00
    """
    Surge amplitude in meters: motion along +X of the world frame.
    Typical range: 0.00-0.05 m. Start small (0.01-0.03) for stability.
    """

    platform_sway_amplitude = 0.0 #was 0.00
    """
    Sway amplitude in meters: motion along +Y of the world frame.
    Typical range: 0.00-0.05 m. Start small (0.01-0.03).
    """

    platform_heave_amplitude = 0.0 #was 0.02
    """
    Heave amplitude in meters: motion along +Z of the world frame (up/down).
    Typical range: 0.00-0.08 m. Start small (0.01-0.03).
    """

    # --- Rotational components (degrees) ---
    platform_roll_amplitude_deg = 13.0
    """
    Roll amplitude in degrees: rotation about the +X axis (tilting left/right).
    Typical range: 0-5 deg. Start at 0-2 deg.
    """

    platform_pitch_amplitude_deg = 13.0 # was 1.0
    """
    Pitch amplitude in degrees: rotation about the +Y axis (tilting forward/back).
    Typical range: 0-5 deg. Start at 0-2 deg.
    """

    # --- Frequency (Hz) ---
    platform_motion_frequency_hz = 0.40
    """
    Base oscillation frequency in Hz for all platform motion components.
    Typical maritime-like range: 0.10-0.50 Hz.
    Higher frequency makes control harder and can destabilize training.
    """

    platform_random_phase = False # Normally put this as rue
    """
    If True, each environment gets an independent random phase per motion axis.
    This prevents all envs moving in sync and improves robustness.
    If False, all envs share the same phase (more reproducible/visualizable).
    """

















    # -------------------------------------------------------------
    # REWARDS
    # -------------------------------------------------------------

    distance_to_goal_reward_scale = 300.0   # Reward approaching robot EE
    smooth_landing_bonus = 180.0            # Bonus when drone is both slow and close
    proximity_bonus = 250.0                 # Bonus when drone is very close
    time_bonus_scale = 5.0                  # Encourage early task completion
    orientation_reward_scale = 25.0        # Encourage robot EE to face upwards
    wrist_height_reward_scale = 0 #180         # Encourage wrists to be at a certain height
    wrist_height_penalty_scale = 0 #-180       # Penalize wrists being too low

    safe_z_alignment_threshold = 0.90

    arm_go_safe_scale = 1.0 

    arm_hold_still_scale = 0.8 #was 0.5 

    arm_near_jitter_scale = 0.25  #was 0.05

    dist_reward_boost_near = 1.0

    # punishments    
    lin_vel_reward_scale = 0              # Penalize high linear velocity (drone) used to be 1.5 ->10
    ang_vel_reward_scale = 0            # Penalize angular velocity (drone)   used to be -0.1 -> 1.10
    time_penalty = -0.01                   # Per-step penalty to encourage speed
    died_penalty = 0.0                   # Penalty for going out of bounds used to be -100.0 -> -10.0

    alignment_reward = 25    # Now more than ever, we want the drone to be aligned with the arm's end-effector (Previously it was 0 as the arm immediately was in the correct initial position)                   
    magnet_reward = 10000                  

    # Old wind scale for testing with old drone. Triple the conditions for the new bigger drone
    magnet_condition_distance = 0.4      # Distance at which the magnet can catch the drone
    magnet_condition_max_speed = 15       # Speed at which the magnet can catch the drone
    magnet_time_threshold_in_seconds = 0.1  # Number of seconds the drone must be within the magnet condition to be considered caught

    # Conditions for the drone to be considered aligned with the arm's end-effector
    approach_zone = 0.90  # Distance at which the drone is considered close enough to the arm's end-effector (90 cm)
    alignment_threshold = 0.70  # Cosine similarity threshold for alignment (0.70 corresponds to ~45° angle (arccos(0.70) ≈ 45°))  ~45.572996 degrees

    # wind scale for no wind 
    lower_wind_scale = 0.0
    upper_wind_scale = 0.0 

    # wind scale for wind
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.2


    # # wind scale for medium wind
    # lower_wind_scale = 0.3
    # upper_wind_scale = 0.45 

    # # # wind scale for strong wind
    # lower_wind_scale = 0.5
    # upper_wind_scale = 0.6

    # wind scale for testing overall performance
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.6

    # -----------------------
    # Wind mode toggle
    # -----------------------
    RealisticWindYesOrNo = True  # True = Option A (turn-rate integrated), False = Option B (cone updates)

    # Wind speed range (Option A & B)
    wind_speed_min = lower_wind_scale
    wind_speed_max = upper_wind_scale

    # -----------------------
    # Option A: realistic (noise -> turn rate)
    # -----------------------
    wind_max_yaw_rate_deg = 25.0   # max direction change speed (deg/s)

    # -----------------------
    # Option B: cone updates (new target every few seconds)
    # -----------------------
    wind_cone_half_angle_deg = 35.0  # max deviation from previous direction (deg)
    wind_update_interval_s = 2.0     # how often we pick a new target direction/speed

    # -----------------------
    # Smoothing (both modes)
    # -----------------------
    wind_direction_tau_s = 0.7  # smaller = snappier direction transitions
    wind_speed_tau_s = 0.9      # smaller = snappier speed transitions

    # Optional: keep gusts from triggering after success
    suppress_gusts_on_win = True
