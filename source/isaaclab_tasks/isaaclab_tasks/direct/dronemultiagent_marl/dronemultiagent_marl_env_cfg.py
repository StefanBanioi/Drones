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
    """
    Configuration for the drone + UR10 multi-agent environment.

    This file is now organized around two goals:

    1. Preserve the current baseline behavior through PACE = 0.
    2. Prepare the environment for future curriculum phases without forcing
       those behaviors into the environment code yet.

    Important design principle:
    ---------------------------
    This config should act as the single source of truth for curriculum setup.
    The environment file should read settings from here rather than hard-coding
    phase logic in many places.

    PACE overview:
    --------------
    PACE 0:
        Current shared-task baseline behavior.

    PACE 1:
        Separated training boxes.
        - Drone learns static point-to-point without wind.
        - Arm learns static reach + orientation without waves.

    PACE 2:
        Separated training boxes with moving targets.
        - Drone learns moving point-to-point.
        - Arm learns moving point-to-point + orientation tracking.

    PACE 3:
        Separated training boxes with disturbances.
        - Drone gets wind/gusts.
        - Arm gets platform motion / waves.

    PACE 4:
        Recombined shared task.
        - Drone and arm are back in the same environment and resume the
          cooperative landing objective.
    """


    # =====================================================================
    # 1) CORE ENVIRONMENT SETTINGS
    # =====================================================================

    decimation = 2
    episode_length_s = 3.0 #was 6 seconds then 3 seconds now 2
    debug_vis = True  

    # multi-agent specification and spaces definition
    possible_agents = ["_DroneRobot", "_Ur10Arm"]
    action_spaces = {"_DroneRobot": 4, "_Ur10Arm": 6}

    # Keep these dimensions stable across PACE phases
    observation_spaces = {"_DroneRobot": 23, "_Ur10Arm": 37}
    state_space = 54  # sum of global state dims
      
    # Optional curriculum-level debug comments
    PACE = 1
    PRINT_PACE_COMMENTS = True
    PRINT_PACE_ON_RESET = True
    PRINT_PACE_GOAL_UPDATES = True

    # Human-readable phase names for logging / debugging
    PACE_NAME_MAP = {
        "0": "PACE_0_SHARED_BASELINE",
        "1": "PACE_1_SEPARATE_STATIC",
        "2": "PACE_2_SEPARATE_MOVING",
        "3": "PACE_3_SEPARATE_DISTURBED",
        "4": "PACE_4_SHARED_FINAL",
    }

    # =====================================================================
    # 2) SIMULATION SETTINGS
    # =====================================================================

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

    # =====================================================================
    # 3) TERRAIN SETTINGS
    # =====================================================================

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
    
    disable_ground_collisions = True
    """
    If True, disables collisions for the terrain plane (/World/ground).
    
    Useful for open-water / boat scenarios, where the manipulator should not
    physically collide with a 'ground' surface. The plane remains as a visual reference.
    """

    # =====================================================================
    # 4) ROBOT ASSET SETTINGS
    # =====================================================================

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
            # Bellow is the "OG"
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

    Drone_CFG: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # =====================================================================
    # 5) SCENE SETTINGS
    # =====================================================================

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4090,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # =====================================================================
    # 6) OBSERVATION / ACTION PROCESSING SETTINGS
    # =====================================================================

    vel_obs_scale = 0.2
    act_moving_average = 0.05 #1.0
    
    # These toggles do not yet change behavior by themselves.
    # They exist so the environment code can later switch observation/state
    # contents cleanly based on curriculum phase.
    INCLUDE_WIND_IN_OBS = True
    INCLUDE_CROSS_AGENT_INFO_IN_OBS = True
    INCLUDE_GOAL_IN_OBS = True
    INCLUDE_GOAL_ORIENTATION_IN_OBS = False

    # =====================================================================
    # 7) CURRICULUM / PACE BEHAVIOR SETTINGS
    # =====================================================================

    # ---------------------------------------------------------------------
    # Shared curriculum philosophy
    # ---------------------------------------------------------------------
    #
    # These settings are phase-readiness settings for now.
    # The env code should later read from them and decide:
    # - how goals are spawned
    # - what disturbances are enabled
    # - what reward modules are active
    # - what reset logic is used
    # - what bounds / death boxes are applied
    #
    # For the first refactor pass, PACE = 0 should preserve the current
    # shared-task baseline as much as possible.
    #

    # --- High-level curriculum toggles ---
    USE_SEPARATED_TRAINING_BOXES = False
    USE_SHARED_GOAL_LOGIC = True
    USE_MOVING_GOALS = False
    USE_DRONE_GOAL = True
    USE_ARM_GOAL = False
    USE_MAGNET_LOGIC = True
    USE_SHARED_SUCCESS_CONDITION = True

    # --- Goal modes ---
    # Intended future values:
    #   "ee_tracking"      -> drone follows live UR10 ee_link
    #   "static_world"     -> static world-space goal
    #   "moving_world"     -> moving world-space goal
    #   "arm_sphere_pose"  -> random arm target pose in reachable sphere
    DRONE_GOAL_MODE = "ee_tracking"
    ARM_GOAL_MODE = "none"

    # --- Goal update timing for future moving-target phases ---
    goal_update_interval_s = 2.0
    goal_hold_time_for_success_s = 0.10

    # =====================================================================
    # 8) TRAINING BOX / BOUNDARY SETTINGS
    # =====================================================================

    # Current baseline shared drone box around env origin.
    # These are written explicitly here so later phases can switch between
    # shared and separated boxes from config rather than hard-coded values.
    shared_box_x_min = -2.0
    shared_box_x_max = 2.0
    shared_box_y_min = -2.0
    shared_box_y_max = 2.0
    shared_box_z_min = 0.1
    shared_box_z_max = 2.0

    # Future separated drone box
    drone_box_x_min = -2.0
    drone_box_x_max = 2.0
    drone_box_y_min = -2.0
    drone_box_y_max = 2.0
    drone_box_z_min = 0.25
    drone_box_z_max = 2.0

    # Future separated arm workspace / goal region
    arm_goal_sphere_radius = 0.50
    arm_goal_min_height = 0.10
    arm_goal_max_height = 1.50

    # Per-environment side split for separated PACE phases
    # Negative X side = drone side
    # Positive X side = arm side
    drone_side_x_center = -1.0
    arm_side_x_center = 1.0
    side_half_width = 0.60

    # Spawn safety margin from boundaries
    reset_spawn_margin_xy = 0.05
    reset_spawn_margin_z = 0.10
    # Emergency world-Z kill switch for disabled-ground setups.
    # If the drone falls below this absolute world Z, terminate/reset it.
    drone_world_z_kill = 0.30

    # =====================================================================
    # 9) RESET SETTINGS
    # =====================================================================

    # Drone reset randomization
    drone_reset_randomize_position = True
    drone_reset_randomize_orientation = False
    drone_reset_x_range = (-1.5, 1.5)
    drone_reset_y_range = (-1.5, 1.5)
    drone_reset_z_range = (0.0, 0.5)

    # UR10 root randomization
    ur10_reset_randomize_root = True
    ur10_reset_root_x_range = (-0.2, 0.2)
    ur10_reset_root_y_range = (-0.2, 0.2)

    # UR10 joint reset randomization
    ur10_reset_randomize_joints = False

    # =====================================================================
    # 10) PLATFORM MOTION / WAVE SETTINGS
    # =====================================================================

    enable_platform_motion = False #True
    """
    If True, applies a smooth, kinematic motion to the UR10 *base* each simulation step.
    This approximates boat/deck motion (heave/sway/surge + optional pitch/roll) without
    requiring a full floating-boat physics model.

    In future PACE phases:
    - disabled in early arm-training phases
    - enabled in disturbed arm-training phases
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

    # =====================================================================
    # 11) REWARD SETTINGS
    # =====================================================================

    # ---------------------------------------------------------------------
    # Drone reward terms
    # ---------------------------------------------------------------------
    distance_to_goal_reward_scale = 300.0   # Reward approaching robot EE
    smooth_landing_bonus = 180.0            # Bonus when drone is both slow and close
    proximity_bonus = 250.0                 # Bonus when drone is very close
    time_bonus_scale = 5.0                  # Encourage early task completion
    alignment_reward = 25                   # Now more than ever, we want the drone to be aligned with the arm's end-effector (Previously it was 0 as the arm immediately was in the correct initial position)                   
    magnet_reward = 10000                  

    lin_vel_reward_scale = -0.03                # Penalize high linear velocity (drone) used to be 1.5 ->10
    ang_vel_reward_scale = -0.01               # Penalize angular velocity (drone)   used to be -0.1 -> 1.10

    # ---------------------------------------------------------------------
    # Arm reward terms
    # ---------------------------------------------------------------------
    orientation_reward_scale = 25.0        # Encourage robot EE to face upwards
    wrist_height_reward_scale = 0 #180         # Encourage wrists to be at a certain height
    wrist_height_penalty_scale = 0 #-180       # Penalize wrists being too low
    safe_z_alignment_threshold = 0.90
    arm_go_safe_scale = 1.0 
    arm_hold_still_scale = 0.02 #was 0.8 
    arm_near_jitter_scale = 0.25  #was 0.05

    # ---------------------------------------------------------------------
    # Shared / penalty terms
    # ---------------------------------------------------------------------
    time_penalty = -0.01                   # Per-step penalty to encourage speed
    died_penalty = -50.0                   # Penalty for going out of bounds used to be -100.0 -> -10.0
    dist_reward_boost_near = 1.0

    # ---------------------------------------------------------------------
    # Landing / capture conditions
    # ---------------------------------------------------------------------
    magnet_condition_distance = 0.4      # Distance at which the magnet can catch the drone
    magnet_condition_max_speed = 15       # Speed at which the magnet can catch the drone
    magnet_time_threshold_in_seconds = 0.1  # Number of seconds the drone must be within the magnet condition to be considered caught

    # Conditions for the drone to be considered aligned with the arm's end-effector
    approach_zone = 0.90  # Distance at which the drone is considered close enough to the arm's end-effector (90 cm)
    alignment_threshold = 0.70  # Cosine similarity threshold for alignment (0.70 corresponds to ~45° angle (arccos(0.70) ≈ 45°))  ~45.572996 degrees

    # ---------------------------------------------------------------------
    # Reward module toggles for future phase control
    # ---------------------------------------------------------------------
    ENABLE_REWARD_DISTANCE_TO_GOAL = True
    ENABLE_REWARD_SMOOTH_LANDING = True
    ENABLE_REWARD_PROXIMITY = True
    ENABLE_REWARD_TIME_SHAPING = True
    ENABLE_REWARD_ALIGNMENT = True
    ENABLE_REWARD_MAGNET = True
    ENABLE_REWARD_ARM_ORIENTATION = True
    ENABLE_REWARD_ARM_WRIST = True
    ENABLE_REWARD_ARM_GO_SAFE = True
    ENABLE_REWARD_ARM_HOLD_STILL = True
    ENABLE_REWARD_ARM_NEAR_JITTER = True
    ENABLE_REWARD_DIED_PENALTY = True

    # =====================================================================
    # 12) WIND SETTINGS
    # =====================================================================

    # For PACE readiness, keep the old wind system but document it better.
    # Later phases can simply enable/disable these from phase logic.

    enable_wind = False #True
    enable_wind_gusts = False #True

    # wind scale for no wind 
    lower_wind_scale = 0.0
    upper_wind_scale = 0.0 

    # Example alternatives:
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.2
    #
    # lower_wind_scale = 0.3
    # upper_wind_scale = 0.45
    #
    # lower_wind_scale = 0.5
    # upper_wind_scale = 0.6
    #
    # wind scale for testing overall performance
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.6

    # -----------------------
    # Wind mode toggle
    # -----------------------
    RealisticWindYesOrNo = True
    """
    True:
        Use noise-driven continuous turn-rate wind updates.
    
    False:
        Use cone-based wind direction target updates.
    """ 

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
    

    # =====================================================================
    # 13) FUTURE PHASE PRESETS
    # =====================================================================
    #
    # These are documentation presets for later implementation in the env.
    # They do not automatically apply behavior yet unless the env reads them.
    #
    # Keeping them here makes curriculum tuning much easier later on.
    #

    PACE_PRESET_DESCRIPTIONS = {
        "0": "Current shared-task baseline.",
        "1": "Separated boxes, static goals, no wind, no waves.",
        "2": "Separated boxes, moving goals, no wind, no waves.",
        "3": "Separated boxes, moving goals, wind for drone, waves for arm.",
        "4": "Shared final task, agents recombined.",
    }