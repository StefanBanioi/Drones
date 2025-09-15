# Created By Stefan Banioi 2025
# University of Maastricht - Robotics
# All rights reserved.

"""Configuration for the Avular Vertex One Quadcopter"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

VERTEX_ONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Users\\UMRobotics\\Desktop\\Stefan Code\\AvularDrone.usd",
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
        pos=(0, 0, 1.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            # The Avular Vertex One V1 doesn't have the Crazyflie spinning rotor joints, 
            # our mock-up has no moving joints. Forces/torques will later be applied programmatically instead.
            # "m1_joint": 200.0,# *5.2,
            # "m2_joint": -200.0,# *5.2,
            # "m3_joint": 200.0,# *5.2,
            # "m4_joint": -200.0,# *5.2,
        },
    ),
    actuators={
        # The Vertex is a rigid body with no articulated joints
        # We use a dummy actuator to satisfy the API
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Avular Vertex One Quadcopter."""
