# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the air4a robot with a gripper."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

AIR4A_WITH_FINGER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/bjae/project/Assets/test_stage/air4a_withfinger.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "air4a_joint1": -0.1096,
            "air4a_joint2": 0.3648,
            "air4a_joint3": 0.5554,
            "air4a_joint4": 0.2520,
            "air4a_joint5": 0.3076,
            "air4a_joint6": -0.0809,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "arm_j1": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint1"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=700,
            damping=0,
        ),
        "arm_j2": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint2"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=700,
            damping=0,
        ),
        "arm_j3": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint3"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=266,
            damping=0,
        ),
        "arm_j4": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint4"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=241,
            damping=0,
        ),
        "arm_j5": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint5"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=44,
            damping=0,
        ),
        "arm_j6": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint6"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=12,
            damping=0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=15,
            damping=0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of air4a robot with a gripper."""
