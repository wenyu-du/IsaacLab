# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the air4a robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

AIR4A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/bjae/asset/air4a/air4a.usd",
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
            "air4a_joint2":  0.3648, 
            "air4a_joint3":  0.5554,
            "air4a_joint4":   0.2520,
            "air4a_joint5":  0.3076,
            "air4a_joint6":  -0.0809
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["air4a_joint[1-6]"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.175,
            stiffness=1200,
            damping=170,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of air4a robot."""
