# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script provides a viewer for the dual-arm tabletop scene.

It allows for visualizing the scene, debugging the setup, and testing different components.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/view_dual_arm_scene.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Viewer for the dual-arm tabletop scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# # TODO: Replace this with the actual import for your ae_robot configuration.
# from isaaclab_assets.robots.air4a import AIR4A_CFG

# # Note: These values may need to be tuned for your specific robot.
# for actuator_cfg in AIR4A_CFG.actuators.values():
#     actuator_cfg.stiffness = 1200
#     actuator_cfg.damping = 170
#     actuator_cfg.effort_limit_sim = 300

from isaaclab_assets.robots.air4a_withfinger import AIR4A_WITH_FINGER_CFG

for actuator_cfg in AIR4A_WITH_FINGER_CFG.actuators.values():
    actuator_cfg.stiffness = 1200
    actuator_cfg.damping = 170
    actuator_cfg.effort_limit_sim = 300

@configclass
class AirRobotTableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a tabletop scene with the air_robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.51)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    dual_arm_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dual_arm",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/bjae/project/Assets/test_stage/dual_arm_body_withmesh.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25017, -0.90137, -1.12303)),
    )
    # robot
    robot_left: ArticulationCfg = AIR4A_WITH_FINGER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_left",
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={
            "air4a_joint1": -0.1096,
            "air4a_joint2":  0.3648, 
            "air4a_joint3":  0.5554,
            "air4a_joint4":   0.2520,
            "air4a_joint5":  0.3076,
            "air4a_joint6":  -0.0809
        },
        pos=(0.0, -0.055, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        ),
    )
    robot_right: ArticulationCfg = AIR4A_WITH_FINGER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot_right",
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={
            "air4a_joint1": -0.1096,
            "air4a_joint2":  0.3648, 
            "air4a_joint3":  0.5554,
            "air4a_joint4":   0.2520,
            "air4a_joint5":  0.3076,
            "air4a_joint6":  -0.0809
        },
        pos=(0.0, 0.055, 0.0), rot=(0.7071, -0.7071, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
        ),
    )

# @configclass
# class AirRobotTableTopSceneCfg(InteractiveSceneCfg):
#     """Configuration for a tabletop scene with the air_robot."""

#     # ground plane
#     ground = AssetBaseCfg(
#         prim_path="/World/defaultGroundPlane",
#         spawn=sim_utils.GroundPlaneCfg(),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.51)),
#     )

#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )
#     dual_arm_scene = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/dual_arm",
#         spawn=sim_utils.UsdFileCfg(usd_path="/home/bjae/project/Assets/test_stage/dual_arm_body_withmesh.usd"),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25017, -0.90137, -1.12303)),
#     )
#     # robot
#     robot_left: ArticulationCfg = AIR4A_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/Robot_left",
#         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.055, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
#     )
#     robot_right: ArticulationCfg = AIR4A_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/Robot_right",
#         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.055, 0.0), rot=(0.7071, -0.7071, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
#     )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device) # using a larger dt for visualization
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.get_physics_context().enable_gpu_dynamics(True)  # 关键行
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 1.0])
    # Design scene
    scene_cfg = AirRobotTableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Get simulation dt
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update scene
        scene.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
