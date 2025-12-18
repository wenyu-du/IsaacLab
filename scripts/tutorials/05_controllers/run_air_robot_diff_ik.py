# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the air_robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_air_robot_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller with air_robot.")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
# TODO: Replace this with the actual import for your ae_robot configuration.
# from isaaclab_assets.robots.air4a import AIR4A_WITH_FINGER_CFG
from isaaclab_assets.robots.air4a_withfinger import AIR4A_WITH_FINGER_CFG
# Increase the stiffness and damping of the robot's actuators for better tracking performance.
# A stiffer robot can hold its pose more accurately against gravity and other forces.
# Note: These values may need to be tuned for your specific robot.
# for actuator_cfg in AIR4A_WITH_FINGER_CFG.actuators.values():
#     actuator_cfg.stiffness = 7000.0
#     actuator_cfg.damping = 0.003
for actuator_cfg in AIR4A_WITH_FINGER_CFG.actuators.values():
    actuator_cfg.stiffness = 1200
    actuator_cfg.damping = 170
    actuator_cfg.effort_limit_sim = 300

# for actuator_cfg in AIR4A_WITH_FINGER_CFG.actuators.values():
#     actuator_cfg.stiffness = 40.0
#     actuator_cfg.damping = 200.0
# @configclass
# class AirRobotTableTopSceneCfg(InteractiveSceneCfg):
#     """Configuration for a tabletop scene with the air_robot."""

#     # ground plane
#     ground = AssetBaseCfg(
#         prim_path="/World/defaultGroundPlane",
#         spawn=sim_utils.GroundPlaneCfg(),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
#     )

#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # mount for left robot
#     table_left = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table_left",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path="/home/bjae/project/Assets/STL_library/body_base/body_base.usd", scale=(2.0, 2.0, 2.0)
#         ),
        
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.0, 0.0)),
#     )

#     # mount for right robot
#     # table_right = AssetBaseCfg(
#     #     prim_path="{ENV_REGEX_NS}/Table_right",
#     #     spawn=sim_utils.UsdFileCfg(
#     #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
#     #     ),
#     #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.4, 0.0)),
#     # )

#     # robot
#     robot_left: ArticulationCfg = AIR4A_WITH_FINGER_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/Robot_left",
#         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.4, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
#     )
#     robot_right: ArticulationCfg = AIR4A_WITH_FINGER_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/Robot_right",
#         init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.4, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
#     )
    


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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot_left = scene["robot_left"]
    robot_right = scene["robot_right"]

    # Create controller
    # Use Damped-Least-Squares (DLS) method for IK.
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 1}
    )
    # diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="trans", ik_params={"k_val": 1})
    diff_ik_controller_left = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    diff_ik_controller_right = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    ee_marker_cfg = FRAME_MARKER_CFG.copy()
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker_left = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_current_left"))
    goal_marker_left = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_goal_left"))
    ee_marker_right = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_current_right"))
    goal_marker_right = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_goal_right"))

    # Define goals for the arm in the robot's base frame.
    # Format: [x, y, z, qx, qy, qz, qw]
    ee_goals_left = [
        [0.66, -0.40, -0.05, 0.707, 0.0, 0.707, 0.0],
        # [0.4, 0.2, 0.4, 0.707, 0.0, 0.707, 0.0],
    ]
    ee_goals_right = [
        [0.66, 0.40, -0.05, 0.707, 0.0, 0.707, 0.0],
        # [0.4, -0.2, 0.4, 0.707, 0.0, 0.707, 0.0],
    ]
    ee_goals_left = torch.tensor(ee_goals_left, device=sim.device)
    ee_goals_right = torch.tensor(ee_goals_right, device=sim.device)
    current_goal_idx = 0

    # Create buffers to store actions
    ik_commands_left = torch.zeros(scene.num_envs, diff_ik_controller_left.action_dim, device=robot_left.device)
    ik_commands_left[:] = ee_goals_left[current_goal_idx]
    ik_commands_right = torch.zeros(scene.num_envs, diff_ik_controller_right.action_dim, device=robot_right.device)
    ik_commands_right[:] = ee_goals_right[current_goal_idx]

    # -- Specify robot-specific parameters
    # TODO: Make sure the joint names and end-effector frame name are correct for your robot.
    robot_entity_cfg_left = SceneEntityCfg("robot_left", joint_names=[".*"], body_names=["air4a_link6"])
    robot_entity_cfg_left.resolve(scene)
    robot_entity_cfg_right = SceneEntityCfg("robot_right", joint_names=["air4a_joint[1-6]"], body_names=["air4a_link6"])
    robot_entity_cfg_right.resolve(scene)
    # Obtain the frame index of the end-effector from the jacobian
    if robot_left.is_fixed_base:
        ee_jacobi_idx_left = robot_entity_cfg_left.body_ids[0] - 1
    else:
        ee_jacobi_idx_left = robot_entity_cfg_left.body_ids[0]
    if robot_right.is_fixed_base:
        ee_jacobi_idx_right = robot_entity_cfg_right.body_ids[0] - 1
    else:
        ee_jacobi_idx_right = robot_entity_cfg_right.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 800 == 0:
            # reset time and goal
            count = 0
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals_left)
            ik_commands_left[:] = ee_goals_left[current_goal_idx]
            ik_commands_right[:] = ee_goals_right[current_goal_idx]
            # reset joint state
            joint_pos_left = robot_left.data.default_joint_pos.clone()
            joint_vel_left = robot_left.data.default_joint_vel.clone()
            robot_left.write_joint_state_to_sim(joint_pos_left, joint_vel_left)
            robot_left.reset()
            joint_pos_right = robot_right.data.default_joint_pos.clone()
            joint_vel_right = robot_right.data.default_joint_vel.clone()
            robot_right.write_joint_state_to_sim(joint_pos_right, joint_vel_right)
            robot_right.reset()
            # reset controller
            diff_ik_controller_left.reset()
            diff_ik_controller_left.set_command(ik_commands_left)
            diff_ik_controller_right.reset()
            diff_ik_controller_right.set_command(ik_commands_right)
            print(f"[INFO] Resetting robot state and moving to goal {current_goal_idx}...")

        # obtain quantities from simulation for left robot
        jacobian_left = robot_left.root_physx_view.get_jacobians()[:, ee_jacobi_idx_left, :, robot_entity_cfg_left.joint_ids]
        ee_pose_w_left = robot_left.data.body_pose_w[:, robot_entity_cfg_left.body_ids[0]]
        root_pose_w_left = robot_left.data.root_pose_w
        joint_pos_left = robot_left.data.joint_pos[:, robot_entity_cfg_left.joint_ids]
        # compute frame in root frame
        ee_pos_b_left, ee_quat_b_left = subtract_frame_transforms(
            root_pose_w_left[:, 0:3], root_pose_w_left[:, 3:7], ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des_left = diff_ik_controller_left.compute(ee_pos_b_left, ee_quat_b_left, jacobian_left, joint_pos_left)

        # obtain quantities from simulation for right robot
        jacobian_right = robot_right.root_physx_view.get_jacobians()[:, ee_jacobi_idx_right, :, robot_entity_cfg_right.joint_ids]
        ee_pose_w_right = robot_right.data.body_pose_w[:, robot_entity_cfg_right.body_ids[0]]
        root_pose_w_right = robot_right.data.root_pose_w
        joint_pos_right = robot_right.data.joint_pos[:, robot_entity_cfg_right.joint_ids]
        # compute frame in root frame
        ee_pos_b_right, ee_quat_b_right = subtract_frame_transforms(
            root_pose_w_right[:, 0:3], root_pose_w_right[:, 3:7], ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des_right = diff_ik_controller_right.compute(
            ee_pos_b_right, ee_quat_b_right, jacobian_right, joint_pos_right
        )

        # apply actions
        robot_left.set_joint_position_target(joint_pos_des_left, joint_ids=robot_entity_cfg_left.joint_ids)
        robot_right.set_joint_position_target(joint_pos_des_right, joint_ids=robot_entity_cfg_right.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # update marker positions
        ee_pose_w_left = robot_left.data.body_state_w[:, robot_entity_cfg_left.body_ids[0], 0:7]
        ee_marker_left.visualize(ee_pose_w_left[:, 0:3], ee_pose_w_left[:, 3:7])
        goal_marker_left.visualize(ik_commands_left[:, 0:3] + scene.env_origins, ik_commands_left[:, 3:7])
        ee_pose_w_right = robot_right.data.body_state_w[:, robot_entity_cfg_right.body_ids[0], 0:7]
        ee_marker_right.visualize(ee_pose_w_right[:, 0:3], ee_pose_w_right[:, 3:7])
        goal_marker_right.visualize(ik_commands_right[:, 0:3] + scene.env_origins, ik_commands_right[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.001, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 1.0])
    # Design scene
    scene_cfg = AirRobotTableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()