# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the operational space controller (OSC) with the air_robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_air_robot_osc.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the operational space controller with air_robot.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
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
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, subtract_frame_transforms, combine_frame_transforms

##
# Pre-defined configs
##
# TODO: Replace this with the actual import for your ae_robot configuration.
from isaaclab_assets.robots.air4a import AIR4A_CFG

# To use OSC, we need to disable the robot's internal PD controllers.
# We do this by setting the stiffness and damping to zero.
# The OSC will then directly command torques to the joints.
for actuator_cfg in AIR4A_CFG.actuators.values():
    actuator_cfg.stiffness = 0.0
    actuator_cfg.damping = 0.0
# We also disable gravity for the robot articulation, as the OSC will handle gravity compensation.
AIR4A_CFG.spawn.rigid_props.disable_gravity = True


@configclass
class AirRobotTableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a tabletop scene with the air_robot."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # robot
    robot: ArticulationCfg = AIR4A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]

    # Create OSC controller
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],  # We will command absolute poses
        impedance_mode="fixed",  # Use fixed gains
        gravity_compensation=True,  # The controller will compensate for gravity
        nullspace_control="none",  # Explicitly disable null-space control
        motion_damping_ratio_task=1.0,  # Set damping to be critically damped (ratio of 1.0)
    )
    osc_controller = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    ee_marker_cfg = FRAME_MARKER_CFG.copy()
    ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(ee_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm in the robot's base frame.
    ee_goals = [
        [0.5, 0.3, 0.5, 0.707, 0.0, 0.707, 0.0],
        [0.5, -0.3, 0.5, 0.707, 0.0, 0.707, 0.0],
        [0.4, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0

    # -- Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["air4a_link6"])
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector from the jacobian
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            count = 0
            # set new goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            osc_controller.set_command(ee_goals[current_goal_idx].repeat(scene.num_envs, 1))
            # reset robot
            default_joint_pos = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            # reset controller
            osc_controller.reset()
            print(f"[INFO] Resetting robot state and moving to goal {current_goal_idx}...")

        # obtain quantities from simulation
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel) = update_robot_states(
            robot, robot_entity_cfg, ee_jacobi_idx
        )

        # compute the joint commands
        joint_efforts = osc_controller.compute(
            jacobian_b=jacobian_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
        )

        # apply actions
        robot.set_joint_effort_target(joint_efforts, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # update marker positions
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # broadcast the goal pose to all environments for visualization
        goal_pos_b = ee_goals[current_goal_idx, :3].repeat(scene.num_envs, 1)
        goal_quat_b = ee_goals[current_goal_idx, 3:7].repeat(scene.num_envs, 1)
        goal_pos_w, goal_quat_w = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b, goal_quat_b)
        # visualize the markers
        # The marker has only one prototype (the frame), so we specify the index 0 for all instances.
        prototype_indices = torch.zeros(scene.num_envs, dtype=torch.int32, device=sim.device)
        ee_marker.visualize(
            translations=ee_pose_w[:, 0:3],
            orientations=ee_pose_w[:, 3:7],
            marker_indices=prototype_indices,
        )
        goal_marker.visualize(
            translations=goal_pos_w,
            orientations=goal_quat_w,
            marker_indices=prototype_indices,
        )

def update_robot_states(robot: Articulation, robot_entity_cfg: SceneEntityCfg, ee_jacobi_idx: int):
    """Helper function to update the robot states for the controller."""
    # obtain dynamics related quantities from simulation
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, robot_entity_cfg.joint_ids, :][:, :, robot_entity_cfg.joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, robot_entity_cfg.joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = robot.data.body_vel_w[:, robot_entity_cfg.body_ids[0], :]
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, ee_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, ee_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]

    return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
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
