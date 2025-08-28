# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add an on-board camera to a robot's end-effector
and view its feed in real-time in a separate window.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/run_ae_robot_with_camera.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding a camera to a robot's end-effector.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

# For creating a real-time camera viewport
from omni.kit.viewport.utility import create_viewport_window

##
# Pre-defined configs
##
# TODO: Replace this with the actual import for your ae_robot configuration.
from isaaclab_assets.robots.air4a import AIR4A_CFG


@configclass
class AERobotCameraSceneCfg(InteractiveSceneCfg):
    """Design the scene with a camera on the robot's end-effector."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = AIR4A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # camera
    # NOTE: The end-effector frame name is set to 'air4a_link6'. Please verify this is correct for your robot.
    realsense_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/air4a_link6/realsense_d405",
        update_period=0,  # Update every frame
        height=360,
        width=480,
        data_types=["rgb"],  # Only need rgb for visualization
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # Offset the camera from the end-effector frame to point outwards.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Get robot and camera from the scene
    robot = scene["robot"]
    camera: Camera = scene["realsense_camera"]

    # Get default joint positions
    default_joint_pos = robot.data.default_joint_pos.clone()

    # Create a flag to check if the viewport is initialized
    viewport_window = None
    viewport_initialized = False

    # Simulate physics
    while simulation_app.is_running():
        # Create the viewport window after a few steps
        if sim.has_gui() and not viewport_initialized and count > 5:
            # The camera sensor needs to be ticked once to get the render product path
            camera.update(dt=-1)
            # Get the render product path from the first camera
            render_product_path = camera.render_product_paths[0]
            # Create the viewport window
            viewport_window = create_viewport_window(
                render_product_path=render_product_path, name="AE Robot Camera", width=480, height=360
            )
            viewport_initialized = True
            print("[INFO] Created a real-time display window for the robot's camera.")

        # Reset
        if count % 800 == 0:
            count = 0
            sim_time = 0.0
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # --- Move the robot's arm in a circular path ---
        joint_pos_target = default_joint_pos.clone()
        joint_pos_target[:, 0] += 0.5 * torch.sin(torch.tensor(sim_time * 2 * math.pi / 5, device=sim.device))
        joint_pos_target[:, 1] += 0.5 * torch.cos(torch.tensor(sim_time * 2 * math.pi / 5, device=sim.device))

        # -- apply action to the robot
        robot.set_joint_position_target(joint_pos_target)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

    # Cleanup
    if viewport_window:
        viewport_window.destroy()


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])
    # design scene
    scene_cfg = AERobotCameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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