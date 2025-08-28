# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a reinforcement learning (RL) environment for a custom robot.

This environment is based on the `AERobotSceneCfg` and is designed to be compatible with RL frameworks
like Stable-Baselines3. It includes definitions for observations, actions, rewards, and terminations.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_ae_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a custom RL environment for the ae_robot.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Robot configuration
# TODO: Replace this with the actual import for your ae_robot configuration.
from isaaclab_assets.robots.air4a import AIR4A_CFG as AE_ROBOT_CFG


##
# Scene definition
##


@configclass
class AERobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with the ae_robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    ae_robot: ArticulationCfg = AE_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# MDP (Markov Decision Process) configuration for the environment
##


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # Use joint efforts to control the robot, targeting all joints (.*)
    joint_effort = mdp.JointEffortActionCfg(asset_name="ae_robot", joint_names=[".*"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Observe the robot's joint positions and velocities
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("ae_robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("ae_robot")})
        # Observe the last action taken
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Custom functions for the environment
##

def joint_pos_l2_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalizes distance from default joint positions."""
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_pos - robot.data.default_joint_pos), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Computes the L2 norm of the action rate."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def joint_effort_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Computes the L2 norm of the joint efforts."""
    # The action manager stores the last action commanded to the robot.
    # For joint effort actions, this is the effort command.
    return torch.sum(torch.square(env.action_manager.action), dim=1)

def time_out_termination_custom(env: ManagerBasedRLEnv, time_out: bool = True) -> torch.Tensor:
    """Terminate the episode when the episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length

def reset_joints_to_default_custom(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Reset the robot's joints to their default positions."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos
    joint_vel = robot.data.default_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


@configclass
class RewardsCfg:
    """Reward specifications for the environment."""

    # (Negative) Reward for staying close to the default joint positions (encourages stability)
    joint_pos_tracking = RewTerm(func=joint_pos_l2_error, weight=-1.0, params={"asset_cfg": SceneEntityCfg("ae_robot")})

    # Penalty for high action rate (encourages smooth actions)
    action_rate_penalty = RewTerm(func=action_rate_l2, weight=-0.01)

    # Penalty for high joint efforts (encourages energy efficiency)
    joint_effort_penalty = RewTerm(func=joint_effort_l2, weight=-1e-5)


@configclass
class TerminationsCfg:
    """Termination conditions for the environment."""

    # Terminate after a fixed number of steps
    time_out = TermTerm(func=time_out_termination_custom, params={"time_out": True})


@configclass
class EventCfg:
    """Configuration for events in the environment."""

    # Reset the robot's joints to their default positions on every reset
    reset_robot_joints = EventTerm(
        func=reset_joints_to_default_custom, mode="reset", params={"asset_cfg": SceneEntityCfg("ae_robot")}
    )


@configclass
class AERobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ae_robot RL environment."""

    # Use the scene defined in the previous tutorial
    scene: AERobotSceneCfg = AERobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Add the MDP components
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post-initialization checks."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        # viewer settings
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 1.0]


def main():
    """Main function."""
    # Create the RL environment
    env_cfg = AERobotEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            actions = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            # print reward from the first environment
            if count % 10 == 0:
                print(f"[Env 0] Reward: {rew[0].item():.2f}")

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()