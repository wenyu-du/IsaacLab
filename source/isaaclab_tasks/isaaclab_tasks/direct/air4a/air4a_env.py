# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# TODO: Import your robot's configuration
from isaaclab_assets.robots.air4a import AIR4A_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# TODO: Import the base environment class
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class Air4aEnvCfg(DirectRLEnvCfg):
    """Configuration for the Air4a environment."""

    # env
    episode_length_s = 20.0
    decimation = 2
    # TODO: Define action and observation spaces based on your robot
    action_scale = 0.5
    action_space = 12  # Example: 12 joints for air4a
    observation_space = 87  # Example: Placeholder value

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = AIR4A_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # TODO: Define joint gears if necessary
    joint_gears: list = [1.0] * 12

    # TODO: Define reward weights
    # Rewards
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2

    # TODO: Define termination conditions
    # Terminations
    death_cost: float = -2.0
    termination_height: float = 0.31 # TODO: Adjust based on robot height

    # TODO: Define observation scaling
    # Observations
    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


class Air4aEnv(LocomotionEnv):
    """Environment for training the Air4a robot."""

    cfg: Air4aEnvCfg

    def __init__(self, cfg: Air4aEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The rendering mode.
            **kwargs: Additional keyword arguments.
        """
        # You can add custom logic here before calling the parent constructor
        super().__init__(cfg, render_mode, **kwargs)

    # TODO: Implement your own reward functions by overriding methods from LocomotionEnv
    # Example:
    # def _reward_lin_vel_z(self):
    #     return super()._reward_lin_vel_z()

    # TODO: Implement your own observation functions by overriding methods from LocomotionEnv
    # Example:
    # def _observe_base_lin_vel(self):
    #    return super()._observe_base_lin_vel()

