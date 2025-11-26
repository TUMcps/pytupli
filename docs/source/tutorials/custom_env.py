import io
import math
import torch
from typing import Optional

import numpy as np
import pandas as pd

from gymnasium import spaces
from gymnasium.envs.classic_control import utils, MountainCarEnv

from pytupli.benchmark import TupliEnvWrapper
from pytupli.storage import TupliStorage
from gymnasium import Env
from pytupli.schema import ArtifactMetadata, EpisodeMetadataCallback


class CustomMountainCarEnv(MountainCarEnv):
    """
    ## Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ## Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

    ## Action Space

    There are 3 discrete deterministic actions:

    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right

    ## Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.

    ## Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.

    ## Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ## Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.

    ## Arguments

    Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<MountainCarEnv<MountainCar-v0>>>>>
    >>> env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})  # default x_init=np.pi, y_init=1.0
    (array([-0.46352962,  0.        ], dtype=float32), {})

    ```

    ## Version History

    * v0: Initial versions release
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(self, data_path: str, render_mode: Optional[str] = None, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.current_step = 0
        self.data = pd.read_csv(data_path, index_col=0, header=None) * 0.01

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(action), f'{action!r} ({type(action)}) invalid'

        position, velocity = self.state
        velocity += (
            (action - 1) * self.force
            + math.cos(3 * position) * (-self.gravity)
            - self.data.loc[self.current_step].to_numpy().flatten()[0] * math.cos(position)
        )
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = -1.0
        self.current_step += 1

        self.state = (position, velocity)
        if self.render_mode == 'human':
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.current_step = 0
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == 'human':
            self.render()
        return np.array(self.state, dtype=np.float32), {}


class MyTupliEnvWrapper(TupliEnvWrapper):
    def _serialize(self, env) -> Env:
        related_data_sources = []
        ds = env.unwrapped.data
        metadata = ArtifactMetadata(name='test')
        data_kwargs = {'header': None}
        try:
            content = ds.to_csv(encoding='utf-8', **data_kwargs)
            content = content.encode(encoding='utf-8')
        except Exception as e:
            raise ValueError(f'Failed to serialize data source: {e}')

        ds_storage_metadata = self.storage.store_artifact(artifact=content, metadata=metadata)
        related_data_sources.append(ds_storage_metadata.id)
        setattr(env.unwrapped, 'data', ds_storage_metadata.id)
        return env, related_data_sources

    @classmethod
    def _deserialize(cls, env: Env, storage: TupliStorage) -> Env:
        data_kwargs = {'header': None, 'index_col': 0}
        ds = storage.load_artifact(env.unwrapped.data)
        ds = ds.decode('utf-8')
        d = io.StringIO(ds)
        df = pd.read_csv(d, **data_kwargs)

        env.unwrapped.data = df
        return env

class MyCallback(EpisodeMetadataCallback):
    def __init__(self, is_expert: bool = False):
        super().__init__()
        # we will compute the cumulative reward for an episode
        self.cum_reward = 0
        # Furthermore, we want to store the fact that the episode was not an expert episode
        self.is_expert = is_expert
    def reset(self):
        # we will compute the cumulative reward for an episode
        self.cum_reward = 0
    def __call__(self, tuple):
        self.cum_reward += tuple.reward
        return {"cum_eps_reward": [self.cum_reward], "is_expert": self.is_expert}


def discretize_observation(bins: int, ranges: tuple, observations: torch.tensor) -> torch.tensor:
    """
    Discretizes continuous observations into specified number of bins.

    Args:
        observations (np.ndarray): Continuous observations to be discretized.
        bins (int): Number of bins to discretize each dimension.
        ranges (tuple): two arrays, one containing the minima and one containing the maxima of each dim.

    Returns:
        np.ndarray: Discretized observations.
    """
    observations = observations.cpu().numpy()
    discretized = np.zeros_like(observations, dtype=int)
    for i in range(observations.shape[1]):
        min_val = ranges[0][i]
        max_val = ranges[1][i]
        discretized[:, i] = np.digitize(
            observations[:, i],
            bins=np.linspace(min_val, max_val, bins + 1)[1:-1]
        )
    return torch.tensor(discretized)
