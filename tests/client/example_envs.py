import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
import pandas as pd
import io
from typing import Any
from pytupli.benchmark import TupliEnvWrapper
from pytupli.schema import (
    ArtifactMetadata,
    RLTuple,
    EpisodeMetadataCallback,
)
from pytupli.storage import TupliStorage

# Test environment class
class SimpleTestEnv(gym.Env):
    """Simple environment for testing."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self.state, {"test_key": "test_value"}

    def step(self, action):
        self.steps += 1
        self.state = np.array([0.1 * action, 0.1 * self.steps], dtype=np.float32)
        terminated = self.steps >= 2
        return self.state, float(action), terminated, False, {"test_key": "test_value"}

# Additional test environments for parameterized testing
class ContinuousTestEnv(gym.Env):
    """Test environment with continuous action space."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.state = np.zeros(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(2)
        return self.state, {}

    def step(self, action):
        self.state = np.array([action[0], self.state[1] + 0.1])
        return self.state, float(action[0]), self.state[1] >= 0.2, False, {}

# Test environment class with artifact
class TestEnvArtifact(SimpleTestEnv):
    """Simple environment for testing."""
    def __init__(self):
        super().__init__()
        # Create a simple CSV as bytes
        csv_data = 't,value\n0,10\n1,20\n2,30\n'
        # Create a dataframe for verification
        df = pd.read_csv(io.StringIO(csv_data), dtype={'t': str})
        df.set_index('t', inplace=True, verify_integrity=True)
        self.artifact_df = df

class CustomTupliEnvWrapper(TupliEnvWrapper):
    def _serialize(self, env) -> Env:
        related_data_sources = []
        ds = env.artifact_df
        metadata = ArtifactMetadata(name='test')
        data_kwargs = {'header': None}
        try:
            content = ds.to_csv(encoding='utf-8', **data_kwargs)
            content = content.encode(encoding='utf-8')
        except Exception as e:
            raise ValueError(f'Failed to serialize data source: {e}')

        ds_storage_metadata = self.storage.store_artifact(artifact=content, metadata=metadata)
        related_data_sources.append(ds_storage_metadata.id)
        setattr(env, 'artifact_df', ds_storage_metadata.id)
        return env, related_data_sources

    @classmethod
    def _deserialize(cls, env: Env, storage: TupliStorage) -> Env:
        data_kwargs = {'header': None, 'index_col': 0}
        ds = storage.load_artifact(env.artifact_df)
        ds = ds.decode('utf-8')
        d = io.StringIO(ds)
        df = pd.read_csv(d, **data_kwargs)

        env.artifact_df = df
        return env

class CustomMetadataCallback(EpisodeMetadataCallback):
    """Test callback that tracks the maximum reward seen."""

    def __init__(self):
        self.max_reward = float('-inf')

    def reset(self) -> None:
        pass

    def __call__(self, last_tuple: RLTuple) -> dict[str, Any]:
        reward = last_tuple.reward
        self.max_reward = max(self.max_reward, reward)
        return {
            'reward': reward,
            'max_reward_seen': self.max_reward
        }
