"""
Module for everything related to dataset management.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Generator, List
import random
import numpy as np

from pytupli.schema import (
    BaseFilter,
    BenchmarkHeader,
    EpisodeHeader,
    EpisodeItem,
    FilterOR,
    RLTuple,
)
from pytupli.storage import TupliStorage

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class BaseTupleParser:
    """Base class for parsing (nested) lists of obs/act/reward/term/trunc into tensors for other formats like numpy arrays, PyTorch tensors, etc.
    This class can be extended to implement specific parsing logic for different formats.
    """

    @classmethod
    def parse_lists(
        cls,
        obs: list[list[float]],
        act: list[list[float]],
        rew: list[float],
        term: list[bool],
        trunc: list[bool],
    ) -> tuple:
        """Parses the input lists into a format suitable for further processing.

        Args:
            obs (list[list[float]]): List of observations.
            act (list[list[float]]): List of actions.
            rew (list[float]): List of rewards.
            term (list[bool]): List of terminal flags.
            trunc (list[bool]): List of timeout flags.

        Returns:
            tuple: A tuple containing the parsed data in a suitable format.
        """
        raise NotImplementedError('This method should be overridden by subclasses.')


class NumpyTupleParser(BaseTupleParser):
    """Parser for converting lists of observations, actions, rewards, terminal flags, and timeout flags into numpy arrays."""

    @classmethod
    def parse_lists(
        cls,
        obs: list[list[float]],
        act: list[list[float]],
        rew: list[float],
        term: list[bool],
        trunc: list[bool],
    ) -> tuple:
        """Converts lists to numpy arrays."""
        obs_array = np.array(obs, dtype=np.float32)
        act_array = np.array(act, dtype=np.float32)
        rew_array = np.array(rew, dtype=np.float32)
        term_array = np.array(term, dtype=np.bool_)
        trunc_array = np.array(trunc, dtype=np.bool_)

        return obs_array, act_array, rew_array, term_array, trunc_array


class TorchTupleParser(BaseTupleParser):
    """Parser for converting lists of observations, actions, rewards, terminal flags, and timeout flags into PyTorch tensors."""

    @classmethod
    def parse_lists(
        cls,
        obs: list[list[float]],
        act: list[list[float]],
        rew: list[float],
        term: list[bool],
        trunc: list[bool],
    ) -> tuple:
        """Converts lists to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is not installed. Please install it with: pip install torch')

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        act_tensor = torch.tensor(act, dtype=torch.float32)
        rew_tensor = torch.tensor(rew, dtype=torch.float32)
        term_tensor = torch.tensor(term, dtype=torch.bool)
        trunc_tensor = torch.tensor(trunc, dtype=torch.bool)

        return obs_tensor, act_tensor, rew_tensor, term_tensor, trunc_tensor


class TensorflowTupleParser(BaseTupleParser):
    """Parser for converting lists of observations, actions, rewards, terminal flags, and timeout flags into TensorFlow tensors."""

    @classmethod
    def parse_lists(
        cls,
        obs: list[list[float]],
        act: list[list[float]],
        rew: list[float],
        term: list[bool],
        trunc: list[bool],
    ) -> tuple:
        """Converts lists to TensorFlow tensors."""
        if not TF_AVAILABLE:
            raise ImportError(
                'TensorFlow is not installed. Please install it with: pip install tensorflow'
            )

        obs_tensor = tf.constant(obs, dtype=tf.float32)
        act_tensor = tf.constant(act, dtype=tf.float32)
        rew_tensor = tf.constant(rew, dtype=tf.float32)
        term_tensor = tf.constant(term, dtype=tf.bool)
        trunc_tensor = tf.constant(trunc, dtype=tf.bool)

        return obs_tensor, act_tensor, rew_tensor, term_tensor, trunc_tensor


class TupliDataset:
    """A dataset class for downloading, managing and filtering offline RL tuple data.

    This class provides functionality to load, filter, and process reinforcement learning
    data including benchmarks, episodes, and tuples. It supports various filtering operations
    and provides methods for batch processing and data conversion.

    Args:
        storage (TupliStorage): The storage backend to fetch data from.
    """

    def __init__(self, storage: TupliStorage):
        self.storage = storage

        self._benchmark_filter: BaseFilter = None
        self._episode_filter: BaseFilter = None
        self._tuple_filter_fcn: Callable = None

        self.benchmarks: list[BenchmarkHeader] = []
        self.episodes: list[EpisodeHeader] | list[EpisodeItem] = []
        self.tuples: list[RLTuple] = []

        self._refetch_benchmarks_flag = True
        self._refetch_episodes_flag = True
        self._refetch_tuples_flag = True
        self._refilter_tuples_flag = True

    def _fetch_episodes(self, with_tuples: bool = False) -> None:
        """Fetches episodes from storage based on current filters.

        This internal method refreshes the episodes list based on the current benchmark
        and episode filters. It can optionally include the tuple data for each episode.

        Args:
            with_tuples (bool): If True, includes tuple data in the fetched episodes.
        """
        if self._refetch_benchmarks_flag:
            self.benchmarks = self.storage.list_benchmarks(self._benchmark_filter)
            self._refetch_benchmarks_flag = False

        if self._refetch_episodes_flag or (self._refetch_tuples_flag and with_tuples):
            episode_filter = FilterOR.from_list(
                self.benchmarks, on_key='benchmark_id', from_key='id'
            )
            if self._episode_filter:
                episode_filter = episode_filter & self._episode_filter

            self.episodes = self.storage.list_episodes(episode_filter, include_tuples=with_tuples)
            self._refetch_episodes_flag = False
            self._refetch_tuples_flag = not with_tuples

    def with_benchmark_filter(self, filter: BaseFilter) -> TupliDataset:
        """Creates a new dataset with an additional benchmark filter.

        Args:
            filter (BaseFilter): The filter to apply to benchmarks.

        Returns:
            TupliDataset: A new dataset instance with the applied filter.
        """
        new_dataset = deepcopy(self)
        new_dataset._benchmark_filter = filter
        new_dataset._refetch_benchmarks_flag = True
        return new_dataset

    def with_episode_filter(self, filter: BaseFilter) -> TupliDataset:
        """Creates a new dataset with an additional episode filter.

        Args:
            filter (BaseFilter): The filter to apply to episodes.

        Returns:
            TupliDataset: A new dataset instance with the applied filter.
        """
        new_dataset = deepcopy(self)
        new_dataset._episode_filter = filter
        new_dataset._refetch_episodes_flag = True
        return new_dataset

    def with_tuple_filter(self, filter_fcn: Callable) -> TupliDataset:
        """Creates a new dataset with an additional tuple filter function.

        Args:
            filter_fcn (Callable): A function that takes a tuple and returns a boolean.

        Returns:
            TupliDataset: A new dataset instance with the applied filter.
        """
        new_dataset = deepcopy(self)
        new_dataset._tuple_filter_fcn = filter_fcn
        new_dataset._refilter_tuples_flag = True
        return new_dataset

    def preview(self) -> list[EpisodeHeader]:
        """Returns a preview of the episodes without loading the full tuple data.

        Returns:
            list[EpisodeHeader]: A list of episode headers matching the current filters.
        """
        self._fetch_episodes(with_tuples=False)
        return self.episodes

    def load(self) -> None:
        """Loads all episode data including tuples and applies any filters.

        This method fetches all episode data and their associated tuples, then applies
        any tuple filters that have been set.
        """
        self._fetch_episodes(with_tuples=True)
        if self._refilter_tuples_flag:
            self.tuples = [
                rl_tuple
                for episode in self.episodes
                for rl_tuple in episode.tuples
                if not self._tuple_filter_fcn or self._tuple_filter_fcn(rl_tuple)
            ]
            self._refilter_tuples_flag = False

    def set_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed to set.
        """
        random.seed(seed)

    def as_batch_generator(
        self, batch_size: int, shuffle: bool = False
    ) -> Generator[List[RLTuple], None, None]:
        """Returns a generator that yields batches of tuples from the dataset.

        Args:
            batch_size (int): The size of each batch.
            shuffle (bool): Whether to shuffle the tuples before creating batches.

        Yields:
            List[RLTuple]: Batches of tuples of the specified size.
        """
        # Make sure tuples are loaded
        self.load()

        # Create a copy of the tuples list that we can shuffle if needed
        tuples_to_batch = list(self.tuples)

        # Shuffle if requested
        if shuffle:
            random.shuffle(tuples_to_batch)

        # Yield batches
        for i in range(0, len(tuples_to_batch), batch_size):
            yield tuples_to_batch[i : i + batch_size]

    def sample_episodes(self, n_samples: int) -> list[EpisodeItem]:
        """Randomly samples episodes from the dataset.

        Args:
            n_samples (int): The number of episodes to sample.

        Returns:
            list[EpisodeItem]: A list of randomly sampled episodes.
        """
        self._fetch_episodes(with_tuples=False)
        return random.sample(self.episodes, min(n_samples, len(self.episodes)))

    def convert_to_tensors(self, parser: type[BaseTupleParser] = NumpyTupleParser) -> tuple:
        """Converts the dataset tuples into tensors of the format specified by handing over the respective parser.

        Args:
            parser (BaseTupleParser): The parser to use for converting tuples.

        Returns:
            tuple: A tuple containing tensors in the specified format:
                - observations: Tensor/Array of state observations
                - actions: Tensor/Array of actions
                - rewards: Tensor/Array of rewards
                - terminals: Tensor/Array of terminal flags
                - timeouts: Tensor/Array of timeout flags

        Raises:
            ValueError: If an unsupported format is specified.
            ImportError: If the required library for the format is not installed.
        """
        # First, convert tuples to lists
        observations = [tuple.state for tuple in self.tuples]
        actions = [tuple.action for tuple in self.tuples]
        rewards = [tuple.reward for tuple in self.tuples]
        terminals = [tuple.terminal for tuple in self.tuples]
        timeouts = [tuple.timeout for tuple in self.tuples]

        # Use the parser to convert lists to the desired format
        return parser.parse_lists(observations, actions, rewards, terminals, timeouts)

    def convert_to_d4rl_format(self) -> dict[str, np.ndarray]:
        """Converts the dataset tuples into the format used by D4RL.

        Returns:
            dict[str, np.ndarray]: A dictionary containing:
                - observations: Array of state observations
                - actions: Array of actions
                - next_observations: Array of next state observations
                - rewards: Array of rewards
                - terminals: Array of terminal flags
        """
        obs, act, rew, term, time = self.convert_to_tensors()

        # Create next_observations by shifting observations by one step
        # For the last observation, we duplicate it (common practice in D4RL)
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

        return {
            'observations': obs,
            'actions': act,
            'next_observations': next_obs,
            'rewards': rew,
            'terminals': np.logical_or(term, time),  # Combine terminal and timeout flags
        }

    def convert_to_dataframe(self):
        """Converts the dataset tuples into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - observations: State observations
                - actions: Actions taken
                - next_observations: Next state observations
                - rewards: Rewards received
                - terminals: Terminal flags

        Raises:
            ImportError: If pandas is not installed.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError('pandas is not installed. Please install it with: pip install pandas')

        d4rl_data = self.convert_to_d4rl_format()

        return pd.DataFrame(
            {
                'observations': list(d4rl_data['observations']),
                'actions': list(d4rl_data['actions']),
                'next_observations': list(d4rl_data['next_observations']),
                'rewards': list(d4rl_data['rewards']),
                'terminals': list(d4rl_data['terminals']),
            }
        )
