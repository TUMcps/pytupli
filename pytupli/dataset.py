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

    def convert_to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts the dataset tuples into numpy arrays.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - observations: Array of state observations
                - actions: Array of actions
                - rewards: Array of rewards
                - terminals: Array of terminal flags
                - timeouts: Array of timeout flags
        """
        observations = np.array([tuple.state for tuple in self.tuples], dtype=np.float64)
        actions = np.array([tuple.action for tuple in self.tuples], dtype=np.float64)
        rewards = np.array([tuple.reward for tuple in self.tuples], dtype=np.float64)
        terminals = np.array([tuple.terminal for tuple in self.tuples], dtype=np.float64)
        timeouts = np.array([tuple.timeout for tuple in self.tuples], dtype=np.float64)
        return observations, actions, rewards, terminals, timeouts
