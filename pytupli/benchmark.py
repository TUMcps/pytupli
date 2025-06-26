"""
Module for everything related to benchmark management.
"""

from __future__ import annotations

import hashlib
from typing import Any, SupportsFloat
from gymnasium import Env, Wrapper
import jsonpickle

from pytupli.schema import (
    Benchmark,
    BenchmarkMetadata,
    BenchmarkQuery,
    RLTuple,
    Episode,
    FilterEQ,
    EpisodeMetadataCallback,
)
from pytupli.storage import TupliStorage


class TupliEnvWrapper(Wrapper):
    """A wrapper for Gymnasium environments that enables serialization and
    deserialization with the goal of creating reproducible benchmarks from environments.
    It handles the interface to the storage backend, including storing, loading, and
    publishing benchmarks. Enables users to record interactions with gymnasium
    environments to the storage such that they can be used as datasets for offline RL.

    Args:
        env (Env): The Gymnasium environment to wrap
        storage (TupliStorage): Storage backend for saving benchmark and episode data
        benchmark_id (str | None): Identifier for the benchmark. Defaults to None
        metadata_callback (EpisodeMetadataCallback | None): Callback for generating
        episode metadata. Defaults to None
        rl_tuple_cls (type[RLTuple]): Class to use for creating RL tuples. Defaults to RLTuple
    """

    def __init__(
        self,
        env: Env,
        storage: TupliStorage,
        benchmark_id: str | None = None,
        metadata_callback: EpisodeMetadataCallback | None = None,
        rl_tuple_cls: type[RLTuple] = RLTuple,
    ):
        super().__init__(env)
        self.storage = storage
        self.tuple_buffer = []  # list of RLTuples
        self._record_episodes = True  # whether to record tuples or not
        self.metadata_callback = metadata_callback
        self.id = benchmark_id  # Benchmark ID once stored
        self.rl_tuple_cls = rl_tuple_cls

    def activate_recording(self):
        """Activates the recording of environment interactions.

        When active, the wrapper will record tuples of (state, action, reward, etc.)
        and store them as episodes.
        """
        self._record_episodes = True

    def deactivate_recording(self):
        """Deactivates the recording of environment interactions.

        When deactivated, the wrapper will not record or store any environment interactions.
        """
        self._record_episodes = False

    def _get_hash(self, obj: Any) -> str:
        """Generates a hash for a given object using JSON serialization.

        Args:
            obj (Any): The object to hash

        Returns:
            str: SHA-256 hash of the serialized object
        """
        return hashlib.sha256(jsonpickle.encode(obj).encode('utf-8')).hexdigest()

    def serialize_env(self, env: Env) -> str:
        """Serializes a Gymnasium environment to a JSON string.

        This method handles the serialization of the environment and any related artifacts.

        Args:
            env (Env): The environment to serialize

        Returns:
            str: JSON string representation of the environment
        """
        env, related_artifacts = self._serialize(env)
        setattr(env.unwrapped, 'related_artifacts', related_artifacts)
        serialized_env = jsonpickle.encode(env, indent=4, warn=True)
        return serialized_env

    @classmethod
    def deserialize_env(cls, serialized_env: str, storage: TupliStorage) -> Env:
        """Deserializes a JSON string back into a Gymnasium environment.

        Args:
            serialized_env (str): The JSON string representation of the environment
            storage (TupliStorage): Storage backend for loading related artifacts

        Returns:
            Env: The deserialized Gymnasium environment
        """
        env = jsonpickle.decode(serialized_env)
        env = cls._deserialize(env, storage)
        return env

    def _serialize(self, env: Env) -> tuple[Env, list]:
        """Internal method for environment serialization.

        This method is meant to be overridden by subclasses to implement custom
        serialization behavior, e.g., for artifacts such as csv files or trained models.

        Args:
            env (Env): The environment to serialize

        Returns:
            tuple[Env, list]: Tuple containing the processed environment and list of related artifacts
        """
        related_artifacts = []
        return env, related_artifacts

    @classmethod
    def _deserialize(cls, env: Env, storage: TupliStorage) -> Env:
        """Internal method for environment deserialization.

        This method is meant to be overridden by subclasses to implement custom
        deserialization behavior, e.g., for artifacts such as csv files or trained models.

        Args:
            env (Env): The environment to deserialize
            storage (TupliStorage): Storage backend for loading artifacts

        Returns:
            Env: The deserialized environment
        """
        return env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Resets the environment and returns the initial observation.

        Args:
            seed (int | None): Random seed for environment reset. Defaults to None
            options (dict[str, Any] | None): Additional options for reset. Defaults to None

        Returns:
            tuple[Any, dict[str, Any]]: Initial observation and info dictionary
        """
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Takes a step in the environment and optionally records the interaction.

        If recording is active, stores the interaction in the tuple buffer and creates
        an episode when the episode terminates.

        Args:
            action (Any): The action to take in the environment

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: Tuple containing:
                - observation: The environment observation
                - reward: The reward for the action
                - terminated: Whether the episode terminated naturally
                - truncated: Whether the episode was artificially terminated
                - info: Additional information from the environment
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        rl_tuple = self.rl_tuple_cls.from_env_step(obs, action, reward, terminated, truncated, info)

        if self._record_episodes:
            episode_metadata = self.metadata_callback(rl_tuple) if self.metadata_callback else {}
            self.tuple_buffer.append(rl_tuple)

            if terminated or truncated:
                episode = Episode(
                    benchmark_id=self.id, metadata=episode_metadata, tuples=self.tuple_buffer
                )
                self.storage.record_episode(episode)
                self.tuple_buffer = []
                if self.metadata_callback:
                    self.metadata_callback.reset()

        return obs, reward, terminated, truncated, info

    def _prepare_storage(self, metadata: BenchmarkMetadata) -> tuple[str, str]:
        """Prepares benchmark data for storage.

        Args:
            metadata (BenchmarkMetadata): Metadata for the benchmark

        Returns:
            tuple[str, str]: Tuple containing the benchmark hash and serialized environment
        """
        serialized_env = self.serialize_env(self.env)
        benchmark_hash = self._get_hash([self.env, metadata])
        return benchmark_hash, serialized_env

    def store(
        self,
        name: str,
        description: str = '',
        difficulty: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] = {},
    ) -> str:
        """Stores the benchmark in the storage backend.

        Args:
            name (str): Name of the benchmark
            description (str, optional): Description of the benchmark. Defaults to ''
            difficulty (str | None, optional): Difficulty level of the benchmark. Defaults to None
            version (str | None, optional): Version of the benchmark. Defaults to None
            metadata (dict[str, Any], optional): Additional metadata. Defaults to {}

        Returns:
            str: The ID of the stored benchmark
        """
        metadata = BenchmarkMetadata(
            name=name,
            description=description,
            difficulty=difficulty,
            version=version,
            extra=metadata,
        )
        benchmark_hash, serialized_env = self._prepare_storage(metadata=metadata)
        object_metadata = self.storage.store_benchmark(
            benchmark_query=BenchmarkQuery(
                hash=benchmark_hash, serialized=serialized_env, metadata=metadata
            )
        )
        self.id = object_metadata.id

    def publish(self) -> None:
        """Publishes the benchmark, making it available for other users depending on their access rights.

        This method should be called after storing the benchmark when it's ready
        to be used by others.
        """
        self.storage.publish_benchmark(self.id)

    @classmethod
    def load(
        cls,
        storage: TupliStorage,
        benchmark_id: str | None = None,
        metadata_callback: EpisodeMetadataCallback | None = None,
        rl_tuple_cls: type[RLTuple] = RLTuple,
    ) -> TupliEnvWrapper:
        """Loads a benchmark from storage.

        Args:
            storage (TupliStorage): Storage backend to load from
            benchmark_id (str | None): ID of the benchmark to load. Defaults to None
            metadata_callback (EpisodeMetadataCallback | None): Callback for generating
                episode metadata. Defaults to None

        Returns:
            TupliEnvWrapper: A new wrapper instance with the loaded benchmark
        """
        stored_benchmark: Benchmark = storage.load_benchmark(benchmark_id)
        env: Env = cls.deserialize_env(stored_benchmark.serialized, storage)

        return cls(env, storage, benchmark_id, metadata_callback, rl_tuple_cls)

    def delete(self, delete_artifacts: bool = False, delete_episodes: bool = True):
        """Deletes the benchmark and optionally its related data from storage.

        Args:
            delete_artifacts (bool, optional): Whether to delete related artifacts.
                Defaults to False
            delete_episodes (bool, optional): Whether to delete related episodes.
                Defaults to True

        Raises:
            Exception: If deletion of benchmark, episodes, or artifacts fails
        """
        if delete_episodes:
            try:
                episode_filter = FilterEQ(key='benchmark_id', value=self.id)
                episodes = self.storage.list_episodes(episode_filter, include_tuples=True)
                for eps in episodes:
                    self.storage.delete_episode(eps.id)
            except Exception as e:
                raise e
        try:
            self.storage.delete_benchmark(self.id)
        except Exception as e:
            raise e
        if delete_artifacts:
            try:
                for ds_id in self.env.unwrapped.related_artifacts:
                    self.storage.delete_artifact(ds_id)
            except Exception as e:
                raise e
