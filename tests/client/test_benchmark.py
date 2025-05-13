"""Test methods of environment wrapper module."""
import pytest
import numpy as np
from pytupli.benchmark import TupliEnvWrapper
from pytupli.storage import TupliStorageError
from pytupli.schema import RLTuple

from tests.client.example_envs import CustomTupliEnvWrapper

def test_serialize_env(parameterized_env):
    """Test serialization and deserialization of environment."""
    benchmark = TupliEnvWrapper(parameterized_env, None)  # No storage needed for this test
    serialized = benchmark.serialize_env(parameterized_env)
    assert isinstance(serialized, str)

    # Test deserialization
    deserialized = TupliEnvWrapper.deserialize_env(serialized, None)
    assert isinstance(deserialized, type(parameterized_env))

def test_serialize_env_with_artifact(test_benchmark_artifact, test_env_artifact, test_storage):
    """Test serialization and deserialization of environment with artifact."""
    serialized = test_benchmark_artifact.serialize_env(test_env_artifact)
    assert isinstance(serialized, str)

    # Test deserialization
    deserialized = CustomTupliEnvWrapper.deserialize_env(serialized, storage=test_storage)
    assert isinstance(deserialized, type(test_env_artifact))
    assert deserialized.artifact_df.iloc[0,0] == 10

def test_load_non_existent_benchmark(test_storage):
    """Test loading a non-existent benchmark."""
    with pytest.raises(Exception):
        TupliEnvWrapper.load(test_storage, "non_existent_id")

def test_store_and_load(test_benchmark, test_storage):
    """Test storing and loading a benchmark."""
    # Store the benchmark
    test_benchmark.store(
        name="Test Benchmark",
        description="A test benchmark",
        difficulty="easy",
        version="1.0",
        metadata={"test_key": "test_value"}
    )
    assert test_benchmark.id is not None

    # Load the benchmark
    loaded_benchmark = TupliEnvWrapper.load(test_storage, test_benchmark.id)
    assert loaded_benchmark.id == test_benchmark.id

    # Test that the loaded environment works
    obs, _ = loaded_benchmark.reset()
    assert np.array_equal(obs, np.array([0.0, 0.0]))

    obs, reward, term, trunc, _ = loaded_benchmark.step(np.int64(1))
    assert reward == 1.0
    assert np.array_equal(obs, np.array([0.1, 0.1], dtype=np.float32))
    test_benchmark.delete()

def test_duplicate_store(test_benchmark):
    """Test storing the same benchmark multiple times."""
    # First store
    test_benchmark.store(name="Test Benchmark", description="Test")
    first_id = test_benchmark.id

    # Second store of same benchmark
    test_benchmark.store(name="Test Benchmark", description="Test")
    second_id = test_benchmark.id

    # IDs should be the same since it's the same benchmark
    assert first_id == second_id
    test_benchmark.delete()

def test_publish(test_benchmark):
    """Test publishing a benchmark."""
    test_benchmark.store(name="Test Benchmark", description="Test")
    test_benchmark.publish()  # Test the publish functionality
    test_benchmark.delete()

def test_convert_to_tuple():
    """Test the tuple conversion functionality."""
    obs = np.array([0.1, 0.2])
    action = np.int64(1)
    reward = 1.0
    terminated = True
    truncated = False
    info = {"test": "info"}

    rl_tuple = RLTuple.from_env_step(obs, action, reward, terminated, truncated, info)

    assert rl_tuple.state == [0.1, 0.2]
    assert rl_tuple.action == 1
    assert rl_tuple.reward == reward
    assert rl_tuple.terminal == terminated
    assert rl_tuple.timeout == truncated
    assert rl_tuple.info == info

def test_invalid_numpy_types():
    """Test handling of invalid numpy types in observations and actions."""
    # Test with invalid observation type (string array)
    with pytest.raises(ValueError, match="Unsupported observation type"):
        RLTuple.from_env_step(
            'invalid',
            np.array([1]),
            0.0,
            False,
            False,
            {},
        )

    # Test with invalid action type (string)
    with pytest.raises(ValueError, match="Unsupported action type"):
        RLTuple.from_env_step(
            np.zeros(2),
            "invalid",
            0.0,
            False,
            False,
            {},
        )

def test_basic_recording(test_benchmark):
    """Test basic recording of episodes."""
    # Store benchmark
    test_benchmark.store(
        name="Test Benchmark",
        description="A test benchmark",
        difficulty="easy",
        version="1.0",
        metadata={"test_key": "test_value"}
    )
    assert test_benchmark.id is not None
    # Reset environment and take a few steps
    obs, _ = test_benchmark.reset()
    assert isinstance(obs, np.ndarray)
    assert len(test_benchmark.tuple_buffer) == 0

    obs, reward, term, trunc, _ = test_benchmark.step(np.int64(1))
    assert len(test_benchmark.tuple_buffer) == 1
    # Verify the tuple was created correctly
    first_tuple = test_benchmark.tuple_buffer[0]
    assert isinstance(first_tuple.state, list)  # Should be converted from numpy
    print(type(first_tuple.action))
    assert isinstance(first_tuple.action, (list, int))  # Should be converted from numpy/int
    assert np.allclose(first_tuple.state, [0.1, 0.1])
    assert first_tuple.action == 1
    assert first_tuple.reward == 1.0
    assert not first_tuple.terminal

    obs, reward, term, trunc, _ = test_benchmark.step(np.int64(0))
    # Episode should be complete and buffer cleared after this step
    assert term
    assert len(test_benchmark.tuple_buffer) == 0
    test_benchmark.delete()

def test_recording_control(test_benchmark):
    """Test activation and deactivation of recording."""
    # Store benchmark
    test_benchmark.store(
        name="Test Benchmark",
        description="A test benchmark",
        difficulty="easy",
        version="1.0",
        metadata={"test_key": "test_value"}
    )
    # Start with recording active (default state)
    obs, _ = test_benchmark.reset()
    test_benchmark.step(np.int64(1))
    assert len(test_benchmark.tuple_buffer) == 1

    # Test deactivation
    test_benchmark.deactivate_recording()
    test_benchmark.step(np.int64(1))
    assert len(test_benchmark.tuple_buffer) == 1  # Should not have increased

    # Test reactivation
    test_benchmark.activate_recording()
    test_benchmark.step(np.int64(1))
    assert len(test_benchmark.tuple_buffer) == 0  # The episode ends after two steps, so we delete the buffer
    test_benchmark.delete()

def test_recording_episodes(test_benchmark):
    """Test that episodes are properly recorded and stored."""
    # Store the benchmark first
    test_benchmark.store(
        name="Test Benchmark",
        description="A test benchmark",
        metadata={}
    )

    # Complete an episode
    obs, _ = test_benchmark.reset()
    test_benchmark.step(np.int64(1))
    test_benchmark.step(np.int64(0))  # This should finish the episode

    # List episodes and verify
    episodes = test_benchmark.storage.list_episodes(include_tuples=True)
    assert len(episodes) == 1
    assert len(episodes[0].tuples) == 2
    assert episodes[0].tuples[0].action == 1
    assert episodes[0].tuples[1].action == 0
    assert episodes[0].tuples[1].terminal
    test_benchmark.delete()

def test_cleanup(test_benchmark, test_storage):
    """Test proper cleanup of resources."""
    # Store benchmark and record some episodes
    test_benchmark.store(name="Test Benchmark", description="Test")
    benchmark_id = test_benchmark.id

    # Record an episode
    obs, _ = test_benchmark.reset()
    test_benchmark.step(np.int64(1))
    test_benchmark.step(np.int64(0))  # This should finish the episode

    # Get episode ID
    episodes = test_storage.list_episodes()
    assert len(episodes) == 1
    episode_id = episodes[0].id

    # Delete the benchmark with cleanup
    test_benchmark.delete(delete_episodes=True)

    # Verify benchmark is gone
    benchmark_ids = test_storage.list_benchmarks()
    assert len(benchmark_ids) == 0

    # Verify episode is gone
    episodes = test_storage.list_episodes()
    assert len(episodes) == 0

def test_delete_with_artifacts(test_benchmark_artifact, test_storage):
    """Test deleting a benchmark with artifacts."""
    test_benchmark_artifact.store(name="Test Benchmark", description="Test")
    # Test deletion with artifacts
    test_benchmark_artifact.delete(delete_artifacts=True)

    # Verify there are no artifacts left
    artifacts = test_storage.list_artifacts()
    assert len(artifacts) == 0

def test_episode_metadata_callback(test_benchmark_with_metadata, test_storage):
    """Test that episode metadata callbacks work correctly."""
    # Store the benchmark
    test_benchmark_with_metadata.store(name="Test Benchmark", description="Test")

    # Run a few episodes
    for _ in range(3):
        obs, _ = test_benchmark_with_metadata.reset()
        for _ in range(2):
            obs, reward, term, trunc, _ = test_benchmark_with_metadata.step(np.int64(1))  # Always take action 1
            if term or trunc:
                break

    # Check the recorded episodes
    episodes = test_storage.list_episodes(include_tuples=True)
    assert len(episodes) == 3

    # Verify metadata is present and correct
    for episode in episodes:
        assert 'reward' in episode.metadata
        assert 'max_reward_seen' in episode.metadata
        assert episode.metadata['reward'] == episode.tuples[-1].reward
        assert episode.metadata['max_reward_seen'] >= episode.metadata['reward']
    test_benchmark_with_metadata.delete()

def test_benchmark_hash_consistency(test_benchmark):
    """Test that the same environment produces the same hash."""
    env = test_benchmark.env
    hash1 = test_benchmark._get_hash(env)
    hash2 = test_benchmark._get_hash(env)
    assert hash1 == hash2

    # Store with same content should produce same ID
    test_benchmark.store(name="Test1", description="Test")
    id1 = test_benchmark.id
    test_benchmark.store(name="Test1", description="Test")
    id2 = test_benchmark.id
    assert id1 == id2  # IDs should be same since environment is same
    test_benchmark.delete()

if __name__ == '__main__':
    pytest.main(['-v', __file__])
