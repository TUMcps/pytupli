"""Test methods of dataset module."""
import pytest
import numpy as np
from pytupli.dataset import TupliDataset
from pytupli.schema import FilterEQ, RLTuple

def test_dataset_initialization(test_storage):
    """Test basic initialization of TupliDataset."""
    dataset = TupliDataset(test_storage)
    assert dataset.storage == test_storage
    assert len(dataset.benchmarks) == 0
    assert len(dataset.episodes) == 0
    assert len(dataset.tuples) == 0

def test_preview_episodes(benchmark_with_episodes, test_storage):
    """Test preview method to list episodes."""
    dataset = TupliDataset(test_storage)
    episodes = dataset.preview()
    assert len(episodes) == 3  # We created 3 episodes
    # Episodes should be headers only, not contain tuples
    assert not hasattr(episodes[0], 'tuples')

def test_load_episodes(benchmark_with_episodes, test_storage):
    """Test loading episodes with tuples."""
    dataset = TupliDataset(test_storage)
    dataset.load()
    assert len(dataset.tuples) == 6  # 3 episodes * 2 steps each
    # Verify tuple structure
    assert isinstance(dataset.tuples[0], RLTuple)

def test_benchmark_filter(benchmark_with_episodes, test_storage):
    """Test filtering by benchmark."""
    dataset = TupliDataset(test_storage)
    benchmark_filter = FilterEQ(key="id", value=benchmark_with_episodes.id)
    filtered_dataset = dataset.with_benchmark_filter(benchmark_filter)

    # Original dataset should be unchanged
    assert dataset._benchmark_filter is None

    # Check filtered dataset
    filtered_dataset.load()
    assert len(filtered_dataset.benchmarks) == 1
    assert filtered_dataset.benchmarks[0].id == benchmark_with_episodes.id

def test_episode_filter(benchmark_with_episodes, test_storage):
    """Test filtering by episode."""
    dataset = TupliDataset(test_storage)
    # Get first episode ID
    episodes = dataset.preview()
    first_episode_id = episodes[0].id

    episode_filter = FilterEQ(key="id", value=first_episode_id)
    filtered_dataset = dataset.with_episode_filter(episode_filter)

    # Original dataset should be unchanged
    assert dataset._episode_filter is None

    # Check filtered dataset
    filtered_dataset.load()
    assert len(filtered_dataset.episodes) == 1
    assert filtered_dataset.episodes[0].id == first_episode_id

def test_tuple_filter(benchmark_with_episodes, test_storage):
    """Test filtering tuples using a custom function."""
    dataset = TupliDataset(test_storage)
    dataset.load()
    initial_tuple_count = len(dataset.tuples)

    # Filter for tuples with action == 1
    filtered_dataset = dataset.with_tuple_filter(lambda t: t.action == 1)
    filtered_dataset.load()

    assert len(filtered_dataset.tuples) < initial_tuple_count
    assert all(t.action == 1 for t in filtered_dataset.tuples)

def test_batch_generator(benchmark_with_episodes, test_storage):
    """Test batch generator functionality."""
    dataset = TupliDataset(test_storage)
    dataset.load()

    batch_size = 2
    generator = dataset.as_batch_generator(batch_size=batch_size)

    batches = list(generator)
    assert len(batches) == 3  # 6 tuples total / 2 batch size = 3 batches
    assert all(len(batch) <= batch_size for batch in batches)
    assert all(isinstance(tuple_, RLTuple) for batch in batches for tuple_ in batch)

def test_batch_generator_with_shuffle(benchmark_with_episodes, test_storage):
    """Test batch generator with shuffle enabled."""
    dataset = TupliDataset(test_storage)
    dataset.load()

    # Set seed for reproducibility
    dataset.set_seed(42)
    gen1 = dataset.as_batch_generator(batch_size=2, shuffle=True)
    batches1 = list(gen1)

    # Reset seed and generate again
    dataset.set_seed(42)
    gen2 = dataset.as_batch_generator(batch_size=2, shuffle=True)
    batches2 = list(gen2)

    # With same seed, shuffled batches should be identical
    assert len(batches1) == len(batches2)
    for b1, b2 in zip(batches1, batches2):
        assert b1 == b2

def test_sample_episodes(benchmark_with_episodes, test_storage):
    """Test sampling episodes."""
    dataset = TupliDataset(test_storage)

    # Test sampling with specific seed
    dataset.set_seed(42)
    samples1 = dataset.sample_episodes(n_samples=2)

    dataset.set_seed(42)
    samples2 = dataset.sample_episodes(n_samples=2)

    assert len(samples1) == 2
    assert len(samples2) == 2
    assert samples1 == samples2  # Should be same with same seed

def test_convert_to_numpy(benchmark_with_episodes, test_storage):
    """Test conversion of tuples to numpy arrays."""
    dataset = TupliDataset(test_storage)
    dataset.load()

    observations, actions, rewards, terminals, timeouts = dataset.convert_to_numpy()

    assert isinstance(observations, np.ndarray)
    assert isinstance(actions, np.ndarray)
    assert isinstance(rewards, np.ndarray)
    assert isinstance(terminals, np.ndarray)
    assert isinstance(timeouts, np.ndarray)

    # Check shapes (6 tuples total from 3 episodes with 2 steps each)
    assert observations.shape[0] == 6
    assert actions.shape[0] == 6
    assert rewards.shape[0] == 6
    assert terminals.shape[0] == 6
    assert timeouts.shape[0] == 6

def test_chained_filters(benchmark_with_episodes, test_storage):
    """Test chaining multiple filters together."""
    dataset = TupliDataset(test_storage)

    # Load all episodes first to find one with action=1
    dataset.load()
    episode_with_action_one = None
    for episode in dataset.episodes:
        # Find tuples belonging to this episode by checking if they're in the episode's tuples list
        episode_tuples = [t for t in episode.tuples]
        # Check if this episode has any tuples with action=1
        has_action_one = any(t.action == 1 for t in episode_tuples)
        if has_action_one:
            episode_with_action_one = episode
            break

    assert episode_with_action_one is not None, "No episode found with action=1"

    # Apply multiple filters
    filtered_dataset = (
        dataset.with_benchmark_filter(FilterEQ(key="id", value=benchmark_with_episodes.id))
        .with_episode_filter(FilterEQ(key="id", value=episode_with_action_one.id))
        .with_tuple_filter(lambda t: t.action == 1)
    )
    filtered_dataset_eps = (
        dataset.with_benchmark_filter(FilterEQ(key="id", value=benchmark_with_episodes.id))
        .with_episode_filter(FilterEQ(key="id", value=episode_with_action_one.id))
    )

    # Load and verify
    filtered_dataset.load()
    filtered_dataset_eps.load()
    assert len(filtered_dataset.tuples) > 0
    assert len(filtered_dataset_eps.episodes) == 1
    assert all(t.action == 1 for t in filtered_dataset.tuples)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
