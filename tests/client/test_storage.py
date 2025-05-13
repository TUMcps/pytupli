import pytest
import os
import hashlib

from pytupli.storage import FileStorage, TupliStorageError
from pytupli.schema import (
    ArtifactMetadataItem,
    Benchmark,
    BenchmarkHeader,
    EpisodeHeader,
    EpisodeItem,
    FilterEQ,
)
from conftest import STORAGE_TYPES, admin_cleanup


# Benchmark Tests
@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_store_benchmark(storage, sample_benchmark_query):
    """Test storing a benchmark."""
    # Execute store_benchmark
    benchmark_header = storage.store_benchmark(sample_benchmark_query)

    # Verify results
    assert benchmark_header is not None
    assert isinstance(benchmark_header, BenchmarkHeader)
    assert benchmark_header.hash == 'test_hash'

    # Clean up
    admin_cleanup(storage, 'benchmark', benchmark_header.id)


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_load_benchmark(storage, sample_benchmark_stored, sample_benchmark_query):
    """Test loading a benchmark."""
    benchmark_id = sample_benchmark_stored

    # Execute load_benchmark
    loaded_benchmark = storage.load_benchmark(benchmark_id)

    # Verify results
    assert loaded_benchmark is not None
    assert isinstance(loaded_benchmark, Benchmark)
    assert loaded_benchmark.hash == sample_benchmark_query.hash
    assert loaded_benchmark.metadata.name == sample_benchmark_query.metadata.name
    assert loaded_benchmark.serialized == sample_benchmark_query.serialized


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_benchmarks(storage, sample_benchmark_stored):
    """Test listing benchmarks."""

    # Execute list_benchmarks
    benchmark_list = storage.list_benchmarks()

    # Verify results
    assert benchmark_list is not None
    assert isinstance(benchmark_list, list)
    assert len(benchmark_list) == 1, 'Expected exactly one benchmark'
    assert isinstance(benchmark_list[0], BenchmarkHeader)
    assert benchmark_list[0].hash == 'test_hash', "Stored benchmark hash doesn't match"


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_benchmarks_with_filter(storage, sample_benchmark_stored):
    """Test listing benchmarks with a filter."""

    # Create a filter
    benchmark_filter = FilterEQ(key='hash', value='test_hash')

    # Execute list_benchmarks with filter
    benchmark_list = storage.list_benchmarks(filter=benchmark_filter)

    # Verify results
    assert benchmark_list is not None
    assert isinstance(benchmark_list, list)
    assert len(benchmark_list) == 1, 'Expected exactly one benchmark matching filter'
    assert isinstance(benchmark_list[0], BenchmarkHeader)
    assert benchmark_list[0].hash == 'test_hash'


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_delete_benchmark(storage, sample_benchmark_stored):
    """Test deleting a benchmark."""
    benchmark_id = sample_benchmark_stored

    # Execute delete_benchmark
    storage.delete_benchmark(benchmark_id)

    # Verify the benchmark was deleted by attempting to load it
    with pytest.raises(TupliStorageError):
        storage.load_benchmark(benchmark_id)


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_publish_benchmark(storage, sample_benchmark_stored):
    """Test publishing a benchmark."""
    benchmark_id = sample_benchmark_stored

    # Execute publish_benchmark
    storage.publish_benchmark(benchmark_id)

    # For API storage, we'd ideally verify that the benchmark is now public
    if not isinstance(storage, FileStorage):
        # Load the benchmark and check its public status
        loaded_benchmark = storage.load_benchmark(benchmark_id)
        assert loaded_benchmark.is_public, 'Benchmark was not published successfully'


# Artifact Tests
@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_store_artifact(storage, sample_artifact_data):
    """Test storing an artifact."""
    artifact_bytes, metadata, _, _ = sample_artifact_data

    # Execute store_artifact
    artifact_item = storage.store_artifact(artifact_bytes, metadata)

    # Verify results
    assert artifact_item is not None
    assert isinstance(artifact_item, ArtifactMetadataItem)
    assert artifact_item.name == 'test_artifact.csv'

    # Clean up
    admin_cleanup(storage, 'artifact', artifact_item.id)


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_load_artifact(storage, sample_artifact_stored, sample_artifact_data):
    """Test loading an artifact."""
    artifact_id = sample_artifact_stored
    _, _, _, expected_hash = sample_artifact_data

    # Execute load_artifact
    artifact_bytes = storage.load_artifact(artifact_id)

    # Verify results
    assert artifact_bytes is not None
    assert isinstance(artifact_bytes, bytes)

    # Verify hash of loaded artifact matches original for file storage
    loaded_artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if isinstance(storage, FileStorage):
        # For file storage, the hash should match exactly
        assert loaded_artifact_hash == expected_hash


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_artifacts(storage, sample_artifact_stored):
    """Test listing artifacts."""

    # Execute list_artifacts
    artifact_list = storage.list_artifacts()

    # Verify results
    assert artifact_list is not None
    assert isinstance(artifact_list, list)
    assert len(artifact_list) == 1, 'Expected exactly one artifact'
    assert isinstance(artifact_list[0], ArtifactMetadataItem)
    assert artifact_list[0].name == 'test_artifact.csv', "Stored artifact name doesn't match"


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_artifacts_with_filter(storage, sample_artifact_stored):
    """Test listing artifacts with a filter."""

    # Create a filter
    artifact_filter = FilterEQ(key='name', value='test_artifact.csv')

    # Execute list_artifacts with filter
    artifact_list = storage.list_artifacts(filter=artifact_filter)

    # Verify results
    assert artifact_list is not None
    assert isinstance(artifact_list, list)
    assert len(artifact_list) == 1, 'Expected exactly one artifact matching filter'
    assert isinstance(artifact_list[0], ArtifactMetadataItem)
    assert artifact_list[0].name == 'test_artifact.csv'


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_delete_artifact(storage, sample_artifact_stored):
    """Test deleting an artifact."""
    artifact_id = sample_artifact_stored

    # Execute delete_artifact
    storage.delete_artifact(artifact_id)

    # Verify the artifact was deleted by attempting to load it
    with pytest.raises(TupliStorageError):
        storage.load_artifact(artifact_id)


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_publish_artifact(storage, sample_artifact_stored):
    """Test publishing an artifact."""
    artifact_id = sample_artifact_stored

    # Execute publish_artifact
    storage.publish_artifact(artifact_id)

    # Load artifact metadata and verify it's now public
    if not isinstance(storage, FileStorage):
        artifact_list = storage.list_artifacts(filter=FilterEQ(key='id', value=artifact_id))
        assert len(artifact_list) == 1, "Couldn't find the published artifact"
        assert artifact_list[0].is_public, 'Artifact was not published successfully'


# Episode Tests
@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_record_episode(storage, sample_episode):
    """Test recording an episode."""
    # Execute record_episode
    episode_header = storage.record_episode(sample_episode)

    # Verify results
    assert episode_header is not None
    assert isinstance(episode_header, EpisodeHeader)
    assert episode_header.benchmark_id == sample_episode.benchmark_id

    # Clean up
    admin_cleanup(storage, 'episode', episode_header.id)


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_episodes(storage, sample_episode_stored):
    """Test listing episodes."""

    # Execute list_episodes
    episode_list = storage.list_episodes()

    # Verify results
    assert episode_list is not None
    assert isinstance(episode_list, list)
    assert len(episode_list) == 1, 'Expected exactly one episode'
    assert isinstance(episode_list[0], EpisodeHeader)
    assert episode_list[0].metadata.get('agent') == 'test_agent', "Episode metadata doesn't match"


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_episodes_with_tuples(storage, sample_episode_stored):
    """Test listing episodes with tuples included."""

    # Execute list_episodes with include_tuples=True
    episode_list = storage.list_episodes(include_tuples=True)

    # Verify results
    assert episode_list is not None
    assert isinstance(episode_list, list)
    assert len(episode_list) == 1, 'Expected exactly one episode'
    assert isinstance(episode_list[0], EpisodeItem)
    assert hasattr(episode_list[0], 'tuples')
    assert len(episode_list[0].tuples) == 2, 'Expected exactly 2 tuples in episode'
    assert episode_list[0].metadata.get('agent') == 'test_agent'
    assert episode_list[0].tuples[0].action == 1
    assert episode_list[0].tuples[1].terminal


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_list_episodes_with_filter(storage, sample_episode_stored):
    """Test listing episodes with a filter."""

    # Create a filter
    episode_filter = FilterEQ(key='metadata.agent', value='test_agent')

    # Execute list_episodes with filter
    episode_list = storage.list_episodes(filter=episode_filter)

    # Verify results
    assert episode_list is not None
    assert isinstance(episode_list, list)
    assert len(episode_list) == 1, 'Expected exactly one episode matching filter'
    assert isinstance(episode_list[0], EpisodeHeader)
    assert episode_list[0].metadata.get('agent') == 'test_agent'


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_publish_episode(storage, sample_episode_stored):
    """Test publishing an episode."""
    episode_id = sample_episode_stored

    # First publish the benchmark since episodes can only be published if their benchmark is public
    # Get the benchmark ID from the episode list
    episode_list = storage.list_episodes(include_tuples=False)
    benchmark_id = None

    # Find our specific episode to get its benchmark ID
    for episode in episode_list:
        if (
            isinstance(storage, FileStorage)
            and episode.id == episode_id.split('/')[-1].replace('episode_', '').replace('.json', '')
        ) or (not isinstance(storage, FileStorage) and episode.id == episode_id):
            benchmark_id = episode.benchmark_id
            # Publish the benchmark first
            storage.publish_benchmark(benchmark_id)
            break

    assert benchmark_id is not None, "Couldn't find benchmark ID for the episode"

    # Execute publish_episode
    storage.publish_episode(episode_id)

    # For API storage, verify the episode is now public where possible
    if not isinstance(storage, FileStorage):
        episode_filter = FilterEQ(key='id', value=episode_id)
        episode_list = storage.list_episodes(filter=episode_filter)
        if len(episode_list) > 0:
            # Only assert if we can actually get the episode's public status
            if hasattr(episode_list[0], 'is_public'):
                assert episode_list[0].is_public, 'Episode was not published successfully'


@pytest.mark.parametrize('storage', STORAGE_TYPES, indirect=True)
def test_delete_episode(storage, sample_episode_stored):
    """Test deleting an episode."""
    episode_id = sample_episode_stored

    # Execute delete_episode
    storage.delete_episode(episode_id)

    # For file storage, verify the file no longer exists
    if isinstance(storage, FileStorage):
        assert not os.path.exists(episode_id)
    else:
        # For API storage, try to list episodes with our metadata and verify it's gone
        episode_filter = FilterEQ(key='metadata.agent', value='test_agent')
        episode_list = storage.list_episodes(filter=episode_filter)
        assert len(episode_list) == 0, 'Episode was not deleted successfully'


if __name__ == '__main__':
    pytest.main(['-v', __file__])
