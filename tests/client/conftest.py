import keyring
import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import io
import json
import hashlib
import uuid
import shutil
from pytupli.benchmark import TupliEnvWrapper
from typing import Any

from pytupli.storage import TupliAPIClient, FileStorage, TupliStorageError, TupliStorage
from pytupli.schema import (
    ArtifactMetadata,
    BenchmarkMetadata,
    BenchmarkQuery,
    Episode,
    RLTuple,
)
from tests.client.example_envs import (
    SimpleTestEnv,
    ContinuousTestEnv,
    TestEnvArtifact,
    CustomTupliEnvWrapper,
    CustomMetadataCallback,
    SimpleTestEnv,
    ContinuousTestEnv,
    TestEnvArtifact,
    CustomTupliEnvWrapper,
    CustomMetadataCallback,
)


@pytest.fixture(scope='session')  # fixture to configure asyncio in pytest
def anyio_backend():
    return 'asyncio'


# Test parameters to run tests with different storage types
STORAGE_TYPES = [
    pytest.param(
        'api',
        id='api_storage',
        marks=pytest.mark.skipif(
            os.environ.get('SKIP_API_TESTS') == 'true', reason='API tests are disabled'
        ),
    ),
    pytest.param('file', id='file_storage'),
]

# Constants for API tests
API_BASE_URL = 'http://localhost:8080'  # Local server URL
API_USERNAME = 'test_user'  # Test user for API tests
API_PASSWORD = 'test_password'  # Test password for API tests
ADMIN_USERNAME = 'admin'  # Admin username for cleanups
ADMIN_PASSWORD = 'pytupli'  # Admin password for cleanups


@pytest.fixture(scope='session')
def temp_dir():
    """Provide a temporary directory for file storage tests."""
    # Create a specific directory that won't be automatically deleted
    tmpdir = Path(tempfile.gettempdir()) / f'pytupli_tests_{uuid.uuid4().hex}'
    tmpdir.mkdir(exist_ok=True, parents=True)
    yield str(tmpdir)
    # Clean up after all tests complete
    try:
        shutil.rmtree(tmpdir)
    except Exception as e:
        print(f'Warning: Failed to clean up temporary directory: {e}')

@pytest.fixture(scope='function')
def clean_keyring():
    """Remove any existing tokens from keyring before tests."""
    try:
        keyring.delete_password('pytupli', 'access_token')
    except:
        pass
    try:
        keyring.delete_password('pytupli', 'refresh_token')
    except:
        pass

def admin_cleanup(storage, resource_type, resource_id):
    """
    Helper function to perform admin cleanup of resources.

    Args:
        storage: The storage instance being used in the test
        resource_type: Type of resource to clean up ('benchmark', 'artifact', 'episode', 'user')
        resource_id: ID of the resource to delete
    """
    # For API storage, we might need admin privileges to delete published resources
    if not isinstance(storage, FileStorage):
        admin_client = TupliAPIClient()
        admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

        try:
            # Use admin client for deletion
            if resource_type == 'benchmark':
                admin_client.delete_benchmark(resource_id)
            elif resource_type == 'artifact':
                admin_client.delete_artifact(resource_id)
            elif resource_type == 'episode':
                admin_client.delete_episode(resource_id)
            elif resource_type == 'user':
                admin_client.delete_user(resource_id)
            elif resource_type == 'group':
                admin_client.delete_group(resource_id)
            elif resource_type == 'role':
                admin_client.delete_role(resource_id)
        except TupliStorageError as e:
            print(f'Admin cleanup failed for {resource_type} {resource_id}: {str(e)}')
    else:
        # For file storage, we can use the regular storage instance
        try:
            if resource_type == 'benchmark':
                storage.delete_benchmark(resource_id)
            elif resource_type == 'artifact':
                storage.delete_artifact(resource_id)
            elif resource_type == 'episode':
                storage.delete_episode(resource_id)
            # No user concept in file storage
        except TupliStorageError as e:
            print(f'Admin cleanup failed for {resource_type} {resource_id}: {str(e)}')


@pytest.fixture(scope='function')
def api_user(clean_keyring):
    """Create a test user for an API test."""

    # Create a real API client connected to local server
    client = TupliAPIClient()

    unique_username = f'{API_USERNAME}_{uuid.uuid4().hex[:8]}'

    # Try to create a new user for this test run
    client.signup(unique_username, API_PASSWORD)

    # Login with the new user
    client.login(unique_username, API_PASSWORD)

    # Store username and creation status for cleanup
    user_info = {'username': unique_username, 'client': client}

    yield user_info, client

    # Cleanup: Delete the user using admin privileges
    admin_cleanup(None, 'user', unique_username)


@pytest.fixture(scope='function')
def storage(request, temp_dir, api_user):
    """Create a storage instance based on the test parameter."""
    storage_type = request.param

    if storage_type == 'api':
        if os.environ.get('SKIP_API_TESTS') == 'true':
            pytest.skip('API tests are disabled')

        # Extract the client from api_user tuple
        _, client = api_user

        # Return the client as TupliStorage instance
        yield client

    else:
        # Create file storage with temporary directory
        storage = FileStorage(storage_base_dir=temp_dir)
        yield storage


@pytest.fixture(scope='function')
def sample_benchmark_query():
    """Create a sample benchmark query for testing."""
    return BenchmarkQuery(
        hash='test_hash',
        metadata=BenchmarkMetadata(
            name='Test Benchmark',
            description='A benchmark for testing',
            difficulty='easy',
            version='1.0',
        ),
        serialized=json.dumps({'test': 'serialized benchmark data'}),
    )


@pytest.fixture(scope='function')
def sample_artifact_data():
    """Create sample artifact data for testing."""
    # Create a simple CSV as bytes
    csv_data = 't,value\n0,10\n1,20\n2,30\n'
    artifact_bytes = csv_data.encode('utf-8')

    # Create metadata for the artifact
    metadata = ArtifactMetadata(
        name='test_artifact.csv',
        description='Test artifact for testing',
    )

    # Create a dataframe for verification
    df = pd.read_csv(io.StringIO(csv_data), dtype={'t': str})
    df.set_index('t', inplace=True, verify_integrity=True)

    # Generate hash for verification
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()

    return artifact_bytes, metadata, df, artifact_hash


@pytest.fixture(scope='function')
def sample_benchmark_stored(request, storage, sample_benchmark_query):
    """Store a benchmark and return its ID for further tests."""
    # Store the benchmark using the real storage
    benchmark_header = storage.store_benchmark(sample_benchmark_query)

    # Return the stored ID for test
    yield benchmark_header.id

    # Cleanup after test finishes
    admin_cleanup(storage, 'benchmark', benchmark_header.id)


@pytest.fixture(scope='function')
def sample_artifact_stored(request, storage, sample_artifact_data):
    """Store an artifact and return its ID for further tests."""
    artifact_bytes, metadata, _, _ = sample_artifact_data

    # Store the artifact using the real storage
    artifact_item = storage.store_artifact(artifact_bytes, metadata)

    # Return the stored ID for test
    yield artifact_item.id

    # Cleanup after test finishes
    admin_cleanup(storage, 'artifact', artifact_item.id)


@pytest.fixture(scope='function')
def sample_episode(sample_benchmark_stored):
    """Create a sample episode for testing."""
    benchmark_id = sample_benchmark_stored

    return Episode(
        benchmark_id=benchmark_id,
        metadata={'agent': 'test_agent', 'version': '1.0'},
        tuples=[
            RLTuple(
                state={'position': 0, 'velocity': 0},
                action=1,
                reward=0.0,
                info={},
                terminal=False,
                timeout=False,
            ),
            RLTuple(
                state={'position': 0.1, 'velocity': 0.1},
                action=1,
                reward=0.1,
                info={},
                terminal=True,
                timeout=False,
            ),
        ],
    )

@pytest.fixture(scope='function')
def sample_episode_2(sample_benchmark_stored):
    """Create a sample episode for testing."""
    benchmark_id = sample_benchmark_stored

    return Episode(
        benchmark_id=benchmark_id,
        metadata={'agent': 'test_agent', 'version': '1.0'},
        tuples=[
            RLTuple(
                state={'position': 0.5, 'velocity': 0.5},
                action=1,
                reward=1.0,
                info={},
                terminal=False,
                timeout=False,
            ),
            RLTuple(
                state={'position': 0.1, 'velocity': 0.1},
                action=1,
                reward=0.1,
                info={},
                terminal=True,
                timeout=False,
            ),
        ],
    )

@pytest.fixture(scope='function')
def sample_episode_stored(request, storage, sample_episode):
    """Store an episode and return its ID for further tests."""
    # Store the episode using the real storage
    episode_header = storage.record_episode(sample_episode)

    # Return the stored ID for test
    yield episode_header.id

    # Cleanup after test finishes
    admin_cleanup(storage, 'episode', episode_header.id)

@pytest.fixture(scope='function')
def sample_episode_2_stored(request, storage, sample_episode_2):
    """Store an episode and return its ID for further tests."""
    # Store the episode using the real storage
    episode_header = storage.record_episode(sample_episode_2)

    # Return the stored ID for test
    yield episode_header.id

    # Cleanup after test finishes
    admin_cleanup(storage, 'episode', episode_header.id)


@pytest.fixture(scope='function')
def test_env():
    """Create a test environment."""
    return SimpleTestEnv()

@pytest.fixture(scope='function')
def test_env_artifact():
    """Create a test environment with artifact."""
    return TestEnvArtifact()

@pytest.fixture(scope='function')
def test_storage(temp_dir):
    """Create a file storage instance for testing."""
    return FileStorage(storage_base_dir=temp_dir)

@pytest.fixture(scope='function')
def test_benchmark(test_env, test_storage) -> TupliEnvWrapper:
    """Create a TupliEnvWrapper instance for testing."""
    return TupliEnvWrapper(test_env, test_storage)

@pytest.fixture(scope='function')
def test_benchmark_artifact(test_env_artifact, test_storage) -> TupliEnvWrapper:
    """Create a TupliEnvWrapper instance with artifact for testing."""
    return CustomTupliEnvWrapper(test_env_artifact, test_storage)

@pytest.fixture(scope='function')
def test_benchmark_with_metadata(test_env, test_storage) -> TupliEnvWrapper:
    """Create a TupliEnvWrapper instance with metadata callback for testing."""
    metadata_callback = CustomMetadataCallback()
    return TupliEnvWrapper(test_env, test_storage, metadata_callback=metadata_callback)

@pytest.fixture(params=[
    SimpleTestEnv,
    ContinuousTestEnv
])
def parameterized_env(request):
    """Fixture that provides different types of environments."""
    return request.param()

@pytest.fixture
def benchmark_with_episodes(test_storage, test_env):
    """Create a benchmark and record some episodes."""
    benchmark = TupliEnvWrapper(test_env, test_storage)
    benchmark.store(
        name="Test Dataset Benchmark",
        description="A benchmark for dataset tests",
        difficulty="easy",
        version="1.0"
    )

    # Record three episodes with different actions
    for episode in range(3):
        obs, _ = benchmark.reset()
        # First episode all ones, second all zeros, third mixed
        actions = [1, 1] if episode == 0 else [0, 0] if episode == 1 else [1, 0]
        for action in actions:
            obs, reward, term, trunc, _ = benchmark.step(np.int64(action))
            if term or trunc:
                break

    yield benchmark

    # Cleanup after tests
    benchmark.delete(delete_episodes=True)
