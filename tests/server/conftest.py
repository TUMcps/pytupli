import httpx
import pytest
from pytupli.schema import (
    Benchmark,
    BenchmarkMetadata,
    BenchmarkQuery,
    ArtifactMetadata,
    ArtifactMetadataItem,
    Group,
    GroupMembership,
    GroupMembershipQuery,
    RLTuple,
    Episode,
    BenchmarkHeader,
    EpisodeHeader,
    EpisodeItem,
)
import json
from pathlib import Path

# officially recommended by FastAPI documentation: https://fastapi.tiangolo.com/advanced/async-tests/#in-detail
import pandas as pd
import io
import hashlib
import datetime


@pytest.fixture(scope='session')  # fixture to configure asyncio in pytest
def anyio_backend():
    return 'asyncio'


@pytest.fixture(scope='function')
async def async_client():
    async with httpx.AsyncClient(
        base_url='http://localhost:8080', timeout=httpx.Timeout(60)
    ) as client:
        yield client


@pytest.fixture(scope='function')
async def admin_headers(async_client):
    return {'Authorization': f'Bearer {await get_JWT_token(async_client)}'}


@pytest.fixture(scope='function')
async def standard_user1_headers(async_client, admin_headers):
    resp = await async_client.post(
        '/access/users/create',
        json={'username': 'test_user_1', 'password': 'test1234'},
        headers=admin_headers,
    )
    assert resp.status_code == 200

    token = await get_JWT_token(async_client, user='test_user_1', password='test1234')
    yield {'Authorization': f'Bearer {token}'}

    # cleanup
    resp = await async_client.delete(
        '/access/users/delete?username=test_user_1', headers=admin_headers
    )
    assert resp.status_code == 200


@pytest.fixture(scope='function')
async def standard_user2_headers(async_client, admin_headers):
    resp = await async_client.post(
        '/access/users/create',
        json={'username': 'test_user_2', 'password': 'test1234'},
        headers=admin_headers,
    )
    assert resp.status_code == 200

    token = await get_JWT_token(async_client, user='test_user_2', password='test1234')
    yield {'Authorization': f'Bearer {token}'}

    # cleanup
    resp = await async_client.delete(
        '/access/users/delete?username=test_user_2', headers=admin_headers
    )
    assert resp.status_code == 200


async def get_JWT_token(async_client, user='admin', password='pytupli'):
    r = await async_client.post('/access/users/token', json={'username': user, 'password': password})
    r.raise_for_status()
    return r.json()['access_token']['token']


# DATATYPE FIXTURES


def process_str_artifact_to_df(content):
    df = pd.read_csv(io.StringIO(content), dtype={'t': str})
    df.set_index('t', inplace=True, verify_integrity=True)
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return df, df_hash


@pytest.fixture(scope='function')
def sample_artifact():
    data_path = Path(__file__).parent.parent / 'data' / 'test_data.csv'
    file = open(data_path, 'r', encoding='utf-8')
    content = file.read()
    df, df_hash = process_str_artifact_to_df(content)

    metadata = ArtifactMetadata(
        name='test_data.csv',
        description='test data source',
    )

    return content, df, df_hash, metadata


@pytest.fixture(scope='function')
def sample_benchmark():
    json_file_path = Path(__file__).parent.parent / 'benchmarks' / 'test.json'
    with open(json_file_path, 'r') as json_file:
        json_object = json.load(json_file)

    return BenchmarkQuery(
        hash='test_hash',
        metadata=BenchmarkMetadata(
            name='test',
            description='test description',
        ),
        serialized=json.dumps(json_object, indent=4),
    )


@pytest.fixture(scope='function')
def sample_episode(published_benchmark_admin):
    _, benchmark = published_benchmark_admin

    return Episode(
        benchmark_id=benchmark.id,
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
def sample_episode_timeout(published_benchmark_admin):
    _, benchmark = published_benchmark_admin

    return Episode(
        benchmark_id=benchmark.id,
        metadata={'agent': 'timeout_agent', 'version': '1.0'},
        tuples=[
            RLTuple(
                state={'position': 0, 'velocity': 0},
                action=0,
                reward=0.0,
                info={},
                terminal=False,
                timeout=False,
            ),
            RLTuple(
                state={'position': 0.0, 'velocity': 0.0},
                action=0,
                reward=0.0,
                info={},
                terminal=False,
                timeout=True,
            ),
        ],
    )


@pytest.fixture(scope='function')
def sample_benchmarks():
    json_file_path = Path.cwd() / 'tests' / 'benchmarks' / 'BenchmarkTest1.json'
    with open(json_file_path, 'r') as json_file:
        json_object_1 = json.load(json_file)

    json_file_path = Path.cwd() / 'tests' / 'benchmarks' / 'BenchmarkTest2.json'
    with open(json_file_path, 'r') as json_file:
        json_object_2 = json.load(json_file)

    json_file_path = Path.cwd() / 'tests' / 'benchmarks' / 'BenchmarkTest3.json'
    with open(json_file_path, 'r') as json_file:
        json_object_3 = json.load(json_file)

    return [
        Benchmark(
            id='id1',
            created_by='test_user',
            published_in=['global'],
            created_at=datetime.datetime.now(datetime.timezone.utc),
            hash='hash1',
            metadata=BenchmarkMetadata(
                name='Benchmark 1',
                description='high reward benchmark',
                difficulty='easy',
                version='1.0',
            ),
            serialized=json.dumps(json_object_1, indent=4),
        ),
        Benchmark(
            id='id2',
            created_by='test_user',
            published_in=[],
            created_at=datetime.datetime.now(datetime.timezone.utc),
            hash='hash2',
            metadata=BenchmarkMetadata(
                name='Benchmark 2',
                description='low reward benchmark',
                difficulty='hard',
                version='2.0',
            ),
            serialized=json.dumps(json_object_2, indent=4),
        ),
        Benchmark(
            id='id3',
            created_by='test_user',
            published_in=['global'],
            created_at=datetime.datetime.now(datetime.timezone.utc),
            hash='hash3',
            metadata=BenchmarkMetadata(
                name='Benchmark 3',
                description='medium reward benchmark',
                difficulty='hard',
                version='1.0',
            ),
            serialized=json.dumps(json_object_3, indent=4),
        ),
    ]


# Benchmark helper functions
async def create_benchmark(async_client, benchmark_data, headers):
    """Create a benchmark and return the response and created benchmark header"""
    response = await async_client.post(
        '/benchmarks/create', json=benchmark_data.model_dump(), headers=headers
    )
    if response.status_code == 200:
        return response, BenchmarkHeader(**response.json())
    return response, None


async def publish_benchmark(async_client, benchmark_id, headers, publish_in='global'):
    """Publish a benchmark and return the response"""
    return await async_client.put(
        f'/benchmarks/publish?benchmark_id={benchmark_id}&publish_in={publish_in}', headers=headers
    )


async def unpublish_benchmark(async_client, benchmark_id, headers, unpublish_from='global'):
    """Unpublish a benchmark and return the response"""
    return await async_client.put(
        f'/benchmarks/unpublish?benchmark_id={benchmark_id}&unpublish_from={unpublish_from}', headers=headers
    )


async def load_benchmark(async_client, benchmark_id, headers):
    """Load a benchmark and return the response and loaded benchmark if successful"""
    response = await async_client.get(
        f'/benchmarks/load?benchmark_id={benchmark_id}', headers=headers
    )
    if response.status_code == 200:
        return response, Benchmark(**response.json())
    return response, None


async def delete_benchmark(async_client, benchmark_id, headers):
    """Delete a benchmark and return the response"""
    return await async_client.delete(
        f'/benchmarks/delete?benchmark_id={benchmark_id}', headers=headers
    )


# Group helper functions
async def create_group(async_client, group_name, headers, description="Test group"):
    """Create a group and return the response"""
    group = Group(name=group_name, description=description)
    return await async_client.post(
        '/access/groups/create', json=group.model_dump(), headers=headers
    )


async def delete_group(async_client, headers, group_name):
    """Delete a group and return the response"""
    return await async_client.delete(
        f'/access/groups/delete?group_name={group_name}', headers=headers
    )


async def add_user_to_group(async_client, group_name, username, headers, roles=None):
    """Add a user to a group with specific roles"""
    if roles is None:
        roles = ['contributor']  # Default role

    membership = GroupMembershipQuery(
        group_name=group_name,
        members=[GroupMembership(
            user=username,
            roles=roles
        )]
    )
    return await async_client.post(
        '/access/groups/add-members', json=membership.model_dump(), headers=headers
    )


# Episode helper functions
async def record_episode(async_client, episode_data, headers):
    """Record an episode and return the response and episode header if successful"""
    response = await async_client.post(
        '/episodes/record', json=episode_data.model_dump(), headers=headers
    )
    if response.status_code == 200:
        return response, EpisodeHeader(**response.json())
    return response, None


async def publish_episode(async_client, episode_id, headers=None, publish_in='global'):
    """Publish an episode by ID or all episodes for a benchmark ID"""
    url = '/episodes/publish'
    url += f'?episode_id={episode_id}&publish_in={publish_in}'
    return await async_client.put(url, headers=headers)


async def unpublish_episode(async_client, episode_id, headers=None, unpublish_from='global'):
    """Unpublish an episode by ID"""
    url = '/episodes/unpublish'
    url += f'?episode_id={episode_id}&unpublish_from={unpublish_from}'
    return await async_client.put(url, headers=headers)


async def delete_episode(async_client, episode_id, headers=None):
    """Delete an episode by ID or all episodes for a benchmark ID"""
    url = '/episodes/delete'
    url += f'?episode_id={episode_id}'
    return await async_client.delete(url, headers=headers)


async def list_episodes(async_client, filter_data=None, include_tuples=False, headers=None):
    """List episodes with optional filtering"""
    url = '/episodes/list'

    json_data = {
        **(filter_data.model_dump() if filter_data else {}),
        'include_tuples': include_tuples,
    }
    response = await async_client.post(url, headers=headers, json=json_data)

    if response.status_code == 200:
        if include_tuples:
            return response, [EpisodeItem(**episode) for episode in response.json()]
        else:
            return response, [EpisodeHeader(**episode) for episode in response.json()]
    return response, None


@pytest.fixture(scope='function')
async def created_benchmark_admin(async_client, sample_benchmark, admin_headers):
    """Create a benchmark as admin and clean up after test"""
    response, benchmark = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 200
    yield response, benchmark
    # Cleanup
    await delete_benchmark(async_client, benchmark.id, admin_headers)


@pytest.fixture(scope='function')
async def created_benchmark_user1(
    async_client, sample_benchmark, standard_user1_headers, admin_headers
):
    """Create a benchmark as standard_user1 and clean up after test"""

    sample_benchmark.hash = 'test_hash_created_user1'

    response, benchmark = await create_benchmark(
        async_client, sample_benchmark, standard_user1_headers
    )
    assert response.status_code == 200
    yield response, benchmark
    # Cleanup (user might be deleted, so use admin for cleanup)
    await delete_benchmark(async_client, benchmark.id, admin_headers)


@pytest.fixture(scope='function')
async def published_benchmark_admin(async_client, sample_benchmark, admin_headers):
    """Create and publish a benchmark as admin and clean up after test"""

    sample_benchmark.hash = 'test_hash_published_admin'

    response, benchmark = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 200

    pub_response = await publish_benchmark(async_client, benchmark.id, admin_headers)
    assert pub_response.status_code == 200

    yield response, benchmark
    # Cleanup
    await delete_benchmark(async_client, benchmark.id, admin_headers)


@pytest.fixture(scope='function')
async def published_benchmark_user1(
    async_client, sample_benchmark, standard_user1_headers, admin_headers
):
    """Create and publish a benchmark as standard_user1 and clean up after test"""

    sample_benchmark.hash = 'test_hash_published_user1'

    response, benchmark = await create_benchmark(
        async_client, sample_benchmark, standard_user1_headers
    )
    assert response.status_code == 200

    pub_response = await publish_benchmark(async_client, benchmark.id, standard_user1_headers)
    assert pub_response.status_code == 200

    yield response, benchmark
    # Cleanup - Use admin for cleanup since published benchmarks can't be deleted by standard users
    await delete_benchmark(async_client, benchmark.id, admin_headers)


@pytest.fixture(scope='function')
async def recorded_episode_admin(
    async_client, sample_episode, admin_headers, published_benchmark_admin
):
    """Record an episode as admin and yield the response and episode"""
    response, episode = await record_episode(async_client, sample_episode, admin_headers)
    assert response.status_code == 200
    yield response, episode
    # Cleanup
    delete_resp = await delete_episode(async_client, episode_id=episode.id, headers=admin_headers)
    assert delete_resp.status_code in [200]


@pytest.fixture(scope='function')
async def recorded_episode_user1(
    async_client, sample_episode, standard_user1_headers, published_benchmark_admin
):
    """Record an episode as standard_user1 and yield the response and episode"""
    response, episode = await record_episode(async_client, sample_episode, standard_user1_headers)
    assert response.status_code == 200
    yield response, episode
    # Cleanup
    delete_resp = await delete_episode(
        async_client, episode_id=episode.id, headers=standard_user1_headers
    )
    assert delete_resp.status_code in [200]


@pytest.fixture(scope='function')
async def published_episode_admin(
    async_client, sample_episode, admin_headers, published_benchmark_admin
):
    """Record and publish an episode as admin"""
    response, episode = await record_episode(async_client, sample_episode, admin_headers)
    assert response.status_code == 200

    pub_response = await publish_episode(async_client, episode_id=episode.id, headers=admin_headers)
    assert pub_response.status_code == 200

    yield response, episode
    # Cleanup
    delete_resp = await delete_episode(async_client, episode_id=episode.id, headers=admin_headers)
    assert delete_resp.status_code in [200]


# Data helper functions
async def upload_artifact(async_client, metadata, data_content, filename='test.csv', headers=None):
    """Upload a artifact and return the response and id if successful"""
    response = await async_client.post(
        '/artifacts/upload',
        files={'data': data_content},
        data={'metadata': metadata.model_dump_json(serialize_as_any=True)},
        headers=headers,
    )
    if response.status_code == 200:
        uploaded_id = response.json()['id']
        return response, uploaded_id
    return response, None


async def download_artifact(async_client, artifact_id, headers=None):
    """Download a artifact and return the response and dataframe if successful"""
    response = await async_client.get(
        f'/artifacts/download?artifact_id={artifact_id}', headers=headers
    )
    if response.status_code == 200:
        # Handle binary response
        content = response.content.decode('utf-8')
        loaded_df, loaded_df_hash = process_str_artifact_to_df(content)

        # Extract metadata from headers
        metadata = None
        if 'X-Metadata' in response.headers:
            metadata = ArtifactMetadataItem.model_validate_json(response.headers['X-Metadata'])

        return response, loaded_df, loaded_df_hash, metadata
    return response, None, None, None


async def publish_artifact(async_client, artifact_id, headers, publish_in='global'):
    """Publish a artifact and return the response"""
    return await async_client.put(f'/artifacts/publish?artifact_id={artifact_id}&publish_in={publish_in}', headers=headers)


async def unpublish_artifact(async_client, artifact_id, headers, unpublish_from='global'):
    """Unpublish an artifact and return the response"""
    return await async_client.put(f'/artifacts/unpublish?artifact_id={artifact_id}&unpublish_from={unpublish_from}', headers=headers)


async def delete_artifact(async_client, artifact_id, headers):
    """Delete a artifact and return the response"""
    return await async_client.delete(
        f'/artifacts/delete?artifact_id={artifact_id}', headers=headers
    )


@pytest.fixture(scope='function')
async def uploaded_artifact_admin(async_client, sample_artifact, admin_headers):
    """Upload a artifact as admin and clean up after test"""
    data, df, df_hash, metadata = sample_artifact
    response, artifact_id = await upload_artifact(
        async_client, metadata, data, headers=admin_headers
    )
    assert response.status_code == 200
    yield response, artifact_id, df, df_hash, metadata
    # Cleanup
    await delete_artifact(async_client, artifact_id, admin_headers)


@pytest.fixture(scope='function')
async def uploaded_artifact_user1(async_client, sample_artifact, standard_user1_headers):
    """Upload a artifact as standard_user1 and clean up after test"""
    data, df, df_hash, metadata = sample_artifact
    response, artifact_id = await upload_artifact(
        async_client, metadata, data, headers=standard_user1_headers
    )
    assert response.status_code == 200
    yield response, artifact_id, df, df_hash
    # Cleanup
    await delete_artifact(async_client, artifact_id, standard_user1_headers)


@pytest.fixture(scope='function')
async def published_artifact_admin(async_client, sample_artifact, admin_headers):
    """Upload and publish a artifact as admin and clean up after test"""
    data, df, _, metadata = sample_artifact
    response, artifact_id = await upload_artifact(
        async_client, metadata, data, headers=admin_headers
    )
    assert response.status_code == 200

    pub_response = await publish_artifact(async_client, artifact_id, admin_headers)
    assert pub_response.status_code == 200

    yield response, artifact_id, df
    # Cleanup
    await delete_artifact(async_client, artifact_id, admin_headers)
