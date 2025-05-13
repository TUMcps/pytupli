import pytest
from pytupli.schema import BenchmarkHeader

# The helper functions are moved to conftest.py
from conftest import (
    create_benchmark,
    publish_benchmark,
    load_benchmark,
    delete_benchmark,
)


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_create(async_client, sample_benchmark, admin_headers):
    response, benchmark = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 200
    assert benchmark is not None

    response = await delete_benchmark(async_client, benchmark.id, admin_headers)
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_create_benchmark_exists_publicly(
    async_client, published_benchmark_user1, standard_user2_headers, sample_benchmark
):
    _, _ = published_benchmark_user1

    # Second user tries to create the same benchmark - should fail with 409
    response, _ = await create_benchmark(async_client, sample_benchmark, standard_user2_headers)
    assert response.status_code == 409


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_duplication(
    async_client, created_benchmark_admin, admin_headers, sample_benchmark
):
    _, created_benchmark = created_benchmark_admin

    # Try creating it again - should fail with 409
    response, _ = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 409


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_load(created_benchmark_admin, async_client, admin_headers):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Load the benchmark
    response, loaded_benchmark = await load_benchmark(
        async_client, created_benchmark.id, admin_headers
    )
    assert response.status_code == 200

    # Verify benchmark data
    assert loaded_benchmark.id == created_benchmark.id
    assert loaded_benchmark.hash == created_benchmark.hash
    assert loaded_benchmark.metadata == created_benchmark.metadata
    assert loaded_benchmark.serialized is not None
    assert loaded_benchmark.is_public is False


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_load_private_insufficient_rights(
    created_benchmark_admin, async_client, standard_user1_headers
):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Attempt to load with user (should fail)
    response, _ = await load_benchmark(async_client, created_benchmark.id, standard_user1_headers)
    assert response.status_code == 403


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_publish(created_benchmark_admin, async_client, admin_headers):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Publish benchmark
    response = await publish_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 200

    # Load and verify it's public
    response, loaded_benchmark = await load_benchmark(
        async_client, created_benchmark.id, admin_headers
    )
    assert response.status_code == 200
    assert loaded_benchmark.is_public is True


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_delete(created_benchmark_admin, async_client, admin_headers):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Delete benchmark
    response = await delete_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 200

    # Verify deletion
    response, _ = await load_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 404


@pytest.mark.anyio(loop_scope='session')
async def test_delete_benchmark_guest_forbidden(created_benchmark_admin, async_client):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Attempt to delete with no token => guest user => 401
    delete_response = await async_client.delete(
        f'/benchmarks/delete?benchmark_id={created_benchmark.id}'
    )
    assert delete_response.status_code == 401


@pytest.mark.anyio(loop_scope='session')
async def test_delete_benchmark_other_user_forbidden(
    async_client, created_benchmark_user1, standard_user2_headers
):
    _, created_benchmark = created_benchmark_user1

    # Try to delete with user 2 - should fail
    response = await delete_benchmark(async_client, created_benchmark.id, standard_user2_headers)
    assert response.status_code == 403


@pytest.mark.anyio(loop_scope='session')
async def test_benchmarks_list(
    async_client, created_benchmark_user1, published_benchmark_admin, admin_headers
):
    _, b1 = created_benchmark_user1
    _, b2 = published_benchmark_admin

    # List benchmarks
    response = await async_client.get('/benchmarks/list', headers=admin_headers)
    assert response.status_code == 200

    benchmarks = [BenchmarkHeader(**b) for b in response.json()]
    assert len(benchmarks) == 2
    assert b1.id in [b.id for b in benchmarks]
    assert b2.id in [b.id for b in benchmarks]


@pytest.mark.anyio(loop_scope='session')
async def test_delete_public_content(
    async_client, published_benchmark_user1, standard_user1_headers, admin_headers
):
    _, created_benchmark = published_benchmark_user1

    # Student tries to delete their public benchmark (should fail)
    response = await delete_benchmark(async_client, created_benchmark.id, standard_user1_headers)
    assert response.status_code == 403

    # Admin deletes public benchmark (should succeed)
    response = await delete_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 200


if __name__ == '__main__':
    pytest.main(['-v', __file__])
