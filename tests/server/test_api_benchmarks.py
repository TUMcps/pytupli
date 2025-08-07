import pytest
from pytupli.schema import BenchmarkHeader

# The helper functions are moved to conftest.py
from conftest import (
    create_benchmark,
    publish_benchmark,
    unpublish_benchmark,
    load_benchmark,
    delete_benchmark,
    create_group,
    delete_group,
    add_user_to_group,
)

pytestmark = pytest.mark.anyio


async def test_benchmarks_create(async_client, sample_benchmark, admin_headers):
    response, benchmark = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 200
    assert benchmark is not None

    response = await delete_benchmark(async_client, benchmark.id, admin_headers)
    assert response.status_code == 200



async def test_benchmarks_create_benchmark_exists_publicly(
    async_client, published_benchmark_user1, standard_user2_headers, sample_benchmark
):
    _, _ = published_benchmark_user1

    # Second user tries to create the same benchmark - should fail with 409
    response, _ = await create_benchmark(async_client, sample_benchmark, standard_user2_headers)
    assert response.status_code == 409



async def test_benchmarks_duplication(
    async_client, created_benchmark_admin, admin_headers, sample_benchmark
):
    _, created_benchmark = created_benchmark_admin

    # Try creating it again - should fail with 409
    response, _ = await create_benchmark(async_client, sample_benchmark, admin_headers)
    assert response.status_code == 409



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
    assert 'admin' in loaded_benchmark.published_in  # Benchmark is published in creator's group



async def test_benchmarks_load_private_insufficient_rights(
    created_benchmark_admin, async_client, standard_user1_headers
):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    response, _ = await load_benchmark(async_client, created_benchmark.id, standard_user1_headers)
    assert response.status_code == 403



async def test_benchmarks_publish(created_benchmark_admin, async_client, admin_headers):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Publish benchmark
    response = await publish_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 200

    # Load and verify it's published in global
    response, loaded_benchmark = await load_benchmark(
        async_client, created_benchmark.id, admin_headers
    )
    assert response.status_code == 200
    assert 'global' in loaded_benchmark.published_in



async def test_benchmarks_delete(created_benchmark_admin, async_client, admin_headers):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Delete benchmark
    response = await delete_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 200

    # Verify deletion
    response, _ = await load_benchmark(async_client, created_benchmark.id, admin_headers)
    assert response.status_code == 404



async def test_delete_benchmark_guest_forbidden(created_benchmark_admin, async_client):
    # Get created benchmark
    _, created_benchmark = created_benchmark_admin

    # Attempt to delete with no token => guest user => 401
    delete_response = await async_client.delete(
        f'/benchmarks/delete?benchmark_id={created_benchmark.id}'
    )
    assert delete_response.status_code == 403



async def test_delete_benchmark_other_user_forbidden(
    async_client, created_benchmark_user1, standard_user2_headers
):
    _, created_benchmark = created_benchmark_user1

    # Try to delete with user 2 - should fail
    response = await delete_benchmark(async_client, created_benchmark.id, standard_user2_headers)
    assert response.status_code == 403



async def test_benchmarks_list(
    async_client, created_benchmark_user1, published_benchmark_admin, admin_headers
):
    _, b1 = created_benchmark_user1
    _, b2 = published_benchmark_admin

    # List benchmarks as admin
    response = await async_client.post('/benchmarks/list', headers=admin_headers, json={})
    assert response.status_code == 200

    benchmarks = [BenchmarkHeader(**b) for b in response.json()]
    # Admin should only see benchmarks they have read access to
    # This includes their own benchmark and globally published ones
    # b1 is created by user1 and published in ['test_user_1'] - admin can't see it
    # b2 is created by admin and published in ['admin', 'global'] - admin can see it
    assert len(benchmarks) >= 1
    admin_benchmark_ids = [b.id for b in benchmarks]
    assert b2.id in admin_benchmark_ids



async def test_delete_public_content(
    async_client, published_benchmark_user1, standard_user1_headers, admin_headers
):
    _, created_benchmark = published_benchmark_user1

    # With the current access system, users can delete their own content even if it's public
    # The creator always has delete rights to their own content
    response = await delete_benchmark(async_client, created_benchmark.id, standard_user1_headers)
    assert response.status_code == 200



async def test_benchmarks_publish_in_user_group(
    async_client, created_benchmark_admin, admin_headers, standard_user1_headers
):
    """Test publishing a benchmark in a user-created group"""
    _, created_benchmark = created_benchmark_admin
    test_group_name = "test_benchmark_group"

    try:
        # Create a test group
        group_response = await create_group(async_client, test_group_name, admin_headers)
        assert group_response.status_code == 200

        # Add user1 to the group with CONTRIBUTOR role (which includes BENCHMARK_CREATE rights)
        from pytupli.schema import DEFAULT_ROLE
        membership_response = await add_user_to_group(
            async_client, test_group_name, "test_user_1", admin_headers, [DEFAULT_ROLE.CONTRIBUTOR.value]
        )
        assert membership_response.status_code == 200

        # Admin publishes benchmark in the test group
        publish_response = await publish_benchmark(
            async_client, created_benchmark.id, admin_headers, test_group_name
        )
        assert publish_response.status_code == 200

        # Verify benchmark is published in the test group
        response, loaded_benchmark = await load_benchmark(
            async_client, created_benchmark.id, admin_headers
        )
        assert response.status_code == 200
        assert test_group_name in loaded_benchmark.published_in

    finally:
        # Clean up the group
        await delete_group(async_client, admin_headers, test_group_name)



async def test_benchmarks_publish_in_nonexistent_group(
    async_client, created_benchmark_admin, admin_headers
):
    """Test publishing a benchmark in a non-existent group (should fail)"""
    _, created_benchmark = created_benchmark_admin
    nonexistent_group = "nonexistent_group"

    # Try to publish in non-existent group (should fail)
    publish_response = await publish_benchmark(
        async_client, created_benchmark.id, admin_headers, nonexistent_group
    )
    assert publish_response.status_code == 403  # Should fail due to lack of permissions



async def test_benchmarks_unpublish_success(
    async_client, created_benchmark_admin, admin_headers
):
    """Test successful unpublishing of a benchmark from global group"""
    _, created_benchmark = created_benchmark_admin

    # First publish the benchmark in global
    publish_response = await publish_benchmark(
        async_client, created_benchmark.id, admin_headers, "global"
    )
    assert publish_response.status_code == 200

    # Verify it's published
    response, loaded_benchmark = await load_benchmark(
        async_client, created_benchmark.id, admin_headers
    )
    assert response.status_code == 200
    assert "global" in loaded_benchmark.published_in

    # Unpublish from global
    unpublish_response = await unpublish_benchmark(
        async_client, created_benchmark.id, admin_headers, "global"
    )
    assert unpublish_response.status_code == 200

    # Verify it's no longer published in global
    response, loaded_benchmark = await load_benchmark(
        async_client, created_benchmark.id, admin_headers
    )
    assert response.status_code == 200
    assert "global" not in loaded_benchmark.published_in



async def test_benchmarks_unpublish_insufficient_permissions(
    async_client, created_benchmark_admin, admin_headers, standard_user1_headers
):
    """Test unpublishing fails when user lacks permissions"""
    _, created_benchmark = created_benchmark_admin

    # Admin publishes the benchmark in global
    publish_response = await publish_benchmark(
        async_client, created_benchmark.id, admin_headers, "global"
    )
    assert publish_response.status_code == 200

    # User1 tries to unpublish (should fail - insufficient permissions)
    unpublish_response = await unpublish_benchmark(
        async_client, created_benchmark.id, standard_user1_headers, "global"
    )
    assert unpublish_response.status_code == 403  # Should fail due to insufficient permissions



async def test_benchmarks_unpublish_from_user_group(
    async_client, created_benchmark_admin, admin_headers, standard_user1_headers
):
    """Test unpublishing from a user-created group"""
    _, created_benchmark = created_benchmark_admin
    test_group_name = "test_unpublish_group"

    try:
        # Create a test group
        group_response = await create_group(async_client, test_group_name, admin_headers)
        assert group_response.status_code == 200

        # Add user1 to the group with CONTENT_ADMIN role (which includes both BENCHMARK_CREATE and BENCHMARK_DELETE rights)
        from pytupli.schema import DEFAULT_ROLE
        membership_response = await add_user_to_group(
            async_client, test_group_name, "test_user_1", admin_headers,
            [DEFAULT_ROLE.CONTENT_ADMIN.value]
        )
        assert membership_response.status_code == 200

        # Admin publishes benchmark in the test group
        publish_response = await publish_benchmark(
            async_client, created_benchmark.id, admin_headers, test_group_name
        )
        assert publish_response.status_code == 200

        # User1 unpublishes from the test group (should succeed)
        unpublish_response = await unpublish_benchmark(
            async_client, created_benchmark.id, standard_user1_headers, test_group_name
        )
        assert unpublish_response.status_code == 200

        # Verify benchmark is no longer published in the test group
        response, loaded_benchmark = await load_benchmark(
            async_client, created_benchmark.id, admin_headers
        )
        assert response.status_code == 200
        assert test_group_name not in loaded_benchmark.published_in

    finally:
        # Clean up the group
        await delete_group(async_client, admin_headers, test_group_name)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
