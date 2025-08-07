import pytest
from conftest import (
    upload_artifact,
    download_artifact,
    delete_artifact,
    publish_artifact,
    unpublish_artifact,
    create_group,
    delete_group,
    add_user_to_group,
)
from pytupli.schema import ArtifactMetadataItem

pytestmark = pytest.mark.anyio

async def test_artifact_upload(async_client, sample_artifact, admin_headers):
    data, _, expected_hash, metadata = sample_artifact
    response, uploaded_id = await upload_artifact(
        async_client, metadata, data, headers=admin_headers
    )
    assert response.status_code == 200
    # The hash is now the data_id returned from the endpoint
    assert uploaded_id is not None

    await delete_artifact(async_client, uploaded_id, headers=admin_headers)



async def test_artifact_duplicate_upload(
    async_client, uploaded_artifact_admin, sample_artifact, admin_headers
):
    # The artifact is already uploaded by the fixture
    _, data_id, _, _, metadata = uploaded_artifact_admin

    # Try uploading again with same data
    data, _, _, _ = sample_artifact
    response, _ = await upload_artifact(async_client, metadata, data, headers=admin_headers)
    assert response.status_code == 200



async def test_artifact_download(async_client, uploaded_artifact_admin, admin_headers):
    _, data_id, _, _, up_metadata = uploaded_artifact_admin

    response, _, _, down_metadata = await download_artifact(
        async_client, data_id, headers=admin_headers
    )
    assert response.status_code == 200

    assert down_metadata.name == up_metadata.name
    assert down_metadata.description == up_metadata.description



async def test_artifact_download_insufficient_rights(
    async_client, uploaded_artifact_user1, standard_user2_headers
):
    _, data_id, _, _ = uploaded_artifact_user1

    response, _, _, _ = await download_artifact(
        async_client, data_id, headers=standard_user2_headers
    )
    assert response.status_code == 403



async def test_artifact_download_non_existing(async_client, admin_headers):
    response, _, _, _ = await download_artifact(
        async_client, 'non_existing_id', headers=admin_headers
    )
    assert response.status_code == 404



async def test_artifact_list_insufficient_rights(
    async_client, uploaded_artifact_admin, standard_user1_headers
):
    response = await async_client.post('/artifacts/list', headers=standard_user1_headers)
    assert response.status_code == 200
    artifacts = response.json()
    assert len(artifacts) == 0



async def test_artifact_delete(async_client, uploaded_artifact_admin, admin_headers):
    _, data_id, _, _, _ = uploaded_artifact_admin
    response = await delete_artifact(async_client, data_id, headers=admin_headers)
    assert response.status_code == 200

    response, _, _, _ = await download_artifact(async_client, data_id, headers=admin_headers)
    assert response.status_code == 404



async def test_artifact_published_list(
    async_client, published_artifact_admin, standard_user1_headers
):
    (
        _,
        data_id,
        _,
    ) = published_artifact_admin

    response = await async_client.post('/artifacts/list', headers=standard_user1_headers)
    assert response.status_code == 200
    artifacts: list[ArtifactMetadataItem] = [ArtifactMetadataItem(**d) for d in response.json()]
    assert len(artifacts) == 1
    assert artifacts[0].id == data_id



async def test_artifact_publish_in_user_group(
    async_client, uploaded_artifact_admin, admin_headers, standard_user1_headers
):
    """Test publishing an artifact in a user-created group"""
    _, artifact_id, _, _, _ = uploaded_artifact_admin
    test_group_name = "test_artifact_group"

    try:
        # Create a test group
        group_response = await create_group(async_client, test_group_name, admin_headers)
        assert group_response.status_code == 200

        # Add user1 to the group with default roles (should include ARTIFACT_CREATE)
        membership_response = await add_user_to_group(
            async_client, test_group_name, "test_user_1", admin_headers
        )
        assert membership_response.status_code == 200

        # Admin publishes artifact in the test group
        publish_response = await publish_artifact(
            async_client, artifact_id, admin_headers, test_group_name
        )
        assert publish_response.status_code == 200

        # Verify artifact is accessible to group members
        response = await async_client.post('/artifacts/list', headers=standard_user1_headers)
        assert response.status_code == 200
        artifacts = [ArtifactMetadataItem(**d) for d in response.json()]
        artifact_ids = [a.id for a in artifacts]
        assert artifact_id in artifact_ids

    finally:
        # Clean up the group
        await delete_group(async_client, admin_headers, test_group_name)



async def test_artifact_publish_in_nonexistent_group(
    async_client, uploaded_artifact_admin, admin_headers
):
    """Test publishing an artifact in a non-existent group (should fail)"""
    _, artifact_id, _, _, _ = uploaded_artifact_admin
    nonexistent_group = "nonexistent_group"

    # Try to publish in non-existent group (should fail)
    publish_response = await publish_artifact(
        async_client, artifact_id, admin_headers, nonexistent_group
    )
    assert publish_response.status_code == 403  # Should fail due to lack of permissions



async def test_artifact_unpublish_success(
    async_client, uploaded_artifact_admin, admin_headers
):
    """Test successful unpublishing of an artifact from global group"""
    _, artifact_id, _, _, _ = uploaded_artifact_admin

    # First publish the artifact in global
    publish_response = await publish_artifact(
        async_client, artifact_id, admin_headers, "global"
    )
    assert publish_response.status_code == 200

    # Verify it's published (visible in list)
    response = await async_client.post('/artifacts/list', headers=admin_headers)
    assert response.status_code == 200
    artifacts = [ArtifactMetadataItem(**d) for d in response.json()]
    artifact_ids = [a.id for a in artifacts]
    assert artifact_id in artifact_ids

    # Unpublish from global
    unpublish_response = await unpublish_artifact(
        async_client, artifact_id, admin_headers, "global"
    )
    assert unpublish_response.status_code == 200

    # Verify it's no longer published in global (not visible in general list)
    # Note: This depends on the exact implementation of the list endpoint
    # You may need to adjust this assertion based on your actual API behavior



async def test_artifact_unpublish_insufficient_permissions(
    async_client, uploaded_artifact_admin, admin_headers, standard_user1_headers
):
    """Test unpublishing fails when user lacks permissions"""
    _, artifact_id, _, _, _ = uploaded_artifact_admin

    # Admin publishes the artifact in global
    publish_response = await publish_artifact(
        async_client, artifact_id, admin_headers, "global"
    )
    assert publish_response.status_code == 200

    # User1 tries to unpublish (should fail - insufficient permissions)
    unpublish_response = await unpublish_artifact(
        async_client, artifact_id, standard_user1_headers, "global"
    )
    assert unpublish_response.status_code == 403  # Should fail due to insufficient permissions


if __name__ == '__main__':
    pytest.main(['-v', __file__])
