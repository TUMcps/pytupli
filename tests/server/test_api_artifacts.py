import pytest
from conftest import (
    upload_artifact,
    download_artifact,
    delete_artifact,
)
from pytupli.schema import ArtifactMetadataItem


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_upload(async_client, sample_artifact, admin_headers):
    data, _, expected_hash, metadata = sample_artifact
    response, uploaded_id = await upload_artifact(
        async_client, metadata, data, headers=admin_headers
    )
    assert response.status_code == 200
    # The hash is now the data_id returned from the endpoint
    assert uploaded_id is not None

    await delete_artifact(async_client, uploaded_id, headers=admin_headers)


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_duplicate_upload(
    async_client, uploaded_artifact_admin, sample_artifact, admin_headers
):
    # The artifact is already uploaded by the fixture
    _, data_id, _, _, metadata = uploaded_artifact_admin

    # Try uploading again with same data
    data, _, _, _ = sample_artifact
    response, _ = await upload_artifact(async_client, metadata, data, headers=admin_headers)
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_download(async_client, uploaded_artifact_admin, admin_headers):
    _, data_id, _, _, up_metadata = uploaded_artifact_admin

    response, _, _, down_metadata = await download_artifact(
        async_client, data_id, headers=admin_headers
    )
    assert response.status_code == 200

    assert down_metadata.name == up_metadata.name
    assert down_metadata.description == up_metadata.description


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_download_insufficient_rights(
    async_client, uploaded_artifact_user1, standard_user2_headers
):
    _, data_id, _, _ = uploaded_artifact_user1

    response, _, _, _ = await download_artifact(
        async_client, data_id, headers=standard_user2_headers
    )
    assert response.status_code == 403


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_download_non_existing(async_client, admin_headers):
    response, _, _, _ = await download_artifact(
        async_client, 'non_existing_id', headers=admin_headers
    )
    assert response.status_code == 404


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_list_insufficient_rights(
    async_client, uploaded_artifact_admin, standard_user1_headers
):
    response = await async_client.get('/artifacts/list', headers=standard_user1_headers)
    assert response.status_code == 200
    artifacts = response.json()
    assert len(artifacts) == 0


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_delete(async_client, uploaded_artifact_admin, admin_headers):
    _, data_id, _, _, _ = uploaded_artifact_admin
    response = await delete_artifact(async_client, data_id, headers=admin_headers)
    assert response.status_code == 200

    response, _, _, _ = await download_artifact(async_client, data_id, headers=admin_headers)
    assert response.status_code == 404


@pytest.mark.anyio(loop_scope='session')
async def test_artifact_published_list(
    async_client, published_artifact_admin, standard_user1_headers
):
    (
        _,
        data_id,
        _,
    ) = published_artifact_admin

    response = await async_client.get('/artifacts/list', headers=standard_user1_headers)
    assert response.status_code == 200
    artifacts: list[ArtifactMetadataItem] = [ArtifactMetadataItem(**d) for d in response.json()]
    assert len(artifacts) == 1
    assert artifacts[0].id == data_id


if __name__ == '__main__':
    pytest.main(['-v', __file__])
