import hashlib
import tempfile
import os
from typing import Annotated

from gridfs import NoFile
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from pytupli.schema import (
    ArtifactMetadata,
    ArtifactMetadataItem,
    UserOut,
    BaseFilter,
    RESOURCE_TYPE,
    RIGHT,
)
from pytupli.server.api.dependencies import get_db_handler
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    ARTIFACTS_COLLECTION_NAME,
)
from pytupli.server.management.authorization import (
    check_authorization,
    check_group_permissions,
    inject_read_permission_filter,
)

router = APIRouter()

filename_template = 'artifact_{id}'


async def _download_artifact_data(
    db_handler: MongoDBHandler, artifact_id: str
) -> tuple[bytes, dict]:
    """Download artifact data and metadata from database."""
    try:
        data, metadata = await db_handler.download_file(
            ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id}
        )
        return data, metadata
    except NoFile as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Could not find file {artifact_id}: {e}'
        )


async def _update_artifact_published_in(
    db_handler: MongoDBHandler, artifact_id: str, published_in_list: list[str]
) -> None:
    """Update artifact's published_in list by re-uploading with new metadata."""
    try:
        # Download current file and metadata
        data, metadata = await _download_artifact_data(db_handler, artifact_id)

        # Update the published_in field in metadata
        metadata['published_in'] = published_in_list

        # Delete old file
        await db_handler.delete_file(ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id})

        # Re-upload with updated metadata
        await db_handler.upload_file(
            ARTIFACTS_COLLECTION_NAME,
            data,
            filename_template.format(id=metadata['id']),
            metadata=metadata,
        )
    except Exception as e:
        if 'Multiple Objects with id' in str(e):
            raise HTTPException(status_code=status.HTTP_300_MULTIPLE_CHOICES, detail=str(e))
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post('/upload')
async def artifact_upload(
    data: Annotated[UploadFile, File()],
    metadata: Annotated[
        str, Form()
    ],  # this must conform to DataSourceMetadata but can only be a str in a multipart form
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.ARTIFACT, RIGHT.ARTIFACT_CREATE)
    ),
) -> ArtifactMetadataItem:
    user, _ = auth_result
    metadata: ArtifactMetadata = ArtifactMetadata.model_validate_json(metadata)

    contents = await data.read()
    try:
        # Hash the raw file contents combined with the username
        file_hash = hashlib.sha256(contents + user.username.encode('utf-8')).hexdigest()

        artifact_metadata = ArtifactMetadataItem.create_new(
            created_by=user.username,
            hash=file_hash,
            **metadata.model_dump(),
        )

        # Write the file to MongoDB using GridFS
        object_metadata = await db_handler.upload_file(
            ARTIFACTS_COLLECTION_NAME,
            contents,
            filename_template.format(id=artifact_metadata.id),
            artifact_metadata.model_dump(),
        )
        return object_metadata
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to save file: {e}',
        )


@router.put('/publish')
async def artifact_publish(
    artifact_id: str,
    publish_in: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, ArtifactMetadataItem | None] = Depends(
        check_authorization(RESOURCE_TYPE.ARTIFACT, RIGHT.ARTIFACT_READ, requires_ownership=True)
    ),
) -> None:
    user, artifact = auth_result

    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Artifact does not exist')

    # Check if user has ARTIFACT_CREATE rights in the specified group
    await check_group_permissions(db_handler, user, publish_in, [RIGHT.ARTIFACT_CREATE])

    # Check if already published in this group
    current_published_in = artifact.published_in
    if publish_in in current_published_in:
        return

    # Add the new group to published_in list
    updated_published_in = current_published_in + [publish_in]
    await _update_artifact_published_in(db_handler, artifact_id, updated_published_in)


@router.put('/unpublish')
async def artifact_unpublish(
    artifact_id: str,
    unpublish_from: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, ArtifactMetadataItem | None] = Depends(
        check_authorization(RESOURCE_TYPE.ARTIFACT, RIGHT.ARTIFACT_READ)
    ),
) -> None:
    user, artifact = auth_result

    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Artifact does not exist')

    # Check if user has CONTENT_DELETE rights in the specified group
    await check_group_permissions(db_handler, user, unpublish_from, [RIGHT.ARTIFACT_DELETE])

    # Check if currently published in this group
    current_published_in = artifact.published_in
    if unpublish_from not in current_published_in:
        return

    # Remove the group from published_in list
    updated_published_in = [group for group in current_published_in if group != unpublish_from]
    await _update_artifact_published_in(db_handler, artifact_id, updated_published_in)


@router.post('/list')
async def artifact_list(
    filter: BaseFilter = BaseFilter(),
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.ARTIFACT)),
) -> list[ArtifactMetadataItem]:
    user, _ = auth_result
    filter.apply_prefix('metadata')
    filter = await inject_read_permission_filter(
        filter, user, RESOURCE_TYPE.ARTIFACT, db_handler, prefix_path='metadata'
    )
    query_filter = db_handler.convert_filter_to_query(filter) if filter else {}
    try:
        files = await db_handler.download_files(ARTIFACTS_COLLECTION_NAME, query_filter)
        metadata = []
        for file in files:
            metadata.append(ArtifactMetadataItem(**file['metadata']))
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to list arifact: {str(e)}')


@router.get('/download')
async def artifact_download(
    artifact_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, ArtifactMetadataItem | None] = Depends(
        check_authorization(RESOURCE_TYPE.ARTIFACT, RIGHT.ARTIFACT_READ)
    ),
) -> FileResponse:
    _, artifact = auth_result
    if not artifact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Artifact does not exist')

    try:
        # Get both data and metadata (resource already verified by authorization)
        data, metadata = await db_handler.download_file(
            ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id}
        )

        metadata = ArtifactMetadataItem(**metadata)

        # Create a temporary file to serve with FileResponse
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        try:
            temp_file.write(data)
            temp_file.flush()
            temp_file_path = temp_file.name

            # Prepare headers with metadata
            headers = {'X-Metadata': metadata.model_dump_json()}

            return FileResponse(
                path=temp_file_path,
                headers=headers,
                filename=metadata.name,
                background=lambda: os.unlink(
                    temp_file_path
                ),  # Clean up the temp file after sending
            )

        except Exception as e:
            # Clean up temp file in case of errors
            os.unlink(temp_file.name)
            raise e
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        if 'Multiple Objects with id' in str(e):
            raise HTTPException(status_code=status.HTTP_300_MULTIPLE_CHOICES, detail=str(e))
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete('/delete')
async def artifact_delete(
    artifact_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, dict] = Depends(
        check_authorization(RESOURCE_TYPE.ARTIFACT, RIGHT.ARTIFACT_DELETE)
    ),
) -> None:
    try:
        # Delete the artifact (existence already verified by authorization)
        _ = await db_handler.delete_file(ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id})
        # if database returns delete_count 0, then the file was not found
        # if r.deleted_count == 0:
        #    raise FileNotFoundError(f'Artifact with id {artifact_id} not found')
        return
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except NoFile as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Could not find file {artifact_id}: {e}'
        )
    except Exception as e:
        if 'Multiple Objects with id' in str(e):
            raise HTTPException(status_code=status.HTTP_300_MULTIPLE_CHOICES, detail=str(e))
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f'Could not delete artifact {artifact_id}: {e}',
            )
