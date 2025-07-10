import hashlib
import tempfile
import os
from typing import Annotated

from gridfs import NoFile
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from pytupli.schema import ArtifactMetadata, ArtifactMetadataItem, User, BaseFilter
from pytupli.server.api.dependencies import get_db_handler
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    ARTIFACTS_COLLECTION_NAME,
)
from pytupli.server.management.security import check_authentication, inject_read_permission_filter

router = APIRouter()

filename_template = 'artifact_{id}'


@router.post('/upload')
async def artifact_upload(
    data: Annotated[UploadFile, File()],
    metadata: Annotated[
        str, Form()
    ],  # this must conform to DataSourceMetadata but can only be a str in a multipart form
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> ArtifactMetadataItem:
    metadata: ArtifactMetadata = ArtifactMetadata.model_validate_json(metadata)

    contents = await data.read()
    try:
        # Hash the raw file contents combined with the username
        file_hash = hashlib.sha256(contents + user.username.encode('utf-8')).hexdigest()

        artifact_metadata = ArtifactMetadataItem.create_new(
            **metadata.model_dump(),
            hash=file_hash,
            created_by=user.username,
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
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    # you can't update files in MongoDB, so we need to download the file, update the metadata and upload it again
    try:
        # download file
        data, metadata = await db_handler.download_file(
            ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id}
        )
        if not data:
            raise FileNotFoundError(f'Artifact with id {artifact_id} not found')
        # update metadata
        metadata['is_public'] = True
        # delete file from db
        await db_handler.delete_file(ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id})

        # create file in db again
        _ = await db_handler.upload_file(
            ARTIFACTS_COLLECTION_NAME,
            data,
            filename_template.format(id=metadata['id']),
            metadata=metadata,
        )
        return {'message': 'Datasource published successfully.'}
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
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post('/list')
async def artifact_list(
    filter: BaseFilter = BaseFilter(),
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> list[ArtifactMetadataItem]:
    filter.apply_prefix('metadata')
    filter = await inject_read_permission_filter(filter, user, db_handler, prefix_path='metadata')
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
    _=Depends(check_authentication),
) -> FileResponse:
    try:
        # Get both data and metadata
        data, metadata = await db_handler.download_file(
            ARTIFACTS_COLLECTION_NAME, {'metadata.id': artifact_id}
        )
        if not data:
            raise FileNotFoundError(f'Artifact with id {artifact_id} not found')

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
    _=Depends(check_authentication),
) -> None:
    try:
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
