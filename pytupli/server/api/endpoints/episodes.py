from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import (
    BenchmarkHeader,
    EpisodeHeader,
    EpisodeItem,
    Episode,
    EpisodesListRequest,
    UserOut,
    RESOURCE_TYPE,
    RIGHT,
)
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
    EPISODES_COLLECTION_NAME,
)
from pytupli.server.management.authorization import (
    check_authorization,
    check_group_permissions,
    inject_read_permission_filter,
    get_rights_in_groups,
)

router = APIRouter()


@router.post('/record')
async def episodes_record(
    episode: Episode,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.EPISODE, RIGHT.EPISODE_CREATE)
    ),
) -> EpisodeHeader:
    user, _ = auth_result

    # check if the benchmark exists
    query = {'id': episode.benchmark_id}
    benchmark_data = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)
    if not benchmark_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark id doesn't exist"
        )

    benchmark = BenchmarkHeader(**benchmark_data)

    # check if user has read access to the benchmark
    user_rights = await get_rights_in_groups(db_handler, user, ['global', *benchmark.published_in])
    if RIGHT.BENCHMARK_READ not in user_rights:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Cannot record episodes for benchmarks without read access',
        )

    # record the episode
    try:
        episode_item = EpisodeItem.create_new(created_by=user.username, **episode.model_dump())
        await db_handler.create_item(EPISODES_COLLECTION_NAME, episode_item.model_dump())
        return episode_item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to record episode: {str(e)}')


@router.put('/publish')
async def episodes_publish(
    episode_id: str,
    publish_in: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, EpisodeItem | None] = Depends(
        check_authorization(RESOURCE_TYPE.EPISODE, RIGHT.EPISODE_READ, requires_ownership=True)
    ),
) -> None:
    user, episode = auth_result

    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Referenced episode {episode_id} does not exist',
        )

    # Check if user has CONTENT_CREATE rights in the specified group
    await check_group_permissions(db_handler, user, publish_in, [RIGHT.EPISODE_CREATE])

    # get referenced benchmark
    query = {'id': episode.benchmark_id}
    benchmark = BenchmarkHeader(**await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query))

    # check if user has read access to the benchmark
    user_rights = await get_rights_in_groups(db_handler, user, ['global', *benchmark.published_in])
    if RIGHT.BENCHMARK_READ not in user_rights:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Insufficient permissions to read benchmark: {episode.benchmark_id}',
        )

    try:
        query = {'id': episode_id}
        published_in = episode.published_in or []
        if publish_in in published_in:
            return

        published_in.append(publish_in)
        update = {'$set': {'published_in': published_in}}
        await db_handler.update_items(EPISODES_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to record episodes: {str(e)}')


@router.put('/unpublish')
async def episodes_unpublish(
    episode_id: str,
    unpublish_from: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, EpisodeItem | None] = Depends(
        check_authorization(RESOURCE_TYPE.EPISODE, RIGHT.EPISODE_READ)
    ),
) -> None:
    user, episode = auth_result

    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Episode {episode_id} does not exist',
        )

    # Check if user has EPISODE_DELETE rights in the specified group
    await check_group_permissions(db_handler, user, unpublish_from, [RIGHT.EPISODE_DELETE])

    # Check if currently published in this group
    current_published_in = episode.published_in or []
    if unpublish_from not in current_published_in:
        return

    # Remove the group from published_in list
    updated_published_in = [group for group in current_published_in if group != unpublish_from]
    update = {'$set': {'published_in': updated_published_in}}

    try:
        query = {'id': episode_id}
        await db_handler.update_items(EPISODES_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to unpublish episode: {str(e)}',
        )


@router.delete('/delete')
async def episodes_delete(
    episode_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, dict] = Depends(
        check_authorization(RESOURCE_TYPE.EPISODE, RIGHT.EPISODE_DELETE)
    ),
) -> None:
    try:
        # Delete the episode(s) from the database (existence already verified by authorization)
        query = {'id': episode_id}
        await db_handler.delete_items(EPISODES_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete episode(s): {str(e)}',
        )


@router.post('/list')
async def episodes_list(
    request: EpisodesListRequest = EpisodesListRequest(),
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.EPISODE)),
) -> list[EpisodeHeader] | list[EpisodeItem]:
    user, _ = auth_result
    filter = await inject_read_permission_filter(request, user, RESOURCE_TYPE.EPISODE, db_handler)
    try:
        episodes = await db_handler.query_items(
            EPISODES_COLLECTION_NAME,
            filter,
            projection={'tuples': 0}
            if not request.include_tuples
            else None,  # Optionally exlude tuples from the result to reduce traffic
        )

        return (
            [EpisodeHeader(**episode) for episode in episodes]
            if not request.include_tuples
            else [EpisodeItem(**episode) for episode in episodes]
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list episodes: {str(e)}',
        )
