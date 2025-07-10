from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import EpisodeHeader, EpisodeItem, Episode, User, BaseFilter
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
    EPISODES_COLLECTION_NAME,
)
from pytupli.server.management.security import check_authentication, inject_read_permission_filter

router = APIRouter()


class EpisodesListRequest(BaseFilter):
    include_tuples: bool = False


@router.post('/record')
async def episodes_record(
    episode: Episode,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> EpisodeHeader:
    # check if the benchmark exists
    query = {'id': episode.benchmark_id}
    benchmark_entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)
    if not benchmark_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark id doesn't exist"
        )

    # check if the benchmark is public or owned by the user
    if benchmark_entry['is_public'] is False and benchmark_entry['created_by'] != user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Cannot record episodes for private benchmarks that are not owned by the user',
        )

    # record the episode
    try:
        episode_item = EpisodeItem.create_new(**episode.model_dump(), created_by=user.username)
        await db_handler.create_item(EPISODES_COLLECTION_NAME, episode_item.model_dump())
        return episode_item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to record episode: {str(e)}')


@router.put('/publish')
async def episodes_publish(
    episode_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    # fetch the benchmark id from the episode
    query = {'id': episode_id}
    episode_entry = await db_handler.get_item(EPISODES_COLLECTION_NAME, query)
    if not episode_entry:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f'Referenced episode {episode_id} does not exist',
        )
    ref_benchmark_id = episode_entry['benchmark_id']

    # check if the benchmark is already published
    benchmark_query = {'id': ref_benchmark_id}
    benchmark_entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, benchmark_query)
    if not benchmark_entry or not benchmark_entry['is_public']:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f'Referenced benchmark {ref_benchmark_id} is not public or does not exist',
        )

    try:
        update = {'$set': {'is_public': True}}
        await db_handler.update_items(EPISODES_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to record episodes: {str(e)}')


@router.delete('/delete')
async def episodes_delete(
    episode_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    try:
        # Delete the episode(s) from the database
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
    user: User = Depends(check_authentication),
) -> list[EpisodeHeader] | list[EpisodeItem]:
    filter = await inject_read_permission_filter(request, user, db_handler)
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
