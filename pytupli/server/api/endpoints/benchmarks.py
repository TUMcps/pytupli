from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import Benchmark, BenchmarkHeader, BenchmarkQuery, User, BaseFilter
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
)
from pytupli.server.management.security import check_authentication, inject_read_permission_filter

router = APIRouter()


@router.post('/create')
async def benchmarks_create(
    benchmark: BenchmarkQuery,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> BenchmarkHeader:
    query = {'hash': benchmark.hash}
    benchmark_entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)
    if benchmark_entry:
        # check if the benchmark already exists
        if benchmark_entry['created_by'] == user.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail='Benchmark already exists'
            )
        elif benchmark_entry['is_public']:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail='Benchmark already exists'
            )
    try:
        new_benchmark = Benchmark.create_new(
            created_by=user.username,
            **benchmark.model_dump(),
        )
        await db_handler.create_item(BENCHMARK_COLLECTION_NAME, new_benchmark.model_dump())
        # return the benchmark item that has just been created
        return BenchmarkHeader(**new_benchmark.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create benchmark: {str(e)}',
        )


@router.get('/load')
async def benchmarks_load(
    benchmark_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> Benchmark:
    try:
        # Load the benchmark from the database
        query = {'id': benchmark_id}
        benchmark_entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)

        # Check if the benchmark exists
        if not benchmark_entry:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Benchmark not found')

        # Return the benchmark
        return Benchmark(**benchmark_entry)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to load benchmark: {str(e)}',
        )


@router.post('/list')
async def benchmarks_list(
    filter: BaseFilter = BaseFilter(),
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> list[BenchmarkHeader]:
    filter = await inject_read_permission_filter(filter, user, db_handler)
    try:
        benchmarks = await db_handler.query_items(BENCHMARK_COLLECTION_NAME, filter)
        return [BenchmarkHeader(**benchmark) for benchmark in benchmarks]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list benchmarks: {str(e)}',
        )


@router.put('/publish')
async def benchmarks_publish(
    benchmark_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    # check if user exists
    try:
        query = {'id': benchmark_id}
        entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail='Benchmark does not exists'
            )

        # Create the update dictionary
        update = {'$set': {'is_public': True}}

        # update the benchmark in the db
        await db_handler.update_item(BENCHMARK_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to publish benchmark: {str(e)}',
        )


@router.delete('/delete')
async def benchmarks_delete(
    benchmark_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    try:
        # Delete the benchmark from the database
        query = {'id': benchmark_id}
        await db_handler.delete_item(BENCHMARK_COLLECTION_NAME, query)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete benchmark: {str(e)}',
        )
