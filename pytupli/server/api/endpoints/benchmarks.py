from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import (
    Benchmark,
    BenchmarkHeader,
    BenchmarkQuery,
    UserOut,
    BaseFilter,
    RESOURCE_TYPE,
    RIGHT,
)
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
)
from pytupli.server.management.authorization import (
    check_authorization,
    check_group_permissions,
    inject_read_permission_filter,
)

router = APIRouter()


@router.post('/create')
async def benchmarks_create(
    benchmark: BenchmarkQuery,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.BENCHMARK, RIGHT.BENCHMARK_CREATE)
    ),
) -> BenchmarkHeader:
    user, _ = auth_result
    query = {'hash': benchmark.hash}
    existing_benchmark = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, query)

    if existing_benchmark:
        benchmark_entry = BenchmarkHeader(**existing_benchmark)
        # check if the benchmark already exists
        if benchmark_entry.created_by == user.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail='Benchmark already exists'
            )
        elif 'global' in benchmark_entry.published_in:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail='Benchmark already exists in global group',
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
    auth_result: tuple[UserOut, BenchmarkHeader | None] = Depends(
        check_authorization(RESOURCE_TYPE.BENCHMARK, RIGHT.BENCHMARK_READ)
    ),
) -> Benchmark:
    user, benchmark_header = auth_result
    try:
        # Check if the benchmark exists (resource was loaded by authorization)
        if not benchmark_header:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Benchmark not found')

        # Load the full benchmark data from the database
        benchmark_entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, {'id': benchmark_id})
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
    auth_result: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.BENCHMARK)),
) -> list[BenchmarkHeader]:
    user, _ = auth_result
    filter = await inject_read_permission_filter(filter, user, RESOURCE_TYPE.BENCHMARK, db_handler)
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
    publish_in: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, BenchmarkHeader | None] = Depends(
        check_authorization(RESOURCE_TYPE.BENCHMARK, RIGHT.BENCHMARK_READ, requires_ownership=True)
    ),
) -> None:
    user, benchmark = auth_result

    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Benchmark does not exists'
        )

    # Check if user has CONTENT_CREATE rights in the specified group
    await check_group_permissions(db_handler, user, publish_in, [RIGHT.BENCHMARK_CREATE])

    try:
        # Create the update dictionary
        published_in = benchmark.published_in
        if publish_in in published_in:
            return

        published_in.append(publish_in)
        update = {'$set': {'published_in': published_in}}

        # update the benchmark in the db
        query = {'id': benchmark_id}
        await db_handler.update_item(BENCHMARK_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to publish benchmark: {str(e)}',
        )


@router.put('/unpublish')
async def benchmarks_unpublish(
    benchmark_id: str,
    unpublish_from: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, BenchmarkHeader | None] = Depends(
        check_authorization(RESOURCE_TYPE.BENCHMARK, RIGHT.BENCHMARK_READ)
    ),
) -> None:
    user, benchmark = auth_result

    if not benchmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Benchmark does not exist'
        )

    # Check if user has BENCHMARK_DELETE rights in the specified group
    await check_group_permissions(db_handler, user, unpublish_from, [RIGHT.BENCHMARK_DELETE])

    # Check if currently published in this group
    current_published_in = benchmark.published_in
    if unpublish_from not in current_published_in:
        return

    # Remove the group from published_in list
    updated_published_in = [group for group in current_published_in if group != unpublish_from]
    update = {'$set': {'published_in': updated_published_in}}

    try:
        query = {'id': benchmark_id}
        await db_handler.update_item(BENCHMARK_COLLECTION_NAME, query, update)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to unpublish benchmark: {str(e)}',
        )


@router.delete('/delete')
async def benchmarks_delete(
    benchmark_id: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, dict] = Depends(
        check_authorization(RESOURCE_TYPE.BENCHMARK, RIGHT.BENCHMARK_DELETE)
    ),
) -> None:
    try:
        # Delete the benchmark from the database (resource existence already verified by authorization)
        query = {'id': benchmark_id}
        await db_handler.delete_item(BENCHMARK_COLLECTION_NAME, query)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete benchmark: {str(e)}',
        )
