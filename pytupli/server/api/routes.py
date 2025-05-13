from fastapi import APIRouter

from pytupli.server.api.endpoints import access, artifacts, benchmarks, episodes

api_router = APIRouter()
api_router.include_router(artifacts.router, prefix='/artifacts', tags=['Artifacts'])
api_router.include_router(benchmarks.router, prefix='/benchmarks', tags=['Benchmarks'])
api_router.include_router(episodes.router, prefix='/episodes', tags=['Episodes'])
api_router.include_router(access.router, prefix='/access', tags=['Access'])
