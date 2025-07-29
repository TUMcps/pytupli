from fastapi import APIRouter

from pytupli.server.api.endpoints import artifacts, benchmarks, episodes, users, groups, roles

api_router = APIRouter()
api_router.include_router(artifacts.router, prefix='/artifacts', tags=['Artifacts'])
api_router.include_router(benchmarks.router, prefix='/benchmarks', tags=['Benchmarks'])
api_router.include_router(episodes.router, prefix='/episodes', tags=['Episodes'])
api_router.include_router(users.router, prefix='/access/users', tags=['Access', 'Users'])
api_router.include_router(groups.router, prefix='/access/groups', tags=['Access', 'Groups'])
api_router.include_router(roles.router, prefix='/access/roles', tags=['Access', 'Roles'])
