from fastapi import Request

from fastapi import FastAPI

app = FastAPI()


async def get_db_handler(request: Request):
    return request.app.db_handler
