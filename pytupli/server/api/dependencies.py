from fastapi import Request


async def get_db_handler(request: Request):
    return request.app.db_handler
