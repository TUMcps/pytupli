import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pytupli.server.api.routes import api_router
from pytupli.server.config import DBHandlerFactory
from pytupli.server.management.roles import initialize_database
from pytupli.server.management.security import initialize_secret_key


@asynccontextmanager
async def db_lifespan(app: FastAPI):
    initialize_secret_key()
    app.db_handler = DBHandlerFactory.get_handler()
    ping_response = await app.db_handler.ping()

    # Add logger to app
    app.logger = logging.getLogger('uvicorn.error')
    if int(ping_response) != 1:
        raise Exception('Problem connecting to database cluster.')
    else:
        app.logger.info('Connected to database cluster.')
    try:
        await initialize_database(app.db_handler)
        yield
    finally:
        # Shutdown
        app.db_handler.close_connection()
        app.logger.info('Connection to database cluster closed.')


app = FastAPI(lifespan=db_lifespan)

app.include_router(api_router)


@app.get('/')
async def read_root():
    return


if __name__ == '__main__':
    import uvicorn

    port = int(os.getenv('PORT', '8080'))
    uvicorn.run(
        'pytupli.server.main:app', host='0.0.0.0', port=port, env_file='./.env', reload=True
    )
