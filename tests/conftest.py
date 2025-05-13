import asyncio
from pathlib import Path
from threading import Thread
import dotenv

import httpx
import pytest_asyncio

import uvicorn

env_file = Path(__file__).parent.parent / 'pytupli' / 'server' / '.env'

dotenv.load_dotenv(dotenv_path=env_file)

async def create_webserver(asgi_str: str, port: int, log_level: str = 'warning'):
    print(f'Starting {asgi_str} service on port {port}')

    server_config = uvicorn.Config(asgi_str, port=port, log_level=log_level, env_file=env_file)

    server = uvicorn.Server(server_config)
    await server.serve()


async def check_service_ready(url: str, timeout: int = 15):
    """Poll a service's health endpoint until it's ready or the timeout expires."""
    async with httpx.AsyncClient() as client:
        for _ in range(timeout):
            try:
                response = await client.get(
                    url,
                )
                if response.status_code == 200:
                    print(f'Service at {url} is ready.')
                    return True
            except httpx.RequestError:
                pass  # Service is not ready yet
            await asyncio.sleep(2)

        print(f'Service at {url} failed to become ready!')
        return False


async def task_servers():
    return [
        asyncio.create_task(create_webserver('pytupli.server.main:app', 8080)),
    ]


async def ensure_all_services_ready():
    # Check that all dependent services are ready
    await asyncio.gather(
        check_service_ready('http://localhost:8080/'),
    )


@pytest_asyncio.fixture(scope='session', autouse=True)
def run_backend():
    def run_event_loop():
        asyncio.run(start_servers_and_manage_lifecycle())

    server_thread = Thread(target=run_event_loop)
    server_thread.daemon = True  # Allow server to exit when tests finish
    server_thread.start()

    # Wait for all services to become ready
    asyncio.run(ensure_all_services_ready())
    print('All services running')

    yield


async def start_servers_and_manage_lifecycle():
    """Start servers and keep them running."""
    # Start all servers in the background
    tasks = await task_servers()

    try:
        # Keep the event loop running indefinitely to maintain the servers
        await asyncio.Future()  # Wait forever until interrupted
    except asyncio.CancelledError:
        print('Shutting down servers...')
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


if __name__ == '__main__':
    asyncio.run(create_webserver('pytupli.server.main:app', 8080))
