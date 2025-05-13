import uvicorn
import threading
import time
import os
import signal
from pytupli.server.main import app
import requests


class UvicornThread(threading.Thread):
    def __init__(self, module='pytupli.server.main:app', host=None, port=8080):
        super().__init__(daemon=True)
        self.module = module
        self.host = host
        self.port = port
        self.server = None
        # Import the app directly to access it later
        self.app = app

    def run(self):
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.module,
                host=self.host,
                port=self.port,
                reload=False,
                workers=1,
                loop='asyncio',
                log_level='error',  # Reduce logging noise
            )
        )
        self.server.run()

    def shutdown(self, timeout=5):
        if self.server:
            self.server.should_exit = True
            self.join(timeout=timeout)
            if self.is_alive():
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.1)
                if self.is_alive():
                    os.kill(os.getpid(), signal.SIGKILL)


def start_api():
    """Start API server and wait until it's ready"""
    # Use CI-friendly host from environment variable
    host = '0.0.0.0'
    server = UvicornThread(host=host, port=8080)
    server.start()

    # Wait for server with better retries
    max_retries = 10
    for i in range(max_retries):
        try:
            # Use correct host for health check
            response = requests.get(f'http://{host}:8080/')
            if response.status_code == 200:
                print(f'Server is up after {i + 1} attempt(s)')
                break
        except requests.ConnectionError:
            print(f'Waiting for server (attempt {i + 1}/{max_retries})')
            time.sleep(2)  # Longer wait for CI environments
    else:
        print('WARNING: Server may not be ready after maximum retries')

    # Give extra time for server to stabilize
    time.sleep(2)
    return server


def stop_api(server):
    server.shutdown()


def get_app(server):
    """Get the FastAPI app instance"""
    return server.app
