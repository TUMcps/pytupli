import os
from pytupli.server.db.db_handler import MongoDBHandler

from dotenv import load_dotenv

load_dotenv(override=True)


class DBHandlerFactory:
    """
    Factory returning an instance of a storage class based on the given storage type.
    """

    @staticmethod
    def get_handler():
        """
        Returns an instance of a storage class based on the given storage type.

        Args:
            storage_type (str): The type of storage to use. Valid options are "file" and "mongo".

        Returns:
            An instance of a storage class based on the given storage type.

        Raises:
            ValueError: If an invalid storage type is provided.
        """

        CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'mongodb://localhost:27017/')
        return MongoDBHandler(CONNECTION_STRING, os.getenv('MONGO_DB_NAME', 'pytupli'))


ARTIFACTS_COLLECTION_NAME = 'artifacts'
EPISODES_COLLECTION_NAME = 'episodes'
BENCHMARK_COLLECTION_NAME = 'benchmarks'
USER_COLLECTION_NAME = 'users'
USER_ROLES_COLLECTION_NAME = 'user_roles'

OPEN_ACCESS_MODE = os.getenv('OPEN_ACCESS_MODE', 'True')
OPEN_SIGNUP_MODE = os.getenv('OPEN_SIGNUP_MODE', 'True')

ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30)
REFRESH_TOKEN_EXPIRE_MINUTES = os.getenv('REFRESH_TOKEN_EXPIRE_MINUTES', 60 * 7 * 30)
