import io
from pymongo.results import DeleteResult
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

from pytupli.schema import BaseFilter, FilterType


class MongoDBHandler:
    """
    DB Handler connecting to MongoDB
    """

    @staticmethod
    def convert_filter_to_query(filter_obj: BaseFilter) -> dict:
        """
        Convert a filter object or list of filters to MongoDB query format.
        Args:
            filter_obj (BaseFilter): The filter object to convert. This can be a single filter or a composite filter
                containing multiple sub-filters.
        Returns:
            dict: A dictionary representing the MongoDB query.
        Supported Filter Types:
            - FilterType.AND: Logical AND of multiple filters.
            - FilterType.OR: Logical OR of multiple filters.
            - FilterType.EQ: Equality filter.
            - FilterType.GEQ: Greater than or equal filter.
            - FilterType.LEQ: Less than or equal filter.
            - FilterType.GT: Greater than filter.
            - FilterType.LT: Less than filter.
            - FilterType.NE: Not equal filter
        """

        match filter_obj.type:
            case FilterType.AND:
                if not filter_obj.filters:
                    return {}
                return {
                    '$and': [MongoDBHandler.convert_filter_to_query(f) for f in filter_obj.filters]
                }

            case FilterType.OR:
                if not filter_obj.filters:
                    return {}
                return {
                    '$or': [MongoDBHandler.convert_filter_to_query(f) for f in filter_obj.filters]
                }

            case FilterType.EQ:
                return {filter_obj.key: filter_obj.value}

            case FilterType.GEQ:
                return {filter_obj.key: {'$gte': filter_obj.value}}

            case FilterType.LEQ:
                return {filter_obj.key: {'$lte': filter_obj.value}}

            case FilterType.GT:
                return {filter_obj.key: {'$gt': filter_obj.value}}

            case FilterType.LT:
                return {filter_obj.key: {'$lt': filter_obj.value}}

            case FilterType.NE:
                return {filter_obj.key: {'$ne': filter_obj.value}}

        return {}

    def __init__(self, connection_string: str, database_name: str):
        self.mongodb_client = AsyncIOMotorClient(connection_string)
        self.database = self.mongodb_client[database_name]

    async def query_items(
        self, collection: str, filter: BaseFilter | None, projection: dict = None
    ):
        """
        Queries items in the collection based on the filter.\n
        The filter is an object of the BaseFilter class, providing complex filtering logic.
        """
        query = self.convert_filter_to_query(filter) if filter else {}
        results = await self.get_items(collection, query, projection)
        return results

    async def create_item(self, collection: str, item: dict):
        return await self.database[collection].insert_one(item)

    async def create_items(self, collection: str, items: list):
        return await self.database[collection].insert_many(items)

    async def get_item(self, collection: str, query: dict[str, any]):
        return await self.database[collection].find_one(query)

    async def get_items(
        self, collection: str, query: dict[str, any], projection: dict[str, any] = None
    ):
        cursor = self.database[collection].find(query, projection=projection)
        cursor_list = await cursor.to_list()
        return cursor_list

    async def update_item(self, collection: str, filter: dict[str, any], update: dict[str, any]):
        result = await self.database[collection].update_one(filter, update)
        if result.matched_count == 0:
            raise FileNotFoundError(
                f'No document matching the filter {filter} found in the collection {collection}.'
            )
        return result

    async def update_items(self, collection: str, filter: dict[str, any], update: dict[str, any]):
        result = await self.database[collection].update_many(filter, update)
        if result.matched_count == 0:
            raise FileNotFoundError(
                f'No documents matching the filter {filter} found in the collection {collection}.'
            )
        return result

    async def delete_item(self, collection: str, filter: dict[str, any]) -> DeleteResult:
        return await self.database[collection].delete_one(filter)

    async def delete_items(self, collection: str, filter: dict[str, any]) -> DeleteResult:
        return await self.database[collection].delete_many(filter)

    async def upload_file(self, collection: str, file: bytes, file_name: str, metadata: dict):
        filter = {'metadata.hash': metadata['hash']}
        fs = AsyncIOMotorGridFSBucket(self.database, collection)
        documents = await fs.find(filter).to_list()
        if len(documents) > 0:
            return documents[0]['metadata']

        await fs.upload_from_stream(file_name, file, metadata=metadata)
        return metadata

    async def download_file(self, collection: str, filter: dict[str, any]):
        fs = AsyncIOMotorGridFSBucket(self.database, collection)
        cursor_list = await self._find_documents(fs, collection, filter)
        if len(cursor_list) == 0:
            return None, None
        if len(cursor_list) > 1:
            raise Exception(
                f'Multiple Objects for filter {filter} found in Database!\
                              If you want to download multiple files, use download_files\
                              (self, collection: str, filter: dict[str, str]).'
            )

        data = cursor_list[0]
        filename = data['filename']
        buffer = io.BytesIO()
        await fs.download_to_stream_by_name(filename, buffer)
        buffer.seek(0)
        file_data = buffer.read()
        return file_data, data['metadata']

    async def download_files(self, collection: str, filter: dict[str, str]):
        fs = AsyncIOMotorGridFSBucket(self.database, collection)
        cursor_list = await self._find_documents(fs, collection, filter)
        if len(cursor_list) == 0:
            return []

        downloaded_files = []

        for data in cursor_list:
            filename = data['filename']
            buffer = io.BytesIO()
            await fs.download_to_stream_by_name(filename, buffer)
            buffer.seek(0)
            file_data = buffer.read()
            downloaded_files.append({'data': file_data, 'metadata': data['metadata']})

        return downloaded_files

    async def delete_file(self, collection: str, filter: dict[str, str]):
        fs = AsyncIOMotorGridFSBucket(self.database, collection)
        cursor_list = await self._find_documents(fs, collection, filter)
        if len(cursor_list) == 0:
            return DeleteResultMock(0)
        if len(cursor_list) > 1:
            raise Exception(
                f'Multiple Objects for filter {filter} found in Database!\
                              If you want to delete multiple files, use delete_files\
                              (self, collection: str, filter: dict[str, str]).'
            )
        data = cursor_list[0]
        file_id = data['_id']
        try:
            await fs.delete(file_id)
            return DeleteResultMock(1)
        except Exception as e:
            print(f'Error deleting file: {e}')
            return DeleteResultMock(0)

    async def delete_files(self, collection: str, filter: dict[str, str]):
        fs = AsyncIOMotorGridFSBucket(self.database, collection)
        cursor_list = await self._find_documents(fs, collection, filter)
        if len(cursor_list) == 0:
            return DeleteResultMock(0)
        deleted_count = 0
        try:
            for data in cursor_list:
                file_id = data['_id']
                await fs.delete(file_id)
                # increment deleted count, if file was deleted successfully
                deleted_count += 1
            return DeleteResultMock(deleted_count)
        except Exception as e:
            print(f'Error deleting files: {e}')
            return DeleteResultMock(deleted_count)

    async def ping(self):
        res = await self.database.command('ping')
        return res['ok']

    def close_connection(self):
        self.mongodb_client.close()

    async def _find_documents(
        self, fs: AsyncIOMotorGridFSBucket, collection: str, filter: dict[str, str]
    ):
        """
        Helper function finding a document in the GridFS collection.\n
        Will throw FileNotFoundError if no document matching the filer is found.
        """
        cursor = fs.find(filter)
        cursor_list = await cursor.to_list()
        return cursor_list


class DeleteResultMock:
    """
    Class mocking the returned result of a delete action in MongoDB.\n
    Contains the number of deleted documents.
    """

    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


class UpdateResultMock:
    """
    Class mocking the returned result of an update action in MongoDB.\n
    Contains the number of matched and modified documents.
    """

    def __init__(self, matched_count: int, modified_count: int):
        self.matched_count = matched_count
        self.modified_count = modified_count
