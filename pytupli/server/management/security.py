# JWT logic following the official FastAPI documentation https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext

from pytupli.server.api.dependencies import get_db_handler
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
    ARTIFACTS_COLLECTION_NAME,
    EPISODES_COLLECTION_NAME,
    USER_COLLECTION_NAME,
    OPEN_ACCESS_MODE,
    OPEN_SIGNUP_MODE,
    USER_ROLES_COLLECTION_NAME,
)
from pytupli.schema import BaseFilter, FilterType, User

http_bearer = HTTPBearer(auto_error=False)

# permissions needed for each endpoint
permissions_dict = {
    '/artifacts/upload': 'write',
    '/artifacts/download': 'read_all',
    '/artifacts/delete': 'delete_all',
    '/artifacts/publish': 'write',
    '/artifacts/list': None,  # public endpoint
    '/benchmarks/create': 'write',
    '/benchmarks/load': 'read_all',
    '/benchmarks/list': None,  # public endpoint
    '/benchmarks/delete': 'delete_all',
    '/benchmarks/publish': 'write',
    '/episodes/record': 'write',
    '/episodes/list': None,  # public endpoint
    '/episodes/publish': 'write',
    '/episodes/delete': 'delete_all',
    '/access/signup': None if OPEN_SIGNUP_MODE else 'user_management',
    '/access/list-users': 'user_management',
    '/access/list-roles': 'user_management',
    '/access/change-password': 'user_management',
    '/access/change-roles': 'user_management',
    '/access/delete-user': 'user_management',
    '/access/refresh-token': None,
}

SECRET_KEY = None  # Initialize as None to avoid errors at import time
ALGORITHM = 'HS256'

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


def initialize_secret_key():
    """
    Initializes secret key needed for token creation and validation.
    """
    global SECRET_KEY
    SECRET_KEY = os.getenv('API_SECRET_KEY')
    if SECRET_KEY is None:
        raise ValueError('API_SECRET_KEY environment variable must be set!')


def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    """
    return pwd_context.hash(password)  # is automatically salted


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a password against a hashed password.\n
    Returns True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_token(data: dict, expires_delta: timedelta = timedelta(minutes=15)) -> str:
    """
    Function to create an access or refresh token.
    Args:
        data (dict): The data to be encoded in the token.
        expires_delta (timedelta): The expiration time of the token.
            Default is 15 minutes.
    Returns:
        str: The generated token.
    """
    to_encode = data.copy()
    # set exiration time of token
    expire = datetime.now(timezone.utc) + expires_delta

    to_encode.update({'exp': expire})
    # create token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def validate_token(token: str, db_handler: MongoDBHandler) -> User:
    """
    Function to validate a JWT token.
    Args:
        token (str): The token to be validated.
        db_handler (AbstractDatabaseHandler): The database handler.
    Returns:
        tuple: The username and rights of the user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        # decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_username: str = payload.get('issuer')
        # if "issuer" is not in payload, token is invalid as it does not come from our token creation logic
        if token_username is None:
            raise credentials_exception
    except InvalidTokenError as e:
        print(e)
        raise credentials_exception
    # get user from database
    user_dict = await db_handler.get_item(USER_COLLECTION_NAME, {'username': token_username})
    if user_dict is None:
        raise credentials_exception

    # Create a User object
    user = User(**user_dict)
    return user


def check_entry_owner_private(user, entry) -> bool:
    """
    Funcion to check if the user is the owner of a private entry.
    Args:
        user (str): The username of the user.
        entry (dict): The entry to be checked.
    Returns:
        bool: True if the user is the owner of the entry and the entry is private, False otherwise.
    """
    if entry is None:
        return False
    return entry['created_by'] == user and not entry['is_public']


async def get_user_rights(user: str | dict | User, db_handler: MongoDBHandler) -> list[str]:
    """
    Function to get the rights of a user.
    Args:
        user (str | dict | User): The username of the user, user dict, or User object.
        db_handler (AbstractDatabaseHandler): The database handler.
    Returns:
        list: The rights of the user.
    """
    # Convert string username to User object
    if isinstance(user, str):
        user_dict = await db_handler.get_item(USER_COLLECTION_NAME, {'username': user})
        if user_dict is None:
            return []
        user = User(**user_dict)
    # Convert dict to User object
    elif isinstance(user, dict):
        user = User(**user)

    if user is None:
        return []

    rights = []
    for role in user.roles:
        role_entry = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, {'role': role})
        if not role_entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f'Role {role} of user invalid. Role not found!',
            )
        rights = rights + role_entry['rights']

    return rights


async def get_entry(request: Request, db_handler: MongoDBHandler):
    """
    Function to get the entry based on the submitted id from the request from the database.
    Args:
        request (Request): The request object.
        db_handler (MongoDBHandler): The database handler.
    Returns:
        dict: The entry from the database.
    """
    entry = None
    if 'artifact' in request.url.path:
        if 'artifact_id' not in request.query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='artifact_id query parameter is missing',
            )
        id = request.query_params['artifact_id']
        _, entry = await db_handler.download_file(ARTIFACTS_COLLECTION_NAME, {'metadata.id': id})
        return entry
    elif 'benchmark' in request.url.path:
        if 'benchmark_id' not in request.query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='benchmark_id query parameter is missing',
            )
        id = request.query_params['benchmark_id']
        entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, {'id': id})
        return entry
    elif 'episodes' in request.url.path:
        if 'episode_id' not in request.query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='episode_id query parameter is missing',
            )
        id = request.query_params['episode_id']
        entry = await db_handler.get_item(EPISODES_COLLECTION_NAME, {'id': id})
        return entry
    elif 'access' in request.url.path:
        return None


def check_entry_is_public(entry):
    """
    Function to check if the entry is public.
    """
    if not entry:
        return False
    return entry['is_public']


async def check_authentication(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
    db_handler: MongoDBHandler = Depends(get_db_handler),
) -> User:
    """
    Function to check if the user is authenticated and has the necessary permissions to access the endpoint.
    Args:
        request (Request): The request object.
        credentials (HTTPAuthorizationCredentials): The credentials of the user.
        db_handler (AbstractDatabaseHandler): The database handler.
    Returns:
        User: The authenticated user
    Throws:
        HTTPException: If the user is not authenticated or does not have the necessary permissions.
    """

    # Do we allow open access?
    if credentials is None and not OPEN_ACCESS_MODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='You need to be authenticated with Bearer scheme to access the platform.',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # if the request is unauthorized, the user is a guest
    # for that, credentials would be None therefore we don't check the scheme
    if credentials is not None:
        # check that the authentication scheme is Bearer
        if not credentials.scheme == 'Bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid authentication scheme',
                headers={'WWW-Authenticate': 'Bearer'},
            )

    # get permissions needed for the currently requested endpoint
    if request.url.path not in permissions_dict:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'No rights found for endpoint {request.url.path}',
        )

    # get entity
    is_delete_request = '/delete' in request.url.path
    is_download_request = '/download' in request.url.path or '/load' in request.url.path
    is_list_request = '/list' in request.url.path
    is_publish_request = '/publish' in request.url.path

    is_public = False
    entry = None

    if is_download_request or is_delete_request or is_publish_request:
        entry = await get_entry(request, db_handler)

    if (is_download_request or is_publish_request) and entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Object not found',
        )

    if is_download_request or is_publish_request:
        is_public = check_entry_is_public(entry)

    # the list endpoints are always public
    # just what gets returned is depending on the user rights
    if is_list_request and 'access' not in request.url.path:
        is_public = True

    # special rule for signup
    if request.url.path == '/access/signup' and OPEN_SIGNUP_MODE:
        is_public = True

    # if the data to read is public, no token is needed
    if is_public and credentials is None:
        # Return guest user
        guest_user = User(username='guest', password='', roles=[])
        return guest_user
    elif not is_public and credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='You need to be authenticated with Bearer scheme to access this endpoint.',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # get user from the token
    token = credentials.credentials
    user: User = await validate_token(token, db_handler)

    # We do not fail a delete operation if the entry does not exist
    # unless someone is trying to delete a user account
    if is_delete_request and entry is None and 'access' not in request.url.path:
        return user

    # Get user rights
    user_rights = await get_user_rights(user, db_handler)

    if (is_download_request or is_publish_request) and (
        is_public or check_entry_owner_private(user.username, entry)
    ):
        return user
    elif is_publish_request:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You do not have permission to access this endpoint.',
        )

    # you are only allowed to delete your own private documents
    if is_delete_request and check_entry_owner_private(user.username, entry):
        return user

    needed_perm = permissions_dict[request.url.path]
    # if user rights include the needed permissions, the user is allowed to do the operation
    if needed_perm is None or (user_rights and needed_perm in user_rights):
        return user

    # check if its a management endpoint (starting with auth)
    if 'access' in request.url.path:
        # if the user wants to delete their own account, they are allowed to do so
        # (but we prevent the initial admin account from being deleted)
        if is_delete_request:
            if request.query_params['username'] == user.username and user.username != 'admin':
                return user
        # if the user wants to change their own password, they are allowed to do so
        if request.url.path == '/access/change-password':
            body = await request.json()
            if body['username'] == user.username:
                return user

    # if none of the above conditions are met, the user is not allowed to access the endpoint
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail='You do not have permission to access this endpoint.',
    )


async def inject_read_permission_filter(
    filter: BaseFilter,
    user: User,
    db_handler: MongoDBHandler,
    prefix_path: str = '',
) -> BaseFilter | None:
    """
    Function to inject read permission filter into the user-provided filter.
    If the user has 'read_all' rights, no filter is applied.

    Args:
        filter (BaseFilter): The user-provided filter.
        user (User): The user object.
        db_handler (MongoDBHandler): The database handler (to fetch user rights).
        prefix_path (str, optional): Prefix path to the fields relevant for permissions, e.g. "metadata".
            Defaults to "".

    Returns:
        BaseFilter | None: The modified filter or None if no filter is needed.
    """
    if 'read_all' not in (await get_user_rights(user, db_handler)):
        prefix_path = prefix_path + '.' if prefix_path else ''
        allowed_filter = BaseFilter(
            type='OR',
            filters=[
                BaseFilter(type='EQ', key=f'{prefix_path}created_by', value=user.username),
                BaseFilter(type='EQ', key=f'{prefix_path}is_public', value=True),
            ],
        )
    else:
        allowed_filter = None

    # Inject allowed filter into the user-provided filter
    if filter:
        if allowed_filter:
            filter = BaseFilter(
                type=FilterType.AND,
                filters=[allowed_filter, filter],
            )
    else:
        filter = allowed_filter

    return filter
