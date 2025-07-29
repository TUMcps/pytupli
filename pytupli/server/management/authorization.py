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
from pytupli.schema import (
    RESOURCE_TYPE,
    ArtifactMetadataItem,
    BaseFilter,
    BenchmarkHeader,
    DBItem,
    EpisodeHeader,
    FilterType,
    Membership,
    UserOut,
    UserRole,
    RIGHT,
    RIGHT_BUNDLE_CONTENT_CREATE,
    RIGHT_BUNDLE_CONTENT_DELETE,
    RIGHT_BUNDLE_CONTENT_READ,
    DEFAULT_ROLE,
)

http_bearer = HTTPBearer(auto_error=False)

SECRET_KEY = None  # Initialize as None to avoid errors at import time
ALGORITHM = 'HS256'

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


AUTH_ERROR = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail='You do not have sufficient permissions for this operation.',
)

GUEST_USER = UserOut(
    username='$$GUEST$$', memberships=[Membership(group='global', roles=[DEFAULT_ROLE.GUEST])]
)


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


async def authenticate_user(
    db_handler: MongoDBHandler,
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
) -> UserOut | None:
    # Do we allow open access?
    if credentials is None and not OPEN_ACCESS_MODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='You need to be authenticated with Bearer scheme to access the platform.',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    if credentials is None and OPEN_ACCESS_MODE:
        return None

    if credentials is not None:
        # check that the authentication scheme is Bearer
        if not credentials.scheme == 'Bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid authentication scheme',
                headers={'WWW-Authenticate': 'Bearer'},
            )

    token = credentials.credentials

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
    except InvalidTokenError:
        raise credentials_exception
    # get user from database
    user_dict = await db_handler.get_item(USER_COLLECTION_NAME, {'username': token_username})
    if user_dict is None:
        # No user found -> guest user
        return None

    return UserOut(**user_dict)


async def get_resource(
    resource_type: RESOURCE_TYPE, request_params: dict[str, str], db_handler: MongoDBHandler
) -> DBItem | None:
    """
    Returns a filter to get the resource by its type and id.
    """

    resource_id: str | None = request_params.get(resource_type.value.lower() + '_id')

    if resource_type == RESOURCE_TYPE.ARTIFACT:
        _, entry = await db_handler.download_file(
            ARTIFACTS_COLLECTION_NAME, {'metadata.id': resource_id}
        )
        entry = ArtifactMetadataItem(**entry) if entry else None
    elif resource_type == RESOURCE_TYPE.BENCHMARK:
        entry = await db_handler.get_item(BENCHMARK_COLLECTION_NAME, {'id': resource_id})
        entry = BenchmarkHeader(**entry) if entry else None
    elif resource_type == RESOURCE_TYPE.EPISODE:
        entry = await db_handler.get_item(EPISODES_COLLECTION_NAME, {'id': resource_id})
        entry = EpisodeHeader(**entry) if entry else None
    return entry


def get_user_memberships(user: UserOut) -> list[Membership]:
    """
    Returns the memberships of the user.
    """
    memberships = user.memberships.copy()

    if user.username == '$$GUEST$$':
        return memberships

    # Add default memberships for global and user-specific groups
    memberships.append(
        Membership(group='global', roles=[DEFAULT_ROLE.GLOBAL_MEMBER, DEFAULT_ROLE.CONTRIBUTOR])
    )
    memberships.append(
        Membership(group=user.username, roles=[DEFAULT_ROLE.USER_ADMIN, DEFAULT_ROLE.CONTRIBUTOR])
    )

    return memberships


async def get_rights_in_groups(
    db_handler: MongoDBHandler,
    user: UserOut,
    groups: list[str],
):
    user_memberships: list[Membership] = get_user_memberships(user)

    # Check if the published_in list contains any of the user's memberships and construct set of rights
    user_roles = set()
    user_rights = set()
    for membership in user_memberships:
        if membership.group in groups:
            user_roles.update(membership.roles)

    for role in user_roles:
        role_entry = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, {'role': role})
        if role_entry:
            user_rights.update(role_entry.get('rights', []))

    return user_rights


async def _extract_request_params(request: Request) -> dict[str, str]:
    """Extract parameters from both query params and request body."""
    params = {}

    # Get all query parameters
    for key, value in request.query_params.items():
        params[key] = value

    # Try to get parameters from request body
    try:
        import json

        body = await request.body()
        if body:
            body_data = json.loads(body)
            if isinstance(body_data, dict):
                params.update(body_data)
    except Exception:
        # If we can't parse the body, continue with just query params
        pass

    return params


def check_authorization(
    resource_type: RESOURCE_TYPE,
    required_right: RIGHT | None = None,
    requires_ownership: bool = False,
):
    async def _check_authorization(
        request: Request,
        db_handler: MongoDBHandler = Depends(get_db_handler),
        credentials: HTTPAuthorizationCredentials = Security(http_bearer),
    ) -> tuple[UserOut, DBItem | None]:
        # Extract parameters from request (query params + body)
        request_params = await _extract_request_params(request)

        # Special rule for signup
        if (
            OPEN_SIGNUP_MODE
            and resource_type == RESOURCE_TYPE.USER
            and required_right == RIGHT.USER_CREATE
        ):
            return GUEST_USER, None

        user: UserOut | None = await authenticate_user(db_handler, credentials)
        if not user:
            user = GUEST_USER

        if not required_right:  # no specific right required
            return user, None

        user_rights = set()
        resource = None

        if resource_type == RESOURCE_TYPE.ROLE:
            # Role management requires global admin rights
            user_rights = await get_rights_in_groups(db_handler, user, ['global'])

        if resource_type in [RESOURCE_TYPE.USER, RESOURCE_TYPE.GROUP]:
            # creating users/groups requires create rights on global scope
            if required_right in [RIGHT.USER_CREATE, RIGHT.GROUP_CREATE]:
                user_rights = await get_rights_in_groups(db_handler, user, ['global'])

            if required_right in [RIGHT.GROUP_DELETE, RIGHT.GROUP_READ, RIGHT.GROUP_UPDATE]:
                # for group management
                group_name = request_params.get('group_name')
                user_rights = await get_rights_in_groups(db_handler, user, ['global', group_name])

            if required_right in [RIGHT.USER_READ, RIGHT.USER_UPDATE, RIGHT.USER_DELETE]:
                # for user management
                if user.username == request_params.get('username'):
                    # user is accessing their own data
                    user_rights = await get_rights_in_groups(
                        db_handler, user, ['global', user.username]
                    )
                else:
                    # otherwise only global admin can access other users
                    user_rights = await get_rights_in_groups(db_handler, user, ['global'])

        if resource_type in [
            RESOURCE_TYPE.ARTIFACT,
            RESOURCE_TYPE.BENCHMARK,
            RESOURCE_TYPE.EPISODE,
        ]:
            if required_right in RIGHT_BUNDLE_CONTENT_CREATE:
                user_rights = await get_rights_in_groups(
                    db_handler, user, ['global', user.username]
                )

            elif required_right in RIGHT_BUNDLE_CONTENT_READ + RIGHT_BUNDLE_CONTENT_DELETE:
                resource: DBItem | None = await get_resource(
                    resource_type, request_params, db_handler
                )
                if resource is None:
                    return user, None

                has_ownership = resource.created_by == user.username

                if requires_ownership and not has_ownership:
                    raise AUTH_ERROR

                if has_ownership:
                    user_rights = set(RIGHT_BUNDLE_CONTENT_DELETE + RIGHT_BUNDLE_CONTENT_READ)
                else:
                    if required_right in RIGHT_BUNDLE_CONTENT_READ:
                        # read rights in published_in groups
                        user_rights = await get_rights_in_groups(
                            db_handler, user, [*resource.published_in]
                        )
                    elif required_right in RIGHT_BUNDLE_CONTENT_DELETE:
                        # only global admin or creator can delete
                        user_rights = await get_rights_in_groups(db_handler, user, ['global'])

        if required_right not in user_rights:
            raise AUTH_ERROR

        return user, resource

    return _check_authorization


async def check_group_permissions(
    db_handler: MongoDBHandler, user: UserOut, group_name: str, required_rights: list[RIGHT]
) -> None:
    """Check if user has required rights in the specified group."""
    user_rights = await get_rights_in_groups(db_handler, user, [group_name])

    # Check if user has any of the required rights
    has_permission = any(right in user_rights for right in required_rights)

    if not has_permission:
        rights_str = ' or '.join([right.value for right in required_rights])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Insufficient permissions in group: {group_name}. Required: {rights_str}',
        )


async def inject_read_permission_filter(
    filter: BaseFilter,
    user: UserOut,
    resource_type: RESOURCE_TYPE,
    db_handler: MongoDBHandler,
    prefix_path: str = '',
) -> BaseFilter | None:
    """
    Function to inject read permission filter into the user-provided filter.

    Args:
        filter (BaseFilter): The user-provided filter.
        user (User): The user object.
        db_handler (MongoDBHandler): The database handler (to fetch user rights).
        prefix_path (str, optional): Prefix path to the fields relevant for permissions, e.g. "metadata".
            Defaults to "".

    Returns:
        BaseFilter | None: The modified filter or None if no filter is needed.
    """

    # Determine the appropriate read right based on resource type
    match resource_type:
        case RESOURCE_TYPE.ARTIFACT:
            resource_read_right = RIGHT.ARTIFACT_READ
        case RESOURCE_TYPE.BENCHMARK:
            resource_read_right = RIGHT.BENCHMARK_READ
        case RESOURCE_TYPE.EPISODE:
            resource_read_right = RIGHT.EPISODE_READ
        case RESOURCE_TYPE.USER:
            resource_read_right = RIGHT.USER_READ
        case RESOURCE_TYPE.GROUP:
            resource_read_right = RIGHT.GROUP_READ
        case _:
            raise ValueError(f'Unsupported resource type: {resource_type}')

    # In which groups does the user have read rights?
    user_roles = {
        membership.group: role
        for membership in get_user_memberships(user)
        for role in membership.roles
    }
    user_read_groups = []
    for group, role in user_roles.items():
        role_entry: dict = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, {'role': role})
        if role_entry:
            role_obj = UserRole(**role_entry)
            if resource_read_right in role_obj.rights:
                user_read_groups.append(group)

    prefix_path = prefix_path + '.' if prefix_path else ''
    allowed_filter = BaseFilter(type='IN', key=f'{prefix_path}published_in', value=user_read_groups)

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
