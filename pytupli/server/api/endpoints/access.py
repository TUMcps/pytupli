# JWT logic following the official FastAPI documentation https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import Token, User, UserCredentials, UserOut, UserRole, UserRoleUpdate
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (  # environment variables, constants and Handler Factory
    BENCHMARK_COLLECTION_NAME,
    ARTIFACTS_COLLECTION_NAME,
    EPISODES_COLLECTION_NAME,
    USER_COLLECTION_NAME,
    USER_ROLES_COLLECTION_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_MINUTES,
)
from pytupli.server.management.security import (
    check_authentication,
    create_token,
    get_user_rights,
    hash_password,
    verify_password,
)

router = APIRouter()
logging.getLogger('passlib').setLevel(logging.ERROR)


@router.post('/signup', response_model=UserOut)
async def user_signup(
    user: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> User:
    user_entry = None
    try:
        # check if username is already taken
        query = {'username': user.username}
        user_entry = await db_handler.get_item(USER_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create user: {str(e)}',
        )
    if user_entry:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='User already exists')

    try:
        # Hash the user's password
        hashed_password = hash_password(user.password)

        # Create the user object for database insertion
        db_user = {
            'username': user.username,
            'password': hashed_password,  # Store only the hashed password
            'roles': ['standard_user'],  # use standard (read only rights) as default role
        }

        # create the user in the db
        await db_handler.create_item(USER_COLLECTION_NAME, db_user)
        return User(**db_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create user: {str(e)}',
        )


@router.get('/list-users', response_model=list[UserOut])
async def list_users(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> list[User]:
    try:
        # get all users
        users = await db_handler.get_items(USER_COLLECTION_NAME, {})
        user_objects = [User(**user) for user in users]
        return user_objects
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list users: {str(e)}',
        )


@router.get('/list-roles')
async def list_roles(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> list[UserRole]:
    try:
        roles = await db_handler.get_items(USER_ROLES_COLLECTION_NAME, {})
        role_objects = [UserRole(**role) for role in roles]
        return role_objects
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list roles: {str(e)}',
        )


@router.put('/change-password')
async def user_change_password(
    user: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    try:
        # check if user exists
        query = {'username': user.username}
        user_entry = await db_handler.get_item(USER_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to change password of user: {str(e)}',
        )

    if not user_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User does not exists')

    try:
        # Hash the user's new password
        hashed_password = hash_password(user.password)

        # Create the update dictionary
        update = {'$set': {'password': hashed_password}}  # Update the hashed password

        # update the user in the db
        await db_handler.update_item(USER_COLLECTION_NAME, query, update)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to change password: {str(e)}',
        )


@router.put('/change-roles')
async def user_change_roles(
    user: UserRoleUpdate,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    try:
        # check if user exists
        query = {'username': user.username}
        user_entry = await db_handler.get_item(USER_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to change roles of user: {str(e)}',
        )

    if not user_entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User does not exists')

    try:
        # Create the update dictionary
        update = {'$set': {'roles': user.roles}}

        # update the user in the db
        await db_handler.update_item(USER_COLLECTION_NAME, query, update)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to change roles: {str(e)}',
        )


@router.delete('/delete-user')
async def delete_user_and_content(
    username: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _=Depends(check_authentication),
) -> None:
    try:
        # check if user exists
        user_query = {'username': username}
        user_entry = await db_handler.get_item(USER_COLLECTION_NAME, user_query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete user: {str(e)}',
        )

    if not user_entry:
        return

    try:
        # delete all private contents of user
        docs_query = {'created_by': username, 'is_public': False}
        benchmarks = await db_handler.delete_items(BENCHMARK_COLLECTION_NAME, docs_query)
        data_sources = await db_handler.delete_files(ARTIFACTS_COLLECTION_NAME, docs_query)
        tuples = await db_handler.delete_items(EPISODES_COLLECTION_NAME, docs_query)

        # Delete the user
        await db_handler.delete_item(USER_COLLECTION_NAME, user_query)

        return {
            'message': (
                f'Deleted user {username} and all their content. '
                + f'Deleted {benchmarks.deleted_count} benchmarks, '
                + f'{data_sources.deleted_count} data sources, and '
                + f'{tuples.deleted_count} tuples'
            )
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete user: {str(e)}',
        )


@router.post('/token')
async def login_for_token(
    form_data: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
) -> dict[str, Token]:
    incorrect_auth_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Incorrect username or password',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        # check if user exists
        user_query = {'username': form_data.username}
        user_entry = await db_handler.get_item(USER_COLLECTION_NAME, user_query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to get token for user: {str(e)}',
        )

    if not user_entry:
        raise incorrect_auth_exception

    # check password
    if not verify_password(form_data.password, user_entry['password']):
        raise incorrect_auth_exception

    try:
        # get user rights attached to all roles of user
        rights = await get_user_rights(user_entry, db_handler)
        # create token
        # we currently attach the roles to the JWT token to not have to query the database for every request of the user
        access_token = create_token(
            data={'issuer': form_data.username, 'rights': rights},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        refresh_token = create_token(
            data={'issuer': form_data.username},
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
        )
        return {
            'access_token': Token(token=access_token, token_type='bearer'),
            'refresh_token': Token(token=refresh_token, token_type='bearer'),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to login: {str(e)}'
        )


@router.post('/refresh-token')
async def refresh_token(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    user: User = Depends(check_authentication),
) -> Token:
    # refresh token is already validated by check_authentication
    # we can just return a new access token

    # The refresh token does not have any rights attached to it
    # So we need to fetch user rights first
    user_rights = await get_user_rights(user, db_handler)

    access_token = create_token(
        data={'issuer': user.username, 'rights': user_rights},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return access_token
