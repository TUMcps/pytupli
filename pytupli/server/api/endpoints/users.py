import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import Token, User, UserCredentials, UserOut, RESOURCE_TYPE, RIGHT
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (
    BENCHMARK_COLLECTION_NAME,
    ARTIFACTS_COLLECTION_NAME,
    EPISODES_COLLECTION_NAME,
    USER_COLLECTION_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_MINUTES,
)
from pytupli.server.management.authorization import (
    check_authorization,
    create_token,
    hash_password,
    verify_password,
)

router = APIRouter()
logging.getLogger('passlib').setLevel(logging.ERROR)


@router.post('/create', response_model=UserOut)
async def create_user(
    user: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.USER, RIGHT.USER_CREATE)),
) -> User:
    """Create a new user."""
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
            'memberships': [],
        }

        # create the user in the db
        await db_handler.create_item(USER_COLLECTION_NAME, db_user)
        return User(**db_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create user: {str(e)}',
        )


@router.get('/list', response_model=list[UserOut])
async def list_users(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.USER, RIGHT.USER_READ)),
) -> list[UserOut]:
    """List all users."""
    try:
        # get all users
        users = await db_handler.get_items(USER_COLLECTION_NAME, {})
        user_objects = [UserOut(**user) for user in users]
        return user_objects
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list users: {str(e)}',
        )


@router.delete('/delete')
async def delete_user(
    username: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.USER, RIGHT.USER_DELETE)),
) -> None:
    """Delete a user and all their content."""
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
        # delete all purely private contents of user
        docs_query = {'created_by': username, 'published_in': [username]}
        _ = await db_handler.delete_items(BENCHMARK_COLLECTION_NAME, docs_query)
        _ = await db_handler.delete_files(ARTIFACTS_COLLECTION_NAME, docs_query)
        _ = await db_handler.delete_items(EPISODES_COLLECTION_NAME, docs_query)

        # Delete the user
        await db_handler.delete_item(USER_COLLECTION_NAME, user_query)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete user: {str(e)}',
        )


@router.put('/change-password')
async def change_password(
    user: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.USER, RIGHT.USER_UPDATE)),
) -> None:
    """Change user password."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User does not exist')

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


@router.post('/token')
async def login_for_token(
    form_data: UserCredentials,
    db_handler: MongoDBHandler = Depends(get_db_handler),
) -> dict[str, Token]:
    """Login and get access and refresh tokens."""
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
        # create token
        access_token = create_token(
            data={'issuer': form_data.username},
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
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.USER, RIGHT.USER_READ)
    ),
) -> Token:
    """Refresh access token using refresh token."""
    # refresh token is already validated by authenticate_user
    # we can just return a new access token
    user, _ = auth_result
    access_token = create_token(
        data={'issuer': user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return Token(token=access_token, token_type='bearer')
