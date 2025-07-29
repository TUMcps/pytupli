import logging

from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import UserRole, UserOut, RESOURCE_TYPE, RIGHT
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import USER_COLLECTION_NAME, USER_ROLES_COLLECTION_NAME
from pytupli.server.management.authorization import check_authorization

router = APIRouter()
logging.getLogger('passlib').setLevel(logging.ERROR)


@router.post('/create')
async def create_role(
    role: UserRole,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.ROLE, RIGHT.ROLE_MANAGEMENT)
    ),
) -> UserRole:
    """Create a new role."""
    try:
        # check if role already exists
        query = {'role': role.role}
        role_entry = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create role: {str(e)}',
        )

    if role_entry:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Role already exists')

    try:
        # create the role in the db
        await db_handler.create_item(USER_ROLES_COLLECTION_NAME, role.model_dump())
        return role
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create role: {str(e)}',
        )


@router.get('/list')
async def list_roles(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.ROLE, RIGHT.ROLE_MANAGEMENT)
    ),
) -> list[UserRole]:
    """List all roles."""
    try:
        roles = await db_handler.get_items(USER_ROLES_COLLECTION_NAME, {})
        role_objects = [UserRole(**role) for role in roles]
        return role_objects
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list roles: {str(e)}',
        )


@router.delete('/delete')
async def delete_role(
    role_name: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.ROLE, RIGHT.ROLE_MANAGEMENT)
    ),
) -> None:
    """Delete a role."""
    try:
        # check if role exists
        role_query = {'role': role_name}
        role_entry = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, role_query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete role: {str(e)}',
        )

    if not role_entry:
        return

    try:
        # Check if any users have this role and remove it
        users = await db_handler.get_items(USER_COLLECTION_NAME, {})
        for user in users:
            user_obj = UserOut(**user)
            updated_memberships = []
            modified = False

            for membership in user_obj.memberships:
                updated_roles = [role for role in membership.roles if role != role_name]
                if len(updated_roles) != len(membership.roles):
                    modified = True
                    # Only keep membership if it still has roles
                    if updated_roles:
                        membership.roles = updated_roles
                        updated_memberships.append(membership)
                else:
                    updated_memberships.append(membership)

            if modified:
                # Update user if memberships changed
                update = {'$set': {'memberships': [m.model_dump() for m in updated_memberships]}}
                await db_handler.update_item(
                    USER_COLLECTION_NAME, {'username': user_obj.username}, update
                )

        # Delete the role
        await db_handler.delete_item(USER_ROLES_COLLECTION_NAME, role_query)
        return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete role: {str(e)}',
        )
