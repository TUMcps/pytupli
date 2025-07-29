import logging

from fastapi import APIRouter, Depends, HTTPException, status

from pytupli.server.api.dependencies import get_db_handler
from pytupli.schema import (
    DEFAULT_ROLE,
    BaseFilter,
    FilterType,
    Group,
    GroupMembershipQuery,
    GroupWithMembers,
    Membership,
    UserOut,
    RESOURCE_TYPE,
    RIGHT,
)
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (
    GROUPS_COLLECTION_NAME,
    USER_COLLECTION_NAME,
    USER_ROLES_COLLECTION_NAME,
)
from pytupli.server.management.authorization import (
    check_authorization,
    get_user_memberships,
    get_rights_in_groups,
)

router = APIRouter()
logging.getLogger('passlib').setLevel(logging.ERROR)


@router.post('/create')
async def create_group(
    group: Group,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.GROUP, RIGHT.GROUP_CREATE)
    ),
) -> Group:
    """Create a new group."""
    current_user, _ = auth_result
    try:
        # check if group name is already taken
        query = {'name': group.name}
        group_entry = await db_handler.get_item(GROUPS_COLLECTION_NAME, query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create group: {str(e)}',
        )

    if group_entry:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Group already exists')

    try:
        # create the group in the db
        await db_handler.create_item(GROUPS_COLLECTION_NAME, group.model_dump())

        # add admin membership for the creator
        creator_membership = Membership(
            group=group.name, roles=[DEFAULT_ROLE.GROUP_ADMIN, DEFAULT_ROLE.CONTRIBUTOR]
        )

        # Update user memberships
        updated_memberships = current_user.memberships + [creator_membership]
        update = {'$set': {'memberships': [m.model_dump() for m in updated_memberships]}}
        await db_handler.update_item(
            USER_COLLECTION_NAME, {'username': current_user.username}, update
        )

        return group
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to create group: {str(e)}',
        )


@router.get('/list')
async def list_groups(
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.GROUP)),
) -> list[Group]:
    """List all groups based on user's access rights."""
    current_user, _ = auth_result

    try:
        # Check if user has global GROUP_READ right
        global_rights = await get_rights_in_groups(db_handler, current_user, ['global'])
        has_global_group_read = RIGHT.GROUP_READ in global_rights

        if has_global_group_read:
            # Get all groups
            accessible_groups = await db_handler.get_items(GROUPS_COLLECTION_NAME, {})
        else:
            # User can only see groups they are a member of
            user_memberships = get_user_memberships(current_user)
            filter = BaseFilter(
                type=FilterType.OR,
                filters=[
                    BaseFilter(field='name', value=membership.group)
                    for membership in user_memberships
                ],
            )
            accessible_groups = await db_handler.query_items(GROUPS_COLLECTION_NAME, filter)

        return [Group(**group) for group in accessible_groups]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to list groups: {str(e)}',
        )


@router.get('/read')
async def read_group(
    group_name: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.GROUP, RIGHT.GROUP_READ)
    ),
) -> GroupWithMembers:
    """Read a specific group with its members if user has appropriate access."""
    current_user, _ = auth_result

    try:
        group = await db_handler.get_item(GROUPS_COLLECTION_NAME, {'name': group_name})
        if not group:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Group not found')

        group = Group(**group)

        # Check if user has global USER_READ right
        global_rights = await get_rights_in_groups(db_handler, current_user, ['global', group.name])
        has_group_read = RIGHT.USER_READ in global_rights

        # Initialize the group response
        group_with_members = GroupWithMembers(**group.model_dump(), members=[])

        # If user has USER_READ in the group, include members
        # This could be made more efficient by expanding the query concept
        if has_group_read:
            # Get all users and filter those who are members of this group
            all_users = await db_handler.get_items(USER_COLLECTION_NAME, {})

            for user_data in all_users:
                user_obj = UserOut(**user_data)
                # Check if user is a member of this group
                for membership in user_obj.memberships:
                    if membership.group == group_name:
                        group_with_members.members.append(user_obj.username)
                        break

        return group_with_members

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to read group: {str(e)}',
        )


@router.delete('/delete')
async def delete_group(
    group_name: str,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    _: tuple[UserOut, None] = Depends(check_authorization(RESOURCE_TYPE.GROUP, RIGHT.GROUP_DELETE)),
) -> None:
    """Delete a group."""

    group = await db_handler.get_item(GROUPS_COLLECTION_NAME, {'name': group_name})
    if not group:
        return

    try:
        # Remove group membership from all users
        # This could be made more efficient by expanding the query concept
        users = await db_handler.get_items(USER_COLLECTION_NAME, {})
        for user in users:
            user_obj = UserOut(**user)
            updated_memberships = [
                membership for membership in user_obj.memberships if membership.group != group_name
            ]
            if len(updated_memberships) != len(user_obj.memberships):
                # Update user if memberships changed
                update = {'$set': {'memberships': [m.model_dump() for m in updated_memberships]}}
                await db_handler.update_item(
                    USER_COLLECTION_NAME, {'username': user_obj.username}, update
                )

        # Delete the group
        await db_handler.delete_item(GROUPS_COLLECTION_NAME, {'name': group_name})
        return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to delete group: {str(e)}',
        )


@router.post('/add-members')
async def add_members(
    membership_query: GroupMembershipQuery,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.GROUP, RIGHT.GROUP_UPDATE)
    ),
) -> GroupWithMembers:
    """Add members to a group with specified roles."""
    current_user, _ = auth_result

    group = await db_handler.get_item(GROUPS_COLLECTION_NAME, {'name': membership_query.group_name})
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Group not found')

    group = Group(**group)

    # Check if user has global USER_READ right (otherwise they can't add users)
    global_user_rights = await get_rights_in_groups(db_handler, current_user, ['global'])
    has_user_read = RIGHT.USER_READ in global_user_rights

    if not has_user_read:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You do not have sufficient rights to add users to this group',
        )

    # Get current user's rights to validate they can assign these roles
    current_user_rights = await get_rights_in_groups(
        db_handler, current_user, ['global', membership_query.group_name]
    )

    # Validate that current user has rights to assign all requested roles
    try:
        for member in membership_query.members:
            for role in member.roles:
                role_entry = await db_handler.get_item(USER_ROLES_COLLECTION_NAME, {'role': role})

                if not role_entry:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, detail=f'Role {role} not found'
                    )

                role_rights = set(role_entry.get('rights', []))
                if not role_rights.issubset(current_user_rights):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f'You do not have sufficient rights to assign role {role}',
                    )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to validate role assignments: {str(e)}',
        )

    # Add members to the group
    try:
        for member in membership_query.members:
            if not member.roles:
                continue  # Skip if no roles are specified

            # Check if user exists
            user_entry = await db_handler.get_item(USER_COLLECTION_NAME, {'username': member.user})
            if not user_entry:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f'User {member.user} not found'
                )

            user_obj = UserOut(**user_entry)

            # Remove existing membership in this group (if any)
            updated_memberships = [
                membership for membership in user_obj.memberships if membership.group != group.name
            ]

            # Add new membership
            new_membership = Membership(group=group.name, roles=member.roles)
            updated_memberships.append(new_membership)

            # Update user
            update = {'$set': {'memberships': [m.model_dump() for m in updated_memberships]}}
            await db_handler.update_item(USER_COLLECTION_NAME, {'username': member.user}, update)

        return await read_group(
            group_name=membership_query.group_name,
            db_handler=db_handler,
            auth_result=auth_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to add members: {str(e)}',
        )


@router.post('/remove-members')
async def remove_members(
    membership_query: GroupMembershipQuery,
    db_handler: MongoDBHandler = Depends(get_db_handler),
    auth_result: tuple[UserOut, None] = Depends(
        check_authorization(RESOURCE_TYPE.GROUP, RIGHT.GROUP_UPDATE)
    ),
) -> GroupWithMembers:
    """Remove members from a group."""
    current_user, _ = auth_result

    group = await db_handler.get_item(GROUPS_COLLECTION_NAME, {'name': membership_query.group_name})
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Group not found')

    group = Group(**group)

    # Check if user has USER_READ rights
    current_user_rights = await get_rights_in_groups(
        db_handler, current_user, ['global', group.name]
    )
    has_user_read = RIGHT.USER_READ in current_user_rights

    if not has_user_read:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='You do not have sufficient rights to remove users from this group',
        )

    # Remove members from the group
    try:
        for member in membership_query.members:
            # Check if user exists
            user_entry = await db_handler.get_item(USER_COLLECTION_NAME, {'username': member.user})
            if not user_entry:
                continue  # Skip if user doesn't exist

            user_obj = UserOut(**user_entry)

            # Remove membership from this group
            updated_memberships = [
                membership for membership in user_obj.memberships if membership.group != group.name
            ]

            # Only update if membership was actually removed
            if len(updated_memberships) < len(user_obj.memberships):
                update = {'$set': {'memberships': [m.model_dump() for m in updated_memberships]}}
                await db_handler.update_item(
                    USER_COLLECTION_NAME, {'username': member.user}, update
                )

        return await read_group(
            group_name=membership_query.group_name,
            db_handler=db_handler,
            auth_result=auth_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to remove members: {str(e)}',
        )
