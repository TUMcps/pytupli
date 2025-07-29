import asyncio
import os

from pytupli.schema import (
    RIGHT_BUNDLE_CONTENT_CREATE,
    RIGHT_BUNDLE_CONTENT_DELETE,
    RIGHT_BUNDLE_CONTENT_READ,
    DEFAULT_ROLE,
    RIGHT,
    Membership,
    User,
)
from pytupli.server.config import USER_COLLECTION_NAME, USER_ROLES_COLLECTION_NAME, DBHandlerFactory
from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.management.authorization import hash_password


default_roles = [
    {
        'role': DEFAULT_ROLE.ADMIN,
        'rights': [
            *RIGHT_BUNDLE_CONTENT_READ,
            *RIGHT_BUNDLE_CONTENT_CREATE,
            *RIGHT_BUNDLE_CONTENT_DELETE,
            RIGHT.USER_CREATE,
            RIGHT.USER_READ,
            RIGHT.USER_UPDATE,
            RIGHT.USER_DELETE,
            RIGHT.GROUP_CREATE,
            RIGHT.GROUP_READ,
            RIGHT.GROUP_UPDATE,
            RIGHT.GROUP_DELETE,
            RIGHT.ROLE_MANAGEMENT,
        ],
        'description': 'Full access to all resources and management functions.',
    },
    {
        'role': DEFAULT_ROLE.CONTENT_ADMIN,
        'rights': [
            *RIGHT_BUNDLE_CONTENT_READ,
            *RIGHT_BUNDLE_CONTENT_CREATE,
            *RIGHT_BUNDLE_CONTENT_DELETE,
        ],
        'description': 'Can create, edit, and delete content.',
    },
    {
        'role': DEFAULT_ROLE.USER_ADMIN,
        'rights': [
            RIGHT.USER_CREATE,
            RIGHT.USER_READ,
            RIGHT.USER_UPDATE,
            RIGHT.USER_DELETE,
        ],
        'description': 'Can manage users.',
    },
    {
        'role': DEFAULT_ROLE.GROUP_ADMIN,
        'rights': [
            RIGHT.GROUP_CREATE,
            RIGHT.GROUP_READ,
            RIGHT.GROUP_UPDATE,
            RIGHT.GROUP_DELETE,
            RIGHT.USER_READ,
        ],
        'description': 'Can manage groups.',
    },
    {
        'role': DEFAULT_ROLE.GUEST,
        'rights': RIGHT_BUNDLE_CONTENT_READ,
        'description': 'Limited access for guest users.',
    },
    {
        'role': DEFAULT_ROLE.CONTRIBUTOR,
        'rights': [
            *RIGHT_BUNDLE_CONTENT_CREATE,
            *RIGHT_BUNDLE_CONTENT_READ,
        ],
        'description': 'Can create and read content.',
    },
    {
        'role': DEFAULT_ROLE.MEMBER,
        'rights': [
            RIGHT.GROUP_READ,
            RIGHT.USER_READ,
            *RIGHT_BUNDLE_CONTENT_READ,
        ],
        'description': 'Can create and read content in a group.',
    },
    {
        'role': DEFAULT_ROLE.GLOBAL_MEMBER,
        'rights': [
            RIGHT.GROUP_CREATE,
            RIGHT.USER_READ,
            *RIGHT_BUNDLE_CONTENT_READ,
        ],
        'description': 'Can read content and create groups.',
    },
]


# Async function for creating roles in the database
async def initialize_roles(db_handler: MongoDBHandler):
    """
    Function initializes the roles in the database.
    """
    r = await db_handler.get_items(USER_ROLES_COLLECTION_NAME, {})
    existing_roles = [role['role'] for role in r]
    all_roles_exist = False
    for role in default_roles:
        if role['role'] not in existing_roles:
            all_roles_exist = False
            break
        all_roles_exist = True
    if all_roles_exist:
        print('Roles already exist')
        return
    res = await db_handler.create_items(USER_ROLES_COLLECTION_NAME, default_roles)
    print(res)


async def create_admin(db_handler: MongoDBHandler, admin_pw: str):
    """
    Function creates an admin user in the database.
    """
    r = await db_handler.get_item(USER_COLLECTION_NAME, {'username': 'admin'})
    if r:
        print('Admin user already exists')
        return
    hashed_password = hash_password(admin_pw)
    res = await db_handler.create_item(
        collection=USER_COLLECTION_NAME,
        item=User(
            username='admin',
            password=hashed_password,
            memberships=[Membership(group='global', roles=[DEFAULT_ROLE.ADMIN.value])],
        ).model_dump(),
    )
    print(res)


async def initialize_database(db_handler: DBHandlerFactory):
    """
    Function initializes the database with roles and an admin user.
    """
    if os.getenv('DB_ADMIN_PW') is None:
        raise ValueError(
            'DB_ADMIN_PW not set in environment. Please set this environment variable to continue.'
        )
    admin_pw = os.getenv('DB_ADMIN_PW')
    await initialize_roles(db_handler)
    await create_admin(db_handler, admin_pw)


# Run the async function
if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv(override=True)

    db_handler = DBHandlerFactory.get_handler()
    asyncio.run(initialize_database(db_handler))
    print('Verified roles and admin user')
