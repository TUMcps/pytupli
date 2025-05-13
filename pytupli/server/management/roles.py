"""
This is a setup script to set up the database with all roles and an admin user.
"""

import asyncio
import logging
import os

from pytupli.server.db.db_handler import MongoDBHandler
from pytupli.server.config import (
    USER_COLLECTION_NAME,
    USER_ROLES_COLLECTION_NAME,
    DBHandlerFactory,
)
from pytupli.server.management.security import hash_password

# passlib bcrypt has little bug of getting verions of bcrypt
# supress these warnings
logging.getLogger('passlib').setLevel(logging.ERROR)

"""
Rights:
All users can read and delete their own contents and can read all public contents.

The following rights enable you to do following:
user_management - for user management
read_all - you can read all contents that are public and private
write - you can upload new content
delete_all - you can delete contents that are not yours in the private section and also to delete public contents
"""
roles_rights = {
    'admin': ['user_management', 'read_all', 'write', 'delete_all'],  # all rights
    'standard_user': ['write'],  # right to create content
    'user_admin': ['user_management'],  # right to manage users
    'content_admin': [
        'read_all',
        'write',
        'delete_all',
    ],  # read, write and delete rights for the whole space - except user management
}

roles = [
    {
        'role': 'admin',
        'rights': roles_rights['admin'],
        'description': 'All rights',
    },
    {
        'role': 'user_admin',
        'rights': roles_rights['user_admin'],
        'description': 'User management rights only',
    },
    {
        'role': 'content_admin',
        'rights': roles_rights['content_admin'],
        'description': 'Read+write+delete for self owned as well as read+delete rights for all user objects',
    },
    {
        'role': 'standard_user',
        'rights': roles_rights['standard_user'],
        'description': 'Read rights for all public and read+write+delete for self owned objects',
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
    for role in roles:
        if role['role'] not in existing_roles:
            all_roles_exist = False
            break
        all_roles_exist = True
    if all_roles_exist:
        print('Roles already exist')
        return
    res = await db_handler.create_items(USER_ROLES_COLLECTION_NAME, roles)
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
        USER_COLLECTION_NAME, {'username': 'admin', 'password': hashed_password, 'roles': ['admin']}
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
