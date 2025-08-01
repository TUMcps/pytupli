import pytest
import uuid
import keyring
import time
from unittest.mock import patch

from pytupli.storage import TupliAPIClient, TupliStorageError
from pytupli.schema import UserOut, UserRole, Group, GroupMembership, GroupMembershipQuery, GroupWithMembers
from conftest import API_BASE_URL, API_USERNAME, API_PASSWORD, ADMIN_USERNAME, ADMIN_PASSWORD, admin_cleanup, clean_keyring


@pytest.fixture(scope="function")
def api_client(clean_keyring):
    """Create a TupliAPIClient instance for testing."""
    client = TupliAPIClient()
    # Set a consistent base URL for testing
    client.set_url(API_BASE_URL)
    return client


def test_signup_success(api_client):
    """Test successful user signup."""
    # Create a unique test username to avoid conflicts
    test_username = f"test_signup_{uuid.uuid4().hex[:8]}"

    try:
        # Call the signup method
        user = api_client.signup(test_username, "test_password")

        # Verify results
        assert user is not None
        assert isinstance(user, UserOut)
        assert user.username == test_username
        # Check that user has memberships (not a direct 'roles' attribute)
        assert hasattr(user, 'memberships')
    finally:
        # Clean up the created user
        admin_cleanup(None, 'user', test_username)


def test_signup_user_exists(api_client):
    """Test signup when user already exists."""
    # Create a test user first
    test_username = f"test_exists_{uuid.uuid4().hex[:8]}"

    try:
        # First signup should succeed
        api_client.signup(test_username, "test_password")

        # Second signup with the same username should fail
        with pytest.raises(TupliStorageError, match="Signup failed"):
            api_client.signup(test_username, "different_password")
    finally:
        # Clean up the created user
        admin_cleanup(None, 'user', test_username)


def test_login_success(api_client, api_user):
    """Test successful login with token storage."""
    user_info, client = api_user
    username = user_info['username']

    # Call login method
    api_client.login(username, API_PASSWORD)

    # Verify tokens were stored in keyring
    assert keyring.get_password('pytupli', 'access_token') is not None
    assert keyring.get_password('pytupli', 'refresh_token') is not None


def test_login_with_url(api_client, api_user):
    """Test login with a provided URL."""
    user_info, client = api_user
    username = user_info['username']

    # Call login with URL parameter (use the same API_BASE_URL for test purposes)
    api_client.login(username, API_PASSWORD, url=API_BASE_URL)

    # Verify URL was set and tokens were stored
    assert api_client.base_url == API_BASE_URL
    assert keyring.get_password('pytupli', 'base_url') == API_BASE_URL
    assert keyring.get_password('pytupli', 'access_token') is not None


def test_login_invalid_credentials(api_client):
    """Test login with invalid credentials."""
    # Verify the correct exception is raised with wrong password
    with pytest.raises(TupliStorageError, match="Login failed"):
        api_client.login(API_USERNAME, "wrong_password")

    # Verify no tokens were stored
    assert keyring.get_password('pytupli', 'access_token') is None
    assert keyring.get_password('pytupli', 'refresh_token') is None


def test_list_users(api_user):
    """Test listing all users."""
    user_info, client = api_user

    # Need admin privileges to list users
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    # Call list_users with admin privileges
    users = admin_client.list_users()

    # Verify results
    assert users is not None
    assert isinstance(users, list)
    assert len(users) > 0  # Should have at least the admin user and our test user
    assert any(u.username == user_info['username'] for u in users)
    assert any(u.username == ADMIN_USERNAME for u in users)
    assert all(isinstance(u, UserOut) for u in users)


def test_list_roles(api_user):
    """Test listing all available roles."""
    user_info, client = api_user

    # Need admin privileges to list roles
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    # Call list_roles
    roles = admin_client.list_roles()

    # Verify results
    assert roles is not None
    assert isinstance(roles, list)
    assert len(roles) > 0  # Should have at least the basic roles defined
    assert all(isinstance(r, UserRole) for r in roles)
    # Check for expected roles
    role_names = [r.role for r in roles]
    assert "admin" in role_names
    assert "guest" in role_names


def test_change_password(api_client):
    """Test successfully changing a user's password."""
    # Create a unique test user for this test
    test_username = f"test_pwd_change_{uuid.uuid4().hex[:8]}"
    initial_password = "initial_password"
    new_password = "new_password_123"

    try:
        # Create the user
        api_client.signup(test_username, initial_password)

        # Login as admin to change the password (regular users can only change their own password)
        api_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

        # Change the password
        api_client.change_password(test_username, new_password)

        # Try logging in with the new password
        api_client.login(test_username, new_password)

        # If we got here without exception, the password change worked
        assert keyring.get_password('pytupli', 'access_token') is not None

    finally:
        # Clean up the created user
        admin_cleanup(None, 'user', test_username)


def test_change_password_unauthorized(api_user):
    """Test changing password with insufficient privileges."""
    user_info, client = api_user

    # Try to change the admin's password as a regular user
    with pytest.raises(TupliStorageError, match="API request failed"):
        client.change_password(ADMIN_USERNAME, "new_admin_password")


def test_delete_user(api_client):
    """Test successfully deleting a user."""
    # Create a unique test user for this test
    test_username = f"test_delete_{uuid.uuid4().hex[:8]}"

    # First create the user
    api_client.signup(test_username, "test_password")

    # Login as admin to delete the user
    api_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    # Make sure the user exists in the list first
    users = api_client.list_users()
    assert any(u.username == test_username for u in users)

    # Delete the user
    api_client.delete_user(test_username)

    # Verify the user was deleted by checking the user list again
    users = api_client.list_users()
    assert not any(u.username == test_username for u in users)


def test_delete_user_unauthorized(api_client, api_user):
    """Test deleting a user with insufficient privileges."""
    user_info, client = api_user

    # Create another test user that we'll try to delete
    test_username = f"test_delete_unauth_{uuid.uuid4().hex[:8]}"

    try:
        # Create the second test user
        api_client.signup(test_username, "test_password")

        # Try to delete it as a regular user, which should fail
        with pytest.raises(TupliStorageError, match="API request failed"):
            client.delete_user(test_username)

        # Verify the user still exists
        admin_client = TupliAPIClient()
        admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)
        users = admin_client.list_users()
        assert any(u.username == test_username for u in users)

    finally:
        # Clean up the created user
        admin_cleanup(None, 'user', test_username)


# Group Management Tests
def test_create_group(api_user):
    """Test creating a group."""
    user_info, client = api_user

    # Need admin privileges to create groups
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    group_name = f"test_group_{uuid.uuid4().hex[:8]}"
    group = Group(name=group_name, description="Test group for client testing")

    try:
        # Create the group
        created_group = admin_client.create_group(group)

        # Verify results
        assert created_group is not None
        assert isinstance(created_group, Group)
        assert created_group.name == group_name
        assert created_group.description == "Test group for client testing"

    finally:
        # Clean up the created group
        admin_cleanup(None, 'group', group_name)


def test_list_groups(api_user):
    """Test listing groups."""
    user_info, client = api_user

    # Need admin privileges to list all groups
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    # Call list_groups
    groups = admin_client.list_groups()

    # Verify results
    assert groups is not None
    assert isinstance(groups, list)
    assert all(isinstance(g, Group) for g in groups)


def test_read_group(api_user):
    """Test reading a specific group."""
    user_info, client = api_user

    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    group_name = f"test_read_group_{uuid.uuid4().hex[:8]}"
    group = Group(name=group_name, description="Test group for reading")

    try:
        # Create the group first
        admin_client.create_group(group)

        # Read the group
        read_group = admin_client.read_group(group_name)

        # Verify results
        assert read_group is not None
        assert isinstance(read_group, GroupWithMembers)
        assert read_group.name == group_name
        assert read_group.description == "Test group for reading"
        assert isinstance(read_group.members, list)

    finally:
        # Clean up the created group
        admin_cleanup(None, 'group', group_name)


def test_delete_group(api_user):
    """Test deleting a group."""
    user_info, client = api_user

    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    group_name = f"test_delete_group_{uuid.uuid4().hex[:8]}"
    group = Group(name=group_name, description="Test group for deletion")

    # Create the group first
    admin_client.create_group(group)

    # Verify it exists
    groups = admin_client.list_groups()
    assert any(g.name == group_name for g in groups)

    # Delete the group
    admin_client.delete_group(group_name)

    # Verify it's deleted
    groups = admin_client.list_groups()
    assert not any(g.name == group_name for g in groups)


def test_add_group_members(api_user):
    """Test adding members to a group."""
    user_info, client = api_user

    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    group_name = f"test_members_group_{uuid.uuid4().hex[:8]}"
    group = Group(name=group_name, description="Test group for member management")
    test_user = f"test_member_{uuid.uuid4().hex[:8]}"

    try:
        # Create the group
        admin_client.create_group(group)

        # Create a test user
        admin_cleanup(None, 'user', test_user)  # Clean up in case it exists
        from pytupli.schema import UserCredentials
        # We'll use signup to create the user
        temp_client = TupliAPIClient()
        temp_client.signup(test_user, "test_password")

        # Add the user to the group
        membership_query = GroupMembershipQuery(
            group_name=group_name,
            members=[GroupMembership(user=test_user, roles=["member"])]
        )

        updated_group = admin_client.add_group_members(membership_query)

        # Verify results
        assert updated_group is not None
        assert isinstance(updated_group, GroupWithMembers)
        assert len(updated_group.members) >= 1
        assert any(member == test_user for member in updated_group.members)

    finally:
        # Clean up
        admin_cleanup(None, 'user', test_user)
        admin_cleanup(None, 'group', group_name)


def test_remove_group_members(api_user):
    """Test removing members from a group."""
    user_info, client = api_user

    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    group_name = f"test_remove_group_{uuid.uuid4().hex[:8]}"
    group = Group(name=group_name, description="Test group for member removal")
    test_user = f"test_remove_member_{uuid.uuid4().hex[:8]}"

    try:
        # Create the group
        admin_client.create_group(group)

        # Create a test user
        admin_cleanup(None, 'user', test_user)  # Clean up in case it exists
        temp_client = TupliAPIClient()
        temp_client.signup(test_user, "test_password")

        # Add the user to the group first
        add_query = GroupMembershipQuery(
            group_name=group_name,
            members=[GroupMembership(user=test_user, roles=["member"])]
        )
        admin_client.add_group_members(add_query)

        # Now remove the user from the group
        remove_query = GroupMembershipQuery(
            group_name=group_name,
            members=[GroupMembership(user=test_user)]
        )

        updated_group = admin_client.remove_group_members(remove_query)

        # Verify the user was removed
        assert updated_group is not None
        assert isinstance(updated_group, GroupWithMembers)
        assert not any(member == test_user for member in updated_group.members)

    finally:
        # Clean up
        admin_cleanup(None, 'user', test_user)
        admin_cleanup(None, 'group', group_name)


# Role Management Tests
def test_create_role():
    """Test creating a role."""
    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    from pytupli.schema import RIGHT

    role_name = f"test_role_{uuid.uuid4().hex[:8]}"
    role = UserRole(
        role=role_name,
        description="Test role for client testing",
        rights=[RIGHT.ARTIFACT_READ]
    )

    try:
        # Create the role
        created_role = admin_client.create_role(role)

        # Verify results
        assert created_role is not None
        assert isinstance(created_role, UserRole)
        assert created_role.role == role_name
        assert created_role.description == "Test role for client testing"
        assert RIGHT.ARTIFACT_READ in created_role.rights

    finally:
        # Clean up the created role
        admin_cleanup(None, 'role', role_name)


def test_delete_role():
    """Test deleting a role."""
    # Need admin privileges
    admin_client = TupliAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

    from pytupli.schema import RIGHT

    role_name = f"test_delete_role_{uuid.uuid4().hex[:8]}"
    role = UserRole(
        role=role_name,
        description="Test role for deletion",
        rights=[RIGHT.ARTIFACT_READ]
    )

    # Create the role first
    admin_client.create_role(role)

    # Verify it exists
    roles = admin_client.list_roles()
    assert any(r.role == role_name for r in roles)

    # Delete the role
    admin_client.delete_role(role_name)

    # Verify it's deleted
    roles = admin_client.list_roles()
    assert not any(r.role == role_name for r in roles)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
