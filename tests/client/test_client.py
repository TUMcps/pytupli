import pytest
import uuid
import keyring
import time
from unittest.mock import patch

from pytupli.storage import TupliAPIClient, TupliStorageError
from pytupli.schema import UserOut, UserRole
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
        assert "standard_user" in user.roles
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
    assert "standard_user" in role_names


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


def test_change_roles(api_client):
    """Test successfully changing a user's roles."""
    # Create a unique test user for this test
    test_username = f"test_role_change_{uuid.uuid4().hex[:8]}"

    try:
        # Create the user
        api_client.signup(test_username, "test_password")

        # Login as admin to change roles
        api_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)

        # Get the available roles first
        roles: list[UserRole] = api_client.list_roles()
        role_names = [r.role for r in roles]

        # Find a role that's not "standard_user" for our test
        test_role = next((r for r in role_names if r != "standard_user" and r != "admin"), None)
        if test_role is None:
            # If no other role found, we'll just use what we have for the test
            test_roles = ["standard_user"]
        else:
            test_roles = [test_role]

        # Change the user's roles
        api_client.change_roles(test_username, test_roles)

        # Fetch the updated user information
        users = api_client.list_users()
        updated_user = next((u for u in users if u.username == test_username), None)

        # Verify the roles were changed
        assert updated_user is not None
        assert isinstance(updated_user, UserOut)
        assert updated_user.username == test_username
        assert set(updated_user.roles) == set(test_roles)

    finally:
        # Clean up the created user
        admin_cleanup(None, 'user', test_username)


def test_change_roles_unauthorized(api_user):
    """Test changing roles with insufficient privileges."""
    user_info, client = api_user

    # Regular users can't change roles, not even their own
    with pytest.raises(TupliStorageError, match="API request failed"):
        client.change_roles(user_info['username'], ["researcher"])


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

if __name__ == '__main__':
    pytest.main(['-v', __file__])
