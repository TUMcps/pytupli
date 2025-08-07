import pytest
from conftest import get_JWT_token

pytestmark = pytest.mark.anyio


async def test_root(async_client):
    response = await async_client.get('/')
    assert response.status_code == 200


# === TOKEN CREATION TESTS ===


async def test_token_creation_success(async_client):
    """Test successful token creation with valid credentials."""
    response = await async_client.post(
        '/access/users/token',
        json={'username': 'admin', 'password': 'pytupli'},
    )
    assert response.status_code == 200

    token_data = response.json()
    assert 'access_token' in token_data
    assert 'refresh_token' in token_data
    assert token_data['access_token']['token_type'] == 'bearer'
    assert token_data['refresh_token']['token_type'] == 'bearer'
    assert len(token_data['access_token']['token']) > 0
    assert len(token_data['refresh_token']['token']) > 0



async def test_token_creation_wrong_password(async_client):
    """Test token creation fails with wrong password."""
    response = await async_client.post(
        '/access/users/token',
        json={'username': 'admin', 'password': 'wrongpassword'},
    )
    assert response.status_code == 401
    assert 'Incorrect username or password' in response.json()['detail']



async def test_token_creation_nonexistent_user(async_client):
    """Test token creation fails with non-existent user."""
    response = await async_client.post(
        '/access/users/token',
        json={'username': 'nonexistent_user', 'password': 'password'},
    )
    assert response.status_code == 401
    assert 'Incorrect username or password' in response.json()['detail']



async def test_token_creation_empty_credentials(async_client):
    """Test token creation fails with empty credentials."""
    response = await async_client.post(
        '/access/users/token',
        json={'username': '', 'password': ''},
    )
    assert response.status_code == 401


# === DATA VALIDATION TESTS ===


async def test_empty_request_bodies(async_client):
    """Test endpoints handle empty request bodies gracefully."""
    # Test token creation with empty body
    response = await async_client.post(
        '/access/users/token',
        json={},
    )
    assert response.status_code == 422  # Validation error


# === USER CREATION TESTS (Now working with server running) ===


async def test_user_creation_success(async_client, admin_headers):
    """Test successful user creation with admin privileges."""
    username = 'test_user_creation_success'

    # Clean up any existing user first
    await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)

    response = await async_client.post(
        '/access/users/create',
        json={'username': username, 'password': 'test1234'},
        headers=admin_headers,
    )
    assert response.status_code == 200

    user_data = response.json()
    assert user_data['username'] == username
    assert 'password' not in user_data  # Password should not be returned
    assert 'memberships' in user_data

    # Cleanup
    cleanup = await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)
    assert cleanup.status_code == 200



async def test_user_creation_duplicate_username(async_client, admin_headers):
    """Test user creation fails when username already exists."""
    username = 'test_duplicate_user'

    # Clean up first
    await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)

    # Create user first time
    response = await async_client.post(
        '/access/users/create',
        json={'username': username, 'password': 'test1234'},
        headers=admin_headers,
    )
    assert response.status_code == 200

    try:
        # Try to create user with same username
        response = await async_client.post(
            '/access/users/create',
            json={'username': username, 'password': 'test1234'},
            headers=admin_headers,
        )
        assert response.status_code == 409  # Conflict
        assert 'already exists' in response.json()['detail']
    finally:
        # Cleanup
        await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)

# === USER LISTING TESTS ===


async def test_user_listing_success(async_client, admin_headers):
    """Test successful user listing with admin privileges."""
    response = await async_client.get('/access/users/list', headers=admin_headers)
    assert response.status_code == 200

    users = response.json()
    assert len(users) >= 1  # At least admin user

    usernames = {user['username'] for user in users}
    assert 'admin' in usernames

    # Check user data structure
    for user in users:
        assert 'username' in user
        assert 'memberships' in user
        assert 'password' not in user  # Password should never be returned



async def test_user_listing_no_auth(async_client):
    """Test user listing fails without authentication."""
    response = await async_client.get('/access/users/list')
    assert response.status_code == 403  # Forbidden


# === USER DELETION TESTS ===


async def test_user_deletion_success(async_client, admin_headers):
    """Test successful user deletion with admin privileges."""
    username = 'test_user_deletion_success'

    # Create user first
    response = await async_client.post(
        '/access/users/create',
        json={'username': username, 'password': 'test1234'},
        headers=admin_headers,
    )
    assert response.status_code == 200

    # Delete user
    response = await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)
    assert response.status_code == 200

    # Verify user no longer exists in list
    response = await async_client.get('/access/users/list', headers=admin_headers)
    assert response.status_code == 200
    usernames = {user['username'] for user in response.json()}
    assert username not in usernames



async def test_user_deletion_nonexistent(async_client, admin_headers):
    """Test deletion of non-existent user returns success (idempotent)."""
    response = await async_client.delete('/access/users/delete?username=nonexistent_user_12345', headers=admin_headers)
    assert response.status_code == 200



async def test_user_deletion_no_auth(async_client):
    """Test user deletion fails without authentication."""
    response = await async_client.delete('/access/users/delete?username=admin')
    assert response.status_code == 403  # Forbidden


# === PASSWORD CHANGE TESTS ===


async def test_change_password_success(async_client, admin_headers):
    """Test successful password change with admin privileges."""
    username = 'test_change_password_user'
    initial_password = 'initial123'
    new_password = 'changed456'

    # Clean up and create user
    await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)
    response = await async_client.post(
        '/access/users/create',
        json={'username': username, 'password': initial_password},
        headers=admin_headers,
    )
    assert response.status_code == 200

    try:
        # Change the password
        response = await async_client.put(
            '/access/users/change-password',
            json={'username': username, 'password': new_password},
            headers=admin_headers,
        )
        assert response.status_code == 200

        # Verify old password no longer works
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': initial_password},
        )
        assert response.status_code == 401

        # Verify new password works
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': new_password},
        )
        assert response.status_code == 200
        assert 'access_token' in response.json()
        assert 'refresh_token' in response.json()

    finally:
        # Cleanup
        await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)



async def test_change_password_nonexistent_user(async_client, admin_headers):
    """Test password change fails for non-existent user."""
    response = await async_client.put(
        '/access/users/change-password',
        json={'username': 'nonexistent_user_99999', 'password': 'newpassword'},
        headers=admin_headers,
    )
    assert response.status_code == 404
    assert 'does not exist' in response.json()['detail']



async def test_change_password_no_auth(async_client):
    """Test password change fails without authentication."""
    response = await async_client.put(
        '/access/users/change-password',
        json={'username': 'admin', 'password': 'newpassword'},
    )
    assert response.status_code == 403  # Forbidden



async def test_change_admin_password_as_standard_user(async_client, admin_headers):
    """Test that a standard user cannot change the admin password."""
    username = 'test_standard_user'
    user_password = 'standarduser123'

    # Clean up and create a standard user
    await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)
    response = await async_client.post(
        '/access/users/create',
        json={'username': username, 'password': user_password},
        headers=admin_headers,
    )
    assert response.status_code == 200

    try:
        # Get token for the standard user
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': user_password},
        )
        assert response.status_code == 200
        user_token = response.json()['access_token']['token']
        user_headers = {'Authorization': f'Bearer {user_token}'}

        # Attempt to change admin password using standard user credentials
        response = await async_client.put(
            '/access/users/change-password',
            json={'username': 'admin', 'password': 'newhackedpassword'},
            headers=user_headers,
        )
        assert response.status_code == 403  # Forbidden - standard user cannot change admin password

        # Verify admin password remains unchanged by testing login
        response = await async_client.post(
            '/access/users/token',
            json={'username': 'admin', 'password': 'pytupli'},
        )
        assert response.status_code == 200
        assert 'access_token' in response.json()

    finally:
        # Cleanup
        await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)


# === INTEGRATION TESTS ===


async def test_user_lifecycle(async_client, admin_headers):
    """Test complete user lifecycle: create, use, change password, delete."""
    username = 'test_lifecycle_user'
    initial_password = 'initial123'
    new_password = 'changed456'

    # Clean up first
    await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)

    try:
        # 1. Create user
        response = await async_client.post(
            '/access/users/create',
            json={'username': username, 'password': initial_password},
            headers=admin_headers,
        )
        assert response.status_code == 200

        # 2. Test login with initial password
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': initial_password},
        )
        assert response.status_code == 200
        user_token = response.json()['access_token']['token']

        # 3. Use token to access root endpoint (basic test)
        user_headers = {'Authorization': f'Bearer {user_token}'}
        response = await async_client.get('/', headers=user_headers)
        assert response.status_code == 200

        # 4. Change password
        response = await async_client.put(
            '/access/users/change-password',
            json={'username': username, 'password': new_password},
            headers=admin_headers,
        )
        assert response.status_code == 200

        # 5. Verify old password no longer works
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': initial_password},
        )
        assert response.status_code == 401

        # 6. Verify new password works
        response = await async_client.post(
            '/access/users/token',
            json={'username': username, 'password': new_password},
        )
        assert response.status_code == 200

        # 7. Verify user appears in user list
        response = await async_client.get('/access/users/list', headers=admin_headers)
        assert response.status_code == 200
        usernames = {user['username'] for user in response.json()}
        assert username in usernames

    finally:
        # 8. Delete user (cleanup)
        response = await async_client.delete(f'/access/users/delete?username={username}', headers=admin_headers)
        assert response.status_code == 200

        # 9. Verify user no longer in list
        response = await async_client.get('/access/users/list', headers=admin_headers)
        assert response.status_code == 200
        usernames = {user['username'] for user in response.json()}
        assert username not in usernames


# === REFRESH TOKEN TESTS ===


async def test_refresh_token_success(async_client, admin_headers):
    """Test successful token refresh with valid refresh token."""
    # Get tokens for a user first
    response = await async_client.post(
        '/access/users/token',
        json={'username': 'admin', 'password': 'pytupli'},
    )
    assert response.status_code == 200
    tokens = response.json()
    refresh_token = tokens['refresh_token']['token']

    # Use refresh token to get new access token
    refresh_headers = {'Authorization': f'Bearer {refresh_token}'}
    response = await async_client.post('/access/users/refresh-token', headers=refresh_headers)
    assert response.status_code == 200

    new_token = response.json()
    assert 'token' in new_token
    assert 'token_type' in new_token
    assert new_token['token_type'] == 'bearer'
    assert len(new_token['token']) > 0



async def test_refresh_token_invalid(async_client):
    """Test refresh token fails with invalid token."""
    invalid_headers = {'Authorization': 'Bearer invalid_token_12345'}
    response = await async_client.post('/access/users/refresh-token', headers=invalid_headers)
    assert response.status_code == 401



async def test_refresh_token_no_auth(async_client):
    """Test refresh token fails without authentication."""
    response = await async_client.post('/access/users/refresh-token')
    assert response.status_code == 403  # Forbidden


# === VALIDATION TESTS FOR EDGE CASES ===


async def test_create_user_invalid_json(async_client, admin_headers):
    """Test user creation with invalid JSON data."""
    response = await async_client.post(
        '/access/users/create',
        json={'username': 'test_invalid', 'invalid_field': 'value'},  # Missing password
        headers=admin_headers,
    )
    assert response.status_code == 422  # Validation error



async def test_change_password_invalid_json(async_client, admin_headers):
    """Test password change with invalid JSON data."""
    response = await async_client.put(
        '/access/users/change-password',
        json={'username': 'admin'},  # Missing password
        headers=admin_headers,
    )
    assert response.status_code == 422  # Validation error



async def test_token_creation_invalid_json(async_client):
    """Test token creation with invalid JSON data."""
    response = await async_client.post(
        '/access/users/token',
        json={'username': 'admin'},  # Missing password
    )
    assert response.status_code == 422  # Validation error


# === EMPTY STRING TESTS ===


async def test_user_deletion_empty_username(async_client, admin_headers):
    """Test deletion with empty username."""
    response = await async_client.delete('/access/users/delete?username=', headers=admin_headers)
    assert response.status_code == 200  # Should be idempotent



async def test_change_password_empty_username(async_client, admin_headers):
    """Test password change with empty username."""
    response = await async_client.put(
        '/access/users/change-password',
        json={'username': '', 'password': 'newpassword'},
        headers=admin_headers,
    )
    assert response.status_code == 404  # User does not exist


if __name__ == '__main__':
    pytest.main(['-v', __file__])
