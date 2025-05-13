import pytest
from conftest import get_JWT_token


# Helper function to create a test user with specified privileges
async def create_test_user(async_client, headers, username):
    return await async_client.post(
        '/access/signup',
        json={'username': username, 'password': 'test1234'},
        headers=headers,
    )


@pytest.mark.anyio(loop_scope='session')
async def test_root(async_client):
    response = await async_client.get('/')
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_no_token(async_client):
    response = await async_client.delete(
        f'/artifacts/delete?artifact_id={123}',
    )
    assert response.status_code == 401


@pytest.mark.anyio(loop_scope='session')
async def test_invalid_auth_type(async_client):
    # don't use Bearer as auth type
    jwt = await get_JWT_token(async_client)
    headers = {'API-Token': jwt + '1234'}
    response = await async_client.delete(f'/artifacts/delete?artifact_id={123}', headers=headers)
    assert response.status_code == 401


@pytest.mark.anyio(loop_scope='session')
async def test_invalid_token(async_client):
    jwt = await get_JWT_token(async_client)
    headers = {'Authorization': f'Bearer {jwt + "1234"}'}
    response = await async_client.delete(f'/artifacts/delete?artifact_id={123}', headers=headers)
    assert response.status_code == 401


@pytest.mark.anyio(loop_scope='session')
async def test_delete_itempotency(async_client, standard_user1_headers):
    # deletion of a non-existing artifact should return 200
    response = await async_client.delete(
        f'/artifacts/delete?artifact_id={123}', headers=standard_user1_headers
    )
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_user_creation(async_client, admin_headers):
    username = 'test_user_creation'
    response = await create_test_user(async_client, admin_headers, username)
    assert response.status_code == 200

    # Cleanup
    cleanup = await async_client.delete(
        f'/access/delete-user?username={username}', headers=admin_headers
    )
    assert cleanup.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_user_listing(async_client, admin_headers, standard_user1_headers):
    response = await async_client.get('/access/list-users', headers=admin_headers)
    assert response.status_code == 200

    users = response.json()
    assert len(users) == 2
    assert users[0]['username'] == 'admin'
    assert users[1]['username'] == 'test_user_1'


@pytest.mark.anyio(loop_scope='session')
async def test_role_listing(async_client, admin_headers):
    response = await async_client.get('/access/list-roles', headers=admin_headers)
    assert response.status_code == 200

    roles = response.json()
    assert len(roles) == 4


@pytest.mark.anyio(loop_scope='session')
async def test_user_change_password(async_client, admin_headers, standard_user1_headers):
    username = 'test_user_1'

    # Change the password
    response = await async_client.put(
        '/access/change-password',
        json={'username': username, 'password': 'test12345'},
        headers=admin_headers,
    )
    assert response.status_code == 200

    # check if the update was successfull by creating a new token
    # with old password we should get an error
    response = await async_client.post(
        '/access/token',
        json={'username': username, 'password': 'test1234'},
    )
    assert response.status_code == 401

    # with new password it should work
    response = await async_client.post(
        '/access/token',
        json={'username': username, 'password': 'test12345'},
    )
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_user_change_roles(
    async_client, admin_headers, sample_benchmark, standard_user1_headers
):
    # create test user to change roles for
    username = 'test_user_1'

    # Change roles to empty list
    response = await async_client.put(
        '/access/change-roles',
        json={'username': username, 'roles': []},
        headers=admin_headers,
    )
    assert response.status_code == 200

    # check if the update was successful by attempting to create a benchmark
    # which should now be forbidden due to lack of roles
    user_token = await get_JWT_token(async_client, user=username, password='test1234')
    user_headers = {'Authorization': f'Bearer {user_token}'}
    response = await async_client.post(
        '/benchmarks/create', json=sample_benchmark.model_dump(), headers=user_headers
    )
    assert response.status_code == 403

    # Cleanup is handled by the standard_user1_headers fixture


@pytest.mark.anyio(loop_scope='session')
async def test_JWT_creation(async_client):
    response = await async_client.post(
        '/access/token',
        json={'username': 'admin', 'password': 'pytupli'},
    )
    assert response.status_code == 200

    # Check if Token is valid
    jwt = response.json()['access_token']['token']
    headers = {'Authorization': f'Bearer {jwt}'}
    response = await async_client.get('/benchmarks/list', headers=headers)
    assert response.status_code == 200


@pytest.mark.anyio(loop_scope='session')
async def test_JWT_creation_invalid_credentials(async_client):
    # Test with wrong password
    response = await async_client.post(
        '/access/token',
        json={'username': 'test_admin', 'password': 'wrongPw'},
    )
    assert response.status_code == 401

    # Test with nonexistent user
    response = await async_client.post(
        '/access/token',
        json={'username': 'user_imaginary', 'password': 'test1234'},
    )
    assert response.status_code == 401


@pytest.mark.anyio(loop_scope='session')
async def test_delete_user_and_contents(
    async_client, admin_headers, created_benchmark_user1, published_benchmark_user1
):
    _, benchmark = created_benchmark_user1
    _, published_benchmark = published_benchmark_user1

    # Delete user
    response = await async_client.delete(
        f'/access/delete-user?username={benchmark.created_by}', headers=admin_headers
    )
    assert response.status_code == 200

    # Check that all private benchmarks created by that user are deleted
    response = await async_client.get(
        f'/benchmarks/load?benchmark_id={benchmark.id}', headers=admin_headers
    )
    assert response.status_code == 404

    # Check that public benchmarks created by that user are not automatically deleted
    response = await async_client.get(
        f'/benchmarks/load?benchmark_id={published_benchmark.id}', headers=admin_headers
    )
    assert response.status_code == 200


if __name__ == '__main__':
    pytest.main(['-v', __file__])
