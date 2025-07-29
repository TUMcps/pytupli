import pytest
from httpx import AsyncClient
from pytupli.schema import RIGHT, UserRole

pytestmark = pytest.mark.anyio

# Fixtures: async_client, admin_headers, standard_user1_headers are expected from conftest.py

@pytest.fixture(scope="function")
def test_role_data():
    return UserRole(
        role="test_role",
        description="A test role",
        rights=[RIGHT.ARTIFACT_READ]
    )

async def test_create_role_success(async_client, admin_headers, test_role_data):
    resp = await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["role"] == test_role_data.role
    # Cleanup
    await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=admin_headers)

async def test_create_role_conflict(async_client, admin_headers, test_role_data):
    # Create role first
    await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    # Try to create again
    resp = await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    assert resp.status_code == 409
    # Cleanup
    await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=admin_headers)

async def test_list_roles(async_client, admin_headers, test_role_data):
    # Create role
    await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    resp = await async_client.get("/access/roles/list", headers=admin_headers)
    assert resp.status_code == 200
    roles = resp.json()
    assert any(r["role"] == test_role_data.role for r in roles)
    # Cleanup
    await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=admin_headers)

async def test_delete_role_success(async_client, admin_headers, test_role_data):
    # Create role
    await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    resp = await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=admin_headers)
    assert resp.status_code == 200
    # Should not exist anymore
    resp = await async_client.get("/access/roles/list", headers=admin_headers)
    assert all(r["role"] != test_role_data.role for r in resp.json())

async def test_delete_role_not_found(async_client, admin_headers):
    resp = await async_client.delete("/access/roles/delete?role_name=nonexistent_role", headers=admin_headers)
    assert resp.status_code == 200

async def test_role_auth_required(async_client, standard_user1_headers, test_role_data):
    # Standard user should not be able to create/list/delete roles
    resp = await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=standard_user1_headers)
    assert resp.status_code == 403
    resp = await async_client.get("/access/roles/list", headers=standard_user1_headers)
    assert resp.status_code == 403
    resp = await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=standard_user1_headers)
    assert resp.status_code == 403

# Additional coverage: test user membership update on role delete
async def test_delete_role_removes_from_user(async_client, admin_headers, test_role_data):
    # Create role
    await async_client.post("/access/roles/create", json=test_role_data.model_dump(), headers=admin_headers)
    # Assign role to user (simulate membership)
    # Create user
    await async_client.post("/access/users/create", json={"username": "role_user", "password": "test1234"}, headers=admin_headers)
    # Manually update user memberships to include the role
    update = {"$set": {"memberships": [{"resource": "global", "roles": [test_role_data.role]}]}}
    await async_client.put("/access/users/update?username=role_user", json=update, headers=admin_headers)
    # Delete role
    resp = await async_client.delete(f"/access/roles/delete?role_name={test_role_data.role}", headers=admin_headers)
    assert resp.status_code == 200
    # Check user memberships
    user_resp = await async_client.get("/access/users/list", headers=admin_headers)
    user = next(u for u in user_resp.json() if u["username"] == "role_user")
    assert all(test_role_data.role not in m["roles"] for m in user["memberships"])
    # Cleanup
    await async_client.delete("/access/users/delete?username=role_user", headers=admin_headers)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
