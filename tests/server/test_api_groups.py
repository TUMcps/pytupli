
import pytest
from pytupli.schema import Group, GroupMembershipQuery, GroupMembership, RIGHT

pytestmark = pytest.mark.anyio

def group_data():
    return Group(name="test_group", description="A test group")

def group_data_no_desc():
    return Group(name="test_group_no_desc")

async def create_group(client, headers, group):
    return await client.post("/access/groups/create", json=group.model_dump(), headers=headers)

async def delete_group(client, headers, name):
    return await client.delete(f"/access/groups/delete?group_name={name}", headers=headers)

async def create_role(client, headers, role):
    return await client.post("/access/roles/create", json=role, headers=headers)

async def delete_role(client, headers, name):
    return await client.delete(f"/access/roles/delete?role_name={name}", headers=headers)

async def create_user(client, headers, username):
    return await client.post("/access/users/create", json={"username": username, "password": "test1234"}, headers=headers)

async def delete_user(client, headers, username):
    return await client.delete(f"/access/users/delete?username={username}", headers=headers)


async def test_create_group_success(async_client, admin_headers):
    group = group_data()
    try:
        resp = await create_group(async_client, admin_headers, group)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == group.name
        assert data["description"] == group.description
    finally:
        await delete_group(async_client, admin_headers, group.name)


async def test_create_group_no_description(async_client, admin_headers):
    group = group_data_no_desc()
    try:
        resp = await create_group(async_client, admin_headers, group)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == group.name
        assert data["description"] is None
    finally:
        await delete_group(async_client, admin_headers, group.name)


async def test_create_group_conflict(async_client, admin_headers):
    group = group_data()
    try:
        await create_group(async_client, admin_headers, group)
        resp = await create_group(async_client, admin_headers, group)
        assert resp.status_code == 409
        assert "Group already exists" in resp.json()["detail"]
    finally:
        await delete_group(async_client, admin_headers, group.name)


async def test_create_group_standard_user_success(async_client, standard_user1_headers):
    group = group_data()
    try:
        resp = await create_group(async_client, standard_user1_headers, group)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == group.name
    finally:
        await delete_group(async_client, standard_user1_headers, group.name)


async def test_list_groups_admin(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    try:
        resp = await async_client.get("/access/groups/list", headers=admin_headers)
        assert resp.status_code == 200
        groups = resp.json()
        assert isinstance(groups, list)
        assert any(g["name"] == group.name for g in groups)
    finally:
        await delete_group(async_client, admin_headers, group.name)


async def test_list_groups_standard_user_empty(async_client, standard_user1_headers):
    resp = await async_client.get("/access/groups/list", headers=standard_user1_headers)
    assert resp.status_code == 200
    groups = resp.json()
    assert isinstance(groups, list)
    assert len(groups) == 0


async def test_list_groups_standard_user_with_membership(async_client, admin_headers, standard_user1_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    test_role = {"role": "test_role", "description": "Test role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="test_user_1", roles=["test_role"])]
    )
    await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    try:
        resp = await async_client.get("/access/groups/list", headers=standard_user1_headers)
        assert resp.status_code == 200
        groups = resp.json()
        assert any(g["name"] == group.name for g in groups)
    finally:
        await delete_role(async_client, admin_headers, "test_role")
        await delete_group(async_client, admin_headers, group.name)


async def test_read_group_success(async_client, admin_headers):
    group = group_data()
    await delete_group(async_client, admin_headers, group.name)
    await create_group(async_client, admin_headers, group)
    try:
        resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == group.name
        assert data["description"] == group.description
        assert "members" in data
        assert isinstance(data["members"], list)
    finally:
        await delete_group(async_client, admin_headers, group.name)


async def test_read_group_not_found(async_client, admin_headers):
    resp = await async_client.get("/access/groups/read?group_name=nonexistent_group", headers=admin_headers)
    assert resp.status_code == 404
    assert "Group not found" in resp.json()["detail"]


async def test_read_group_unauthorized(async_client, standard_user1_headers):
    group = group_data()
    resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=standard_user1_headers)
    assert resp.status_code == 403


async def test_delete_group_success(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    resp = await delete_group(async_client, admin_headers, group.name)
    assert resp.status_code == 200
    resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
    assert resp.status_code == 404


async def test_delete_group_not_found(async_client, admin_headers):
    resp = await delete_group(async_client, admin_headers, "nonexistent_group")
    assert resp.status_code == 200


async def test_delete_group_unauthorized(async_client, standard_user1_headers):
    group = group_data()
    resp = await delete_group(async_client, standard_user1_headers, group.name)
    assert resp.status_code == 403


async def test_delete_group_removes_user_memberships(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    test_role = {"role": "test_role", "description": "Test role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    await create_user(async_client, admin_headers, "group_user1")
    await create_user(async_client, admin_headers, "group_user2")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[
            GroupMembership(user="group_user1", roles=["test_role"]),
            GroupMembership(user="group_user2", roles=["test_role"])
        ]
    )
    await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    users_resp = await async_client.get("/access/users/list", headers=admin_headers)
    users = users_resp.json()
    user1 = next(u for u in users if u["username"] == "group_user1")
    user2 = next(u for u in users if u["username"] == "group_user2")
    assert any(m["group"] == group.name for m in user1["memberships"])
    assert any(m["group"] == group.name for m in user2["memberships"])
    resp = await delete_group(async_client, admin_headers, group.name)
    assert resp.status_code == 200
    users_resp = await async_client.get("/access/users/list", headers=admin_headers)
    users = users_resp.json()
    user1 = next(u for u in users if u["username"] == "group_user1")
    user2 = next(u for u in users if u["username"] == "group_user2")
    assert not any(m["group"] == group.name for m in user1["memberships"])
    assert not any(m["group"] == group.name for m in user2["memberships"])
    await delete_user(async_client, admin_headers, "group_user1")
    await delete_user(async_client, admin_headers, "group_user2")
    await delete_role(async_client, admin_headers, "test_role")


async def test_add_members_success(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    test_role = {"role": "test_role", "description": "Test role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=["test_role"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == group.name
    assert data["description"] == group.description
    read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert len(read_data["members"]) == 2
    assert read_data["members"][1] == "member_user"
    await delete_user(async_client, admin_headers, "member_user")
    await delete_role(async_client, admin_headers, "test_role")
    await delete_group(async_client, admin_headers, group.name)


async def test_add_members_group_not_found(async_client, admin_headers):
    membership_query = GroupMembershipQuery(
        group_name="nonexistent_group",
        members=[GroupMembership(user="test_user", roles=["test_role"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 404
    assert "Group not found" in resp.json()["detail"]


async def test_add_members_user_not_found(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    test_role = {"role": "test_role", "description": "Test role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="nonexistent_user", roles=["test_role"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 404
    assert "User nonexistent_user not found" in resp.json()["detail"]
    await delete_role(async_client, admin_headers, "test_role")
    await delete_group(async_client, admin_headers, group.name)


async def test_add_members_role_not_found(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=["nonexistent_role"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 404
    assert "Role nonexistent_role not found" in resp.json()["detail"]
    await delete_user(async_client, admin_headers, "member_user")
    await delete_group(async_client, admin_headers, group.name)


async def test_add_members_no_roles_skipped(async_client, admin_headers):
    group = group_data()
    await delete_group(async_client, admin_headers, group.name)
    await delete_user(async_client, admin_headers, "member_user")
    await create_group(async_client, admin_headers, group)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=[])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert len(read_data["members"]) == 1
    assert read_data["members"][0] == "admin"  # only the creator remains
    await delete_user(async_client, admin_headers, "member_user")
    await delete_group(async_client, admin_headers, group.name)


async def test_add_members_unauthorized(async_client, standard_user1_headers):
    group = group_data()
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="test_user", roles=["test_role"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=standard_user1_headers)
    assert resp.status_code == 403


async def test_add_members_replaces_existing_membership(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    test_role1 = {"role": "test_role1", "description": "Test role 1", "rights": [RIGHT.ARTIFACT_READ]}
    test_role2 = {"role": "test_role2", "description": "Test role 2", "rights": [RIGHT.ARTIFACT_CREATE]}
    await create_role(async_client, admin_headers, test_role1)
    await create_role(async_client, admin_headers, test_role2)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=["test_role1"])]
    )
    await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=["test_role2"])]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    users_resp = await async_client.get("/access/users/list", headers=admin_headers)
    users = users_resp.json()
    user = next(u for u in users if u["username"] == "member_user")
    group_memberships = [m for m in user["memberships"] if m["group"] == group.name]
    assert len(group_memberships) == 1
    assert "test_role2" in group_memberships[0]["roles"]
    assert "test_role1" not in group_memberships[0]["roles"]
    await delete_user(async_client, admin_headers, "member_user")
    await delete_role(async_client, admin_headers, "test_role1")
    await delete_role(async_client, admin_headers, "test_role2")
    await delete_group(async_client, admin_headers, group.name)


async def test_remove_members_success(async_client, admin_headers):
    group = group_data()
    await delete_group(async_client, admin_headers, group.name)
    await delete_user(async_client, admin_headers, "member_user")
    await delete_role(async_client, admin_headers, "test_role")
    await create_group(async_client, admin_headers, group)
    test_role = {"role": "test_role", "description": "Test role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user", roles=["test_role"])]
    )
    await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=admin_headers)
    read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert len(read_data["members"]) == 2
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user")]
    )
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=admin_headers)
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert len(read_data["members"]) == 1
    assert read_data["members"][0] == "admin"  # only the creator remains
    await delete_user(async_client, admin_headers, "member_user")
    await delete_role(async_client, admin_headers, "test_role")
    await delete_group(async_client, admin_headers, group.name)


async def test_remove_members_group_not_found(async_client, admin_headers):
    membership_query = GroupMembershipQuery(
        group_name="nonexistent_group",
        members=[GroupMembership(user="test_user")]
    )
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 404
    assert "Group not found" in resp.json()["detail"]


async def test_remove_members_user_not_found(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="nonexistent_user")]
    )
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    await delete_group(async_client, admin_headers, group.name)


async def test_remove_members_not_in_group(async_client, admin_headers):
    group = group_data()
    await create_group(async_client, admin_headers, group)
    await create_user(async_client, admin_headers, "member_user")
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="member_user")]
    )
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=admin_headers)
    assert resp.status_code == 200
    await delete_user(async_client, admin_headers, "member_user")
    await delete_group(async_client, admin_headers, group.name)


async def test_remove_members_unauthorized(async_client, standard_user1_headers):
    group = group_data()
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="test_user")]
    )
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=standard_user1_headers)
    assert resp.status_code == 403


async def test_group_auth_required_no_headers(async_client):
    group = group_data()
    resp = await async_client.post("/access/groups/create", json=group.model_dump())
    assert resp.status_code == 403
    resp = await async_client.get("/access/groups/list")
    assert resp.status_code == 200
    assert resp.json() == []
    resp = await async_client.get(f"/access/groups/read?group_name={group.name}")
    assert resp.status_code == 403
    resp = await async_client.delete(f"/access/groups/delete?group_name={group.name}")
    assert resp.status_code == 403
    membership_query = GroupMembershipQuery(
        group_name=group.name,
        members=[GroupMembership(user="test_user")]
    )
    resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump())
    assert resp.status_code == 403
    resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump())
    assert resp.status_code == 403


async def test_group_creator_can_manage_members(async_client, admin_headers, standard_user1_headers, standard_user2_headers):
    """Test that a user who creates a group gets admin rights and can add/remove members."""
    group = group_data()
    await delete_group(async_client, admin_headers, group.name)

    # Create test role and users
    test_role = {"role": "test_member_role", "description": "Test member role", "rights": [RIGHT.ARTIFACT_READ]}
    await create_role(async_client, admin_headers, test_role)
    await create_user(async_client, admin_headers, "test_member_user")

    try:
        # Standard user 1 creates a group (should get group admin rights automatically)
        resp = await create_group(async_client, standard_user1_headers, group)
        assert resp.status_code == 200

        # Verify the creator can read the group
        read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=standard_user1_headers)
        assert read_resp.status_code == 200
        group_data_response = read_resp.json()
        assert group_data_response["name"] == group.name

        # The group creator (standard_user1) should be able to add members to their group
        membership_query = GroupMembershipQuery(
            group_name=group.name,
            members=[GroupMembership(user="test_member_user", roles=["test_member_role"])]
        )
        resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=standard_user1_headers)
        assert resp.status_code == 200

        # Verify the member was added (should be 2 members: creator + added member)
        read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=standard_user1_headers)
        assert read_resp.status_code == 200
        group_data_response = read_resp.json()
        assert len(group_data_response["members"]) == 2

        # Check that both the creator and the added member are in the group
        member_usernames = {member for member in group_data_response["members"]}
        assert "test_user_1" in member_usernames  # the group creator
        assert "test_member_user" in member_usernames  # the added member

        # The group creator should be able to remove members from their group
        membership_query = GroupMembershipQuery(
            group_name=group.name,
            members=[GroupMembership(user="test_member_user")]
        )
        resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=standard_user1_headers)
        assert resp.status_code == 200

        # Verify the member was removed (should be 1 member: just the creator)
        read_resp = await async_client.get(f"/access/groups/read?group_name={group.name}", headers=standard_user1_headers)
        assert read_resp.status_code == 200
        group_data_response = read_resp.json()
        assert len(group_data_response["members"]) == 1
        assert group_data_response["members"][0] == "test_user_1"  # only the creator remains

        # However, a different non-admin user (standard_user2) should NOT be able to add members to user1's group
        membership_query = GroupMembershipQuery(
            group_name=group.name,
            members=[GroupMembership(user="test_member_user", roles=["test_member_role"])]
        )
        resp = await async_client.post("/access/groups/add-members", json=membership_query.model_dump(), headers=standard_user2_headers)
        assert resp.status_code == 403

        # And standard_user2 should NOT be able to remove members from user1's group
        resp = await async_client.post("/access/groups/remove-members", json=membership_query.model_dump(), headers=standard_user2_headers)
        assert resp.status_code == 403

    finally:
        # Cleanup
        await delete_user(async_client, admin_headers, "test_member_user")
        await delete_role(async_client, admin_headers, "test_member_role")
        await delete_group(async_client, admin_headers, group.name)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
