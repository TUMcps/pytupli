# Access Management in PyTupli

This document provides a comprehensive guide to the Role-Based Access Control (RBAC) system implemented in PyTupli. The system provides fine-grained access control for all resources and operations through a hierarchical group-based permission model.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Rights Reference](#rights-reference)
4. [Default Roles](#default-roles)
5. [Use Cases and Examples](#use-cases-and-examples)
6. [Group Management](#group-management)
7. [Content Publishing and Access](#content-publishing-and-access)
8. [Advanced Scenarios](#advanced-scenarios)

## Overview

The RBAC system controls access to all resources in Pytupli through a combination of:
- **Rights**: Specific permissions to perform actions (e.g., `ARTIFACT_READ`, `USER_CREATE`)
- **Roles**: Collections of rights (e.g., `admin`, `contributor`, `member`)
- **Groups**: Organizational units where users can have different roles
- **Memberships**: User assignments to groups with specific roles

### Key Features

- **Hierarchical Permissions**: Global, group-specific, and user-specific scopes
- **Ownership-Based Access**: Resource creators get full control over their content
- **Dynamic Group Creation**: Users can create groups and become group administrators
- **Role Validation**: Users can only assign roles they have rights for
- **Automatic Memberships**: Default memberships for global and user-specific groups

## Core Concepts

### Groups and Scopes

The system operates on three main scopes:

1. **Global Scope** (`global`): Platform-wide permissions
2. **Group Scope** (`group_name`): Permissions within specific groups
3. **User Scope** (`username`): Personal resource management

### Automatic Memberships

Every authenticated user automatically receives:
- **Global Member** role in the `global` group
- **Contributor** role in the `global` group
- **User Admin** role in their personal group (`username`)
- **Contributor** role in their personal group (`username`)

### Resource Types

The system manages access to six resource types:
- `ARTIFACT`: Data artifacts and files
- `BENCHMARK`: Benchmark definitions and metadata
- `EPISODE`: Execution episodes and results
- `USER`: User accounts and profiles
- `GROUP`: Group definitions and memberships
- `ROLE`: Role definitions and permissions

## Rights Reference

### Content Rights

| Right | Description |
|-------|-------------|
| `ARTIFACT_CREATE` | Create new artifacts |
| `ARTIFACT_READ` | Read artifacts |
| `ARTIFACT_DELETE` | Delete artifacts |
| `BENCHMARK_CREATE` | Create new benchmarks |
| `BENCHMARK_READ` | Read benchmarks |
| `BENCHMARK_DELETE` | Delete benchmarks |
| `EPISODE_CREATE` | Create new episodes |
| `EPISODE_READ` | Read episodes |
| `EPISODE_DELETE` | Delete episodes |

### User Management Rights

| Right | Description |
|-------|-------------|
| `USER_CREATE` | Create new user accounts |
| `USER_READ` | Read user information |
| `USER_UPDATE` | Update user information |
| `USER_DELETE` | Delete user accounts |

### Group Management Rights

| Right | Description |
|-------|-------------|
| `GROUP_CREATE` | Create new groups |
| `GROUP_READ` | Read group information |
| `GROUP_UPDATE` | Modify group memberships |
| `GROUP_DELETE` | Delete groups |

### System Rights

| Right | Description |
|-------|-------------|
| `ROLE_MANAGEMENT` | Create, modify, delete roles |

## Default Roles

### System Roles

| Role | Rights | Description |
|------|--------|-------------|
| `ADMIN` | All rights | Full platform administration |
| `CONTENT_ADMIN` | All content rights | Manage all content |
| `USER_ADMIN` | All user rights | User account management |
| `GROUP_ADMIN` | All group rights + USER_READ | Group management |

### Member Roles

| Role | Rights | Description |
|------|--------|-------------|
| `CONTRIBUTOR` | CREATE + READ content | Create and read content |
| `MEMBER` | READ content + GROUP_READ + USER_READ | Basic group member |
| `GLOBAL_MEMBER` | READ content + GROUP_CREATE + USER_READ | Platform member |
| `GUEST` | READ content only | Limited guest access |

### Automatic Roles

These roles are automatically assigned:

| Role | Automatic Assignment | Rights |
|------|---------------------|--------|
| `GLOBAL_MEMBER` | All users → `global` group | GROUP_CREATE, USER_READ, READ content |
| `CONTRIBUTOR` | All users → `global` + personal groups | CREATE + READ content |
| `USER_ADMIN` | All users → personal group | All user rights |

## Use Cases and Examples

### 1. Publishing Content

**Scenario**: A researcher wants to store and publish a benchmark for their research group.

```python
# User stores a benchmark - automatically published to their personal group
benchmark_query = BenchmarkQuery(
    name="ML Classification Benchmark",
    description="Benchmark for machine learning classification",
    # benchmark data here
)
benchmark_header = client.store_benchmark(benchmark_query)

# To share with research group, publish to that group
client.publish_benchmark(benchmark_header.id, publish_in="ml_research_group")
```

**Access Control**:
- ✅ Creator: Full access (read, delete, publish)
- ✅ `ml_research_group` members with READ rights: Can read
- ❌ Other users: No access
- ✅ Global admins: Full access

### 2. Adding Users to a Group

**Scenario**: A group administrator wants to add new members with different roles.

```python
# Only users with GROUP_UPDATE rights in the target group can do this
membership_query = GroupMembershipQuery(
    group_name="ml_research_group",
    members=[
        GroupMembership(user="new_researcher", roles=["contributor"]),
        GroupMembership(user="phd_student", roles=["member"])
    ]
)
client.add_group_members(membership_query)
```

**Authorization Requirements**:
- User must have `GROUP_UPDATE` right in `ml_research_group` OR global scope
- User must have `USER_READ` right globally (to add users)
- User must have all rights contained in assigned roles (can't assign roles with more rights than you have)

### 3. Creating a New Research Group

**Scenario**: A researcher wants to create a new group for their project.

```python
# User needs GROUP_CREATE right (automatic for authenticated users)
group = Group(
    name="new_project_group",
    description="Collaborative research project"
)
created_group = client.create_group(group)
```

**Automatic Behavior**:
- ✅ Creator becomes group administrator automatically
- ✅ Creator gets `GROUP_ADMIN` and `CONTRIBUTOR` roles in the new group
- ✅ Creator can now add/remove members and manage the group

### 4. Managing User Permissions

**Scenario**: An administrator wants to grant a user elevated permissions.

```python
# Create custom role (requires ROLE_MANAGEMENT right)
custom_role = UserRole(
    role="senior_researcher",
    rights=[
        RIGHT.ARTIFACT_CREATE, RIGHT.ARTIFACT_READ,
        RIGHT.BENCHMARK_CREATE, RIGHT.BENCHMARK_READ,
        RIGHT.GROUP_CREATE, RIGHT.USER_READ
    ],
    description="Senior researcher with extended permissions"
)
client.create_role(custom_role)

# Assign role to user in a group
membership_query = GroupMembershipQuery(
    group_name="research_institute",
    members=[GroupMembership(user="senior_user", roles=["senior_researcher"])]
)
client.add_group_members(membership_query)
```

### 5. Accessing Published Content

**Scenario**: A user wants to read benchmarks published to their groups.

**Automatic Filtering**: The system automatically filters content based on user's group memberships and rights:

```python
# System automatically includes only accessible benchmarks
benchmarks = client.list_benchmarks()
# Optional: use filters to narrow results
from pytupli.schema import BaseFilter, FilterType
filter_obj = BaseFilter(type=FilterType.EQ, field="name", value="ML Classification")
filtered_benchmarks = client.list_benchmarks(filter=filter_obj)
```

**Access Logic**:
- ✅ Content published to groups where user has READ rights
- ✅ Content created by the user (ownership)
- ❌ Content in groups where user lacks READ rights

### 6. Deleting Resources

**Scenario**: Different users trying to delete a benchmark.

**Content Creator**:
```python
# Creator can always delete their content
client.delete_benchmark(benchmark_id)  # ✅ Success
```

**Global Admin**:
```python
# Global admin can delete any content
client.delete_benchmark(benchmark_id)  # ✅ Success
```

**Regular User**:
```python
# Regular user cannot delete others' content
client.delete_benchmark(benchmark_id)  # ❌ 403 Forbidden
```

### 7. Working with Artifacts and Episodes

**Scenario**: Creating and managing different types of content.

```python
# Store artifacts with metadata
artifact_data = b"binary data here"
metadata = ArtifactMetadata(
    name="dataset.csv",
    content_type="text/csv",
    description="Training dataset"
)
artifact_item = client.store_artifact(artifact_data, metadata)

# Publish artifact to a group
client.publish_artifact(artifact_item.id, publish_in="ml_research_group")

# Record episodes
episode = Episode(
    name="training_run_1",
    description="First training run"
    # episode data here
)
episode_header = client.record_episode(episode)

# Publish episode
client.publish_episode(episode_header.id, publish_in="ml_research_group")
```

## Group Management

### Creating Groups

Any authenticated user can create groups (automatic `GROUP_CREATE` right):

```python
group = Group(name="my_team", description="My research team")
client.create_group(group)
```

### Group Administration

Group creators automatically receive administrative rights:

- ✅ Add/remove members
- ✅ Assign roles to members
- ✅ Read group information
- ✅ Delete the group

### Member Management

```python
# Add members with specific roles
add_query = GroupMembershipQuery(
    group_name="my_team",
    members=[
        GroupMembership(user="colleague1", roles=["contributor"]),
        GroupMembership(user="colleague2", roles=["member"])
    ]
)
client.add_group_members(add_query)

# Remove members
remove_query = GroupMembershipQuery(
    group_name="my_team",
    members=[GroupMembership(user="colleague1")]
)
client.remove_group_members(remove_query)
```

### Role Assignment Validation

The system validates that users can only assign roles they have rights for:

```python
# This will fail if current user doesn't have all rights in "admin" role
membership = GroupMembership(user="new_user", roles=["admin"])
# HTTPException: You do not have sufficient rights to assign role admin
```

## Advanced Scenarios

### Multi-Group Collaboration

**Scenario**: Content shared across multiple research groups.

```python
# Publish content to multiple groups using separate publish calls
benchmark_id = "benchmark_hash_id_here"
client.publish_benchmark(benchmark_id, publish_in="group_a")
client.publish_benchmark(benchmark_id, publish_in="group_b")
client.publish_benchmark(benchmark_id, publish_in="group_c")

# Or publish to global for everyone
client.publish_benchmark(benchmark_id, publish_in="global")
```

**Access**: Users need READ rights in ANY of the published groups.

### Hierarchical Group Management

**Scenario**: Department admin managing multiple research groups.

```python
# Department admin creates multiple groups
dept_groups = ["ml_team", "nlp_team", "vision_team"]
for group_name in dept_groups:
    group = Group(name=group_name, description=f"{group_name} research group")
    client.create_group(group)

# Admin can manage all groups they created
for group_name in dept_groups:
    client.add_group_members(GroupMembershipQuery(
        group_name=group_name,
        members=[GroupMembership(user="new_researcher", roles=["contributor"])]
    ))
```

### Custom Role Creation

**Scenario**: Organization-specific roles.

```python
# Create specialized roles for different user types
reviewer_role = UserRole(
    role="paper_reviewer",
    rights=[RIGHT.BENCHMARK_READ, RIGHT.ARTIFACT_READ, RIGHT.EPISODE_READ],
    description="Can review submitted papers and benchmarks"
)

annotator_role = UserRole(
    role="data_annotator",
    rights=[RIGHT.ARTIFACT_CREATE, RIGHT.ARTIFACT_READ],
    description="Can create and read annotation artifacts"
)

# Requires ROLE_MANAGEMENT right (global admin only)
client.create_role(reviewer_role)
client.create_role(annotator_role)
```

## Security Considerations

### Authentication
- JWT-based token authentication
- Configurable open access mode for public instances
- Secure password hashing with bcrypt

### Authorization
- All operations require explicit rights checking
- Ownership-based access for resource creators
- Role assignment validation prevents privilege escalation
- Automatic permission filtering for list operations

### Data Integrity
- Group deletion removes all related memberships
- Role deletion removes role from all user memberships
- User deletion removes all created content
- Referential integrity maintained across all operations

## Configuration

### Environment Variables

- `OPEN_ACCESS_MODE`: Allow unauthenticated access (default: false)
- `OPEN_SIGNUP_MODE`: Allow public user registration (default: false)
- `API_SECRET_KEY`: JWT signing secret (required)

### Default Admin Setup

The system automatically creates a default admin user on first startup. This admin has full `ADMIN` role privileges and can manage all aspects of the system.
