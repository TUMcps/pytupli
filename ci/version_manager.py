#!/usr/bin/env python3
"""
Version management script for the pytupli project.
This script handles version bumping based on commit message tags.
"""

import os
import re
import sys
import semver
import shlex


def get_current_version(pyproject_path):
    """Extract current version from pyproject.toml."""
    with open(pyproject_path, 'r') as file:
        content = file.read()

    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        print('ERROR: Could not find version in pyproject.toml')
        sys.exit(1)

    return version_match.group(1)


def determine_increment_type(commit_message):
    """Determine version increment type based on commit message."""
    if '[major-release]' in commit_message:
        return 'major'
    elif '[minor-release]' in commit_message:
        return 'minor'
    else:
        return 'patch'


def calculate_new_version(current_version, increment_type):
    """Calculate new version based on semver rules."""
    version_info = semver.VersionInfo.parse(current_version)

    if increment_type == 'major':
        new_version = version_info.bump_major()
    elif increment_type == 'minor':
        new_version = version_info.bump_minor()
    else:
        new_version = version_info.bump_patch()

    return str(new_version)


def update_pyproject_toml(pyproject_path, current_version, new_version):
    """Update version in pyproject.toml."""
    with open(pyproject_path, 'r') as file:
        content = file.read()

    updated_content = content.replace(
        f'version = "{current_version}"', f'version = "{new_version}"'
    )

    with open(pyproject_path, 'w') as file:
        file.write(updated_content)


def update_docs_conf(conf_path, current_version, new_version):
    """Update version in docs/source/conf.py."""
    if not os.path.exists(conf_path):
        print(f'WARNING: {conf_path} does not exist, skipping')
        return

    with open(conf_path, 'r') as file:
        content = file.read()

    updated_content = content.replace(
        f"release = '{current_version}'", f"release = '{new_version}'"
    )

    with open(conf_path, 'w') as file:
        file.write(updated_content)


def main():
    """Main function to handle version bumping."""
    # Get paths from environment or use defaults
    project_root = os.environ.get('PROJECT_ROOT', '.')
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    docs_conf_path = os.path.join(project_root, 'docs/source/conf.py')
    commit_message = os.environ.get('CI_COMMIT_MESSAGE', '')

    # Get current version
    current_version = get_current_version(pyproject_path)
    print(f'Current version: {current_version}')

    # Determine increment type
    increment_type = determine_increment_type(commit_message)
    print(f'Increment type: {increment_type}')

    # Calculate new version
    new_version = calculate_new_version(current_version, increment_type)
    print(f'New version: {new_version}')

    # Update files
    update_pyproject_toml(pyproject_path, current_version, new_version)
    update_docs_conf(docs_conf_path, current_version, new_version)

    # Save the original commit message to use in RELEASE_DESCRIPTION
    cleaned_message = commit_message.replace('[major-release]', '').replace('[minor-release]', '').strip()
    if not cleaned_message:
        cleaned_message = f"Release version {new_version}"

    # Output for GitLab CI - properly escape the release description for shell
    with open('version.env', 'w') as f:
        f.write(f'NEW_VERSION={new_version}\n')
        f.write(f'RELEASE_DESCRIPTION={shlex.quote(cleaned_message)}\n')

    print(f'Version bumped from {current_version} to {new_version}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
