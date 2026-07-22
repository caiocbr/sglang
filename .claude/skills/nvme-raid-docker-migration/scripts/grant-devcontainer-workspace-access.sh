#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 WORKSPACE [CONTAINER_UID]" >&2
    exit 2
fi

workspace=$(realpath -e -- "$1")
container_uid=${2:-}

if [[ ! -d $workspace ]]; then
    echo "Workspace is not a directory: $workspace" >&2
    exit 1
fi
case "$workspace" in
    / | /home | "$HOME")
        echo "Refusing to grant recursive access to broad path: $workspace" >&2
        exit 1
        ;;
esac

if ! command -v setfacl >/dev/null; then
    echo "setfacl is required; install the acl package first" >&2
    exit 1
fi

if [[ -z $container_uid ]]; then
    if ! command -v docker >/dev/null; then
        echo "Docker is required to discover the container UID" >&2
        exit 1
    fi

    mapfile -t container_ids < <(
        docker ps -q --filter "label=devcontainer.local_folder=$workspace"
    )
    if [[ ${#container_ids[@]} -ne 1 ]]; then
        echo "Expected one running Dev Container for $workspace; found ${#container_ids[@]}" >&2
        echo "Pass CONTAINER_UID explicitly when granting a subdirectory" >&2
        exit 1
    fi
    container_uid=$(docker exec "${container_ids[0]}" id -u)
fi

if [[ ! $container_uid =~ ^[0-9]+$ || $container_uid -eq 0 ]]; then
    echo "Container UID must be a non-root numeric UID: $container_uid" >&2
    exit 1
fi

sudo find "$workspace" -type d \
    -exec setfacl -m "u:$container_uid:rwx,d:u:$container_uid:rwx" {} +
sudo find "$workspace" -type f ! -perm /111 \
    -exec setfacl -m "u:$container_uid:rw-" {} +
sudo find "$workspace" -type f -perm /111 \
    -exec setfacl -m "u:$container_uid:rwx" {} +

echo "Granted UID $container_uid inherited read/write access to $workspace"
