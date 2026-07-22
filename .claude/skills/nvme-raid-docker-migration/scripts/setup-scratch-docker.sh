#!/usr/bin/env bash
set -Eeuo pipefail

expected_data_disks=${EXPECTED_DATA_DISKS:-5}
mount_point=${MOUNT_POINT:-/mnt/scratch}
array_device=${ARRAY_DEVICE:-/dev/md0}
docker_root=${DOCKER_ROOT:-$mount_point/docker}
containerd_root=${CONTAINERD_ROOT:-$mount_point/containerd}
current_user=$(id -un)

for command in findmnt lsblk wipefs mdadm mkfs.xfs blkid rsync python3 docker dockerd containerd ctr; do
    if ! command -v "$command" >/dev/null; then
        echo "Required command is not installed: $command" >&2
        exit 1
    fi
done
if ! sudo -n true; then
    echo "Passwordless sudo is required for unattended migration" >&2
    exit 1
fi
if [[ $(id -u) -ne 0 ]] &&
   ! id -nG "$current_user" | tr ' ' '\n' | grep -qx docker; then
    sudo usermod -aG docker "$current_user"
fi
if ! systemctl is-active --quiet docker; then
    echo "Docker must be active before migration so its contents can be verified" >&2
    exit 1
fi
if ! systemctl is-active --quiet containerd; then
    echo "containerd must be active before migration so its contents can be verified" >&2
    exit 1
fi

root_source=$(findmnt -n -o SOURCE /)
root_disk=$(lsblk -n -o PKNAME "$root_source" | head -n 1)
if [[ -z $root_disk ]]; then
    echo "Could not resolve the operating-system disk from $root_source" >&2
    exit 1
fi

mapfile -t data_disks < <(
    lsblk -dn -o NAME,TYPE |
        awk '$2 == "disk" && $1 ~ /^nvme/ { print "/dev/" $1 }' |
        grep -v "^/dev/${root_disk}$"
)

echo "OS disk: /dev/$root_disk"
printf 'RAID0 data disks: %s\n' "${data_disks[*]}"

if [[ ! -e $array_device ]]; then
    if [[ ${#data_disks[@]} -ne $expected_data_disks ]]; then
        echo "Expected $expected_data_disks data NVMe disks, found ${#data_disks[@]}" >&2
        exit 1
    fi

    for disk in "${data_disks[@]}"; do
        if lsblk -n -o TYPE "$disk" | grep -q '^part$'; then
            echo "Refusing to overwrite partitioned disk: $disk" >&2
            exit 1
        fi
        if [[ -n $(sudo wipefs -n "$disk") ]]; then
            echo "Refusing to overwrite disk with an existing signature: $disk" >&2
            exit 1
        fi
    done

    sudo mdadm --create "$array_device" \
        --level=0 \
        --chunk=512 \
        --raid-devices="${#data_disks[@]}" \
        "${data_disks[@]}" \
        --force \
        --run
else
    raid_level=$(sudo mdadm --detail "$array_device" |
        awk -F ' : ' '/Raid Level/ { print $2 }')
    raid_devices=$(sudo mdadm --detail "$array_device" |
        awk -F ' : ' '/Raid Devices/ { print $2 }')
    if [[ $raid_level != raid0 || $raid_devices -ne $expected_data_disks ]]; then
        echo "Existing $array_device is not the expected RAID0 array" >&2
        exit 1
    fi
fi

filesystem=$(sudo blkid -s TYPE -o value "$array_device" 2>/dev/null || true)
if [[ -z $filesystem ]]; then
    sudo mkfs.xfs -f -n ftype=1 "$array_device"
elif [[ $filesystem != xfs ]]; then
    echo "Unexpected filesystem on $array_device: $filesystem" >&2
    exit 1
fi

sudo mkdir -p "$mount_point"
if mountpoint -q "$mount_point"; then
    mounted_source=$(findmnt -n -o SOURCE "$mount_point")
    if [[ $(readlink -f "$mounted_source") != $(readlink -f "$array_device") ]]; then
        echo "$mount_point is already mounted from $mounted_source" >&2
        exit 1
    fi
else
    sudo mount "$array_device" "$mount_point"
fi
sudo chown "$current_user:$(id -gn)" "$mount_point"

sudo mkdir -p /etc/mdadm
sudo touch /etc/mdadm/mdadm.conf
array_uuid=$(sudo mdadm --detail "$array_device" |
    awk -F ' : ' '/^[[:space:]]*UUID/ { print $2 }')
array_line=$(sudo mdadm --detail --scan |
    grep "UUID=$array_uuid" |
    head -n 1 || true)
if [[ -z $array_line ]]; then
    echo "Could not generate mdadm configuration for $array_device" >&2
    exit 1
fi
if ! sudo grep -Fqx "$array_line" /etc/mdadm/mdadm.conf; then
    printf '%s\n' "$array_line" | sudo tee -a /etc/mdadm/mdadm.conf >/dev/null
    sudo update-initramfs -u
fi

filesystem_uuid=$(sudo blkid -s UUID -o value "$array_device")
fstab_line="UUID=$filesystem_uuid $mount_point xfs defaults,nofail 0 2"
existing_fstab=$(sudo awk -v target="$mount_point" '$2 == target { print }' /etc/fstab)
if [[ -z $existing_fstab ]]; then
    printf '%s\n' "$fstab_line" | sudo tee -a /etc/fstab >/dev/null
elif [[ $existing_fstab != "$fstab_line" ]]; then
    echo "Existing fstab entry for $mount_point does not match $array_device" >&2
    exit 1
fi

current_docker_root=$(sudo docker info --format '{{.DockerRootDir}}')
if [[ $current_docker_root != "$docker_root" ]]; then
    before_containers=$(sudo docker ps -aq | sort)
    before_images=$(sudo docker image ls -aq | sort -u)
    config_backup=/etc/docker/daemon.json.pre-scratch
    data_backup=${current_docker_root}.pre-scratch
    migration_active=0
    config_existed=0

    rollback() {
        status=$?
        if [[ $status -eq 0 || $migration_active -eq 0 ]]; then
            return
        fi

        trap - EXIT
        set +e
        echo "Migration failed; restoring the previous Docker configuration" >&2
        sudo systemctl stop docker docker.socket
        if [[ $config_existed -eq 1 && -e $config_backup ]]; then
            sudo mv "$config_backup" /etc/docker/daemon.json
        elif [[ $config_existed -eq 0 ]]; then
            sudo rm -f /etc/docker/daemon.json
        fi
        if [[ -e $data_backup && ! -e $current_docker_root ]]; then
            sudo mv "$data_backup" "$current_docker_root"
        fi
        sudo systemctl start docker
    }
    trap rollback EXIT

    if [[ -e $config_backup || -e $data_backup ]]; then
        echo "A previous migration backup exists; refusing to overwrite it" >&2
        exit 1
    fi

    migration_active=1
    sudo systemctl stop docker docker.socket
    sudo mkdir -p "$docker_root"
    sudo rsync -aHAXS --numeric-ids --delete \
        "$current_docker_root/" "$docker_root/"
    if [[ -e /etc/docker/daemon.json ]]; then
        sudo cp -a /etc/docker/daemon.json "$config_backup"
        config_existed=1
    fi

    sudo python3 - "$docker_root" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path("/etc/docker/daemon.json")
data = json.loads(path.read_text()) if path.exists() else {}
data["data-root"] = sys.argv[1]
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(data, indent=4) + "\n")
os.chmod(temporary, 0o644)
os.replace(temporary, path)
PY

    sudo dockerd --validate --config-file=/etc/docker/daemon.json
    sudo mv "$current_docker_root" "$data_backup"
    sudo systemctl start docker

    docker_ready=0
    for _ in $(seq 1 30); do
        if sudo docker info >/dev/null 2>&1; then
            docker_ready=1
            break
        fi
        sleep 1
    done
    if [[ $docker_ready -ne 1 ]]; then
        echo "Docker did not become ready within 30 seconds" >&2
        exit 1
    fi

    actual_docker_root=$(sudo docker info --format '{{.DockerRootDir}}')
    after_containers=$(sudo docker ps -aq | sort)
    after_images=$(sudo docker image ls -aq | sort -u)
    if [[ $actual_docker_root != "$docker_root" ||
          $before_containers != "$after_containers" ||
          $before_images != "$after_images" ]]; then
        echo "Docker data verification failed" >&2
        exit 1
    fi

    migration_active=0
    trap - EXIT
    sudo rm -rf --one-file-system "$data_backup"
    sudo rm -f "$config_backup"
fi

containerd_effective_root() {
    sudo containerd config dump |
        sed -n -E "s/^root = ['\"]([^'\"]+)['\"]$/\\1/p" |
        head -n 1
}

current_containerd_root=$(containerd_effective_root)
if [[ -z $current_containerd_root ]]; then
    echo "Could not determine containerd's effective root directory" >&2
    exit 1
fi

if [[ $current_containerd_root != "$containerd_root" ]]; then
    before_containers=$(sudo docker ps -aq | sort)
    before_images=$(sudo docker image ls -aq | sort -u)
    before_namespaces=$(sudo ctr namespaces list -q | sort)
    containerd_config_backup=/etc/containerd/config.toml.pre-scratch
    containerd_data_backup=${current_containerd_root}.pre-scratch
    containerd_migration_active=0
    containerd_config_existed=0

    rollback_containerd() {
        status=$?
        if [[ $status -eq 0 || $containerd_migration_active -eq 0 ]]; then
            return
        fi

        trap - EXIT
        set +e
        echo "containerd migration failed; restoring its previous configuration" >&2
        sudo systemctl stop docker docker.socket containerd
        if [[ $containerd_config_existed -eq 1 &&
              -e $containerd_config_backup ]]; then
            sudo mv "$containerd_config_backup" /etc/containerd/config.toml
        elif [[ $containerd_config_existed -eq 0 ]]; then
            sudo rm -f /etc/containerd/config.toml
        fi
        if [[ -e $containerd_data_backup &&
              ! -e $current_containerd_root ]]; then
            sudo mv "$containerd_data_backup" "$current_containerd_root"
        fi
        sudo systemctl start containerd
        sudo systemctl start docker
    }
    trap rollback_containerd EXIT

    if [[ -e $containerd_config_backup ||
          -e $containerd_data_backup ]]; then
        echo "A previous containerd migration backup exists; refusing to overwrite it" >&2
        exit 1
    fi

    containerd_migration_active=1
    sudo systemctl stop docker docker.socket containerd
    sudo mkdir -p "$containerd_root"
    sudo rsync -aHAXS --numeric-ids --delete \
        "$current_containerd_root/" "$containerd_root/"
    if [[ -e /etc/containerd/config.toml ]]; then
        sudo cp -a /etc/containerd/config.toml "$containerd_config_backup"
        containerd_config_existed=1
    fi

    sudo python3 - "$containerd_root" <<'PY'
import os
import re
import subprocess
import sys
from pathlib import Path

path = Path("/etc/containerd/config.toml")
if path.exists():
    text = path.read_text()
    mode = path.stat().st_mode & 0o777
else:
    text = subprocess.check_output(
        ["containerd", "config", "default"],
        text=True,
    )
    mode = 0o644

replacement = f'root = "{sys.argv[1]}"'
updated, count = re.subn(
    r'(?m)^root\s*=\s*["\'][^"\']*["\']\s*$',
    replacement,
    text,
    count=1,
)
if count != 1:
    updated = replacement + "\n" + text

temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(updated)
os.chmod(temporary, mode)
os.replace(temporary, path)
PY

    sudo containerd config dump >/dev/null
    sudo mv "$current_containerd_root" "$containerd_data_backup"
    sudo systemctl start containerd

    containerd_ready=0
    for _ in $(seq 1 30); do
        if sudo ctr version >/dev/null 2>&1; then
            containerd_ready=1
            break
        fi
        sleep 1
    done
    if [[ $containerd_ready -ne 1 ]]; then
        echo "containerd did not become ready within 30 seconds" >&2
        exit 1
    fi

    sudo systemctl start docker
    docker_ready=0
    for _ in $(seq 1 30); do
        if sudo docker info >/dev/null 2>&1; then
            docker_ready=1
            break
        fi
        sleep 1
    done
    if [[ $docker_ready -ne 1 ]]; then
        echo "Docker did not become ready after containerd migration" >&2
        exit 1
    fi

    actual_containerd_root=$(containerd_effective_root)
    after_containers=$(sudo docker ps -aq | sort)
    after_images=$(sudo docker image ls -aq | sort -u)
    after_namespaces=$(sudo ctr namespaces list -q | sort)
    if [[ $actual_containerd_root != "$containerd_root" ||
          $before_containers != "$after_containers" ||
          $before_images != "$after_images" ||
          $before_namespaces != "$after_namespaces" ]]; then
        echo "containerd data verification failed" >&2
        exit 1
    fi

    containerd_migration_active=0
    trap - EXIT
    sudo rm -rf --one-file-system "$containerd_data_backup"
    sudo rm -f "$containerd_config_backup"
fi

if ! docker info >/dev/null 2>&1; then
    if command -v setfacl >/dev/null; then
        sudo setfacl -m "u:$current_user:rw" /var/run/docker.sock
    else
        echo "Reconnect the login or VS Code session to activate docker group membership" >&2
    fi
fi

echo "host=$(hostname)"
echo "docker_root=$(sudo docker info --format '{{.DockerRootDir}}')"
echo "containerd_root=$(containerd_effective_root)"
df -h / "$mount_point"
cat /proc/mdstat
