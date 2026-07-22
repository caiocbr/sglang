---
name: nvme-raid-docker-migration
description: Set up blank local NVMe disks as RAID0 scratch storage, migrate Docker and containerd data, and fix numeric UID permissions for bind-mounted Dev Container workspaces. Use for MNNVL or HPC nodes with small OS disks and local NVMe disks.
---

# NVMe RAID0 Docker migration

Use the bundled `scripts/setup-scratch-docker.sh` instead of assembling ad hoc
commands when the target layout matches this workflow.

## Safety requirements

1. Inspect `lsblk`, `findmnt /`, `sudo wipefs -n`, `sudo docker info`, and
   `/etc/docker/daemon.json` before making changes.
2. Dynamically identify and exclude the operating-system disk.
3. Never format a disk that has partitions or filesystem/RAID signatures.
4. Explain that RAID0 has no redundancy and local cloud NVMe data can disappear
   after host repair, redeployment, or deallocation. Use it only for scratch,
   caches, Docker data, or reproducible artifacts.
5. Confirm the user accepts formatting the blank disks and a brief Docker outage
   unless that authorization was already explicit.
6. Do not remove the original Docker data until Docker starts from the new root
   and the container/image ID sets match.
7. Ensure the invoking user belongs to the `docker` group. Because existing
   VS Code/SSH processes do not gain new supplementary groups, grant that user
   a socket ACL for the current session when `setfacl` is available; a fresh
   login or VS Code reconnect is still the persistent activation step.
8. Grant Dev Container ACLs at the complete bind-mounted workspace root, not
   only at a package subdirectory. Editors, Git, builds, and generated files may
   need write access anywhere in the repository, including `.git`.

## Run the bundled script

Defaults:

- Five data NVMe disks
- RAID device `/dev/md0`
- XFS mount `/mnt/scratch`
- Docker data root `/mnt/scratch/docker`
- containerd data root `/mnt/scratch/containerd`

Local execution:

```bash
bash ~/.copilot/skills/nvme-raid-docker-migration/scripts/setup-scratch-docker.sh
```

Remote execution:

```bash
ssh HOST 'bash -s' \
  < ~/.copilot/skills/nvme-raid-docker-migration/scripts/setup-scratch-docker.sh
```

Override defaults only when the inspected layout requires it:

```bash
EXPECTED_DATA_DISKS=4 \
MOUNT_POINT=/mnt/local \
ARRAY_DEVICE=/dev/md0 \
DOCKER_ROOT=/mnt/local/docker \
CONTAINERD_ROOT=/mnt/local/containerd \
bash ~/.copilot/skills/nvme-raid-docker-migration/scripts/setup-scratch-docker.sh
```

The script:

1. Finds the OS NVMe disk and refuses to include it.
2. Requires the expected number of whole, signature-free data NVMe disks.
3. Creates a 512 KiB-chunk RAID0 array and an XFS filesystem with `ftype=1`.
4. Persists RAID assembly in `/etc/mdadm/mdadm.conf` and the mount in
   `/etc/fstab` using the filesystem UUID.
5. Copies Docker and containerd data with hard links, ACLs, xattrs, sparse
   files, and numeric ownership preserved.
6. Updates Docker `daemon.json` structurally so existing settings such as the
   NVIDIA runtime remain intact.
7. Updates containerd's persistent `root` while leaving its transient `state`
   under `/run/containerd`.
8. Validates both configurations, compares container/image state, and rolls
   back automatically on failure.
9. Preserves non-root Docker access after service restarts by ensuring `docker`
   group membership and, when needed, applying a user ACL to the current socket.

## Grant a Dev Container access to a bind-mounted workspace

Bind mounts preserve numeric ownership. If the host workspace belongs to UID
1000 while the container user is UID 1003, mode `775` does not make the
workspace writable because UID 1003 is neither its owner nor a member of the
host file's group.

Do not use `chmod -R 777` or change the source tree's owner. Grant the container
UID a named POSIX ACL to the entire repository. If its matching Dev Container
is running, the helper discovers the UID automatically:

```bash
bash .claude/skills/nvme-raid-docker-migration/scripts/grant-devcontainer-workspace-access.sh \
  /home/azhpcuser/sglang
```

If the container is not running, pass its numeric UID explicitly while still
targeting the complete workspace:

```bash
bash .claude/skills/nvme-raid-docker-migration/scripts/grant-devcontainer-workspace-access.sh \
  /home/azhpcuser/sglang \
  1003
```

The helper preserves host ownership, grants the named UID read/write access,
preserves execution only on files already marked executable, and installs
default directory ACLs so newly created files and directories inherit access.
Apply it to the path used by the `devcontainer.local_folder` label, including
the repository's `.git` directory.

## Verify completion

```bash
sudo docker info --format '{{.DockerRootDir}}'
sudo containerd config dump | grep -E '^(root|state) = '
systemctl is-active docker
systemctl is-active containerd
docker version --format '{{.Server.Version}}'
findmnt /mnt/scratch
sudo mdadm --detail /dev/md0
grep /mnt/scratch /etc/fstab
grep 'ARRAY /dev/md0' /etc/mdadm/mdadm.conf
getfacl -p /path/to/workspace
```

Expected results are active Docker and containerd services, both persistent
data roots under the requested scratch mount, `raid0` in a clean state, and an
XFS mount at the requested mount point.
