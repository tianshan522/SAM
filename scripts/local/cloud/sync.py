"""
云端文件同步脚本
通过 SFTP/SSH 将本地项目代码同步到云端 NFS 存储，无需挂载 NFS。

用法（在 SAM conda 环境中运行）：
    conda run -p .conda\sam python scripts/local/cloud/sync.py
"""

import os
import sys
import stat
import fnmatch

# ──────────────────────────────────────────────────────────────────────────────
# 路径解析（绝对路径，避免 cwd 问题）
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))          # cloud/
_LOCAL_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "..")     # scripts/local/cloud/../../../.. = SAM 根目录
_LOCAL_DIR = os.path.normpath(_LOCAL_DIR)                        # D:\work\SAM

# 将 scripts/local 添加到 sys.path 以便导入 dlp_config
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
import dlp_config  # noqa: E402

try:
    import paramiko
except ImportError:
    print("[错误] 缺少 paramiko 依赖，请先运行：conda run -p .conda\\sam pip install paramiko")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 配置
NFS_HOST      = "172.30.2.93"
NFS_PORT      = 22
NFS_USER      = dlp_config.DLP_USERNAME
NFS_PASSWORD  = dlp_config.DLP_PASSWORD
REMOTE_DIR    = dlp_config.CLOUD_REPO_DIR.replace("\\", "/")   # Linux 路径

# 不同步的目录/文件（相对于项目根目录的 glob 模式）
EXCLUDE_PATTERNS = [
    ".conda",
    ".git",
    "experiments",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    "workspace/datasets",
    "scripts/local/dlp_config.py",
]

# ──────────────────────────────────────────────────────────────────────────────

def is_excluded(rel_path: str) -> bool:
    """判断相对路径是否应被排除"""
    # 统一使用正斜杠
    rel_path_fwd = rel_path.replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        # 匹配路径前缀（目录）
        if rel_path_fwd == pattern or rel_path_fwd.startswith(pattern + "/"):
            return True
        # 匹配 glob（文件名）
        basename = os.path.basename(rel_path_fwd)
        if fnmatch.fnmatch(basename, pattern):
            return True
    return False


def ensure_remote_dirs(sftp: "paramiko.SFTPClient", remote_path: str):
    """递归创建远端目录（类似 mkdir -p）"""
    parts = remote_path.split("/")
    current = ""
    for part in parts:
        if not part:
            current += "/"
            continue
        current = current.rstrip("/") + "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def sync(local_root: str, remote_root: str, sftp: "paramiko.SFTPClient"):
    uploaded = 0
    skipped  = 0

    for dirpath, dirnames, filenames in os.walk(local_root):
        # 过滤排除的子目录（原地修改 dirnames 以阻止 os.walk 深入）
        rel_dir = os.path.relpath(dirpath, local_root)
        if rel_dir == ".":
            rel_dir = ""

        # 排除整个目录
        dirnames[:] = [
            d for d in dirnames
            if not is_excluded(os.path.join(rel_dir, d) if rel_dir else d)
        ]

        for filename in filenames:
            rel_file = os.path.join(rel_dir, filename) if rel_dir else filename
            if is_excluded(rel_file):
                skipped += 1
                continue

            local_path  = os.path.join(dirpath, filename)
            remote_path = remote_root.rstrip("/") + "/" + rel_file.replace("\\", "/")
            remote_dir  = remote_path.rsplit("/", 1)[0]

            ensure_remote_dirs(sftp, remote_dir)
            sftp.put(local_path, remote_path)
            print(f"  ↑ {rel_file}")
            uploaded += 1

    return uploaded, skipped


def main():
    print(f"[同步] 本地: {_LOCAL_DIR}")
    print(f"[同步] 远端: {NFS_USER}@{NFS_HOST}:{REMOTE_DIR}")
    print(f"[同步] 排除: {EXCLUDE_PATTERNS}\n")

    transport = paramiko.Transport((NFS_HOST, NFS_PORT))
    transport.connect(username=NFS_USER, password=NFS_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        uploaded, skipped = sync(_LOCAL_DIR, REMOTE_DIR, sftp)
    finally:
        sftp.close()
        transport.close()

    print(f"\n[完成] 上传 {uploaded} 个文件，跳过 {skipped} 个文件。")
    print(f"[完成] 云端路径：{REMOTE_DIR}")


if __name__ == "__main__":
    main()
