#!/bin/zsh
# DLP 平台登录 + smoke test（macOS / Linux）
# 账号信息从 scripts/local/dlp_config.py 读取
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_DIR="$REPO_DIR/scripts/local"

CONDA_ENV="sam-dlp"
PIP_INDEX_URL="http://pypi.truesightai.com/simple"
TRUSTED_HOST="pypi.truesightai.com"

export DLP_GPU_TYPE="${DLP_GPU_TYPE:-}"
export DLP_GPU_NUM="${DLP_GPU_NUM:-}"

# ── 读取 dlp_config.py ────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_DIR/dlp_config.py" ]]; then
  echo "错误：找不到 $CONFIG_DIR/dlp_config.py，请先填写配置文件。"
  exit 1
fi

eval "$(python3 - <<'PY'
import sys
sys.path.insert(0, '$CONFIG_DIR')
import dlp_config
print(f"DLP_USERNAME={dlp_config.DLP_USERNAME!r}")
print(f"DLP_PASSWORD={dlp_config.DLP_PASSWORD!r}")
if not DLP_GPU_TYPE:
    print(f"DLP_GPU_TYPE={dlp_config.DLP_GPU_TYPE!r}")
if not DLP_GPU_NUM:
    print(f"DLP_GPU_NUM={dlp_config.DLP_GPU_NUM}")
PY
)"

# 如果 config 里为空，从命令行读取
if [[ -z "$DLP_USERNAME" ]]; then
  printf "dlp_config.py 中 DLP_USERNAME 为空，请输入用户名: "
  read -r DLP_USERNAME
fi
if [[ -z "$DLP_PASSWORD" ]]; then
  printf "dlp_config.py 中 DLP_PASSWORD 为空，请输入密码: "
  read -rs DLP_PASSWORD
  echo
fi

# ── 检查 conda ────────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
  echo "未检测到 conda，请先安装 Miniforge/Conda"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "未找到 conda 环境: $CONDA_ENV"
  echo "请先执行: conda create -n $CONDA_ENV python=3.11 -y"
  exit 1
fi
echo "[info] conda 环境 $CONDA_ENV 已存在"

# ── 安装/更新 dlpctl ──────────────────────────────────────────────────────────
if ! conda run -n "$CONDA_ENV" python -c "import dlpctl" >/dev/null 2>&1; then
  echo "[info] 安装 dlpctl..."
  conda run -n "$CONDA_ENV" python -m pip install \
    -i "$PIP_INDEX_URL" --trusted-host "$TRUSTED_HOST" dlpctl
fi
echo "[info] dlpctl 版本: $(conda run -n "$CONDA_ENV" dlpctl version 2>/dev/null || echo unknown)"

# ── 登录 ──────────────────────────────────────────────────────────────────────
echo "[info] 登录 DLP 平台（用户: $DLP_USERNAME）..."
conda run -n "$CONDA_ENV" dlpctl login -u "$DLP_USERNAME" -p "$DLP_PASSWORD"
echo "[info] 登录成功"

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[info] 提交 smoke test（GPU=$DLP_GPU_TYPE ×$DLP_GPU_NUM）..."
cd "$REPO_DIR"
DLP_GPU_TYPE="$DLP_GPU_TYPE" DLP_GPU_NUM="$DLP_GPU_NUM" \
  conda run -n "$CONDA_ENV" python scripts/local/cloud/trial/smoke_test.py

echo ""
echo "=========================================="
echo "  全部检查通过，可以提交训练任务："
echo "  # 快速验证（500步，~10min）"
echo "  conda run -n $CONDA_ENV python scripts/local/cloud/trial/quick_train.py"
echo ""
echo "  # 正式训练（50000步）"
echo "  conda run -n $CONDA_ENV python scripts/local/cloud/train.py"
echo "=========================================="
