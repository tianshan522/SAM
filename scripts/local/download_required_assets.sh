#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"
mkdir -p pretrained_models

if ! command -v gdown >/dev/null 2>&1; then
  python3 -m pip install gdown
fi

gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
gdown "https://drive.google.com/u/0/uc?id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0&export=download" -O pretrained_models/psp_ffhq_encode.pt
gdown "https://drive.google.com/u/0/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT&export=download" -O pretrained_models/stylegan2-ffhq-config-f.pt
gdown "https://drive.google.com/u/0/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn&export=download" -O pretrained_models/model_ir_se50.pth
gdown "https://drive.google.com/u/0/uc?id=1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh&export=download" -O pretrained_models/dex_age_classifier.pth

if command -v wget >/dev/null 2>&1; then
  wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat" -O shape_predictor_68_face_landmarks.dat
elif command -v curl >/dev/null 2>&1; then
  curl -L "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat" -o shape_predictor_68_face_landmarks.dat
else
  echo "缺少 wget 或 curl，无法下载 shape_predictor_68_face_landmarks.dat" >&2
  exit 1
fi

echo "模型资产下载完成。"
