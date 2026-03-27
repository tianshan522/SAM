import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
PRETRAINED_ROOT = os.path.join(REPO_ROOT, "pretrained_models")

# ------------------------------------------------------------------
# 训练数据集路径（Humans 数据集，已整理为 256x256 PNG）
# ------------------------------------------------------------------
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "datasets", "humans_aligned")

dataset_paths = {
    "ffhq": os.path.join(DATASET_ROOT, "train"),
    "celeba_test": os.path.join(DATASET_ROOT, "test"),
}

model_paths = {
    "pretrained_psp_encoder": os.path.join(PRETRAINED_ROOT, "psp_ffhq_encode.pt"),
    "pretrained_psp": os.path.join(PRETRAINED_ROOT, "psp_ffhq_encode.pt"),
    "ir_se50": os.path.join(PRETRAINED_ROOT, "model_ir_se50.pth"),
    "stylegan_ffhq": os.path.join(PRETRAINED_ROOT, "stylegan2-ffhq-config-f.pt"),
    "shape_predictor": os.path.join(REPO_ROOT, "shape_predictor_68_face_landmarks.dat"),
    "age_predictor": os.path.join(PRETRAINED_ROOT, "dex_age_classifier.pth"),
}
