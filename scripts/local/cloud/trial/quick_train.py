#!/usr/bin/env python3
"""
【临时验证】提交 500 步快速训练，验证云端环境、数据路径、模型加载是否全部正常。

约 5-10 分钟跑完。确认无误后再运行 cloud/train.py 执行完整训练。

用法:
    conda run -p .conda\sam python scripts/local/cloud/trial/quick_train.py
"""
import sys
import time
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CONFIG_DIR))
try:
    import dlp_config
except ImportError:
    print("错误：找不到 scripts/local/dlp_config.py，请先参考模板填写配置。")
    sys.exit(1)

if not dlp_config.CLOUD_REPO_DIR or dlp_config.CLOUD_REPO_DIR == "/cloud/tmp/your_name/SAM":
    print("错误：请在 scripts/local/dlp_config.py 中将 CLOUD_REPO_DIR 修改为您的实际路径。")
    sys.exit(1)

# 强制使用 Linux 路径分隔符（脚本在 Windows 上运行，目标是 Linux 云端）
CLOUD_REPO_DIR = dlp_config.CLOUD_REPO_DIR.replace("\\", "/")
CLOUD_EXP_DIR  = f"{CLOUD_REPO_DIR}/experiments/quick_trial"
CLOUD_CKPT     = f"{CLOUD_REPO_DIR}/pretrained_models/sam_ffhq_aging.pt"

JOB_CONFIG_GPU_NUM  = dlp_config.DLP_GPU_NUM
JOB_CONFIG_GPU_TYPE = dlp_config.DLP_GPU_TYPE
JOB_CONFIG_ENVS = {
    "PYTHONPATH": CLOUD_REPO_DIR,
}

# DLP 用 exec $ENTRYPOINT 启动（按空格分词，不解析 shell）
# 不能用 bash -c 'cmd'，单引号会被切断；直接用绝对路径调 python，无需 cd
# paths_config.py 通过 __file__ 计算绝对路径，所以不依赖工作目录
JOB_CONFIG_ENTRYPOINT = (
    f"python {CLOUD_REPO_DIR}/scripts/train.py "
    f"--dataset_type=ffhq_aging "
    f"--exp_dir={CLOUD_EXP_DIR} "
    f"--checkpoint_path={CLOUD_CKPT} "
    f"--workers=6 "
    f"--batch_size=6 "
    f"--test_batch_size=6 "
    f"--test_workers=6 "
    f"--val_interval=100 "
    f"--save_interval=500 "
    f"--max_steps=500 "
    f"--start_from_encoded_w_plus "
    f"--id_lambda=0.1 "
    f"--lpips_lambda=0.1 "
    f"--lpips_lambda_aging=0.05 "
    f"--lpips_lambda_crop=0.6 "
    f"--l2_lambda=0.25 "
    f"--l2_lambda_aging=0.1 "
    f"--l2_lambda_crop=1 "
    f"--w_norm_lambda=0.005 "
    f"--aging_lambda=8 "
    f"--cycle_lambda=1 "
    f"--input_nc=4 "
    f"--target_age=baby_biased "
    f"--use_weighted_id_loss"
)

POLL_INTERVAL_SECONDS  = 5
JOB_TERMINAL_STATUSES  = {"Stopped", "Failed", "Finished"}
TASK_TERMINAL_STATUSES = {"Terminated", "Failed", "Stopped", "Finished", "Succeeded"}
TASK_RUNNING_STATUSES  = {"Running"}


def main() -> None:
    try:
        from dlpctl.api.jobs.api_create_job import create_job
        from dlpctl.api.jobs.api_get_jobs import get_job_detail
        from dlpctl.api.tasks.api_get_task_log import get_task_logs
        from dlpctl.api.tasks.api_get_tasks import get_tasks
        from dlpctl.flow.helper import generate_job_config
    except Exception as exc:
        print(f"无法导入 dlpctl，请先运行 login.ps1 完成安装和登录: {exc}")
        sys.exit(1)

    job_config, err = generate_job_config(
        gpu_num=JOB_CONFIG_GPU_NUM,
        entrypoint=JOB_CONFIG_ENTRYPOINT,
        gpu_type=JOB_CONFIG_GPU_TYPE,
        envs=JOB_CONFIG_ENVS,
    )
    if err:
        print(err)
        sys.exit(1)

    print("==========================================")
    print("  SAM 快速验证训练（500 步）")
    print("==========================================")
    print(f"云端仓库:  {CLOUD_REPO_DIR}")
    print(f"实验目录:  {CLOUD_EXP_DIR}")
    print(f"GPU 类型:  {JOB_CONFIG_GPU_TYPE}  ×{JOB_CONFIG_GPU_NUM}")
    print("==========================================\n")

    print("[info] 提交快速验证作业...")
    job = create_job(job_config)
    print(f"[info] 作业已创建: job-{job.id}")
    print(f"[info] 作业链接: http://dlp.truesightai.com/jobs/info?id={job.id}")

    task_name = None
    while True:
        current_job = get_job_detail(job.id)
        print(f"[poll] job-{current_job.id} 状态: {current_job.status}")

        tasks = get_tasks(current_job, page_size=20)
        if tasks:
            task = tasks[0]
            task_name = task.name
            print(f"[poll] task {task.name} 状态: {task.status}")

            if task.status in TASK_RUNNING_STATUSES:
                print("[info] 任务已启动，实时拉取训练日志 ──────────────────────────────")
                for line in get_task_logs(task, stream="stdout", follow=True):
                    print(line, flush=True)
                break

            if task.status in TASK_TERMINAL_STATUSES:
                print("[info] 任务已终止，拉取完整日志 ────────────────────────────────────")
                for line in get_task_logs(task, stream="stdout", follow=False):
                    print(line, flush=True)
                print("[info] stderr:")
                for line in get_task_logs(task, stream="stderr", follow=False):
                    print(line, flush=True)
                break

        if current_job.status in JOB_TERMINAL_STATUSES:
            print(f"[info] 作业进入终态: {current_job.status}")
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    if task_name:
        print(f"\n[done] 手动查看日志: dlpctl logs task {task_name}")


if __name__ == "__main__":
    main()
