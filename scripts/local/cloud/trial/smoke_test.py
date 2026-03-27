#!/usr/bin/env python3
"""
【临时验证】提交环境冒烟测试：仅检查 shell、Python、nvidia-smi 是否可用。

不运行任何训练代码，用于确认 DLP 云端基础环境是否正常。

用法:
    conda run -p .conda\sam python scripts/local/cloud/trial/smoke_test.py
"""
import os
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

JOB_CONFIG_GPU_NUM  = dlp_config.DLP_GPU_NUM
JOB_CONFIG_GPU_TYPE = dlp_config.DLP_GPU_TYPE
JOB_CONFIG_ENVS     = {}
JOB_CONFIG_ENTRYPOINT = (
    "echo [smoke] shell_ok; "
    "if command -v python3 >/dev/null 2>&1; then python3 -V; "
    "elif command -v python >/dev/null 2>&1; then python -V; "
    "else echo python_not_found; fi; "
    "if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; "
    "else echo nvidia_smi_not_found; fi; "
    "exit 0"
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
    print("  DLP 环境冒烟测试")
    print("==========================================")
    print(f"GPU 类型:  {JOB_CONFIG_GPU_TYPE}  ×{JOB_CONFIG_GPU_NUM}")
    print(f"入口命令:  {JOB_CONFIG_ENTRYPOINT}")
    print("==========================================\n")

    print("[info] 提交 smoke test 作业...")
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
                print("[info] 任务已启动，拉取输出 ──────────────────────────────────────")
                for line in get_task_logs(task, stream="stdout", follow=True):
                    print(line, flush=True)
                break

            if task.status in TASK_TERMINAL_STATUSES:
                for line in get_task_logs(task, stream="stdout", follow=False):
                    print(line, flush=True)
                break

        if current_job.status in JOB_TERMINAL_STATUSES:
            print(f"[info] 作业终态: {current_job.status}")
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    if task_name:
        print(f"\n[done] 手动查看日志: dlpctl logs task {task_name}")


if __name__ == "__main__":
    main()
