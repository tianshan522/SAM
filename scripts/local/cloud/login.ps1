param(
    [switch]$SkipLogin,
    [switch]$SkipSmoke
)

<#
.SYNOPSIS
    DLP 平台登录 + smoke test（Windows PowerShell）

.DESCRIPTION
    1. 从 scripts/local/dlp_config.py 读取账号信息
    2. 检查本地 conda 环境（.conda\sam）是否存在
    3. 在该环境中安装/更新 dlpctl
    4. 登录 DLP 平台
    5. 可选：提交 smoke test 验证云端 GPU 环境

.EXAMPLE
    # 读取 dlp_config.py 中的账密自动登录并跑 smoke test
    .\scripts\local\cloud\login.ps1

    # 跳过登录（已有缓存 token），直接跑 smoke test
    .\scripts\local\cloud\login.ps1 -SkipLogin

    # 只登录，不跑 smoke test
    .\scripts\local\cloud\login.ps1 -SkipSmoke
#>

$ErrorActionPreference = "Stop"

$PIP_INDEX_URL = "http://pypi.truesightai.com/simple"
$TRUSTED_HOST  = "pypi.truesightai.com"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir   = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\..\.."))
$configDir = Join-Path $repoDir "scripts\local"

# 使用本地路径环境（.conda\sam），用 -p 而不是 -n
$ENV_PATH = Join-Path $repoDir ".conda\sam"

# ── 检查本地环境路径 ──────────────────────────────────────────────────────────
if (-not (Test-Path $ENV_PATH)) {
    Write-Error "未找到本地 conda 环境: $ENV_PATH"
}
Write-Host "[info] 使用本地环境: $ENV_PATH"

# ── 读取 dlp_config.py ────────────────────────────────────────────────────────
$configFile = Join-Path $configDir "dlp_config.py"
if (-not (Test-Path $configFile)) {
    Write-Error (
        "找不到配置文件: $configFile`n" +
        "请将 scripts/local/dlp_config.py 中的占位符替换为您的实际配置。"
    )
}

# conda run 在 Windows 上不支持 python -c 传多行代码，改为写临时文件再执行
$tmpScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
@"
import sys
sys.path.insert(0, r'$configDir')
import dlp_config
print(dlp_config.DLP_USERNAME)
print(dlp_config.DLP_PASSWORD)
print(dlp_config.DLP_GPU_TYPE)
print(str(dlp_config.DLP_GPU_NUM))
"@ | Set-Content -Encoding UTF8 $tmpScript

$configValues = conda run -p $ENV_PATH python $tmpScript 2>&1
Remove-Item $tmpScript -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) { Write-Error "读取 dlp_config.py 失败: $configValues" }

$configLines = $configValues -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
$DLP_USERNAME = $configLines[0]
$DLP_PASSWORD = $configLines[1]
$GPU_TYPE     = $configLines[2]
$GPU_NUM      = $configLines[3]

if (-not $SkipLogin) {
    if ([string]::IsNullOrWhiteSpace($DLP_USERNAME)) {
        $DLP_USERNAME = Read-Host "dlp_config.py 中 DLP_USERNAME 为空，请手动输入用户名"
    }
    if ([string]::IsNullOrWhiteSpace($DLP_PASSWORD)) {
        $securePass   = Read-Host "dlp_config.py 中 DLP_PASSWORD 为空，请手动输入密码" -AsSecureString
        $DLP_PASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                            [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass))
    }
}

# ── 安装/更新 dlpctl ──────────────────────────────────────────────────────────
Write-Host "[info] 检查 dlpctl..."
$tmpCheck = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
"import dlpctl; print(dlpctl.__version__)" | Set-Content -Encoding UTF8 $tmpCheck
$dlpctlCheck = conda run -p $ENV_PATH python $tmpCheck 2>&1
Remove-Item $tmpCheck -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) {
    Write-Host "[info] 安装 dlpctl..."
    conda run -p $ENV_PATH python -m pip install `
        -i $PIP_INDEX_URL --trusted-host $TRUSTED_HOST dlpctl
    if ($LASTEXITCODE -ne 0) { Write-Error "dlpctl 安装失败" }
} else {
    Write-Host "[info] dlpctl 版本: $($dlpctlCheck.Trim())"
}

# ── 登录 ──────────────────────────────────────────────────────────────────────
if (-not $SkipLogin) {
    Write-Host "[info] 登录 DLP 平台（用户: $DLP_USERNAME）..."
    conda run -p $ENV_PATH dlpctl login -u $DLP_USERNAME -p $DLP_PASSWORD
    if ($LASTEXITCODE -ne 0) { Write-Error "DLP 登录失败，请检查 dlp_config.py 中的账号密码" }
    Write-Host "[info] 登录成功"
} else {
    Write-Host "[info] 跳过登录，复用当前 dlpctl 登录态"
}

# ── Smoke test ────────────────────────────────────────────────────────────────
if (-not $SkipSmoke) {
    Write-Host ""
    Write-Host "[info] 提交 smoke test（GPU=$GPU_TYPE ×$GPU_NUM）..."
    Set-Location $repoDir
    conda run -p $ENV_PATH python scripts/local/cloud/trial/smoke_test.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Smoke test 失败，请检查云端环境" }
    Write-Host "[info] Smoke test 通过 ✓"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  全部检查通过，可以提交训练任务："
Write-Host "  # 快速验证（500步，~10min）"
Write-Host "  conda run -p $ENV_PATH python scripts/local/cloud/trial/quick_train.py"
Write-Host ""
Write-Host "  # 正式训练（50000步）"
Write-Host "  conda run -p $ENV_PATH python scripts/local/cloud/train.py"
Write-Host "=========================================="
