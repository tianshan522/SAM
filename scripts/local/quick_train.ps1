param(
    [string]$CheckpointPath,
    [string]$ExpDir,
    [string]$EnvPath,
    [int]$BatchSize,
    [int]$MaxSteps
)

$ErrorActionPreference = "Stop"

# ==================== 可直接修改的配置区 ====================

# ---- 数据与模型 ----
$DefaultDatasetType    = "ffhq_aging"
$DefaultCheckpointPath = "pretrained_models\sam_ffhq_aging.pt"
$DefaultExpDir         = "experiments\adult_to_baby_finetune"

# ---- 训练参数（RTX 5060 8GB 推荐值） ----
$DefaultBatchSize      = 2
$DefaultTestBatchSize  = 2
$DefaultWorkers        = 0
$DefaultTestWorkers    = 0
$DefaultMaxSteps       = 50000
$DefaultValInterval    = 2500
$DefaultSaveInterval   = 5000

# ---- 损失权重 ----
$DefaultAgingLambda      = 8.0      # 年龄损失，增大 → 婴儿效果更明显
$DefaultIdLambda         = 0.1      # 身份保持，增大 → 更像原图
$DefaultCycleLambda      = 1.0      # 循环一致性
$DefaultLpipsLambda      = 0.1      # 全图感知损失
$DefaultLpipsLambdaAging = 0.05     # 年龄变换感知损失
$DefaultLpipsLambdaCrop  = 0.6      # 面部区域感知损失
$DefaultL2Lambda         = 0.25     # 全图像素损失
$DefaultL2LambdaAging    = 0.1      # 年龄变换像素损失
$DefaultL2LambdaCrop     = 1.0      # 面部区域像素损失
$DefaultWNormLambda      = 0.005    # W+ 潜码正则

# ---- 高级选项 ----
$DefaultTargetAge              = "baby_biased"       # 目标年龄策略（60%采样0-5岁）
$DefaultInputNC                = 4                  # 输入通道数（3色彩 + 1年龄）
$DefaultStartFromEncodedWPlus  = $true              # 从 pSp 编码的 W+ 开始
$DefaultUseWeightedIdLoss      = $true              # 年龄差越大，ID loss 权重越低

# ---- Conda 环境 ----
$DefaultEnvPath = ".conda\sam"

# ==========================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))
Set-Location $repoDir

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoDir $PathValue))
}

$CheckpointPath = if (-not [string]::IsNullOrWhiteSpace($CheckpointPath)) { $CheckpointPath } else { $DefaultCheckpointPath }
$ExpDir         = if (-not [string]::IsNullOrWhiteSpace($ExpDir))         { $ExpDir }         else { $DefaultExpDir }
$EnvPath        = if (-not [string]::IsNullOrWhiteSpace($EnvPath))        { $EnvPath }        else { $DefaultEnvPath }

if ($PSBoundParameters.ContainsKey('BatchSize') -and $BatchSize -gt 0) { } else { $BatchSize = $DefaultBatchSize }
if ($PSBoundParameters.ContainsKey('MaxSteps') -and $MaxSteps -gt 0)   { } else { $MaxSteps  = $DefaultMaxSteps }

$checkpointFullPath = Resolve-RepoPath $CheckpointPath
$expFullPath        = Resolve-RepoPath $ExpDir
$envFullPath        = Resolve-RepoPath $EnvPath

if (-not (Test-Path $checkpointFullPath)) {
    throw "模型权重不存在: $checkpointFullPath`n请先运行 scripts/local/download_required_assets.sh 下载预训练模型"
}
if (-not (Test-Path $envFullPath)) {
    throw "Conda 环境不存在: $envFullPath"
}

New-Item -ItemType Directory -Force -Path $expFullPath | Out-Null

$commandArgs = @(
    "run",
    "-p", $envFullPath,
    "python", "scripts\train.py",
    "--dataset_type",             $DefaultDatasetType,
    "--exp_dir",                  $expFullPath,
    "--checkpoint_path",          $checkpointFullPath,
    "--workers",                  $DefaultWorkers,
    "--batch_size",               $BatchSize,
    "--test_batch_size",          $DefaultTestBatchSize,
    "--test_workers",             $DefaultTestWorkers,
    "--val_interval",             $DefaultValInterval,
    "--save_interval",            $DefaultSaveInterval,
    "--max_steps",                $MaxSteps,
    "--id_lambda",                $DefaultIdLambda,
    "--lpips_lambda",             $DefaultLpipsLambda,
    "--lpips_lambda_aging",       $DefaultLpipsLambdaAging,
    "--lpips_lambda_crop",        $DefaultLpipsLambdaCrop,
    "--l2_lambda",                $DefaultL2Lambda,
    "--l2_lambda_aging",          $DefaultL2LambdaAging,
    "--l2_lambda_crop",           $DefaultL2LambdaCrop,
    "--w_norm_lambda",            $DefaultWNormLambda,
    "--aging_lambda",             $DefaultAgingLambda,
    "--cycle_lambda",             $DefaultCycleLambda,
    "--input_nc",                 $DefaultInputNC,
    "--target_age",               $DefaultTargetAge
)

if ($DefaultStartFromEncodedWPlus) {
    $commandArgs += "--start_from_encoded_w_plus"
}
if ($DefaultUseWeightedIdLoss) {
    $commandArgs += "--use_weighted_id_loss"
}

Write-Host "=========================================="
Write-Host "  SAM 本地训练"
Write-Host "=========================================="
Write-Host "仓库目录:   $repoDir"
Write-Host "Conda 环境: $envFullPath"
Write-Host "起始权重:   $checkpointFullPath"
Write-Host "实验目录:   $expFullPath"
Write-Host ""
Write-Host "训练参数:"
Write-Host "  batch_size   = $BatchSize"
Write-Host "  max_steps    = $MaxSteps"
Write-Host "  val_interval = $DefaultValInterval"
Write-Host "  target_age   = $DefaultTargetAge"
Write-Host ""
Write-Host "损失权重:"
Write-Host "  aging=$DefaultAgingLambda  id=$DefaultIdLambda  cycle=$DefaultCycleLambda"
Write-Host "  lpips=$DefaultLpipsLambda  lpips_aging=$DefaultLpipsLambdaAging  lpips_crop=$DefaultLpipsLambdaCrop"
Write-Host "  l2=$DefaultL2Lambda  l2_aging=$DefaultL2LambdaAging  l2_crop=$DefaultL2LambdaCrop"
Write-Host "  w_norm=$DefaultWNormLambda"
Write-Host "=========================================="
Write-Host ""
Write-Host "开始训练..."

& conda @commandArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "训练执行失败，退出码: $exitCode"
}

Write-Host ""
Write-Host "训练完成。实验目录:"
Write-Host $expFullPath
Write-Host ""
Write-Host "查看 TensorBoard:"
Write-Host "  tensorboard --logdir=$expFullPath\logs"
