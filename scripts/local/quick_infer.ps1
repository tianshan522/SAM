param(
    [string]$InputPath,
    [string]$OutputDir,
    [string]$TargetAge,
    [string]$CheckpointPath = "pretrained_models\sam_ffhq_aging.pt",
    [string]$EnvPath = ".conda\sam",
    [int]$BatchSize = 1,
    [int]$Workers = 0,
    [switch]$ResizeOutputs,
    [switch]$NoCoupleOutputs
)

$ErrorActionPreference = "Stop"

# ==================== 可直接修改的配置区 ====================
# 不想输入命令时，直接修改下面这几行，然后双击或执行本脚本即可。
$DefaultInputPath = "notebooks\images\1287.jpg"
$DefaultOutputDir = "workspace\runs\quick_infer_manual"
$DefaultTargetAge = "1"
$DefaultResizeOutputs = $true
$DefaultCoupleOutputs = $true
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

function Prompt-IfEmpty {
    param(
        [string]$Value,
        [string]$PromptText,
        [string]$DefaultValue = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        return $Value
    }

    if ([string]::IsNullOrWhiteSpace($DefaultValue)) {
        return (Read-Host $PromptText).Trim()
    }

    $userInput = Read-Host "$PromptText [$DefaultValue]"
    if ([string]::IsNullOrWhiteSpace($userInput)) {
        return $DefaultValue
    }
    return $userInput.Trim()
}

$InputPath = if (-not [string]::IsNullOrWhiteSpace($InputPath)) { $InputPath } else { $DefaultInputPath }
$OutputDir = if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir } else { $DefaultOutputDir }
$TargetAge = if (-not [string]::IsNullOrWhiteSpace($TargetAge)) { $TargetAge } else { $DefaultTargetAge }

if (-not $ResizeOutputs.IsPresent -and $DefaultResizeOutputs) {
    $ResizeOutputs = $true
}
if (-not $NoCoupleOutputs.IsPresent -and -not $DefaultCoupleOutputs) {
    $NoCoupleOutputs = $true
}

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Prompt-IfEmpty -Value $InputPath -PromptText "请输入图片文件路径或图片目录路径"
}
if ([string]::IsNullOrWhiteSpace($TargetAge)) {
    $TargetAge = Prompt-IfEmpty -Value $TargetAge -PromptText "请输入目标年龄，支持单个值或逗号分隔，例如 20 或 10,30,50"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Prompt-IfEmpty -Value $OutputDir -PromptText "请输入输出目录" -DefaultValue "workspace\runs\quick_infer_manual"
}

$inputFullPath = Resolve-RepoPath $InputPath
$outputFullPath = Resolve-RepoPath $OutputDir
$checkpointFullPath = Resolve-RepoPath $CheckpointPath
$envFullPath = Resolve-RepoPath $EnvPath

if (-not (Test-Path $inputFullPath)) {
    throw "输入路径不存在: $inputFullPath"
}
if (-not (Test-Path $checkpointFullPath)) {
    throw "模型权重不存在: $checkpointFullPath"
}
if (-not (Test-Path $envFullPath)) {
    throw "Conda 环境不存在: $envFullPath"
}

New-Item -ItemType Directory -Force -Path $outputFullPath | Out-Null

$dataPathForInference = $inputFullPath
$tempInputDir = $null

if (Test-Path $inputFullPath -PathType Leaf) {
    $tempInputDir = Join-Path $repoDir (Join-Path "workspace\tmp" ("quick_infer_input_" + (Get-Date -Format "yyyyMMdd_HHmmss")))
    New-Item -ItemType Directory -Force -Path $tempInputDir | Out-Null
    Copy-Item -Path $inputFullPath -Destination (Join-Path $tempInputDir (Split-Path $inputFullPath -Leaf)) -Force
    $dataPathForInference = $tempInputDir
}

$commandArgs = @(
    "run",
    "-p", $envFullPath,
    "python",
    "scripts\inference.py",
    "--exp_dir", $outputFullPath,
    "--checkpoint_path", $checkpointFullPath,
    "--data_path", $dataPathForInference,
    "--test_batch_size", $BatchSize,
    "--test_workers", $Workers,
    "--target_age", $TargetAge
)

if ($ResizeOutputs.IsPresent) {
    $commandArgs += "--resize_outputs"
}
if (-not $NoCoupleOutputs.IsPresent) {
    $commandArgs += "--couple_outputs"
}

Write-Host "仓库目录: $repoDir"
Write-Host "输入路径: $inputFullPath"
Write-Host "推理数据路径: $dataPathForInference"
Write-Host "输出目录: $outputFullPath"
Write-Host "目标年龄: $TargetAge"
Write-Host "模型权重: $checkpointFullPath"
Write-Host "Conda 环境: $envFullPath"
Write-Host ""
Write-Host "开始执行推理..."

& conda @commandArgs
$exitCode = $LASTEXITCODE

if ($tempInputDir -and (Test-Path $tempInputDir)) {
    Remove-Item -Path $tempInputDir -Recurse -Force
}

if ($exitCode -ne 0) {
    throw "推理执行失败，退出码: $exitCode"
}

Write-Host ""
Write-Host "推理完成。输出目录:"
Write-Host $outputFullPath
