param(
    [string]$Server = "retinascope-server",
    [string]$RemoteRoot = "/root/wangchen/tiansukai",
    [string]$RemoteArchive = "/root/wangchen/tiansukai/retinascope_exp/artifacts/binocular_label_graph/retinascope_binocular_results.tar.gz",
    [string]$LocalDir = "artifacts/server_trained_models/binocular_label_graph"
)

$ErrorActionPreference = "Stop"

if (-not $RemoteArchive.StartsWith($RemoteRoot)) {
    throw "RemoteArchive must stay under $RemoteRoot"
}

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
scp "$Server`:$RemoteArchive" $LocalDir
Write-Host "Fetched binocular experiment results into $LocalDir"
