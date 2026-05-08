param(
    [switch]$Rebuild,
    [switch]$SetupOnly,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "genai-project"

Set-Location $AppDir

Write-Host ""
Write-Host "==========================================="
Write-Host " Worker Selection App Launcher"
Write-Host "==========================================="
Write-Host ""

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
}

if ($Rebuild) {
    Write-Host "Rebuild mode enabled. Removing local virtual environment and trained ML model..."
    Remove-Item -Recurse -Force (Join-Path $RepoRoot ".venv") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $AppDir ".venv") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $AppDir "app\ml_legacy\model.pkl") -ErrorAction SilentlyContinue
}

Write-Host "Syncing Python environment from pyproject.toml / uv.lock..."
& uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SetupOnly) {
    Write-Host "SetupOnly mode enabled. Environment is ready. Exiting without starting server."
    exit 0
}

$RunArgs = @(
    "python",
    "-m",
    "uvicorn",
    "main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000"
)

if ($Dev) {
    Write-Host "Development mode: --reload enabled."
    $RunArgs += "--reload"
}

Write-Host ""
Write-Host "Starting server..."
Write-Host "Open: http://127.0.0.1:8000/"
Write-Host ""

& uv run @RunArgs
exit $LASTEXITCODE