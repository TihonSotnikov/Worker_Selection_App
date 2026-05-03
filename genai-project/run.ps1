param(
    [switch]$Rebuild,
    [switch]$SetupOnly,
    [switch]$Dev,
    [switch]$ClearModelCache
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$VenvPython = ".\.venv\Scripts\python.exe"
$SetupMarker = ".\.setup_complete"
$ModelPath = ".\app\ml_legacy\model.pkl"
$QwenCachePath = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507"

Write-Host ""
Write-Host "==========================================="
Write-Host " Worker Selection App Launcher"
Write-Host "==========================================="
Write-Host ""

if ($ClearModelCache) {
    Write-Host "Clearing Hugging Face Qwen cache..."
    Remove-Item -Recurse -Force $QwenCachePath -ErrorAction SilentlyContinue
}

if ($Rebuild) {
    Write-Host "Rebuild mode enabled."

    Write-Host "Removing setup marker..."
    Remove-Item $SetupMarker -ErrorAction SilentlyContinue

    Write-Host "Removing old ML model..."
    Remove-Item $ModelPath -ErrorAction SilentlyContinue
}

if (!(Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

if (!(Test-Path $VenvPython)) {
    throw "Virtual environment was not created. Check that Python is installed and available as 'python'."
}

if (!(Test-Path "requirements.txt")) {
    throw "requirements.txt not found. Create it in the project root first."
}

if ($Rebuild -or !(Test-Path $SetupMarker)) {
    Write-Host ""
    Write-Host "First setup started. This may take time only once."
    Write-Host ""

    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip wheel
    & $VenvPython -m pip install --upgrade "setuptools<82"

    Write-Host "Installing project dependencies..."
    & $VenvPython -m pip install -r requirements.txt

    Write-Host "Checking dependencies..."
    & $VenvPython -m pip check

    Write-Host ""
    Write-Host "Preloading Qwen model into Hugging Face cache..."
    & $VenvPython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507'); print('Qwen model is cached')"

    Write-Host ""
    Write-Host "Training ML model if needed..."
    & $VenvPython -c "from app.ml_legacy.predictor import RetentionPredictor; p=RetentionPredictor(); p.train_model(); p.save_model(); print('ML model is trained and saved')"

    Write-Host ""
    Write-Host "Writing setup marker..."
    Set-Content $SetupMarker "setup completed at $(Get-Date -Format s)"

    Write-Host ""
    Write-Host "Setup complete."
} else {
    Write-Host "Setup already completed. Skipping dependency install, model download and training."
}

if ($SetupOnly) {
    Write-Host "SetupOnly mode enabled. Exiting without starting server."
    exit 0
}

Write-Host ""
Write-Host "Starting server..."
Write-Host "Open: http://127.0.0.1:8000/"
Write-Host ""

$UvicornArgs = @("main:app", "--host", "127.0.0.1", "--port", "8000")

if ($Dev) {
    Write-Host "Development mode: --reload enabled."
    $UvicornArgs += "--reload"
}

& $VenvPython -m uvicorn @UvicornArgs