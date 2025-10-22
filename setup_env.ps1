# PowerShell helper to create the project environment for the TTC repo
# Usage: Open PowerShell as admin/user and run:
#   .\setup_env.ps1
# The script will try to use conda (environment.yml). If conda is not found, it will create a venv and use pip.

param(
    [switch]$ForceVenv
)

function Write-Ok($s){ Write-Host $s -ForegroundColor Green }
function Write-Err($s){ Write-Host $s -ForegroundColor Red }

# Determine repo root
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not $ForceVenv) {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
} else {
    $conda = $null
}

if ($conda) {
    Write-Host "Conda detected. Creating conda environment from environment.yml..." -ForegroundColor Cyan
    $envName = "TTC"
    $envFile = Join-Path $RepoRoot "environment.yml"
    if (-Not (Test-Path $envFile)) {
        Write-Err "environment.yml not found in repo root. Aborting."
        exit 1
    }
    Write-Host "Running: conda env create -f environment.yml -n $envName" -ForegroundColor Yellow
    conda env create -f $envFile -n $envName
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Conda env creation failed. You can try opening Anaconda Prompt and running the above command manually."
        exit $LASTEXITCODE
    }
    Write-Ok "Conda env '$envName' created. To activate run:`nconda activate $envName"
    Write-Host "Now installing pip-only requirements from requirements.txt inside the environment..." -ForegroundColor Cyan
    Write-Host "Run the following commands in an activated shell:" -ForegroundColor Yellow
    Write-Host "conda activate $envName" -ForegroundColor Yellow
    Write-Host "pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host "# If you have git LFS or other needs, install them as needed." -ForegroundColor Yellow
    exit 0
}

# Fallback to venv + pip
Write-Host "Conda not found (or --ForceVenv specified). Creating a Python venv and installing pip packages..." -ForegroundColor Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Err "Python not found in PATH. Please install Python 3.11+ or use conda."
    exit 1
}

$venvDir = Join-Path $RepoRoot "venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtualenv at $venvDir" -ForegroundColor Yellow
    python -m venv $venvDir
}

$activate = Join-Path $venvDir "Scripts" "Activate.ps1"
Write-Host "Activating venv: $activate" -ForegroundColor Yellow
& $activate

# Upgrade pip and install requirements
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

$reqFile = Join-Path $RepoRoot "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Err "requirements.txt not found in repo root. Aborting."
    exit 1
}

Write-Host "Installing pip requirements... This may take several minutes." -ForegroundColor Yellow
pip install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Err "pip install failed. Check the output for errors and install missing system libraries (CUDA/toolkit) if required." 
    exit $LASTEXITCODE
}

Write-Ok "Virtual environment ready. To use it in PowerShell run:`n& $activate"
Write-Host "Then you can run: python run_minidata.py" -ForegroundColor Cyan

exit 0
