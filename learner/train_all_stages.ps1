# Windows (PowerShell) equivalent of train_all_stages.sh.
# Runs the full co-evolution pipeline: snake vs static prey -> prey/amphibia vs
# that snake -> snake vs the trained prey/amphibia.
#
# Unlike the bash version, this also performs the one-time environment
# initialization that the Taskfile handles on macOS/Linux (uv venv + deps, and
# the PyO3 `build-sim` step), so it works from a fresh checkout on Windows.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\train_all_stages.ps1
#   powershell -ExecutionPolicy Bypass -File .\train_all_stages.ps1 -Steps 2048   # quick smoke run
#
# Requires `uv` (https://docs.astral.sh/uv/) and a Rust toolchain on PATH.

param(
    [int]$Steps = 1000000
)

# Abort on the first failing command (mirrors `set -e`).
$ErrorActionPreference = "Stop"

# Always operate from the learner/ directory this script lives in.
Set-Location -Path $PSScriptRoot

# --- Environment initialization ------------------------------------------------
# Python 3.14 is newer than pyo3 0.22 supports; this forward-compat flag is
# required for every cargo/PyO3 build (the Taskfile sets it too).
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
# Package layout puts the `learner` package under src/.
$env:PYTHONPATH = "src"

Write-Host "Env init: creating/syncing the uv virtual environment..."
uv sync

Write-Host "Env init: building the PyO3 simulation module into the venv (task build-sim)..."
uv pip install -e ../animals_simulation

# --- Training pipeline ----------------------------------------------------------
Write-Host "Phase 1: Training Snake v15 (Static Prey)"
uv run python -m learner.main --steps $Steps --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --amphibias-per-game 4 --max-amphibias 100 --num-procs 4 --model-path models/v15.zip

Write-Host "Phase 2: Training Prey"
uv run python -m learner.train_prey --steps $Steps --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --snake-model models/v15.zip --model-path models/prey_model.zip

Write-Host "Phase 3: Training Amphibia"
uv run python -m learner.train_amphibia --steps $Steps --num-games 16 --snakes-per-game 4 --amphibias-per-game 4 --max-amphibias 100 --snake-model models/v15.zip --model-path models/amphibia_model.zip

Write-Host "Phase 4: Training Snake v16 (Dynamic Prey & Amphibia)"
uv run python -m learner.main --steps $Steps --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --amphibias-per-game 4 --max-amphibias 100 --num-procs 4 --model-path models/v16.zip --prey-model models/prey_model.zip --amphibia-model models/amphibia_model.zip

Write-Host "All training complete!"
