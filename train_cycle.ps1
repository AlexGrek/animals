$ErrorActionPreference = "Stop"

function Run-Command {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Command
    )
    Write-Host "Running: $Command"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Command failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "Building simulation..."
Run-Command "task build-sim"

cd learner
$env:PYTHONPATH = "src"

Write-Host "1. Training Snake from scratch: 500k steps with static prey..."
Run-Command "uv run python -m learner.main --steps 500000 --num-games 2 --snakes-per-game 8 --prey-model `"none`" --amphibia-model `"none`""

Write-Host "2. Resuming Snake training: 500k steps with AI-powered prey..."
Run-Command "uv run python -m learner.main --steps 500000 --num-games 2 --snakes-per-game 8 --resume --prey-model `"models/prey_model.zip`" --amphibia-model `"models/amphibia_model.zip`""

Write-Host "3. Training Prey: 500k steps..."
# Use fewer games to prevent OOM. 2 games * 100 max_preys = 200 envs
Run-Command "uv run python -m learner.train_prey --steps 500000 --resume --num-games 2"

Write-Host "4. Training Amphibia: 500k steps..."
Run-Command "uv run python -m learner.train_amphibia --steps 500000 --resume --num-games 2"

Write-Host "5. Resuming Snake training: 500k steps..."
Run-Command "uv run python -m learner.main --steps 500000 --num-games 2 --snakes-per-game 8 --resume --prey-model `"models/prey_model.zip`" --amphibia-model `"models/amphibia_model.zip`""

Write-Host "Training cycle complete!"
