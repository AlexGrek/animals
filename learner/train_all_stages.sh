#!/bin/bash
set -e

cd /Users/user/personal/animals/learner
export PYTHONPATH=src

echo "Phase 1: Training Snake v15 (Static Prey)"
uv run python -m learner.main --steps 200000 --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --amphibias-per-game 4 --max-amphibias 100 --num-procs 4 --model-path models/v15.zip

echo "Phase 2: Training Prey"
uv run python -m learner.train_prey --steps 200000 --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --snake-model models/v15.zip --model-path models/prey_model.zip

echo "Phase 3: Training Amphibia"
uv run python -m learner.train_amphibia --steps 200000 --num-games 16 --snakes-per-game 4 --amphibias-per-game 4 --max-amphibias 100 --snake-model models/v15.zip --model-path models/amphibia_model.zip

echo "Phase 4: Training Snake v16 (Dynamic Prey & Amphibia)"
uv run python -m learner.main --steps 200000 --num-games 16 --snakes-per-game 4 --preys-per-game 4 --max-preys 100 --amphibias-per-game 4 --max-amphibias 100 --num-procs 4 --model-path models/v16.zip --prey-model models/prey_model.zip --amphibia-model models/amphibia_model.zip

echo "All training complete!"
