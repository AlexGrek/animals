# How to Add a New Species

Adding a new species to the ecosystem requires updates across the entire stack: the Rust core engine, the PyO3 Python bindings, the Python RL training loop, and the Bevy graphical visualizer.

This guide outlines the step-by-step process, using a generic `NewSpecies` as an example.

## 1. Core Engine (`animals_engine`)

1. **Define the State Struct**
   In `animals_engine/src/game.rs`, define the state for an individual instance of your species:
   ```rust
   pub struct NewSpeciesState {
       pub pos: (f32, f32),
       pub is_dead: bool,
       pub points: i32,
       pub family_id: u32,
   }
   ```
2. **Update GameState**
   Add a `Vec<NewSpeciesState>` to `GameState`.
   Update `GameState::new` to accept a `num_new_species: usize` parameter and spawn the initial entities into open map coordinates.
3. **Observation & Actions**
   Define the size of your species' observation vector (e.g., `pub const NEW_SPECIES_OBS_SIZE: usize = 13;`) in `animals_engine/src/lib.rs`.
   Add observation extraction functions (e.g., `pub fn get_new_species_observation(&self, idx: usize) -> [f32; NEW_SPECIES_OBS_SIZE]`) to `game.rs`.
4. **Update the Step Loop**
   In `GameState::step`, apply the actions provided for your species (movements, state changes). Check for deaths, handle scoring, spawn new entities if they replicate, and clean up corpses if applicable.

## 2. Python Bindings (`animals_simulation`)

1. **Update the PyClass**
   In `animals_simulation/src/lib.rs`, update `Simulation::new` to accept `num_new_species`.
2. **Update the Protocol**
   Both `reset` and `step` must be updated to return the new species' data back to Python. The standard pattern is to return a nested tuple of `(obs, rewards, dones, terminal_obs)` for each distinct actor group:
   ```rust
   pub fn step(
       &mut self,
       snake_actions: Vec<usize>,
       prey_actions: Vec<usize>,
       new_species_actions: Vec<usize>,
   ) -> (
       (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<Vec<f32>>), // Snakes
       (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<Vec<f32>>), // Preys/Amphibias
       (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<Vec<f32>>), // NewSpecies
   )
   ```
3. **Reward Logic**
   Implement any reward shaping (e.g., penalties for dying, rewards for eating) within the `step()` function before constructing the return tuples.

## 3. Python Learning Environment (`learner`)

1. **Constants**
   Add `NEW_SPECIES_OBS_SIZE` and `NEW_SPECIES_NUM_ACTIONS` to `learner/src/learner/constants.py`.
2. **Environment definition**
   Copy `prey_environment.py` into `new_species_environment.py`. Create a `VecEnv` subclass that handles feeding the new species' actions directly to the simulation while predicting and freezing the actions of opponents (like Snakes) using pre-trained models.
3. **Training Script**
   Copy `train_prey.py` to `train_new_species.py`. Import your new environment and set up the PPO architecture. If the observation space is small, an `MlpPolicy` is usually sufficient.
4. **Taskfile.yml**
   Add a `train-new-species` task to the root `Taskfile.yml` so you can easily invoke training:
   ```yaml
   train-new-species:
     desc: "Train the new species ML model."
     dir: learner
     cmds:
       - PYTHONPATH=src uv run python -m learner.train_new_species --steps 250000
   ```

## 4. Bevy Visualizer (`animals_game`)

1. **Components**
   Add a Sprite component to `animals_game/src/components.rs` (e.g., `pub struct NewSpeciesSprite { pub idx: usize, }`).
2. **Resources & UI**
   Update `animals_game/src/resources.rs` to include a `Vec<String>` for storing model paths in `MatchConfig`. Parse CLI arguments in `animals_game/src/main.rs`.
3. **TCP Protocol (Rust)**
   Update `animals_game/src/ai.rs`. Add `num_new_species` to the `spawn_ai_server` startup. Modify `spawn_ai_worker` and `queue_ai_inference` to encode the observation lengths, entity counts, and `family_ids` of the new species into the byte stream being sent to Python.
4. **TCP Protocol (Python)**
   Update `learner/src/learner/play.py`. Add `argparse` flags for the new models. Unpack the modified binary payload from `ai.rs`, reshape the tensors for the new species, query the loaded PPO models, and pack the predictions back into the response bytes.
5. **Game Logic & Action Dispatch**
   Update `animals_game/src/utils.rs` (`gather_observations`) to append the new species' observations. Update `animals_game/src/logic.rs` to extract the `new_species_actions` from the returned AI payload and pipe them into `engine.0.step()`.
6. **Rendering**
   In `animals_game/src/render.rs`, modify `render_sync` to iterate over `engine.0.new_species`. Read their positions, calculate `Transform`s, and spawn Sprites with custom colors to display them on the Bevy map.
