use pyo3::prelude::*;
use animals_engine::{GameState, RelativeAction};

#[pyclass]
pub struct Simulation {
    game_state: GameState,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(num_snakes: usize) -> Self {
        Self {
            game_state: GameState::new(100, 100, num_snakes),
        }
    }

    fn reset(&mut self) -> PyResult<Vec<Vec<f32>>> {
        let num_snakes = self.game_state.snakes.len();
        self.game_state = GameState::new(100, 100, num_snakes);
        let mut obs = Vec::new();
        for i in 0..num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }
        Ok(obs)
    }

    /// Steps the simulation. Snakes never truly terminate the game any more:
    /// on death a snake is immediately respawned (fresh body of length 1) so
    /// the other snakes' episodes aren't truncated by one snake's mistake.
    ///
    /// Returns `(obs, rewards, dones, terminal_obs)`:
    /// - `obs[i]` is always a valid "next" observation for snake i. For a
    ///   snake that died this tick, that means its *post-respawn* observation
    ///   (i.e. the reset observation for its new episode).
    /// - `dones[i]` is true exactly on the tick snake i died.
    /// - `terminal_obs[i]` is only meaningful when `dones[i]` is true: it is
    ///   the snake's observation at the moment of death, *before* respawn,
    ///   i.e. the true terminal observation of the just-ended episode. When
    ///   `dones[i]` is false, `terminal_obs[i]` is an all-zero placeholder.
    fn step(&mut self, actions: Vec<usize>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<Vec<f32>>)> {
        let num_snakes = self.game_state.snakes.len();

        for (i, &action) in actions.iter().enumerate() {
            if i < num_snakes {
                let rel_action = RelativeAction::from_usize(action);
                let new_dir = rel_action.to_absolute_direction(self.game_state.snakes[i].direction);
                self.game_state.set_direction(i, new_dir);
            }
        }

        let prev_scores: Vec<u32> = self.game_state.snakes.iter().map(|s| s.score).collect();
        let prev_kills: Vec<u32> = self.game_state.snakes.iter().map(|s| s.kills).collect();

        self.game_state.step(1.0);

        // Reward function
        let calc_reward = |snake: &animals_engine::SnakeState, prev_score: u32, prev_kills: u32| -> f32 {
            if snake.is_dead {
                -10.0 // Wall, Self, Opponent, or Head-to-head
            } else if snake.kills > prev_kills {
                50.0 // Huge reward for killing the opponent!
            } else if snake.score > prev_score {
                10.0 // Big reward for eating apple
            } else {
                -0.01 // Small penalty to encourage efficiency
            }
        };

        let mut rewards = Vec::new();
        let mut dones = Vec::new();
        let mut terminal_obs = Vec::new();

        for i in 0..num_snakes {
            let done = self.game_state.snakes[i].is_dead;
            let reward = calc_reward(&self.game_state.snakes[i], prev_scores[i], prev_kills[i]);
            rewards.push(reward);
            dones.push(done);
            if done {
                // Capture the terminal observation before this snake is respawned.
                terminal_obs.push(self.game_state.get_relative_observation(i).to_vec());
            } else {
                terminal_obs.push(vec![0.0; 130]);
            }
        }

        // Respawn any snakes that died this tick so every snake always has a
        // live episode to continue playing next step.
        self.game_state.respawn_dead();

        let mut obs = Vec::new();
        for i in 0..num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }

        Ok((obs, rewards, dones, terminal_obs))
    }

    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        let mut stats = Vec::new();
        for snake in &self.game_state.snakes {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("length", snake.body.len())?;
            dict.set_item("score", snake.score)?;
            dict.set_item("kills", snake.kills)?;
            dict.set_item("is_dead", snake.is_dead)?;
            dict.set_item("death_by_wall", snake.death_by_wall)?;
            dict.set_item("death_by_self", snake.death_by_self)?;
            dict.set_item("death_by_opponent", snake.death_by_opponent)?;
            stats.push(dict);
        }
        Ok(stats)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn animals_simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulation>()?;
    Ok(())
}
