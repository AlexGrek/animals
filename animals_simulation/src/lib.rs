use pyo3::prelude::*;
use animals_engine::snake::{GameState, RelativeAction};

#[pyclass]
pub struct Simulation {
    game_state: GameState,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new() -> Self {
        Self {
            game_state: GameState::new(100, 100),
        }
    }

    fn reset(&mut self) -> PyResult<Vec<f32>> {
        self.game_state = GameState::new(100, 100);
        Ok(self.game_state.get_relative_observation().to_vec())
    }

    fn step(&mut self, action: usize) -> PyResult<(Vec<f32>, f32, bool, bool)> {
        if self.game_state.game_over {
            return Ok((
                self.game_state.get_relative_observation().to_vec(),
                -10.0,
                true,
                false,
            ));
        }

        // Map the python relative action (0: straight, 1: right, 2: left)
        let rel_action = RelativeAction::from_usize(action);
        let new_dir = rel_action.to_absolute_direction(self.game_state.direction);
        self.game_state.set_direction(new_dir);

        let prev_score = self.game_state.score;
        self.game_state.step();

        let observation = self.game_state.get_relative_observation().to_vec();
        let terminated = self.game_state.game_over;
        let truncated = false;

        // Reward function
        let reward = if terminated {
            -10.0 // Big penalty for dying
        } else if self.game_state.score > prev_score {
            10.0  // Big reward for eating apple
        } else {
            -0.01 // Small penalty to encourage efficiency
        };

        Ok((observation, reward, terminated, truncated))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn animals_simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulation>()?;
    Ok(())
}
