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

    fn reset(&mut self) -> PyResult<(Vec<f32>, Vec<f32>)> {
        self.game_state = GameState::new(100, 100);
        let obs0 = self.game_state.get_relative_observation(0).to_vec();
        let obs1 = self.game_state.get_relative_observation(1).to_vec();
        Ok((obs0, obs1))
    }

    fn step(&mut self, action0: usize, action1: usize) -> PyResult<((Vec<f32>, Vec<f32>), (f32, f32), (bool, bool))> {
        if self.game_state.game_over {
            return Ok((
                (
                    self.game_state.get_relative_observation(0).to_vec(),
                    self.game_state.get_relative_observation(1).to_vec(),
                ),
                (0.0, 0.0),
                (true, true),
            ));
        }

        let rel_action0 = RelativeAction::from_usize(action0);
        let new_dir0 = rel_action0.to_absolute_direction(self.game_state.snakes[0].direction);
        self.game_state.set_direction(0, new_dir0);

        let rel_action1 = RelativeAction::from_usize(action1);
        let new_dir1 = rel_action1.to_absolute_direction(self.game_state.snakes[1].direction);
        self.game_state.set_direction(1, new_dir1);

        let prev_score0 = self.game_state.snakes[0].score;
        let prev_score1 = self.game_state.snakes[1].score;

        let prev_kills0 = self.game_state.snakes[0].kills;
        let prev_kills1 = self.game_state.snakes[1].kills;

        self.game_state.step();

        let obs0 = self.game_state.get_relative_observation(0).to_vec();
        let obs1 = self.game_state.get_relative_observation(1).to_vec();

        let terminated = self.game_state.game_over;

        // Reward function
        let calc_reward = |snake: &animals_engine::snake::SnakeState, prev_score: u32, prev_kills: u32| -> f32 {
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

        let reward0 = calc_reward(&self.game_state.snakes[0], prev_score0, prev_kills0);
        let reward1 = calc_reward(&self.game_state.snakes[1], prev_score1, prev_kills1);

        Ok(((obs0, obs1), (reward0, reward1), (terminated, terminated)))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn animals_simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulation>()?;
    Ok(())
}
