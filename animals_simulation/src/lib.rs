use pyo3::prelude::*;
use animals_engine::{GameState, RelativeAction};

#[pyclass]
pub struct Simulation {
    game_state: GameState,
    num_snakes: usize,
    num_preys: usize,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(num_snakes: usize, num_preys: usize) -> Self {
        Self {
            game_state: GameState::new(100, 100, num_snakes, num_preys),
            num_snakes,
            num_preys,
        }
    }

    fn reset(&mut self) -> PyResult<Vec<Vec<f32>>> {
        self.game_state = GameState::new(100, 100, self.num_snakes, self.num_preys);
        let mut obs = Vec::new();
        for i in 0..self.num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }
        Ok(obs)
    }

    fn get_all_prey_observations(&self) -> PyResult<Vec<Vec<f32>>> {
        let mut obs = Vec::new();
        for i in 0..self.num_preys {
            obs.push(self.game_state.get_prey_observation(i).to_vec());
        }
        Ok(obs)
    }

    /// Steps the simulation.
    /// Returns `(obs, rewards, dones, terminal_obs, prey_obs, prey_rewards, prey_dones)`
    fn step(
        &mut self,
        actions: Vec<usize>,
        prey_actions: Vec<usize>,
    ) -> PyResult<(
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
    )> {
        for (i, &action) in actions.iter().enumerate() {
            if i < self.num_snakes {
                let rel_action = RelativeAction::from_usize(action);
                let new_dir = rel_action.to_absolute_direction(self.game_state.snakes[i].direction);
                self.game_state.set_direction(i, new_dir);
            }
        }

        let get_min_dist = |state: &GameState, pos: (f32, f32)| -> f32 {
            let mut min_d = f32::MAX;
            for p in &state.preys {
                if !p.is_dead {
                    let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                    let dx = pos.0 - p_grid.0 as f32;
                    let dy = pos.1 - p_grid.1 as f32;
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            if min_d == f32::MAX { 0.0 } else { min_d }
        };

        let prev_scores: Vec<u32> = self.game_state.snakes.iter().map(|s| s.score).collect();
        let prev_kills: Vec<u32> = self.game_state.snakes.iter().map(|s| s.kills).collect();
        let prev_dists: Vec<f32> = self.game_state.snakes.iter().map(|s| get_min_dist(&self.game_state, s.head_pos)).collect();

        self.game_state.step(1.0, &prey_actions);

        // Reward function for snakes
        let calc_reward = |snake: &animals_engine::SnakeState, prev_score: u32, prev_kills: u32, prev_dist: f32| -> f32 {
            if snake.is_dead {
                if snake.death_by_hunger {
                    -5.0
                } else {
                    -3.0
                }
            } else if snake.kills > prev_kills {
                50.0
            } else if snake.score > prev_score {
                30.0
            } else {
                let current_dist = get_min_dist(&self.game_state, snake.head_pos);
                let dist_reward = (prev_dist - current_dist) * 0.15;
                let hunger_penalty = -0.01 * (snake.steps_since_last_eat as f32 / 50.0);
                hunger_penalty + dist_reward
            }
        };

        let mut rewards = Vec::new();
        let mut dones = Vec::new();
        let mut terminal_obs = Vec::new();

        for i in 0..self.num_snakes {
            let done = self.game_state.snakes[i].is_dead;
            let reward = calc_reward(&self.game_state.snakes[i], prev_scores[i], prev_kills[i], prev_dists[i]);
            rewards.push(reward);
            dones.push(done);
            if done {
                terminal_obs.push(self.game_state.get_relative_observation(i).to_vec());
            } else {
                terminal_obs.push(vec![0.0; 66]);
            }
        }

        // Respawn any snakes that died this tick
        self.game_state.respawn_dead();

        let mut obs = Vec::new();
        for i in 0..self.num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }

        // Prey done and rewards
        let mut prey_obs = Vec::new();
        let mut prey_rewards = Vec::new();
        let mut prey_dones = Vec::new();

        for i in 0..self.num_preys {
            let prey_done = self.game_state.prey_died_this_tick[i];
            let prey_reward = if prey_done {
                -10.0
            } else {
                0.1
            };
            prey_obs.push(self.game_state.get_prey_observation(i).to_vec());
            prey_rewards.push(prey_reward);
            prey_dones.push(prey_done);
        }

        Ok((obs, rewards, dones, terminal_obs, prey_obs, prey_rewards, prey_dones))
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
            dict.set_item("death_by_hunger", snake.death_by_hunger)?;
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
