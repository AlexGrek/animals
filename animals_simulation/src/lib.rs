#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use animals_engine::{GameState, RelativeAction, HUNGER_LIMIT, PREY_OBS_SIZE, SNAKE_OBS_SIZE};

/// Distance from `pos` to the nearest alive prey, or `None` if none are alive.
fn min_dist_to_prey(state: &GameState, pos: (f32, f32)) -> Option<f32> {
    let mut min_d = f32::MAX;
    for p in &state.preys {
        if !p.is_dead {
            let dx = pos.0 - p.pos.0.round();
            let dy = pos.1 - p.pos.1.round();
            let d = (dx * dx + dy * dy).sqrt();
            if d < min_d {
                min_d = d;
            }
        }
    }
    if min_d == f32::MAX { None } else { Some(min_d) }
}

/// Distance from `pos` to the nearest alive snake's head, or `None` if none.
fn min_dist_to_snake_head(state: &GameState, pos: (f32, f32)) -> Option<f32> {
    let mut min_d = f32::MAX;
    for s in &state.snakes {
        if s.is_dead || s.body.is_empty() {
            continue;
        }
        let h = s.body[0];
        let dx = pos.0 - h.0 as f32;
        let dy = pos.1 - h.1 as f32;
        let d = (dx * dx + dy * dy).sqrt();
        if d < min_d {
            min_d = d;
        }
    }
    if min_d == f32::MAX { None } else { Some(min_d) }
}

#[pyclass]
pub struct Simulation {
    game_state: GameState,
    num_snakes: usize,
    num_preys: usize,
    num_amphibias: usize,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(num_snakes: usize, num_preys: usize, num_amphibias: usize) -> Self {
        Self {
            game_state: GameState::new(100, 100, num_snakes, num_preys, num_amphibias),
            num_snakes,
            num_preys,
            num_amphibias,
        }
    }

    fn reset(&mut self) -> PyResult<Vec<Vec<f32>>> {
        self.game_state = GameState::new(100, 100, self.num_snakes, self.num_preys, self.num_amphibias);
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

    fn get_all_amphibia_observations(&self) -> PyResult<Vec<Vec<f32>>> {
        let mut obs = Vec::new();
        for i in self.num_preys..(self.num_preys + self.num_amphibias) {
            obs.push(self.game_state.get_prey_observation(i).to_vec());
        }
        Ok(obs)
    }

    /// Steps the simulation one tick.
    ///
    /// Returns a 12-tuple:
    /// `(obs, rewards, dones, terminal_obs,`
    ///  ` prey_obs, prey_rewards, prey_dones,`
    ///  ` amphibia_obs, amphibia_rewards, amphibia_dones,`
    ///  ` prey_terminal_obs, amphibia_terminal_obs)`.
    ///
    /// The `*_obs` are post-respawn (next-episode) observations; the
    /// `*_terminal_obs` are the true pre-respawn observations, meaningful only
    /// where the matching `*_dones` entry is true.
    fn step(
        &mut self,
        actions: Vec<usize>,
        prey_actions: Vec<usize>,
        amphibia_actions: Vec<usize>,
    ) -> PyResult<(
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
        Vec<Vec<f32>>,
        Vec<f32>,
        Vec<bool>,
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
    )> {
        for (i, &action) in actions.iter().enumerate() {
            if i < self.num_snakes {
                let rel_action = RelativeAction::from_usize(action);
                let new_dir = rel_action.to_absolute_direction(self.game_state.snakes[i].direction);
                self.game_state.set_direction(i, new_dir);
            }
        }

        // Pre-step snapshots for reward shaping.
        let prev_scores: Vec<u32> = self.game_state.snakes.iter().map(|s| s.score).collect();
        let prev_kills: Vec<u32> = self.game_state.snakes.iter().map(|s| s.kills).collect();
        let prev_snake_dists: Vec<Option<f32>> = self
            .game_state
            .snakes
            .iter()
            .map(|s| min_dist_to_prey(&self.game_state, s.head_pos))
            .collect();
        // Each prey's distance to the nearest snake head, indexed as
        // [preys..., amphibias...] to match the engine's `preys` vector.
        let prev_prey_threat: Vec<Option<f32>> = self
            .game_state
            .preys
            .iter()
            .map(|p| min_dist_to_snake_head(&self.game_state, p.pos))
            .collect();

        let mut all_prey_actions = prey_actions;
        all_prey_actions.extend(amphibia_actions);
        self.game_state.step(1.0, &all_prey_actions);

        // ---- Snake rewards + terminal observations (captured while still dead) ----
        let mut rewards = Vec::with_capacity(self.num_snakes);
        let mut dones = Vec::with_capacity(self.num_snakes);
        let mut terminal_obs = Vec::with_capacity(self.num_snakes);

        for i in 0..self.num_snakes {
            let snake = &self.game_state.snakes[i];
            let done = snake.is_dead;

            let reward = if snake.is_dead {
                if snake.death_by_hunger { -5.0 } else { -3.0 }
            } else {
                let mut r = 0.0;
                // Kills and eats are independent events: sum them so a
                // same-tick kill-and-eat is fully credited.
                r += 50.0 * (snake.kills - prev_kills[i]) as f32;
                r += 30.0 * (snake.score - prev_scores[i]) as f32;
                if snake.score == prev_scores[i] {
                    // Only shape toward prey on ticks where we didn't eat — an
                    // eat resets which prey is nearest, making the delta noise.
                    if let (Some(prev), Some(cur)) =
                        (prev_snake_dists[i], min_dist_to_prey(&self.game_state, snake.head_pos))
                    {
                        r += (prev - cur).clamp(-2.0, 2.0) * 0.15;
                    }
                    r += -0.01 * (snake.steps_since_last_eat as f32 / (HUNGER_LIMIT as f32 / 4.0));
                }
                r
            };

            rewards.push(reward);
            dones.push(done);
            if done {
                terminal_obs.push(self.game_state.get_relative_observation(i).to_vec());
            } else {
                terminal_obs.push(vec![0.0; SNAKE_OBS_SIZE]);
            }
        }

        // ---- Prey / amphibia rewards + terminal observations (pre-respawn) ----
        // Threat shaping: reward moving away from the nearest snake head. For
        // amphibia this naturally rewards fleeing into water, where snakes
        // crawl at 0.2 but amphibia swim at 1.0.
        let prey_reward = |idx: usize| -> f32 {
            if self.game_state.prey_died_this_tick[idx] {
                -10.0
            } else {
                let shaping = match (
                    prev_prey_threat[idx],
                    min_dist_to_snake_head(&self.game_state, self.game_state.preys[idx].pos),
                ) {
                    (Some(prev), Some(cur)) => (cur - prev).clamp(-2.0, 2.0) * 0.1,
                    _ => 0.0,
                };
                0.1 + shaping
            }
        };

        let mut prey_rewards = Vec::with_capacity(self.num_preys);
        let mut prey_dones = Vec::with_capacity(self.num_preys);
        let mut prey_terminal_obs = Vec::with_capacity(self.num_preys);
        for i in 0..self.num_preys {
            let done = self.game_state.prey_died_this_tick[i];
            prey_rewards.push(prey_reward(i));
            prey_dones.push(done);
            prey_terminal_obs.push(if done {
                self.game_state.get_prey_observation(i).to_vec()
            } else {
                vec![0.0; PREY_OBS_SIZE]
            });
        }

        let mut amphibia_rewards = Vec::with_capacity(self.num_amphibias);
        let mut amphibia_dones = Vec::with_capacity(self.num_amphibias);
        let mut amphibia_terminal_obs = Vec::with_capacity(self.num_amphibias);
        for i in self.num_preys..(self.num_preys + self.num_amphibias) {
            let done = self.game_state.prey_died_this_tick[i];
            amphibia_rewards.push(prey_reward(i));
            amphibia_dones.push(done);
            amphibia_terminal_obs.push(if done {
                self.game_state.get_prey_observation(i).to_vec()
            } else {
                vec![0.0; PREY_OBS_SIZE]
            });
        }

        // Respawn dead actors, then read the fresh (next-episode) observations.
        self.game_state.respawn_dead();
        self.game_state.respawn_dead_preys();

        let mut obs = Vec::with_capacity(self.num_snakes);
        for i in 0..self.num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }
        let mut prey_obs = Vec::with_capacity(self.num_preys);
        for i in 0..self.num_preys {
            prey_obs.push(self.game_state.get_prey_observation(i).to_vec());
        }
        let mut amphibia_obs = Vec::with_capacity(self.num_amphibias);
        for i in self.num_preys..(self.num_preys + self.num_amphibias) {
            amphibia_obs.push(self.game_state.get_prey_observation(i).to_vec());
        }

        Ok((
            obs,
            rewards,
            dones,
            terminal_obs,
            prey_obs,
            prey_rewards,
            prey_dones,
            amphibia_obs,
            amphibia_rewards,
            amphibia_dones,
            prey_terminal_obs,
            amphibia_terminal_obs,
        ))
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
