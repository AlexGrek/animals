#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashSet;

use pyo3::prelude::*;
use animals_engine::{
    GameState, RelativeAction, HUNGER_LIMIT, PREY_OBS_SIZE, SMELL_RANGE, SNAKE_OBS_SIZE,
};

/// Distance from `pos` to the nearest alive prey **within `SMELL_RANGE`
/// torus-wrapped Manhattan cells**, or `None` if nothing is in smell range.
fn min_dist_to_smelled_prey(state: &GameState, pos: (f32, f32)) -> Option<f32> {
    let from = (pos.0.round() as i32, pos.1.round() as i32);
    let mut min_d = f32::MAX;
    for p in &state.preys {
        if !p.is_dead {
            let to = (p.pos.0.round() as i32, p.pos.1.round() as i32);
            let (dx, dy) = state.torus_delta(from, to);
            if dx.abs() + dy.abs() > SMELL_RANGE {
                continue;
            }
            let d = ((dx * dx + dy * dy) as f32).sqrt();
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

fn min_dist_to_other_prey(state: &GameState, pos: (f32, f32), self_idx: usize) -> Option<f32> {
    let mut min_d = f32::MAX;
    let w = state.grid_width as f32;
    let h = state.grid_height as f32;
    for (i, p) in state.preys.iter().enumerate() {
        if i == self_idx || p.is_dead {
            continue;
        }
        let mut dx = (pos.0 - p.pos.0).abs();
        let mut dy = (pos.1 - p.pos.1).abs();
        if dx > w / 2.0 { dx = w - dx; }
        if dy > h / 2.0 { dy = h - dy; }
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
    initial_preys: usize,
    max_preys: usize,
    initial_amphibias: usize,
    max_amphibias: usize,
    visited: Vec<HashSet<(i32, i32)>>,
}

#[pymethods]
impl Simulation {
    #[new]
    fn new(num_snakes: usize, initial_preys: usize, max_preys: usize, initial_amphibias: usize, max_amphibias: usize) -> Self {
        Self {
            game_state: GameState::new(100, 100, num_snakes, initial_preys, max_preys, initial_amphibias, max_amphibias, true),
            num_snakes,
            initial_preys,
            max_preys,
            initial_amphibias,
            max_amphibias,
            visited: vec![HashSet::new(); num_snakes],
        }
    }

    fn reset(&mut self) -> PyResult<Vec<Vec<f32>>> {
        self.game_state = GameState::new(100, 100, self.num_snakes, self.initial_preys, self.max_preys, self.initial_amphibias, self.max_amphibias, true);
        self.visited = vec![HashSet::new(); self.num_snakes];
        let mut obs = Vec::new();
        for i in 0..self.num_snakes {
            obs.push(self.game_state.get_relative_observation(i).to_vec());
        }
        Ok(obs)
    }

    fn get_all_prey_observations(&self) -> PyResult<Vec<Vec<f32>>> {
        let mut obs = Vec::new();
        for i in 0..self.max_preys {
            obs.push(self.game_state.get_prey_observation(i).to_vec());
        }
        Ok(obs)
    }

    fn get_all_amphibia_observations(&self) -> PyResult<Vec<Vec<f32>>> {
        let mut obs = Vec::new();
        for i in self.max_preys..(self.max_preys + self.max_amphibias) {
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
            .map(|s| min_dist_to_smelled_prey(&self.game_state, s.head_pos))
            .collect();
        // Each prey's distance to the nearest snake head, indexed as
        // [preys..., amphibias...] to match the engine's `preys` vector.
        let prev_prey_threat: Vec<Option<f32>> = self
            .game_state
            .preys
            .iter()
            .map(|p| if p.is_dead { None } else { min_dist_to_snake_head(&self.game_state, p.pos) })
            .collect();
            
        // Also track each prey's distance to the nearest OTHER prey (crowding)
        let prev_prey_crowding: Vec<Option<f32>> = self
            .game_state
            .preys
            .iter()
            .enumerate()
            .map(|(i, p)| if p.is_dead { None } else { min_dist_to_other_prey(&self.game_state, p.pos, i) })
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
            let ate = !done && snake.score != prev_scores[i];

            let mut reward = if done {
                if snake.death_by_hunger { -5.0 } else { -3.0 }
            } else {
                let mut r = 0.0;
                // Kills and eats are independent events: sum them so a
                // same-tick kill-and-eat is fully credited.
                r += 50.0 * (snake.kills - prev_kills[i]) as f32;
                r += 30.0 * (snake.score - prev_scores[i]) as f32;
                if !ate {
                    // Only shape toward prey on ticks where we didn't eat — an
                    // eat resets which prey is nearest, making the delta noise.
                    let cur_smell = min_dist_to_smelled_prey(&self.game_state, snake.head_pos);
                    if let (Some(prev), Some(cur)) = (prev_snake_dists[i], cur_smell) {
                        r += (prev - cur).clamp(-2.0, 2.0) * 0.15;
                    }
                    r += -0.01 * (snake.steps_since_last_eat as f32 / (HUNGER_LIMIT as f32 / 4.0));

                    // Exploration bonus: only when nothing is smelled (smell ->
                    // pursue via shaping above, no smell -> explore), reward
                    // entering a not-yet-visited coarse (4x4) grid cell.
                    // If no new cell is entered, penalize slightly to discourage spinning.
                    if cur_smell.is_none() {
                        let head = snake.body[0];
                        let coarse = (head.0 / 4, head.1 / 4);
                        if self.visited[i].insert(coarse) {
                            r += 0.1;
                        } else {
                            r -= 0.03;
                        }
                    }
                }
                r
            };

            // A snake's visited set only reflects "since last meal, this life" —
            // clear it on death (respawns elsewhere) and on eating (fresh hunt).
            if done || ate {
                self.visited[i].clear();
            }

            // Mitosis (body reached the split threshold) is the pinnacle event, so
            // it stays the largest single reward — but only just above a kill (50)
            // rather than 2x it. Each mitosis already sits on top of ~6-8 eats worth
            // of +30, so an outsized spike here mostly inflates value-function
            // variance without changing the incentive ordering.
            if self.game_state.snakes[i].mitosis_count > 0 {
                reward += 60.0 * self.game_state.snakes[i].mitosis_count as f32;
                self.game_state.snakes[i].mitosis_count = 0;
            }

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
                
                let (penalty, shaping_crowding) = match (
                    prev_prey_crowding[idx],
                    min_dist_to_other_prey(&self.game_state, self.game_state.preys[idx].pos, idx),
                ) {
                    (Some(prev), Some(cur)) => {
                        let p = if cur < 10.0 { -0.1 } else { 0.0 };
                        let s = (cur - prev).clamp(-2.0, 2.0) * 0.05;
                        (p, s)
                    },
                    _ => (0.0, 0.0),
                };
                
                0.1 + shaping + penalty + shaping_crowding
            }
        };

        let mut prey_rewards = Vec::with_capacity(self.max_preys);
        let mut prey_dones = Vec::with_capacity(self.max_preys);
        let mut prey_terminal_obs = Vec::with_capacity(self.max_preys);
        for i in 0..self.max_preys {
            let done = self.game_state.prey_died_this_tick[i];
            prey_rewards.push(prey_reward(i));
            prey_dones.push(done);
            prey_terminal_obs.push(if done {
                self.game_state.get_prey_observation(i).to_vec()
            } else {
                vec![0.0; PREY_OBS_SIZE]
            });
        }

        let mut amphibia_rewards = Vec::with_capacity(self.max_amphibias);
        let mut amphibia_dones = Vec::with_capacity(self.max_amphibias);
        let mut amphibia_terminal_obs = Vec::with_capacity(self.max_amphibias);
        for i in self.max_preys..(self.max_preys + self.max_amphibias) {
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
        let mut prey_obs = Vec::with_capacity(self.max_preys);
        for i in 0..self.max_preys {
            prey_obs.push(self.game_state.get_prey_observation(i).to_vec());
        }
        let mut amphibia_obs = Vec::with_capacity(self.max_amphibias);
        for i in self.max_preys..(self.max_preys + self.max_amphibias) {
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
