use rand::Rng;
use std::collections::HashSet;
use crate::map::{Map, Terrain};
use crate::snake::SnakeState;
use crate::species::Species;
use crate::direction::Direction;
use crate::{HUNGER_LIMIT, PREY_OBS_SIZE, SNAKE_OBS_SIZE};

#[derive(Clone, Debug)]
pub struct PreyState {
    pub pos: (f32, f32),
    pub is_dead: bool,
    pub species: Species,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub snakes: Vec<SnakeState>,
    pub preys: Vec<PreyState>,
    pub grid_width: i32,
    pub grid_height: i32,
    pub map: Map,
    pub game_over: bool,
    pub prey_died_this_tick: Vec<bool>,
}

impl GameState {
    pub fn new(width: i32, height: i32, num_snakes: usize, num_preys: usize, num_amphibias: usize) -> Self {
        let map = Map::new(width, height);

        let mut preys = Vec::new();
        for _ in 0..num_preys {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: false, species: Species::Prey });
        }
        for _ in 0..num_amphibias {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: false, species: Species::Amphibia });
        }
        let total_preys = num_preys + num_amphibias;

        let mut state = Self {
            snakes: vec![SnakeState::new((0, 0), Direction::Up); num_snakes],
            preys,
            grid_width: width,
            grid_height: height,
            map,
            game_over: false,
            prey_died_this_tick: vec![false; total_preys],
        };

        for i in 0..num_snakes {
            let (pos, direction) = state.spawn_position(i);
            state.snakes[i] = SnakeState::new(pos, direction);
        }
        for i in 0..total_preys {
            state.spawn_prey(i);
        }
        state
    }

    /// The deterministic "evenly spaced columns, mid-height" layout used both
    /// for the initial game setup and as the preferred respawn location.
    fn initial_spawn(&self, index: usize, num_snakes: usize) -> ((i32, i32), Direction) {
        let spacing = self.grid_width / (num_snakes as i32 + 1);
        let x = spacing * (index as i32 + 1);
        let direction = if index % 2 == 0 { Direction::Up } else { Direction::Down };
        ((x, self.grid_height / 2), direction)
    }

    /// Whether `pos` is free of any snake body, optionally excluding one snake
    /// (used when respawning that same snake, so its own stale body doesn't
    /// block its new spawn cell).
    fn is_cell_free(&self, pos: (i32, i32), exclude: Option<usize>) -> bool {
        for p in &self.preys {
            if !p.is_dead {
                let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                if pos == p_grid {
                    return false;
                }
            }
        }
        let terrain = self.map.get_terrain(pos.0, pos.1);
        if terrain == Terrain::Rock || terrain == Terrain::Water {
            return false;
        }
        for (i, s) in self.snakes.iter().enumerate() {
            if Some(i) == exclude {
                continue;
            }
            if s.body.contains(&pos) {
                return false;
            }
        }
        true
    }

    /// Picks a spawn position for `index`: prefer the deterministic evenly
    /// spaced column used at game start; if occupied, fall back to a random
    /// free cell.
    fn spawn_position(&self, index: usize) -> ((i32, i32), Direction) {
        let (preferred, direction) = self.initial_spawn(index, self.snakes.len());
        if self.is_cell_free(preferred, Some(index)) {
            return (preferred, direction);
        }

        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x, y);
            if self.is_cell_free(pos, Some(index)) {
                return (pos, direction);
            }
        }
    }

    /// Respawns every snake currently marked dead: fresh body of length 1,
    /// score/kills/death flags reset for the new life. Does not touch snakes
    /// that are still alive.
    pub fn respawn_dead(&mut self) {
        let dead_indices: Vec<usize> = self
            .snakes
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_dead)
            .map(|(i, _)| i)
            .collect();

        for i in dead_indices {
            let (pos, direction) = self.spawn_position(i);
            self.snakes[i] = SnakeState::new(pos, direction);
        }
    }

    pub fn set_direction(&mut self, snake_index: usize, new_dir: Direction) {
        if let Some(snake) = self.snakes.get_mut(snake_index) {
            if snake.direction.opposite() != new_dir {
                snake.direction = new_dir;
            }
        }
    }

    pub fn step(&mut self, dt: f32, prey_actions: &[usize]) {
        if self.game_over {
            return;
        }

        for p in &mut self.prey_died_this_tick {
            *p = false;
        }

        // Increment hunger and check for hunger death
        for i in 0..self.snakes.len() {
            if !self.snakes[i].is_dead {
                self.snakes[i].steps_since_last_eat += 1;
                if self.snakes[i].steps_since_last_eat >= HUNGER_LIMIT {
                    self.snakes[i].is_dead = true;
                    self.snakes[i].death_by_hunger = true;
                }
            }
        }

        // 1. Move preys
        for i in 0..self.preys.len() {
            if self.preys[i].is_dead { continue; }
            let prey_action = prey_actions.get(i).copied().unwrap_or(0);
            
            let dir_vec = match prey_action {
                1 => (0, 1),   // Up
                2 => (1, 0),   // Right
                3 => (0, -1),  // Down
                4 => (-1, 0),  // Left
                _ => (0, 0),   // Stand
            };

            if dir_vec != (0, 0) {
                let prev_pos = self.preys[i].pos;
                let px_before = prev_pos.0.round() as i32;
                let py_before = prev_pos.1.round() as i32;
                let terrain = self.map.get_terrain(px_before, py_before);
                let speed = self.preys[i].species.speed_on(terrain);

                self.preys[i].pos.0 += dir_vec.0 as f32 * speed * dt;
                self.preys[i].pos.1 += dir_vec.1 as f32 * speed * dt;

                // Clamp to grid boundaries
                if self.preys[i].pos.0 < 0.0 { self.preys[i].pos.0 = 0.0; }
                if self.preys[i].pos.0 >= self.grid_width as f32 { self.preys[i].pos.0 = self.grid_width as f32 - 1.0; }
                if self.preys[i].pos.1 < 0.0 { self.preys[i].pos.1 = 0.0; }
                if self.preys[i].pos.1 >= self.grid_height as f32 { self.preys[i].pos.1 = self.grid_height as f32 - 1.0; }

                // Collide with rocks and water — restore to pre-move position
                let px_after = self.preys[i].pos.0.round() as i32;
                let py_after = self.preys[i].pos.1.round() as i32;
                let terrain_after = self.map.get_terrain(px_after, py_after);
                if terrain_after == Terrain::Rock {
                    self.preys[i].pos = prev_pos;
                }
            }
        }

        // 2. Move snakes
        let mut cell_changed = vec![false; self.snakes.len()];
        let mut new_heads = vec![(0, 0); self.snakes.len()];

        for i in 0..self.snakes.len() {
            let snake = &mut self.snakes[i];
            if snake.is_dead {
                new_heads[i] = snake.body[0];
                continue;
            }
            
            let q_head_before = snake.body[0];
            let terrain = self.map.get_terrain(q_head_before.0, q_head_before.1);
            let speed = Species::Snake.speed_on(terrain);
            let dir_vec = snake.direction.to_vector();
            
            snake.head_pos.0 += dir_vec.0 as f32 * speed * dt;
            snake.head_pos.1 += dir_vec.1 as f32 * speed * dt;
            
            let q_head_after = (snake.head_pos.0.round() as i32, snake.head_pos.1.round() as i32);
            new_heads[i] = q_head_after;
            if q_head_after != q_head_before {
                cell_changed[i] = true;
            }
        }

        let was_alive: Vec<bool> = self.snakes.iter().map(|s| !s.is_dead).collect();
        let mut head_to_head = vec![false; self.snakes.len()];
        for i in 0..self.snakes.len() {
            if !was_alive[i] || !cell_changed[i] { continue; }
            for j in 0..self.snakes.len() {
                if i == j || !was_alive[j] { continue; }
                if new_heads[i] == new_heads[j] {
                    head_to_head[i] = true;
                    head_to_head[j] = true;
                }
            }
        }

        // Check collisions
        for i in 0..self.snakes.len() {
            if self.snakes[i].is_dead { continue; }
            let head = new_heads[i];

            if head_to_head[i] {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_opponent = true;
                continue;
            }

            if !cell_changed[i] { continue; }

            // Wall collision
            if head.0 < 0 || head.0 >= self.grid_width || head.1 < 0 || head.1 >= self.grid_height {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_wall = true;
                continue;
            }

            // Rock collision
            if self.map.get_terrain(head.0, head.1) == Terrain::Rock {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_wall = true;
                continue;
            }

            // Body collision
            for j in 0..self.snakes.len() {
                let snake_j = &self.snakes[j];
                if snake_j.body.contains(&head) {
                    self.snakes[i].is_dead = true;
                    if i == j {
                        self.snakes[i].death_by_self = true;
                    } else {
                        self.snakes[i].death_by_opponent = true;
                        self.snakes[j].kills += 1;
                    }
                    break;
                }
            }
        }

        // Move and eat prey
        for i in 0..self.snakes.len() {
            if self.snakes[i].is_dead || !cell_changed[i] { continue; }
            let head = new_heads[i];
            self.snakes[i].body.insert(0, head);

            let mut ate = false;
            for p_idx in 0..self.preys.len() {
                if !self.preys[p_idx].is_dead {
                    let prey_grid_pos = (self.preys[p_idx].pos.0.round() as i32, self.preys[p_idx].pos.1.round() as i32);
                    let dx = (head.0 - prey_grid_pos.0).abs();
                    let dy = (head.1 - prey_grid_pos.1).abs();
                    if dx <= 1 && dy <= 1 {
                        self.snakes[i].score += 1;
                        self.snakes[i].steps_since_last_eat = 0;
                        self.preys[p_idx].is_dead = true;
                        self.prey_died_this_tick[p_idx] = true;
                        ate = true;
                        break;
                    }
                }
            }

            if !ate {
                self.snakes[i].body.pop();
            }
        }
    }

    /// Respawns every prey currently marked dead at a random free cell. Callers
    /// invoke this after `step()` (the training simulation captures a dead
    /// prey's true pre-respawn terminal observation first; the visualizer just
    /// respawns immediately). Mirrors `respawn_dead()` for snakes.
    pub fn respawn_dead_preys(&mut self) {
        for p_idx in 0..self.preys.len() {
            if self.preys[p_idx].is_dead {
                self.spawn_prey(p_idx);
            }
        }
    }

    pub fn spawn_prey(&mut self, index: usize) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x, y);
            
            let terrain = self.map.get_terrain(x, y);
            if terrain == Terrain::Rock || terrain == Terrain::Water {
                continue;
            }

            let mut free = true;
            for s in &self.snakes {
                if s.body.contains(&pos) {
                    free = false;
                    break;
                }
            }
            for (pi, p) in self.preys.iter().enumerate() {
                if pi != index && !p.is_dead {
                    let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                    if pos == p_grid {
                        free = false;
                        break;
                    }
                }
            }
            if free {
                self.preys[index].pos = (x as f32, y as f32);
                self.preys[index].is_dead = false;
                break;
            }
        }
    }

    /// Snake observation (`SNAKE_OBS_SIZE` floats):
    /// - `[0..64)`  — 8x8 grid in the snake's rotated frame (4 cells ahead,
    ///   3 behind, 4 right, 3 left). Cell encoding: prey `1.0`; wall/rock/own
    ///   body `-1.0`; enemy head `-0.8`; enemy body `-0.5`; else passable
    ///   terrain `Species::Snake.speed_on(terrain) * 0.5`.
    /// - `[64]`/`[65]` — unit direction to the nearest alive prey (forward /
    ///   right components), zero when no prey exists.
    /// - `[66]` — distance to that prey normalized by the larger grid
    ///   dimension (`1.0` when no prey exists).
    /// - `[67]` — hunger: `steps_since_last_eat / HUNGER_LIMIT`.
    /// - `[68]` — own length / 100, capped at 1.
    pub fn get_relative_observation(&self, snake_index: usize) -> [f32; SNAKE_OBS_SIZE] {
        let mut obs = [0.0; SNAKE_OBS_SIZE];
        let snake = &self.snakes[snake_index];
        if snake.body.is_empty() { return obs; }

        let head = snake.body[0];
        let dir = snake.direction;
        let vec_straight = dir.to_vector();
        let vec_right = dir.turn_right().to_vector();

        // Build occupancy sets once (O(total_length)), then each of the 64 grid
        // cells is an O(1) lookup instead of a linear scan over every body Vec.
        let own_body: HashSet<(i32, i32)> = snake.body.iter().copied().collect();
        let mut enemy_bodies: HashSet<(i32, i32)> = HashSet::new();
        let mut enemy_heads: HashSet<(i32, i32)> = HashSet::new();
        for (j, s) in self.snakes.iter().enumerate() {
            if j == snake_index || s.is_dead { continue; }
            if let Some(&h) = s.body.first() {
                enemy_heads.insert(h);
            }
            enemy_bodies.extend(s.body.iter().copied());
        }
        let mut prey_cells: HashSet<(i32, i32)> = HashSet::new();
        for p in &self.preys {
            if !p.is_dead {
                prey_cells.insert((p.pos.0.round() as i32, p.pos.1.round() as i32));
            }
        }

        let mut idx = 0;
        for f in -3..=4 {
            for r in -3..=4 {
                let cx = head.0 + f * vec_straight.0 + r * vec_right.0;
                let cy = head.1 + f * vec_straight.1 + r * vec_right.1;
                let cell = (cx, cy);

                let out_of_bounds = cx < 0 || cx >= self.grid_width || cy < 0 || cy >= self.grid_height;
                let terrain = if out_of_bounds { Terrain::Rock } else { self.map.get_terrain(cx, cy) };

                obs[idx] = if prey_cells.contains(&cell) {
                    1.0
                } else if out_of_bounds || terrain == Terrain::Rock || own_body.contains(&cell) {
                    -1.0
                } else if enemy_heads.contains(&cell) {
                    -0.8
                } else if enemy_bodies.contains(&cell) {
                    -0.5
                } else {
                    Species::Snake.speed_on(terrain) * 0.5
                };
                idx += 1;
            }
        }

        // Unit direction + normalized distance to the nearest alive prey. A unit
        // vector keeps the heading signal strong at any range (the old
        // `dx / max_dim` encoding shrank to ~0.01 for nearby prey).
        let mut closest: Option<(i32, i32, f32)> = None;
        for p in &self.preys {
            if !p.is_dead {
                let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                let dx = p_grid.0 - head.0;
                let dy = p_grid.1 - head.1;
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                if closest.map_or(true, |(_, _, cd)| d < cd) {
                    closest = Some((dx, dy, d));
                }
            }
        }

        let max_dim = self.grid_width.max(self.grid_height) as f32;
        if let Some((dx, dy, dist)) = closest {
            let d = dist.max(1e-6);
            obs[64] = (dx * vec_straight.0 + dy * vec_straight.1) as f32 / d;
            obs[65] = (dx * vec_right.0 + dy * vec_right.1) as f32 / d;
            obs[66] = (dist / max_dim).min(1.0);
        } else {
            obs[66] = 1.0;
        }
        obs[67] = (snake.steps_since_last_eat as f32 / HUNGER_LIMIT as f32).min(1.0);
        obs[68] = (snake.body.len() as f32 / 100.0).min(1.0);

        obs
    }

    /// Prey observation (`PREY_OBS_SIZE` floats), shared by Prey and Amphibia
    /// (terrain values are already species-relative, so the two variants read
    /// the same map differently without needing a species flag):
    /// - `[0..64)` — 8x8 grid in the absolute frame (up is north). Cell
    ///   encoding: OOB/rock `-1.0`; snake head (the lethal part) `-0.8`; snake
    ///   body `-0.5`; else `prey.species.speed_on(terrain) * 0.5` (so water
    ///   reads ~0.1 to a Prey but ~0.5 to an Amphibia).
    /// - `[64]`/`[65]` — unit direction (east / north) to the nearest alive
    ///   snake head, zero when no snake is alive.
    /// - `[66]` — distance to that head normalized by the larger grid
    ///   dimension (`1.0` when no snake is alive). Global threat sense the prey
    ///   needs to flee predators outside its local 8x8 patch.
    pub fn get_prey_observation(&self, prey_index: usize) -> [f32; PREY_OBS_SIZE] {
        let mut obs = [0.0; PREY_OBS_SIZE];
        let prey = &self.preys[prey_index];
        let prey_grid_pos = (prey.pos.0.round() as i32, prey.pos.1.round() as i32);

        let mut snake_bodies: HashSet<(i32, i32)> = HashSet::new();
        let mut snake_heads: HashSet<(i32, i32)> = HashSet::new();
        for snake in &self.snakes {
            if snake.is_dead { continue; }
            if let Some(&h) = snake.body.first() {
                snake_heads.insert(h);
            }
            snake_bodies.extend(snake.body.iter().copied());
        }

        let mut idx = 0;
        for dy in -3..=4 { // North-South (Up is always North)
            for dx in -3..=4 { // East-West
                let cx = prey_grid_pos.0 + dx;
                let cy = prey_grid_pos.1 + dy;
                let cell = (cx, cy);

                let out_of_bounds = cx < 0 || cx >= self.grid_width || cy < 0 || cy >= self.grid_height;
                let terrain = if out_of_bounds { Terrain::Rock } else { self.map.get_terrain(cx, cy) };

                obs[idx] = if out_of_bounds || terrain == Terrain::Rock {
                    -1.0
                } else if snake_heads.contains(&cell) {
                    -0.8
                } else if snake_bodies.contains(&cell) {
                    -0.5
                } else {
                    prey.species.speed_on(terrain) * 0.5
                };
                idx += 1;
            }
        }

        // Unit direction + normalized distance to the nearest alive snake head.
        let mut closest: Option<(i32, i32, f32)> = None;
        for h in &snake_heads {
            let dx = h.0 - prey_grid_pos.0;
            let dy = h.1 - prey_grid_pos.1;
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if closest.map_or(true, |(_, _, cd)| d < cd) {
                closest = Some((dx, dy, d));
            }
        }

        let max_dim = self.grid_width.max(self.grid_height) as f32;
        if let Some((dx, dy, dist)) = closest {
            let d = dist.max(1e-6);
            obs[64] = dx as f32 / d;
            obs[65] = dy as f32 / d;
            obs[66] = (dist / max_dim).min(1.0);
        } else {
            obs[66] = 1.0;
        }

        obs
    }
}
