use rand::Rng;
use std::collections::HashSet;
use crate::map::{Map, Terrain};
use crate::snake::SnakeState;
use crate::species::Species;
use crate::direction::Direction;
use crate::{HUNGER_LIMIT, PREY_OBS_SIZE, SMELL_RANGE, SNAKE_OBS_SIZE};

#[derive(Clone, Debug)]
pub struct PreyState {
    pub pos: (f32, f32),
    pub is_dead: bool,
    pub species: Species,
    pub lifespan: i32,
    pub steps_alive: i32,
    pub death_by_reproduction: bool,
    pub just_revived: bool,
    pub grass_eaten: f32,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub snakes: Vec<SnakeState>,
    pub preys: Vec<PreyState>,
    pub grid_width: i32,
    pub grid_height: i32,
    pub map: Map,
    pub game_over: bool,
    pub cell_changed: Vec<bool>,
    pub prey_died_this_tick: Vec<bool>,
    pub is_training: bool,
}

impl GameState {
    pub fn new(width: i32, height: i32, num_snakes: usize, initial_preys: usize, max_preys: usize, initial_amphibias: usize, max_amphibias: usize, is_training: bool) -> Self {
        let map = Map::new(width, height);

        let mut preys = Vec::with_capacity(max_preys + max_amphibias);
        for i in 0..max_preys {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: i >= initial_preys, species: Species::Prey, lifespan: 0, steps_alive: 0, death_by_reproduction: false, just_revived: false, grass_eaten: 0.0 });
        }
        for i in 0..max_amphibias {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: i >= initial_amphibias, species: Species::Amphibia, lifespan: 0, steps_alive: 0, death_by_reproduction: false, just_revived: false, grass_eaten: 0.0 });
        }
        let total_preys = max_preys + max_amphibias;

        let mut state = Self {
            snakes: vec![SnakeState::new((0, 0), Direction::Up); num_snakes],
            preys,
            grid_width: width,
            grid_height: height,
            map,
            game_over: false,
            cell_changed: vec![false; num_snakes],
            prey_died_this_tick: vec![false; total_preys],
            is_training,
        };

        for i in 0..num_snakes {
            let (pos, direction) = state.spawn_position(i);
            state.snakes[i] = SnakeState::new(pos, direction);
        }
        for i in 0..total_preys {
            if !state.preys[i].is_dead {
                state.spawn_prey(i);
            }
        }
        state.update_targets();
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

    /// Whether `pos` lies within the grid bounds.
    fn in_bounds(&self, pos: (i32, i32)) -> bool {
        pos.0 >= 0 && pos.0 < self.grid_width && pos.1 >= 0 && pos.1 < self.grid_height
    }

    /// The 3 body cells (head, head-v, head-2v) a snake would occupy if
    /// spawned at `head` facing `direction`. Mirrors `SnakeState::new`.
    fn spawn_body_cells(head: (i32, i32), direction: Direction) -> [(i32, i32); 3] {
        let v = direction.to_vector();
        [head, (head.0 - v.0, head.1 - v.1), (head.0 - 2 * v.0, head.1 - 2 * v.1)]
    }

    /// Whether the full 3-cell body a snake would spawn with at `head` facing
    /// `direction` is entirely valid: every cell in-bounds and free per
    /// `is_cell_free` (not Rock/Water terrain, no overlapping snake body, no
    /// overlapping alive prey). `is_cell_free` already treats out-of-bounds
    /// terrain as Rock, but we still guard bounds explicitly to be safe.
    fn is_spawn_body_free(&self, head: (i32, i32), direction: Direction, exclude: Option<usize>) -> bool {
        Self::spawn_body_cells(head, direction)
            .iter()
            .all(|&cell| self.in_bounds(cell) && self.is_cell_free(cell, exclude))
    }

    /// Picks a spawn position for `index`: prefer the deterministic evenly
    /// spaced column used at game start; if the full 3-cell body doesn't fit
    /// there, fall back to a random free cell whose full 3-cell body fits.
    fn spawn_position(&self, index: usize) -> ((i32, i32), Direction) {
        let (preferred, direction) = self.initial_spawn(index, self.snakes.len());
        if self.is_spawn_body_free(preferred, direction, Some(index)) {
            return (preferred, direction);
        }

        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x, y);
            if self.is_spawn_body_free(pos, direction, Some(index)) {
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
        self.update_targets();
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

        self.update_targets();

        for p in &mut self.prey_died_this_tick {
            *p = false;
        }
        for p in &mut self.preys {
            p.death_by_reproduction = false;
            p.just_revived = false;
        }

        let mut preys_to_revive = Vec::new();

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

        // 0. Regenerate grass
        for i in 0..(self.map.width * self.map.height) as usize {
            if self.map.tiles[i] == Terrain::Grass {
                if self.map.grass_health[i] == 0.0 {
                    self.map.grass_empty_timer[i] += 1;
                    if self.map.grass_empty_timer[i] >= 300 {
                        self.map.grass_health[i] += 0.05;
                    }
                } else if self.map.grass_health[i] < 1.0 {
                    self.map.grass_empty_timer[i] = 0;
                    self.map.grass_health[i] += 0.05;
                    if self.map.grass_health[i] > 1.0 {
                        self.map.grass_health[i] = 1.0;
                    }
                }
            }
        }

        // 1. Move preys
        for i in 0..self.preys.len() {
            if self.preys[i].is_dead { continue; }
            
            self.preys[i].steps_alive += 1;
            
            if self.preys[i].grass_eaten >= 8.0 {
                let mut snake_near = false;
                for s in &self.snakes {
                    if s.is_dead || s.body.is_empty() { continue; }
                    let head = s.body[0];
                    let (dx, dy) = self.torus_delta(head, (self.preys[i].pos.0.round() as i32, self.preys[i].pos.1.round() as i32));
                    if dx.abs() <= 8 && dy.abs() <= 8 {
                        snake_near = true;
                        break;
                    }
                }
                
                if !snake_near {
                    self.preys[i].is_dead = true;
                    self.preys[i].death_by_reproduction = true;
                    self.prey_died_this_tick[i] = true;
                    
                    for _ in 0..3 {
                        preys_to_revive.push((self.preys[i].species, self.preys[i].pos));
                    }
                    continue; // Skip movement, this prey is dead
                }
            }

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

                self.preys[i].pos.0 = self.preys[i].pos.0.rem_euclid(self.grid_width as f32);
                self.preys[i].pos.1 = self.preys[i].pos.1.rem_euclid(self.grid_height as f32);

                // Collide with rocks and water — restore to pre-move position
                let px_after = self.preys[i].pos.0.round() as i32;
                let py_after = self.preys[i].pos.1.round() as i32;
                let terrain_after = self.map.get_terrain(px_after, py_after);
                if terrain_after == Terrain::Rock {
                    self.preys[i].pos = prev_pos;
                }
            }

            // Prey feeding
            let px_after = self.preys[i].pos.0.round() as i32;
            let py_after = self.preys[i].pos.1.round() as i32;
            let tile_idx = (py_after.rem_euclid(self.grid_height) * self.grid_width + px_after.rem_euclid(self.grid_width)) as usize;
            if self.map.tiles[tile_idx] == Terrain::Grass && self.map.grass_health[tile_idx] > 0.0 {
                let eat_amount = 0.5f32.min(self.map.grass_health[tile_idx]);
                self.map.grass_health[tile_idx] -= eat_amount;
                if self.map.grass_health[tile_idx] < 0.0001 {
                    self.map.grass_health[tile_idx] = 0.0;
                    self.map.grass_empty_timer[tile_idx] = 0;
                }
                self.preys[i].grass_eaten += eat_amount;
            }
        }

        // Process revived preys
        let mut alive_preys = 0;
        let mut alive_amphibias = 0;
        for p in &self.preys {
            if !p.is_dead {
                if p.species == Species::Prey { alive_preys += 1; }
                else { alive_amphibias += 1; }
            }
        }
        
        if alive_preys < 5 {
            preys_to_revive.push((Species::Prey, (-1.0, -1.0)));
        }
        if alive_amphibias < 5 {
            preys_to_revive.push((Species::Amphibia, (-1.0, -1.0)));
        }

        let mut rng = rand::thread_rng();
        for (species, pos) in preys_to_revive {
            if let Some(idx) = self.preys.iter().position(|p| p.is_dead && p.species == species && !p.death_by_reproduction) {
                if pos.0 < 0.0 {
                    self.spawn_prey(idx);
                } else {
                    let mut px = pos.0 + rng.gen_range(-1.0..=1.0);
                    let mut py = pos.1 + rng.gen_range(-1.0..=1.0);
                    px = px.rem_euclid(self.grid_width as f32);
                    py = py.rem_euclid(self.grid_height as f32);
                    self.preys[idx].pos = (px, py);
                    self.preys[idx].is_dead = false;
                    self.preys[idx].lifespan = rng.gen_range(200..=500);
                    self.preys[idx].steps_alive = 0;
                    self.preys[idx].just_revived = true;
                    self.preys[idx].death_by_reproduction = false;
                    self.preys[idx].grass_eaten = 0.0;
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
            
            snake.head_pos.0 = snake.head_pos.0.rem_euclid(self.grid_width as f32);
            snake.head_pos.1 = snake.head_pos.1.rem_euclid(self.grid_height as f32);
            
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

            // Wall collision (removed due to toroidal map)

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

        // Snake Mitosis Check
        let mut new_snakes = Vec::new();
        let num_snakes = self.snakes.len();
        for i in 0..num_snakes {
            if self.snakes[i].body.len() >= 9 {
                if self.is_training {
                    self.snakes[i].body.truncate(3);
                    self.snakes[i].mitosis_count += 1;
                } else {
                    let body2 = self.snakes[i].body[3..6].to_vec();
                    let body3 = self.snakes[i].body[6..9].to_vec();
                    self.snakes[i].body.truncate(3);
                    
                    let dir = self.snakes[i].direction;
                    let s2 = SnakeState::new_with_body(body2, dir);
                    let s3 = SnakeState::new_with_body(body3, dir);
                    new_snakes.push(s2);
                    new_snakes.push(s3);
                }
            }
        }
        
        if !new_snakes.is_empty() {
            let mut all_cell_changed = vec![true; new_snakes.len()];
            self.snakes.extend(new_snakes);
            self.cell_changed.append(&mut all_cell_changed);
        }
    }

    /// Respawns every prey currently marked dead at a random free cell. Callers
    /// invoke this after `step()` (the training simulation captures a dead
    /// prey's true pre-respawn terminal observation first; the visualizer just
    /// respawns immediately). Mirrors `respawn_dead()` for snakes.
    pub fn respawn_dead_preys(&mut self) {
        // Preys are now dynamically managed via reproduction in `step()`.
        // Dead preys stay in the inactive pool until revived.
        self.update_targets();
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
                self.preys[index].lifespan = rng.gen_range(200..=500);
                self.preys[index].steps_alive = 0;
                self.preys[index].just_revived = true;
                self.preys[index].death_by_reproduction = false;
                self.preys[index].grass_eaten = 0.0;
                break;
            }
        }
    }

    /// Torus-wrapped `(dx, dy)` from grid cell `from` to grid cell `to`, each
    /// component in `(-dim/2, dim/2]` — the shortest path across the wraparound
    /// map edge.
    pub fn torus_delta(&self, from: (i32, i32), to: (i32, i32)) -> (i32, i32) {
        let mut dx = to.0 - from.0;
        let mut dy = to.1 - from.1;
        if dx > self.grid_width / 2 { dx -= self.grid_width; }
        else if dx < -self.grid_width / 2 { dx += self.grid_width; }
        if dy > self.grid_height / 2 { dy -= self.grid_height; }
        else if dy < -self.grid_height / 2 { dy += self.grid_height; }
        (dx, dy)
    }

    fn torus_manhattan(&self, a: (i32, i32), b: (i32, i32)) -> i32 {
        let (dx, dy) = self.torus_delta(a, b);
        dx.abs() + dy.abs()
    }

    /// Refreshes each alive snake's `tracked_target`: a snake only smells prey
    /// within `SMELL_RANGE` torus-wrapped Manhattan cells of its head, so a
    /// target is dropped the tick it (or its prey) leaves that range, and a
    /// new one is acquired only from prey currently within range.
    pub fn update_targets(&mut self) {
        for s in 0..self.snakes.len() {
            if self.snakes[s].is_dead { continue; }
            let head = self.snakes[s].body[0];
            let mut target_idx = self.snakes[s].tracked_target;
            if let Some(idx) = target_idx {
                let drop = match self.preys.get(idx) {
                    None => true,
                    Some(p) => {
                        if p.is_dead {
                            true
                        } else {
                            let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                            self.torus_manhattan(head, p_grid) > SMELL_RANGE
                        }
                    }
                };
                if drop { target_idx = None; }
            }
            if target_idx.is_none() {
                let mut closest_dist = f32::MAX;
                for (i, p) in self.preys.iter().enumerate() {
                    if !p.is_dead {
                        let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                        if self.torus_manhattan(head, p_grid) > SMELL_RANGE { continue; }
                        let (dx, dy) = self.torus_delta(head, p_grid);
                        let d = ((dx * dx + dy * dy) as f32).sqrt();
                        if d < closest_dist {
                            closest_dist = d;
                            target_idx = Some(i);
                        }
                    }
                }
                self.snakes[s].tracked_target = target_idx;
            }
        }
    }

    /// Snake observation (`SNAKE_OBS_SIZE` floats):
    /// - `[0..64)`  — 8x8 grid in the snake's rotated frame (4 cells ahead,
    ///   3 behind, 4 right, 3 left). Cell encoding: prey `1.0`; wall/rock/own
    ///   body `-1.0`; **alive** enemy head `-0.8`; any enemy body cell
    ///   (including a dead snake's frozen corpse — still a solid obstacle in
    ///   `step()`'s collision check) `-0.5`; else passable terrain
    ///   `Species::Snake.speed_on(terrain) * 0.5`.
    /// - `[64]`/`[65]` — unit direction to the nearest prey the snake can
    ///   smell (forward / right components), zero when nothing is smelled.
    ///   A snake only smells prey within `SMELL_RANGE` torus-wrapped
    ///   Manhattan cells of its head (see `update_targets`).
    /// - `[66]` — distance to that prey normalized by `SMELL_RANGE` (`1.0`
    ///   when nothing is smelled).
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
        //
        // Dead snakes are NOT skipped here: a corpse's body stays on the grid
        // as a solid obstacle (the Bevy visualizer never respawns snakes; it
        // freezes them in place until the whole match ends), and the collision
        // check in `step()` doesn't exempt dead bodies either — so a corpse
        // that's invisible in this observation is a wall the snake can't see
        // and will walk straight into. Only the "-0.8 head" danger marker is
        // alive-only, since a dead snake can't trigger a head-to-head kill.
        let own_body: HashSet<(i32, i32)> = snake.body.iter().copied().collect();
        let mut enemy_bodies: HashSet<(i32, i32)> = HashSet::new();
        let mut enemy_heads: HashSet<(i32, i32)> = HashSet::new();
        for (j, s) in self.snakes.iter().enumerate() {
            if j == snake_index { continue; }
            if !s.is_dead {
                if let Some(&h) = s.body.first() {
                    enemy_heads.insert(h);
                }
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
                let cx_wrapped = cx.rem_euclid(self.grid_width);
                let cy_wrapped = cy.rem_euclid(self.grid_height);
                let cell = (cx_wrapped, cy_wrapped);
                let terrain = self.map.get_terrain(cx_wrapped, cy_wrapped);

                obs[idx] = if prey_cells.contains(&cell) {
                    1.0
                } else if terrain == Terrain::Rock || own_body.contains(&cell) {
                    -1.0
                } else if enemy_heads.contains(&cell) {
                    -0.8
                } else if enemy_bodies.contains(&cell) {
                    -0.5
                } else {
                    Species::Snake.speed_on(terrain) * 0.5
                };
                obs[69 + idx] = self.map.grass_health[(cy_wrapped * self.grid_width + cx_wrapped) as usize];
                idx += 1;
            }
        }

        // Unit direction + normalized distance to the nearest alive prey. A unit
        // vector keeps the heading signal strong at any range (the old
        // `dx / max_dim` encoding shrank to ~0.01 for nearby prey).
        let mut closest: Option<(i32, i32, f32)> = None;
        if let Some(idx) = snake.tracked_target {
            let p = &self.preys[idx];
            let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
            let (dx, dy) = self.torus_delta(head, p_grid);
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            closest = Some((dx, dy, d));
        }

        if let Some((dx, dy, dist)) = closest {
            let d = dist.max(1e-6);
            obs[64] = (dx * vec_straight.0 + dy * vec_straight.1) as f32 / d;
            obs[65] = (dx * vec_right.0 + dy * vec_right.1) as f32 / d;
            obs[66] = (dist / SMELL_RANGE as f32).min(1.0);
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
                let cx_wrapped = cx.rem_euclid(self.grid_width);
                let cy_wrapped = cy.rem_euclid(self.grid_height);
                let cell = (cx_wrapped, cy_wrapped);
                let terrain = self.map.get_terrain(cx_wrapped, cy_wrapped);

                obs[idx] = if terrain == Terrain::Rock {
                    -1.0
                } else if snake_heads.contains(&cell) {
                    -0.8
                } else if snake_bodies.contains(&cell) {
                    -0.5
                } else {
                    prey.species.speed_on(terrain) * 0.5
                };
                obs[67 + idx] = self.map.grass_health[(cy_wrapped * self.grid_width + cx_wrapped) as usize];
                idx += 1;
            }
        }

        // Unit direction + normalized distance to the nearest alive snake head.
        let mut closest: Option<(i32, i32, f32)> = None;
        for h in &snake_heads {
            let mut dx = h.0 - prey_grid_pos.0;
            let mut dy = h.1 - prey_grid_pos.1;
            
            if dx > self.grid_width / 2 { dx -= self.grid_width; }
            else if dx < -self.grid_width / 2 { dx += self.grid_width; }
            if dy > self.grid_height / 2 { dy -= self.grid_height; }
            else if dy < -self.grid_height / 2 { dy += self.grid_height; }
            
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The Bevy visualizer never calls `respawn_dead()`: a dead snake's body
    /// stays frozen on the grid as a corpse until the whole match ends. A
    /// living snake's observation must still mark that corpse as an obstacle
    /// (`step()`'s collision check doesn't exempt dead bodies either), or the
    /// snake walks straight into a wall it can't see. Regression test for a
    /// bug where the observation builder skipped dead snakes entirely.
    #[test]
    fn dead_snake_corpse_is_visible_as_obstacle() {
        let mut state = GameState::new(20, 20, 2, 0, 0);

        // Snake 1 (index 1) dies via a wall collision, leaving a body behind.
        state.snakes[1].body = vec![(5, 5), (5, 4), (5, 3)];
        state.snakes[1].is_dead = true;

        // Position snake 0 so the corpse falls inside its forward-facing patch.
        state.snakes[0].body = vec![(5, 8)];
        state.snakes[0].head_pos = (5.0, 8.0);
        state.snakes[0].direction = Direction::Down; // facing toward (5,5)

        let obs = state.get_relative_observation(0);

        // (5,5) is 3 cells straight ahead of (5,8) facing Down -> f=3, r=0 -> idx (3+3)*8+(0+3)=51
        let idx = ((3 + 3) * 8 + (0 + 3)) as usize;
        assert_eq!(obs[idx], -0.5, "dead snake's body cell must read as an obstacle, not open terrain");
    }

    #[test]
    fn prey_beyond_smell_range_is_not_sensed() {
        let mut state = GameState::new(100, 100, 1, 1, 0);
        state.snakes[0].body = vec![(50, 50)];
        state.snakes[0].head_pos = (50.0, 50.0);
        state.snakes[0].direction = Direction::Up;
        state.snakes[0].tracked_target = None;
        state.preys[0].pos = (50.0, 85.0); // Manhattan distance 35 > SMELL_RANGE (30)
        state.preys[0].is_dead = false;

        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, None, "prey beyond SMELL_RANGE must not be acquired");

        let obs = state.get_relative_observation(0);
        assert_eq!(obs[64], 0.0);
        assert_eq!(obs[65], 0.0);
        assert_eq!(obs[66], 1.0);
    }

    #[test]
    fn prey_within_smell_range_sets_direction() {
        let mut state = GameState::new(100, 100, 1, 1, 0);
        state.snakes[0].body = vec![(50, 50)];
        state.snakes[0].head_pos = (50.0, 50.0);
        state.snakes[0].direction = Direction::Up;
        state.snakes[0].tracked_target = None;
        state.preys[0].pos = (50.0, 60.0); // 10 cells straight ahead, within SMELL_RANGE
        state.preys[0].is_dead = false;

        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, Some(0));

        let obs = state.get_relative_observation(0);
        assert!((obs[64] - 1.0).abs() < 1e-5, "forward component should be ~1.0, got {}", obs[64]);
        assert!(obs[65].abs() < 1e-5, "right component should be ~0.0, got {}", obs[65]);
        assert!((obs[66] - 10.0 / 30.0).abs() < 1e-5, "distance should be 10/SMELL_RANGE, got {}", obs[66]);
    }

    #[test]
    fn target_dropped_when_prey_leaves_smell_range() {
        let mut state = GameState::new(100, 100, 1, 1, 0);
        state.snakes[0].body = vec![(50, 50)];
        state.snakes[0].head_pos = (50.0, 50.0);
        state.snakes[0].direction = Direction::Up;
        state.snakes[0].tracked_target = None;
        state.preys[0].pos = (50.0, 60.0);
        state.preys[0].is_dead = false;
        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, Some(0));

        state.preys[0].pos = (50.0, 81.0); // Manhattan distance 31 > SMELL_RANGE
        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, None, "target must be dropped once out of smell range");
    }

    #[test]
    fn smell_wraps_around_torus_edge() {
        let mut state = GameState::new(100, 100, 1, 1, 0);
        state.snakes[0].body = vec![(1, 50)];
        state.snakes[0].head_pos = (1.0, 50.0);
        state.snakes[0].direction = Direction::Up;
        state.snakes[0].tracked_target = None;
        state.preys[0].pos = (98.0, 50.0); // raw Manhattan 97, torus-wrapped 3
        state.preys[0].is_dead = false;

        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, Some(0), "prey must be sensed across the torus wrap");
    }
}
