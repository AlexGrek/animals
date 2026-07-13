use rand::Rng;
use std::collections::HashSet;
use crate::map::{Map, Terrain};
use crate::snake::SnakeState;
use crate::species::Species;
use crate::direction::Direction;
use crate::{HUNGER_DEATH_LIMIT, HUNGER_LIMIT, PREY_OBS_SIZE, SMELL_RANGE, SNAKE_OBS_SIZE, VISIT_HORIZON};

/// Cumulative `grass_eaten` at which a prey/amphibia reproduces (see the
/// trigger in `GameState::step`). Also used as the denominator for the
/// reproduction-progress observation scalar (`get_prey_observation`'s
/// `[67]`), so the two stay in lockstep by construction.
pub const PREY_REPRODUCTION_GRASS: f32 = 50.0;

#[derive(Clone, Debug)]
pub struct PreyState {
    pub pos: (f32, f32),
    pub is_dead: bool,
    pub species: Species,
    pub death_by_reproduction: bool,
    pub just_revived: bool,
    pub grass_eaten: f32,
    pub family_id: u32,
}

#[derive(Clone, Debug)]
pub struct EggState {
    pub pos: (i32, i32),
    pub ticks_alive: i32,
    pub is_dead: bool,
}

#[derive(Clone, Debug)]
pub struct CorpsefagState {
    pub pos: (f32, f32),
    pub is_dead: bool,
    pub points: i32,
    pub family_id: u32,
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
    pub auto_steer: bool,
    pub steps: u64,
    /// Grid cells occupied by dead snakes' bodies (the visualizer's ecosystem
    /// model). A dead snake's *entity* is removed from `snakes` by
    /// `remove_dead_snakes`, but its body cells linger here as a static obstacle
    /// that living snakes read as `-0.5` and die on collision. Stays empty in
    /// training (which respawns instead of reaping), so it costs nothing there.
    pub corpses: HashSet<(i32, i32)>,
    pub eggs: Vec<EggState>,
    pub corpsefags: Vec<CorpsefagState>,
    pub dead_snake_heads: Vec<(f32, f32)>,
    pub cf_births: Vec<(f32, f32)>,
    pub snake_births: Vec<(f32, f32)>,
    pub cf_eats: Vec<(f32, f32)>,
    pub egg_eats: Vec<(f32, f32)>,
}

impl GameState {
    pub fn new(width: i32, height: i32, num_snakes: usize, initial_preys: usize, max_preys: usize, initial_amphibias: usize, max_amphibias: usize, num_corpsefags: usize, is_training: bool, auto_steer: bool) -> Self {
        let map = Map::new(width, height);

        let mut preys = Vec::with_capacity(max_preys + max_amphibias);
        for i in 0..max_preys {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: i >= initial_preys, species: Species::Prey, death_by_reproduction: false, just_revived: false, grass_eaten: 0.0, family_id: i as u32 });
        }
        for i in 0..max_amphibias {
            preys.push(PreyState { pos: (0.0, 0.0), is_dead: i >= initial_amphibias, species: Species::Amphibia, death_by_reproduction: false, just_revived: false, grass_eaten: 0.0, family_id: i as u32 });
        }
        let total_preys = max_preys + max_amphibias;

        let mut state = Self {
            snakes: vec![SnakeState::new((0, 0), Direction::Up, 0); num_snakes],
            preys,
            grid_width: width,
            grid_height: height,
            map,
            game_over: false,
            cell_changed: vec![false; num_snakes],
            prey_died_this_tick: vec![false; total_preys],
            is_training,
            auto_steer,
            steps: 0,
            corpses: HashSet::new(),
            eggs: Vec::new(),
            corpsefags: Vec::with_capacity(num_corpsefags),
            dead_snake_heads: Vec::new(),
            cf_births: Vec::new(),
            snake_births: Vec::new(),
            cf_eats: Vec::new(),
            egg_eats: Vec::new(),
        };

        for i in 0..num_snakes {
            let (pos, direction) = state.spawn_position(i);
            state.snakes[i] = SnakeState::new(pos, direction, i as u32);
        }
        for i in 0..total_preys {
            if !state.preys[i].is_dead {
                state.spawn_prey(i);
            }
        }
        for _ in 0..num_corpsefags {
            state.spawn_corpsefag();
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
            let family_id = self.snakes[i].family_id;
            self.snakes[i] = SnakeState::new(pos, direction, family_id);
        }
        self.update_targets();
    }

    /// Reaps every dead snake: its *entity* leaves the world entirely (removed
    /// from `snakes`, so it's uncounted, undriven, and can't grow memory), but
    /// its body cells are left behind in `corpses` as a static obstacle. This
    /// is the visualizer's ecosystem model: snake population is governed purely
    /// by births (mitosis, `step`) and deaths (hunger/collision/corpse), so it
    /// self-balances against prey abundance rather than being pinned to a fixed
    /// count — letting the game cycle indefinitely.
    ///
    /// `cell_changed` (a per-snake parallel Vec) is resized to match. Snake
    /// indices shift when earlier snakes are removed; callers that track snakes
    /// by index across ticks (e.g. the Bevy renderer) rebuild their index-keyed
    /// state each frame, so this is safe there.
    pub fn remove_dead_snakes(&mut self) {
        for snake in self.snakes.iter().filter(|s| s.is_dead) {
            self.corpses.extend(snake.body.iter().copied());
        }
        self.snakes.retain(|s| !s.is_dead);
        // `cell_changed` is a per-snake parallel Vec that's only ever written
        // (never read across ticks), so just resize it to match rather than
        // filtering in lockstep — keeps the invariant `cell_changed.len() ==
        // snakes.len()` without depending on its stale contents.
        self.cell_changed.resize(self.snakes.len(), false);
        self.update_targets();
    }

    pub fn set_direction(&mut self, snake_index: usize, new_dir: Direction) {
        if let Some(snake) = self.snakes.get_mut(snake_index) {
            if snake.direction.opposite() != new_dir {
                snake.direction = new_dir;
            }
        }
    }

    /// Grid cell snake `i`'s head would land on next tick if it moved in `dir`,
    /// mirroring the head-position update in `step` (continuous move, toroidal
    /// wrap, then round to the nearest cell).
    fn predicted_head_cell(&self, i: usize, dir: Direction, dt: f32) -> (i32, i32) {
        let snake = &self.snakes[i];
        let head_before = snake.body[0];
        let terrain = self.map.get_terrain(head_before.0, head_before.1);
        let speed = Species::Snake.speed_on(terrain);
        let v = dir.to_vector();
        let hx = (snake.head_pos.0 + v.0 as f32 * speed * dt).rem_euclid(self.grid_width as f32);
        let hy = (snake.head_pos.1 + v.1 as f32 * speed * dt).rem_euclid(self.grid_height as f32);
        (hx.round() as i32, hy.round() as i32)
    }

    /// Whether moving snake `i` in `dir` avoids immediate death this tick, using
    /// the same rock/body rules as the collision pass in `step`. A move that
    /// doesn't change cells is safe (the engine skips collision on an unchanged
    /// cell). Head-to-head collisions with other *moving* snakes aren't predicted
    /// here — only static obstacles (rocks, snake bodies) are.
    fn snake_dir_is_safe(&self, i: usize, dir: Direction, dt: f32) -> bool {
        let head_before = self.snakes[i].body[0];
        let cell = self.predicted_head_cell(i, dir, dt);
        if cell == head_before {
            return true;
        }
        if self.map.get_terrain(cell.0, cell.1) == Terrain::Rock {
            return false;
        }
        // Mirror the engine's body check: it tests every snake's current body
        // (self, others, and not-yet-cleared corpses) before any tail pop.
        for s in &self.snakes {
            if s.body.contains(&cell) {
                return false;
            }
        }
        if self.corpses.contains(&cell) {
            return false;
        }
        true
    }

    pub fn step(&mut self, dt: f32, prey_actions: &[usize], corpsefag_actions: &[usize]) {
        if self.game_over {
            return;
        }

        self.steps += 1;
        self.update_targets();

        for p in &mut self.prey_died_this_tick {
            *p = false;
        }
        self.dead_snake_heads.clear();
        self.cf_births.clear();
        self.snake_births.clear();
        self.cf_eats.clear();
        self.egg_eats.clear();
        
        let was_dead_snakes: Vec<bool> = self.snakes.iter().map(|s| s.is_dead).collect();
        for p in &mut self.preys {
            p.death_by_reproduction = false;
            p.just_revived = false;
        }

        let mut preys_to_revive = Vec::new();

        // Tick eggs
        let mut hatched_positions = Vec::new();
        for egg in &mut self.eggs {
            if egg.is_dead { continue; }
            egg.ticks_alive += 1;
            if egg.ticks_alive >= 200 {
                egg.is_dead = true; // hatched
                hatched_positions.push(egg.pos);
            }
        }
        for pos in hatched_positions {
            if let Some(idx) = self.corpsefags.iter().position(|c| c.is_dead) {
                self.corpsefags[idx].pos = (pos.0 as f32, pos.1 as f32);
                self.corpsefags[idx].is_dead = false;
                self.corpsefags[idx].points = 0;
            } else {
                let next_idx = self.corpsefags.len();
                self.corpsefags.push(CorpsefagState {
                    pos: (pos.0 as f32, pos.1 as f32),
                    is_dead: false,
                    points: 0,
                    family_id: (next_idx % 2) as u32,
                });
            }
        }

        // Increment hunger and check for hunger death
        for i in 0..self.snakes.len() {
            if !self.snakes[i].is_dead {
                self.snakes[i].steps_since_last_eat += 1;
                if self.snakes[i].steps_since_last_eat >= HUNGER_DEATH_LIMIT {
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

        // Count currently alive before processing preys
        let mut alive_preys = 0;
        let mut alive_amphibias = 0;
        for p in &self.preys {
            if !p.is_dead {
                if p.species == Species::Prey { alive_preys += 1; }
                else { alive_amphibias += 1; }
            }
        }

        // 1. Move preys
        for i in 0..self.preys.len() {
            if self.preys[i].is_dead { continue; }

            if self.preys[i].grass_eaten >= PREY_REPRODUCTION_GRASS {
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
                    let is_under_cap = if self.preys[i].species == Species::Prey {
                        alive_preys < 200
                    } else {
                        alive_amphibias < 200
                    };

                    if is_under_cap {
                        self.preys[i].grass_eaten = 0.0;
                        self.preys[i].death_by_reproduction = true;
                        self.prey_died_this_tick[i] = true;
                        preys_to_revive.push((self.preys[i].species, self.preys[i].pos, Some(self.preys[i].family_id)));
                        if self.preys[i].species == Species::Prey {
                            alive_preys += 1;
                        } else {
                            alive_amphibias += 1;
                        }
                    }
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

        if alive_preys < 5 {
            preys_to_revive.push((Species::Prey, (-1.0, -1.0), None));
        }
        if alive_amphibias < 5 {
            preys_to_revive.push((Species::Amphibia, (-1.0, -1.0), None));
        }

        let mut rng = rand::thread_rng();
        for (species, pos, family_id_opt) in preys_to_revive {
            if let Some(idx) = self.preys.iter().position(|p| p.is_dead && p.species == species) {
                if pos.0 < 0.0 {
                    self.spawn_prey(idx);
                    self.preys[idx].family_id = family_id_opt.unwrap_or(idx as u32);
                } else {
                    let mut px = pos.0 + rng.gen_range(-1.0..=1.0);
                    let mut py = pos.1 + rng.gen_range(-1.0..=1.0);
                    px = px.rem_euclid(self.grid_width as f32);
                    py = py.rem_euclid(self.grid_height as f32);
                    self.preys[idx].pos = (px, py);
                    self.preys[idx].is_dead = false;
                    self.preys[idx].just_revived = true;
                    self.preys[idx].death_by_reproduction = false;
                    self.preys[idx].grass_eaten = 0.0;
                    self.preys[idx].family_id = family_id_opt.unwrap_or(idx as u32);
                }
            } else {
                let family_id = family_id_opt.unwrap_or(self.preys.len() as u32);
                let new_prey = PreyState {
                    pos: (0.0, 0.0),
                    is_dead: true,
                    species,
                    death_by_reproduction: false,
                    just_revived: false,
                    grass_eaten: 0.0,
                    family_id,
                };
                self.preys.push(new_prey);
                let idx = self.preys.len() - 1;
                self.prey_died_this_tick.push(false);

                if pos.0 < 0.0 {
                    self.spawn_prey(idx);
                } else {
                    let mut px = pos.0 + rng.gen_range(-1.0..=1.0);
                    let mut py = pos.1 + rng.gen_range(-1.0..=1.0);
                    px = px.rem_euclid(self.grid_width as f32);
                    py = py.rem_euclid(self.grid_height as f32);
                    self.preys[idx].pos = (px, py);
                    self.preys[idx].is_dead = false;
                    self.preys[idx].just_revived = true;
                    self.preys[idx].death_by_reproduction = false;
                    self.preys[idx].grass_eaten = 0.0;
                }
            }
        }

        // 1.5 Move Corpsefags
        let mut new_eggs = Vec::new();
        for i in 0..self.corpsefags.len() {
            if self.corpsefags[i].is_dead { continue; }

            let c_action = corpsefag_actions.get(i).copied().unwrap_or(0);
            let dir_vec = match c_action {
                1 => (0, 1),   // Up
                2 => (1, 0),   // Right
                3 => (0, -1),  // Down
                4 => (-1, 0),  // Left
                _ => (0, 0),   // Stand
            };

            if dir_vec != (0, 0) {
                let prev_pos = self.corpsefags[i].pos;
                let px_before = prev_pos.0.round() as i32;
                let py_before = prev_pos.1.round() as i32;
                let terrain = self.map.get_terrain(px_before, py_before);
                let speed = Species::Corpsefag.speed_on(terrain);

                self.corpsefags[i].pos.0 += dir_vec.0 as f32 * speed * dt;
                self.corpsefags[i].pos.1 += dir_vec.1 as f32 * speed * dt;
                self.corpsefags[i].pos.0 = self.corpsefags[i].pos.0.rem_euclid(self.grid_width as f32);
                self.corpsefags[i].pos.1 = self.corpsefags[i].pos.1.rem_euclid(self.grid_height as f32);

                let px_after = self.corpsefags[i].pos.0.round() as i32;
                let py_after = self.corpsefags[i].pos.1.round() as i32;
                let terrain_after = self.map.get_terrain(px_after, py_after);
                
                let grid_pos_after = (px_after, py_after);
                let mut blocked = terrain_after == Terrain::Rock || terrain_after == Terrain::Water;
                if !blocked {
                    for p in &self.preys {
                        if !p.is_dead && (p.pos.0.round() as i32, p.pos.1.round() as i32) == grid_pos_after {
                            blocked = true;
                            break;
                        }
                    }
                }
                if !blocked {
                    for (j, cf) in self.corpsefags.iter().enumerate() {
                        if i != j && !cf.is_dead && (cf.pos.0.round() as i32, cf.pos.1.round() as i32) == grid_pos_after {
                            blocked = true;
                            break;
                        }
                    }
                }

                if blocked {
                    self.corpsefags[i].pos = prev_pos;
                }
            }

            // Corpsefag eating corpses
            let cx = self.corpsefags[i].pos.0.round() as i32;
            let cy = self.corpsefags[i].pos.1.round() as i32;
            let mut eaten = false;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    let tx = (cx + dx).rem_euclid(self.grid_width);
                    let ty = (cy + dy).rem_euclid(self.grid_height);
                    if self.corpses.remove(&(tx, ty)) {
                        self.cf_eats.push((tx as f32, ty as f32));
                        eaten = true;
                    }
                }
            }
            if eaten {
                self.corpsefags[i].points += 1;
            }

            // Laying egg
            if self.corpsefags[i].points >= 3 {
                self.corpsefags[i].points -= 3;
                let c_pos = self.corpsefags[i].pos;
                self.cf_births.push(c_pos);
                new_eggs.push(EggState {
                    pos: (cx, cy),
                    ticks_alive: 0,
                    is_dead: false,
                });
            }
        }
        self.eggs.extend(new_eggs);

        // Anti-suicide steering: if the current direction runs the head into a
        // rock or a body, turn to a safe side instead of driving into it. Gated
        // drives in training, so exposing it only at play time produces
        // out-of-distribution orbiting behavior around obstacles, but the user requested it.
        // It's on for both manual play and AI play.
        if !self.is_training {
            for i in 0..self.snakes.len() {
                if self.snakes[i].is_dead || self.snakes[i].body.is_empty() {
                    continue;
                }
                let dir = self.snakes[i].direction;
                if self.snake_dir_is_safe(i, dir, dt) {
                    continue;
                }
                // Both alternatives are 90° turns (a 180° reversal isn't a legal
                // single move). Try right then left; keep the original heading if
                // the snake is boxed in on all sides (it dies as before).
                for cand in [dir.turn_right(), dir.turn_left()] {
                    if self.snake_dir_is_safe(i, cand, dt) {
                        self.snakes[i].direction = cand;
                        break;
                    }
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

            // Corpse collision: a reaped dead snake's leftover body is still a
            // solid obstacle (visualizer ecosystem model). No kill is credited —
            // its owner is already gone.
            if !self.snakes[i].is_dead && self.corpses.contains(&head) {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_opponent = true;
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
                for e_idx in 0..self.eggs.len() {
                    if !self.eggs[e_idx].is_dead {
                        let dx = (head.0 - self.eggs[e_idx].pos.0).abs();
                        let dy = (head.1 - self.eggs[e_idx].pos.1).abs();
                        if dx <= 1 && dy <= 1 {
                            self.snakes[i].score += 1;
                            self.snakes[i].steps_since_last_eat = 0;
                            self.eggs[e_idx].is_dead = true;
                            self.egg_eats.push((self.eggs[e_idx].pos.0 as f32, self.eggs[e_idx].pos.1 as f32));
                            ate = true;
                            break;
                        }
                    }
                }
            }

            if !ate {
                for c_idx in 0..self.corpsefags.len() {
                    if !self.corpsefags[c_idx].is_dead {
                        let c_grid_pos = (self.corpsefags[c_idx].pos.0.round() as i32, self.corpsefags[c_idx].pos.1.round() as i32);
                        let dx = (head.0 - c_grid_pos.0).abs();
                        let dy = (head.1 - c_grid_pos.1).abs();
                        if dx <= 1 && dy <= 1 {
                            self.snakes[i].score += 1;
                            self.snakes[i].steps_since_last_eat = 0;
                            self.corpsefags[c_idx].is_dead = true;
                            ate = true;
                            break;
                        }
                    }
                }
            }

            if !ate {
                self.snakes[i].body.pop();
            }
        }

        // Track coarse (4x4-tile) cell visitation: drives both the
        // exploration reward (`animals_simulation`) and the visitation
        // observation channel (`get_relative_observation`). Recomputed every
        // tick from the current head cell (not just on cell-change ticks) so a
        // snake sitting still keeps re-marking its own cell as visited.
        for i in 0..self.snakes.len() {
            if self.snakes[i].is_dead { continue; }
            let head = self.snakes[i].body[0];
            let coarse = (head.0 / 4, head.1 / 4);
            let last_visit = self.snakes[i].visited.get(&coarse).copied();
            self.snakes[i].entered_new_patch = match last_visit {
                None => true,
                Some(t) => self.steps.saturating_sub(t) > VISIT_HORIZON,
            };
            self.snakes[i].visited.insert(coarse, self.steps);
        }

        // Snake Mitosis Check
        let mut new_snakes = Vec::new();
        let num_snakes = self.snakes.len();
        for i in 0..num_snakes {
            if self.snakes[i].body.len() >= 12 {
                if let Some(&head) = self.snakes[i].body.get(0) {
                    self.snake_births.push((head.0 as f32, head.1 as f32));
                }
                if self.is_training {
                    self.snakes[i].body.truncate(3);
                    self.snakes[i].mitosis_count += 1;
                } else {
                    let body2 = self.snakes[i].body[3..6].to_vec();
                    let body3 = self.snakes[i].body[6..9].to_vec();
                    let body4 = self.snakes[i].body[9..12].to_vec();
                    self.snakes[i].body.truncate(3);

                    let dir = self.snakes[i].direction;
                    let family_id = self.snakes[i].family_id;
                    let s2 = SnakeState::new_with_body(body2, dir, family_id);
                    let s3 = SnakeState::new_with_body(body3, dir, family_id);
                    let s4 = SnakeState::new_with_body(body4, dir, family_id);
                    new_snakes.push(s2);
                    new_snakes.push(s3);
                    new_snakes.push(s4);
                }
            }
        }

        if !new_snakes.is_empty() {
            let mut all_cell_changed = vec![true; new_snakes.len()];
            self.snakes.extend(new_snakes);
            self.cell_changed.append(&mut all_cell_changed);
        }

        // Collect newly dead snakes for particles
        for i in 0..was_dead_snakes.len() {
            if self.snakes[i].is_dead && !was_dead_snakes[i] {
                if let Some(&head) = self.snakes[i].body.get(0) {
                    self.dead_snake_heads.push((head.0 as f32, head.1 as f32));
                }
            }
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
            // Spawn-camping guard: don't drop a prey inside a live snake's
            // kill box. The population-floor path can otherwise place a
            // fresh prey adjacent to a snake head, where it dies before it
            // can act. Mirrors the reproduction "no snake nearby" gate above.
            if free {
                for s in &self.snakes {
                    if s.is_dead || s.body.is_empty() { continue; }
                    let head = s.body[0];
                    let (dx, dy) = self.torus_delta(head, pos);
                    if dx.abs() <= 8 && dy.abs() <= 8 {
                        free = false;
                        break;
                    }
                }
            }
            if free {
                self.preys[index].pos = (x as f32, y as f32);
                self.preys[index].is_dead = false;
                self.preys[index].just_revived = true;
                self.preys[index].death_by_reproduction = false;
                self.preys[index].grass_eaten = 0.0;
                break;
            }
        }
    }

    pub fn spawn_corpsefag(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x as f32, y as f32);

            let terrain = self.map.get_terrain(x, y);
            if terrain == Terrain::Rock || terrain == Terrain::Water {
                continue;
            }

            let grid_pos = (x, y);
            let mut free = true;
            for s in &self.snakes {
                if s.body.contains(&grid_pos) {
                    free = false;
                    break;
                }
            }
            for p in &self.preys {
                if !p.is_dead && (p.pos.0.round() as i32, p.pos.1.round() as i32) == grid_pos {
                    free = false;
                    break;
                }
            }
            for cf in &self.corpsefags {
                if !cf.is_dead && (cf.pos.0.round() as i32, cf.pos.1.round() as i32) == grid_pos {
                    free = false;
                    break;
                }
            }
            
            if free {
                let next_idx = self.corpsefags.len();
                self.corpsefags.push(CorpsefagState {
                    pos,
                    is_dead: false,
                    points: 0,
                    family_id: (next_idx % 2) as u32,
                });
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
            let mut target_pos = self.snakes[s].tracked_target;
            if let Some(pos) = target_pos {
                // Drop if it moved too far or no longer exists (eaten).
                // It's eaten if neither preys nor eggs have a living entity at this pos.
                let mut exists = false;
                for p in &self.preys {
                    if !p.is_dead && (p.pos.0.round() as i32, p.pos.1.round() as i32) == pos {
                        exists = true;
                        break;
                    }
                }
                if !exists {
                    for e in &self.eggs {
                        if !e.is_dead && e.pos == pos {
                            exists = true;
                            break;
                        }
                    }
                }
                
                if !exists || self.torus_manhattan(head, pos) > SMELL_RANGE {
                    target_pos = None;
                }
            }
            if target_pos.is_none() {
                let mut closest_dist = f32::MAX;
                let mut closest_pos = None;
                for p in &self.preys {
                    if !p.is_dead {
                        let p_grid = (p.pos.0.round() as i32, p.pos.1.round() as i32);
                        if self.torus_manhattan(head, p_grid) > SMELL_RANGE { continue; }
                        let (dx, dy) = self.torus_delta(head, p_grid);
                        let d = ((dx * dx + dy * dy) as f32).sqrt();
                        if d < closest_dist {
                            closest_dist = d;
                            closest_pos = Some(p_grid);
                        }
                    }
                }
                for e in &self.eggs {
                    if !e.is_dead {
                        if self.torus_manhattan(head, e.pos) > SMELL_RANGE { continue; }
                        let (dx, dy) = self.torus_delta(head, e.pos);
                        let d = ((dx * dx + dy * dy) as f32).sqrt();
                        if d < closest_dist {
                            closest_dist = d;
                            closest_pos = Some(e.pos);
                        }
                    }
                }
                target_pos = closest_pos;
            }
            self.snakes[s].tracked_target = target_pos;
        }
    }

    /// Snake observation (`SNAKE_OBS_SIZE` floats):
    /// - `[0..64)`  — 8x8 grid in the snake's rotated frame (4 cells ahead,
    ///   3 behind, 4 right, 3 left). Cell encoding: prey `1.0`; wall/rock/own
    ///   body `-1.0`; **alive** enemy head `-0.8`; any enemy body cell, or a
    ///   `corpses` cell (a reaped dead snake's leftover body — still a solid
    ///   obstacle in `step()`'s collision check), `-0.5`; else passable terrain
    ///   `Species::Snake.speed_on(terrain) * 0.5`.
    /// - `[64]`/`[65]` — unit direction to the nearest prey the snake can
    ///   smell (forward / right components), zero when nothing is smelled.
    ///   A snake only smells prey within `SMELL_RANGE` torus-wrapped
    ///   Manhattan cells of its head (see `update_targets`).
    /// - `[66]` — distance to that prey normalized by `SMELL_RANGE` (`1.0`
    ///   when nothing is smelled).
    /// - `[67]` — hunger: `steps_since_last_eat / HUNGER_LIMIT`.
    /// - `[68]` — own length / 100, capped at 1.
    /// - `[69..133)` — grass-health of each of the same 64 fine-grid cells.
    /// - `[133..197)` — 8x8 coarse visitation-recency grid, same rotated frame
    ///   and cell order as `[0..64)` but each cell spans an 8x8-tile block (a
    ///   2x2 group of the 4x4 coarse patches tracked in `SnakeState::visited`).
    ///   Value is `1.0` for a patch just entered, decaying linearly to `0.0`
    ///   over `VISIT_HORIZON` ticks (or if never visited) — the freshest
    ///   (max) recency among the patches the cell covers. This externalizes
    ///   the exploration reward's memory (see `animals_simulation`'s reward
    ///   function) so a memoryless policy can act on "where have I been".
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
        for e in &self.eggs {
            if !e.is_dead {
                prey_cells.insert(e.pos);
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
                } else if enemy_bodies.contains(&cell) || self.corpses.contains(&cell) {
                    // A corpse (dead snake's leftover body) is a solid obstacle
                    // just like a living snake's body — same -0.5 the model saw
                    // when dead snakes still lived in the `snakes` Vec.
                    -0.5
                } else {
                    Species::Snake.speed_on(terrain) * 0.5
                };
                obs[69 + idx] = self.map.grass_health[(cy_wrapped * self.grid_width + cx_wrapped) as usize];
                idx += 1;
            }
        }

        // Unit direction + normalized distance to the nearest alive prey/egg.
        let mut closest: Option<(i32, i32, f32)> = None;
        if let Some(p_grid) = snake.tracked_target {
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

        // Coarse (8x8-tile) visitation-recency grid at [133..197), same
        // rotated frame/order as the fine grid. Each coarse observation cell
        // covers a 2x2 block of the 4x4 patches keyed in `snake.visited`.
        let mut vidx = 133;
        for f in -3..=4 {
            for r in -3..=4 {
                let mut freshest: f32 = 0.0;
                for df in [0, 4] {
                    for dr in [0, 4] {
                        let cx = head.0 + (f * 8 + df) * vec_straight.0 + (r * 8 + dr) * vec_right.0;
                        let cy = head.1 + (f * 8 + df) * vec_straight.1 + (r * 8 + dr) * vec_right.1;
                        let patch = (
                            cx.rem_euclid(self.grid_width) / 4,
                            cy.rem_euclid(self.grid_height) / 4,
                        );
                        if let Some(&last_visit) = snake.visited.get(&patch) {
                            let age = self.steps.saturating_sub(last_visit) as f32;
                            let recency = (1.0 - age / VISIT_HORIZON as f32).clamp(0.0, 1.0);
                            if recency > freshest {
                                freshest = recency;
                            }
                        }
                    }
                }
                obs[vidx] = freshest;
                vidx += 1;
            }
        }

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
    /// - `[66]` — distance to that head, normalized over a 0-150 cell band
    ///   (`1.0` when no snake is alive or farther than 150 cells away) so
    ///   distant threats stay distinguishable instead of all saturating to
    ///   the same "far away" reading.
    /// - `[67]` — reproduction progress: `grass_eaten / PREY_REPRODUCTION_GRASS`,
    ///   clamped to `1.0`. Makes the reproduction goal (see `step`) directly
    ///   observable instead of latent state the policy has to infer blind.
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
                obs[68 + idx] = self.map.grass_health[(cy_wrapped * self.grid_width + cx_wrapped) as usize];
                idx += 1;
            }
        }

        // Unit direction (torus-wrapped shortest path) + normalized distance to
        // the nearest alive snake head.
        let mut closest: Option<(i32, i32, f32)> = None;
        for h in &snake_heads {
            let (dx, dy) = self.torus_delta(prey_grid_pos, *h);
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if closest.map_or(true, |(_, _, cd)| d < cd) {
                closest = Some((dx, dy, d));
            }
        }

        // Distance normalized over a 0-150 cell band, giving full resolution
        // close-in without every snake beyond a short radius saturating to
        // the same "far away" 1.0 reading.
        if let Some((dx, dy, dist)) = closest {
            let d = dist.max(1e-6);
            obs[64] = dx as f32 / d;
            obs[65] = dy as f32 / d;
            obs[66] = (dist / 150.0).min(1.0);
        } else {
            obs[66] = 1.0;
        }

        // Reproduction progress: how close this prey is to the grass-eaten
        // threshold that triggers reproduction (see `PREY_REPRODUCTION_GRASS`
        // and the trigger in `step`).
        obs[67] = (prey.grass_eaten / PREY_REPRODUCTION_GRASS).min(1.0);

        obs
    }

    pub fn get_corpsefag_observation(&self, index: usize) -> [f32; 18] {
        let mut obs = [0.0; 18];
        let cf = &self.corpsefags[index];
        let pos = (cf.pos.0.round() as i32, cf.pos.1.round() as i32);

        let mut obstacles: HashSet<(i32, i32)> = HashSet::new();
        for s in &self.snakes {
            if !s.is_dead {
                obstacles.extend(s.body.iter().copied());
            }
        }
        for p in &self.preys {
            if !p.is_dead {
                obstacles.insert((p.pos.0.round() as i32, p.pos.1.round() as i32));
            }
        }
        for (i, c) in self.corpsefags.iter().enumerate() {
            if i != index && !c.is_dead {
                obstacles.insert((c.pos.0.round() as i32, c.pos.1.round() as i32));
            }
        }

        let mut idx = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = pos.0 + dx;
                let cy = pos.1 + dy;
                let cx_wrapped = cx.rem_euclid(self.grid_width);
                let cy_wrapped = cy.rem_euclid(self.grid_height);
                let cell = (cx_wrapped, cy_wrapped);
                let terrain = self.map.get_terrain(cx_wrapped, cy_wrapped);

                obs[idx] = if terrain == Terrain::Rock || terrain == Terrain::Water || obstacles.contains(&cell) {
                    -1.0
                } else {
                    Species::Corpsefag.speed_on(terrain) * 0.5
                };
                idx += 1;
            }
        }

        let mut ray_distances = [f32::MAX; 8];
        for &corpse in &self.corpses {
            let (dx, dy) = self.torus_delta(pos, corpse);
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d <= 133.0 {
                let mut angle = (dy as f32).atan2(dx as f32);
                if angle < 0.0 {
                    angle += std::f32::consts::TAU;
                }
                let sector = ((angle + std::f32::consts::TAU / 16.0) / (std::f32::consts::TAU / 8.0)).floor() as usize % 8;
                if d < ray_distances[sector] {
                    ray_distances[sector] = d;
                }
            }
        }

        for i in 0..8 {
            if ray_distances[i] <= 133.0 {
                obs[9 + i] = 1.0 - (ray_distances[i] / 133.0);
            } else {
                obs[9 + i] = 0.0;
            }
        }

        obs[17] = (cf.points as f32 / 3.0).min(1.0);
        
        obs
    }

    pub fn spawn_corpses(&mut self, num: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..num {
            let mut x;
            let mut y;
            loop {
                x = rng.gen_range(0..self.grid_width);
                y = rng.gen_range(0..self.grid_height);
                let terrain = self.map.get_terrain(x, y);
                if terrain != Terrain::Rock && terrain != Terrain::Water {
                    break;
                }
            }
            self.corpses.insert((x, y));
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// `remove_dead_snakes` drops exactly the dead snakes' *entities* (keeping
    /// the living ones in order, `cell_changed` length-matched), while leaving
    /// their body cells behind in `corpses`. This is the ecosystem primitive:
    /// population shrinks on death (no respawn) so it self-balances against
    /// births (mitosis), and the reaped body stays as a static obstacle.
    #[test]
    fn remove_dead_snakes_drops_only_the_dead() {
        // 100x100 (not 20x20): a small map's random terrain can leave no free
        // spawn cell, hanging `GameState::new`'s spawn loop — a pre-existing
        // flakiness the larger smell tests avoid.
        let mut state = GameState::new(100, 100, 3, 0, 0, 0, 0, 0, false, false);
        assert_eq!(state.snakes.len(), 3);
        assert_eq!(state.cell_changed.len(), 3);

        // Kill the middle snake; tag the survivors so we can check ordering.
        state.snakes[0].score = 10;
        state.snakes[1].body = vec![(7, 7), (7, 6), (7, 5)];
        state.snakes[1].is_dead = true;
        state.snakes[2].score = 20;

        state.remove_dead_snakes();

        assert_eq!(state.snakes.len(), 2, "one dead snake should be removed");
        assert_eq!(state.cell_changed.len(), 2, "cell_changed must stay length-matched");
        assert!(state.snakes.iter().all(|s| !s.is_dead), "no dead snakes may remain");
        assert_eq!(state.snakes[0].score, 10, "surviving snakes keep their order");
        assert_eq!(state.snakes[1].score, 20);
        // The reaped snake's body cells become corpse obstacles.
        for cell in [(7, 7), (7, 6), (7, 5)] {
            assert!(state.corpses.contains(&cell), "reaped body cell {cell:?} must become a corpse");
        }
    }

    /// A corpse cell (dead snake reaped out of the `snakes` Vec, body left in
    /// `corpses`) must still read as an obstacle (`-0.5`) in a living snake's
    /// observation — the same value a live enemy body reads as, so the model's
    /// input distribution is unchanged.
    #[test]
    fn corpse_cell_is_visible_as_obstacle() {
        let mut state = GameState::new(100, 100, 1, 0, 0, 0, 0, 0, false, false);
        // Force all-grass terrain so the tested cell is deterministically not a
        // rock (rock takes precedence over the corpse marker in the obs).
        state.map.tiles = vec![Terrain::Grass; (state.grid_width * state.grid_height) as usize];

        // A corpse sits 3 cells straight ahead of snake 0.
        state.snakes[0].body = vec![(5, 8)];
        state.snakes[0].head_pos = (5.0, 8.0);
        state.snakes[0].direction = Direction::Down; // facing toward (5,5)
        state.corpses.insert((5, 5));

        let obs = state.get_relative_observation(0);

        // (5,5) is 3 ahead of (5,8) facing Down -> f=3, r=0 -> idx (3+3)*8+(0+3)=51
        let idx = ((3 + 3) * 8 + (0 + 3)) as usize;
        assert_eq!(obs[idx], -0.5, "a corpse cell must read as an obstacle in the observation");
    }

    /// The Bevy visualizer never calls `respawn_dead()`: a dead snake's body
    /// stays frozen on the grid as a corpse until the whole match ends. A
    /// living snake's observation must still mark that corpse as an obstacle
    /// (`step()`'s collision check doesn't exempt dead bodies either), or the
    /// snake walks straight into a wall it can't see. Regression test for a
    /// bug where the observation builder skipped dead snakes entirely.
    #[test]
    fn dead_snake_corpse_is_visible_as_obstacle() {
        let mut state = GameState::new(20, 20, 2, 0, 0, 0, 0, 0, true, false);

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
        let mut state = GameState::new(100, 100, 1, 1, 1, 0, 0, true, false);
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
        let mut state = GameState::new(100, 100, 1, 1, 1, 0, 0, true, false);
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
        let mut state = GameState::new(100, 100, 1, 1, 1, 0, 0, true, false);
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
        let mut state = GameState::new(100, 100, 1, 1, 1, 0, 0, true, false);
        state.snakes[0].body = vec![(1, 50)];
        state.snakes[0].head_pos = (1.0, 50.0);
        state.snakes[0].direction = Direction::Up;
        state.snakes[0].tracked_target = None;
        state.preys[0].pos = (98.0, 50.0); // raw Manhattan 97, torus-wrapped 3
        state.preys[0].is_dead = false;

        state.update_targets();
        assert_eq!(state.snakes[0].tracked_target, Some(0), "prey must be sensed across the torus wrap");
    }

    pub fn spawn_corpses(&mut self, num: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..num {
            let mut x;
            let mut y;
            loop {
                x = rng.gen_range(0..self.grid_width);
                y = rng.gen_range(0..self.grid_height);
                let terrain = self.map.get_terrain(x, y);
                if terrain != Terrain::Rock && terrain != Terrain::Water {
                    break;
                }
            }
            self.corpses.push((x, y));
        }
    }

}
