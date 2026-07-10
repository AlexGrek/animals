use rand::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn opposite(&self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    pub fn to_vector(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, 1),
            Direction::Down => (0, -1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    pub fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }

    pub fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelativeAction {
    Straight = 0,
    TurnRight = 1,
    TurnLeft = 2,
}

impl RelativeAction {
    pub fn from_usize(val: usize) -> Self {
        match val {
            1 => RelativeAction::TurnRight,
            2 => RelativeAction::TurnLeft,
            _ => RelativeAction::Straight,
        }
    }

    pub fn to_absolute_direction(&self, current: Direction) -> Direction {
        match self {
            RelativeAction::Straight => current,
            RelativeAction::TurnRight => current.turn_right(),
            RelativeAction::TurnLeft => current.turn_left(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SnakeState {
    pub body: Vec<(i32, i32)>,
    pub direction: Direction,
    pub is_dead: bool,
    pub score: u32,
    pub kills: u32,
    pub death_by_wall: bool,
    pub death_by_self: bool,
    pub death_by_opponent: bool,
}

impl SnakeState {
    pub fn new(start_pos: (i32, i32), direction: Direction) -> Self {
        Self {
            body: vec![start_pos],
            direction,
            is_dead: false,
            score: 0,
            kills: 0,
            death_by_wall: false,
            death_by_self: false,
            death_by_opponent: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub snakes: Vec<SnakeState>,
    pub apple_pos: (i32, i32),
    pub grid_width: i32,
    pub grid_height: i32,
    pub game_over: bool,
}

impl GameState {
    pub fn new(width: i32, height: i32, num_snakes: usize) -> Self {
        let mut snakes = Vec::new();

        let mut state = Self {
            snakes: Vec::new(),
            apple_pos: (0, 0),
            grid_width: width,
            grid_height: height,
            game_over: false,
        };

        for i in 0..num_snakes {
            let (pos, direction) = state.initial_spawn(i, num_snakes);
            snakes.push(SnakeState::new(pos, direction));
        }
        state.snakes = snakes;
        state.spawn_apple();
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
        if pos == self.apple_pos {
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

    pub fn step(&mut self) {
        if self.game_over {
            return;
        }

        let mut new_heads = Vec::new();
        for snake in &self.snakes {
            if snake.is_dead {
                new_heads.push(snake.body[0]);
                continue;
            }
            let head = snake.body[0];
            let new_head = match snake.direction {
                Direction::Up => (head.0, head.1 + 1),
                Direction::Down => (head.0, head.1 - 1),
                Direction::Left => (head.0 - 1, head.1),
                Direction::Right => (head.0 + 1, head.1),
            };
            new_heads.push(new_head);
        }

        // Snapshot pre-step alive status and precompute head-to-head collision
        // pairs against that snapshot, so that when two snakes collide head-on
        // BOTH die (previously, marking the lower-indexed snake dead first made
        // the `!is_dead` check skip the higher-indexed snake's own check).
        let was_alive: Vec<bool> = self.snakes.iter().map(|s| !s.is_dead).collect();
        let mut head_to_head = vec![false; self.snakes.len()];
        for i in 0..self.snakes.len() {
            if !was_alive[i] { continue; }
            for j in (i + 1)..self.snakes.len() {
                if !was_alive[j] { continue; }
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

            // Wall collision
            if head.0 < 0 || head.0 >= self.grid_width || head.1 < 0 || head.1 >= self.grid_height {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_wall = true;
                continue;
            }

            // Head-to-head collision
            if head_to_head[i] {
                self.snakes[i].is_dead = true;
                self.snakes[i].death_by_opponent = true;
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

        // Move and eat apples
        for i in 0..self.snakes.len() {
            if self.snakes[i].is_dead { continue; }
            let head = new_heads[i];
            self.snakes[i].body.insert(0, head);

            if head == self.apple_pos {
                self.snakes[i].score += 1;
                self.spawn_apple();
            } else {
                self.snakes[i].body.pop();
            }
        }
    }

    fn spawn_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x, y);
            let mut free = true;
            for s in &self.snakes {
                if s.body.contains(&pos) {
                    free = false;
                    break;
                }
            }
            if free {
                self.apple_pos = pos;
                break;
            }
        }
    }

    pub fn get_relative_observation(&self, snake_index: usize) -> [f32; 66] {
        let mut obs = [0.0; 66];
        let snake = &self.snakes[snake_index];
        if snake.body.is_empty() { return obs; }

        let head = snake.body[0];
        let dir = snake.direction;
        let vec_straight = dir.to_vector();
        let vec_right = dir.turn_right().to_vector();

        let mut idx = 0;
        for f in -3..=4 {
            for r in -3..=4 {
                let cx = head.0 + f * vec_straight.0 + r * vec_right.0;
                let cy = head.1 + f * vec_straight.1 + r * vec_right.1;
                let cell = (cx, cy);

                if cell == self.apple_pos {
                    obs[idx] = 1.0;
                } else if cx < 0 || cx >= self.grid_width || cy < 0 || cy >= self.grid_height {
                    obs[idx] = -1.0;
                } else if snake.body.contains(&cell) {
                    obs[idx] = -1.0;
                } else {
                    let mut is_enemy = false;
                    for j in 0..self.snakes.len() {
                        if snake_index != j && self.snakes[j].body.contains(&cell) {
                            is_enemy = true;
                            break;
                        }
                    }
                    if is_enemy {
                        obs[idx] = -0.5;
                    } else {
                        obs[idx] = 0.0;
                    }
                }
                idx += 1;
            }
        }

        let dx = self.apple_pos.0 - head.0;
        let dy = self.apple_pos.1 - head.1;
        let max_dim = self.grid_width.max(self.grid_height) as f32;
        obs[64] = (dx * vec_straight.0 + dy * vec_straight.1) as f32 / max_dim;
        obs[65] = (dx * vec_right.0 + dy * vec_right.1) as f32 / max_dim;

        obs
    }
}
