use rand::Rng;
use serde::Deserialize;

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
pub struct GameState {
    pub snake_body: Vec<(i32, i32)>,
    pub apple_pos: (i32, i32),
    pub direction: Direction,
    pub grid_width: i32,
    pub grid_height: i32,
    pub score: u32,
    pub game_over: bool,
}

impl GameState {
    pub fn new(width: i32, height: i32) -> Self {
        let mut state = Self {
            snake_body: vec![(width / 2, height / 2)],
            apple_pos: (0, 0),
            direction: Direction::Up,
            grid_width: width,
            grid_height: height,
            score: 0,
            game_over: false,
        };
        state.spawn_apple();
        state
    }

    pub fn set_direction(&mut self, new_dir: Direction) {
        if self.direction.opposite() != new_dir {
            self.direction = new_dir;
        }
    }

    pub fn step(&mut self) {
        if self.game_over {
            return;
        }

        let head = self.snake_body[0];
        let new_head = match self.direction {
            Direction::Up => (head.0, head.1 + 1),
            Direction::Down => (head.0, head.1 - 1),
            Direction::Left => (head.0 - 1, head.1),
            Direction::Right => (head.0 + 1, head.1),
        };

        // Check wall collision
        if new_head.0 < 0
            || new_head.0 >= self.grid_width
            || new_head.1 < 0
            || new_head.1 >= self.grid_height
        {
            self.game_over = true;
            return;
        }

        // Check self collision
        if self.snake_body.contains(&new_head) {
            self.game_over = true;
            return;
        }

        self.snake_body.insert(0, new_head);

        // Check apple eating
        if new_head == self.apple_pos {
            self.score += 1;
            self.spawn_apple();
        } else {
            self.snake_body.pop();
        }
    }

    fn spawn_apple(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..self.grid_width);
            let y = rng.gen_range(0..self.grid_height);
            let pos = (x, y);
            if !self.snake_body.contains(&pos) {
                self.apple_pos = pos;
                break;
            }
        }
    }

    pub fn is_impassable(&self, pos: (i32, i32)) -> bool {
        pos.0 < 0
            || pos.0 >= self.grid_width
            || pos.1 < 0
            || pos.1 >= self.grid_height
            || self.snake_body.contains(&pos)
    }

    /// Exports the 8-dimensional relative observation vector for machine learning.
    pub fn get_relative_observation(&self) -> [f32; 8] {
        if self.snake_body.is_empty() {
            return [0.0; 8];
        }

        let head = self.snake_body[0];
        let dir = self.direction;

        // 1. Directions relative to snake heading
        let vec_straight = dir.to_vector();
        let vec_left = dir.turn_left().to_vector();
        let vec_right = dir.turn_right().to_vector();

        // Cells directly relative to the head
        let cell_straight = (head.0 + vec_straight.0, head.1 + vec_straight.1);
        let cell_left = (head.0 + vec_left.0, head.1 + vec_left.1);
        let cell_right = (head.0 + vec_right.0, head.1 + vec_right.1);

        // Danger observations
        let danger_straight = if self.is_impassable(cell_straight) { 1.0 } else { 0.0 };
        let danger_left = if self.is_impassable(cell_left) { 1.0 } else { 0.0 };
        let danger_right = if self.is_impassable(cell_right) { 1.0 } else { 0.0 };

        // 2. Food relative direction components
        let dx = self.apple_pos.0 - head.0;
        let dy = self.apple_pos.1 - head.1;

        // Project food vector onto heading and left/right vectors
        let forward_comp = dx * vec_straight.0 + dy * vec_straight.1;
        let left_comp = dx * vec_left.0 + dy * vec_left.1;

        let food_ahead = if forward_comp > 0 { 1.0 } else { 0.0 };
        let food_behind = if forward_comp < 0 { 1.0 } else { 0.0 };
        let food_left = if left_comp > 0 { 1.0 } else { 0.0 };
        let food_right = if left_comp < 0 { 1.0 } else { 0.0 };

        // 3. Distance to food
        let distance = ((dx * dx + dy * dy) as f32).sqrt();
        let normalized_distance = 1.0 / (distance + 1.0);

        [
            danger_straight,
            danger_left,
            danger_right,
            food_ahead,
            food_behind,
            food_left,
            food_right,
            normalized_distance,
        ]
    }
}
