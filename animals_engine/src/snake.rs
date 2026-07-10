use crate::direction::Direction;

#[derive(Clone, Debug)]
pub struct SnakeState {
    pub head_pos: (f32, f32),
    pub body: Vec<(i32, i32)>,
    pub direction: Direction,
    pub is_dead: bool,
    pub score: u32,
    pub kills: u32,
    pub death_by_wall: bool,
    pub death_by_self: bool,
    pub death_by_opponent: bool,
    pub death_by_hunger: bool,
    pub steps_since_last_eat: u32,
}

impl SnakeState {
    pub fn new(start_pos: (i32, i32), direction: Direction) -> Self {
        Self {
            head_pos: (start_pos.0 as f32, start_pos.1 as f32),
            body: vec![start_pos],
            direction,
            is_dead: false,
            score: 0,
            kills: 0,
            death_by_wall: false,
            death_by_self: false,
            death_by_opponent: false,
            death_by_hunger: false,
            steps_since_last_eat: 0,
        }
    }
}
