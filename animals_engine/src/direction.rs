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
