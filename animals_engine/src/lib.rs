pub mod direction;
pub mod game;
pub mod map;
pub mod snake;
pub mod species;

// Re-export common types for easier usage from other crates
pub use direction::{Direction, RelativeAction};
pub use game::GameState;
pub use map::{Map, Terrain};
pub use snake::SnakeState;
pub use species::Species;

/// Size of a snake's observation vector. Mirrored in
/// `learner/src/learner/constants.py` — keep in sync.
pub const SNAKE_OBS_SIZE: usize = 69;
/// Size of a prey's observation vector (shared by Prey and Amphibia). Mirrored
/// in `learner/src/learner/constants.py` — keep in sync.
pub const PREY_OBS_SIZE: usize = 67;
/// Steps a snake can go without eating before it dies of hunger.
pub const HUNGER_LIMIT: u32 = 1200;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
