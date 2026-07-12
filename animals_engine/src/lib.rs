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
pub const SNAKE_OBS_SIZE: usize = 197;
/// Size of a prey's observation vector (shared by Prey and Amphibia). Mirrored
/// in `learner/src/learner/constants.py` — keep in sync.
pub const PREY_OBS_SIZE: usize = 131;
/// Steps a snake can go without eating before it dies of hunger.
pub const HUNGER_LIMIT: u32 = 1200;
/// Smell radius: a snake only senses prey within this torus-wrapped
/// Manhattan distance. Mirrored in `learner/src/learner/constants.py`.
pub const SMELL_RANGE: i32 = 60;
/// A coarse (4x4-tile) cell counts as "recently visited" for this many ticks
/// after last being entered; older visits read as unexplored again. Drives
/// both the exploration reward (`animals_simulation`) and the visitation
/// observation channel (`GameState::get_relative_observation`).
pub const VISIT_HORIZON: u64 = 1500;

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
