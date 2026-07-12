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
pub const CORPSEFAG_OBS_SIZE: usize = 18;
/// Normalizes the hunger observation scalar (`obs[67]`) and the hunger reward
/// penalty. Kept separate from `HUNGER_DEATH_LIMIT` so the actual starvation
/// timing can be tuned without changing what a trained model reads as "close
/// to starving" (an already-trained policy's hunger sense was calibrated
/// against this value; changing it would require a retrain to stay correct).
pub const HUNGER_LIMIT: u32 = 1200;
/// Steps a snake can go without eating before it actually dies of hunger.
/// Lower than `HUNGER_LIMIT` on purpose: this speeds up starvation in play
/// without retraining, at the cost of a trained model never seeing the
/// hunger observation climb past `HUNGER_DEATH_LIMIT / HUNGER_LIMIT` before
/// dying (its "danger" calibration was learned over the full 0..1 range).
pub const HUNGER_DEATH_LIMIT: u32 = HUNGER_LIMIT;
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
