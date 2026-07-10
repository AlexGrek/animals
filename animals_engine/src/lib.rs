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
