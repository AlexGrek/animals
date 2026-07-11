use bevy::prelude::*;

/// The on-screen line that reflects `AppStatus` (loading / error messages).
#[derive(Component)]
pub struct StatusText;

#[derive(Component)]
pub struct SnakeSegment;

#[derive(Component)]
pub struct Particle {
    pub velocity: Vec2,
    pub lifetime: Timer,
}

#[derive(Component, PartialEq, Clone, Copy)]
pub struct SpeedButton(pub f32);

#[derive(Component)]
pub enum ToolbarButton {
    ToggleNames,
    ToggleTargets,
    ToggleNn,
}

/// Which layer of the NN a given overlay cell belongs to.
#[derive(Clone, Copy, PartialEq)]
pub enum NnLayer {
    InputEntity,
    InputGrass,
    Scalars,
    Features,
    Pi0,
    Pi1,
    Action,
}

/// A single colored cell in the NN overlay, addressed by layer + index.
#[derive(Component)]
pub struct NnCell {
    pub layer: NnLayer,
    pub index: usize,
}

/// Root node of the NN overlay panel (toggled via `Node.display`).
#[derive(Component)]
pub struct NnPanel;

/// The text line under the panel showing the chosen action + probability.
#[derive(Component)]
pub struct NnActionText;

#[derive(Component)]
pub struct SnakeHead {
    pub snake_idx: usize,
}

#[derive(Component)]
pub struct Apple {
    pub prey_idx: usize,
}

#[derive(Component)]
pub struct MapTile;

/// Marks a sprite that should be smoothly interpolated between two world
/// positions over the course of the current tick interval.
#[derive(Component)]
pub struct Interp {
    pub from: Vec3,
    pub to: Vec3,
}
