pub const GRID_WIDTH: i32 = 400;
pub const GRID_HEIGHT: i32 = 400;
pub const TILE_SIZE: f32 = 6.0;

/// Camera pan speed in world units/sec at zoom scale 1.0 (scaled by the
/// current projection scale so panning feels constant on screen at any zoom).
pub const PAN_SPEED: f32 = 500.0;

/// Multiplicative zoom step applied per "notch" of mouse-wheel scroll.
pub const ZOOM_STEP: f32 = 1.1;
pub const MIN_ZOOM: f32 = 0.2;
pub const MAX_ZOOM: f32 = 8.0;

/// Initial orthographic scale so a good chunk of the 400x400 field is framed
/// on load (field is 2400x2400 world units, window is 800x800).
pub const INITIAL_ZOOM: f32 = 4.0;

/// Fixed activation-vector layout streamed from `learner/play.py`. Keep in sync.
pub const NN_FEATURES: usize = 128;
pub const NN_PI0: usize = 256;
pub const NN_PI1: usize = 256;
pub const NN_LOGITS: usize = 3;
pub const NN_ACT_LEN: usize = NN_FEATURES + NN_PI0 + NN_PI1 + NN_LOGITS; // 643
