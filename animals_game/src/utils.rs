use bevy::prelude::*;
use animals_engine::{GameState, PREY_OBS_SIZE, SNAKE_OBS_SIZE};
use crate::resources::SelectedSnake;

/// Colour used for a snake's head and its header line, so the two are visually
/// linked. Matches the hue formula used when drawing bodies in `render_sync`.
pub fn snake_color(_idx: usize, _total: usize) -> Color {
    Color::srgb(1.0, 0.6, 0.0) // Orange
}

/// Flattens every snake observation followed by every prey observation into one
/// buffer, in the same layout the Python inference server expects.
pub fn gather_observations(game: &GameState) -> Vec<f32> {
    let mut obs = Vec::with_capacity(game.snakes.len() * SNAKE_OBS_SIZE + game.preys.len() * PREY_OBS_SIZE);
    for s in 0..game.snakes.len() {
        obs.extend_from_slice(&game.get_relative_observation(s));
    }
    for p in 0..game.preys.len() {
        obs.extend_from_slice(&game.get_prey_observation(p));
    }
    obs
}

/// The selected snake index as an i32 for the wire protocol (-1 = none / out of range).
pub fn selected_i32(selected: &SelectedSnake, num_snakes: usize) -> i32 {
    match selected.0 {
        Some(i) if i < num_snakes => i as i32,
        _ => -1,
    }
}

/// Turns a model path like `models/v1.zip` into the short `v1` shown in the UI.
pub fn model_display_name(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.strip_suffix(".zip").unwrap_or(file).to_string()
}

/// Builds one label per snake describing who controls it.
pub fn controller_labels(args: &[String], num_snakes: usize, is_ai: bool) -> Vec<String> {
    if !is_ai {
        return (0..num_snakes).map(|_| "No input".to_string()).collect();
    }

    let mut model_paths: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_paths.push(args[i + 1].clone());
            i += 1;
        }
        i += 1;
    }

    // Same defaulting/replication rules as learner/play.py.
    if model_paths.is_empty() {
        model_paths.push("models/snake_model".to_string());
    }
    if model_paths.len() == 1 {
        model_paths = vec![model_paths[0].clone(); num_snakes];
    }

    (0..num_snakes)
        .map(|i| match model_paths.get(i) {
            Some(p) => format!("Model: {}", model_display_name(p)),
            None => "Model: (unassigned)".to_string(),
        })
        .collect()
}

/// Maps a per-layer-normalized activation (~[-1,1]) to a diverging color:
/// negative -> blue, ~0 -> near-black, positive -> orange.
pub fn activation_color(t: f32) -> Color {
    let t = t.clamp(-1.0, 1.0);
    if t >= 0.0 {
        Color::srgb(0.05 + 0.95 * t, 0.05 + 0.55 * t, 0.05)
    } else {
        let a = -t;
        Color::srgb(0.05, 0.05 + 0.45 * a, 0.05 + 0.95 * a)
    }
}

/// Largest absolute value in a slice, floored at a small epsilon.
pub fn max_abs(s: &[f32]) -> f32 {
    s.iter().fold(1e-6_f32, |m, &v| m.max(v.abs()))
}
