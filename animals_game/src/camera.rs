use bevy::prelude::*;
use bevy::input::mouse::MouseWheel;
use crate::constants::*;

/// Pans and zooms the camera every frame. WASD/Arrow keys pan (scaled by the
/// current zoom so it feels constant on screen at any zoom level); the mouse
/// wheel zooms in/out by adjusting the orthographic projection's `scale`.
pub fn camera_control(
    time: Res<Time>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut scroll_events: MessageReader<MouseWheel>,
    mut camera_query: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
) {
    let Ok((mut transform, mut projection)) = camera_query.single_mut() else {
        return;
    };
    let Projection::Orthographic(ortho) = &mut *projection else {
        return;
    };

    let mut pan = Vec2::ZERO;
    if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
        pan.y += 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
        pan.y -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) {
        pan.x -= 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) {
        pan.x += 1.0;
    }
    if pan != Vec2::ZERO {
        let delta = pan.normalize() * PAN_SPEED * ortho.scale * time.delta_secs();
        transform.translation.x += delta.x;
        transform.translation.y += delta.y;
    }

    for ev in scroll_events.read() {
        // Scrolling up (positive y) zooms in, i.e. shrinks the scale.
        if ev.y > 0.0 {
            ortho.scale = (ortho.scale / ZOOM_STEP).clamp(MIN_ZOOM, MAX_ZOOM);
        } else if ev.y < 0.0 {
            ortho.scale = (ortho.scale * ZOOM_STEP).clamp(MIN_ZOOM, MAX_ZOOM);
        }
    }
}
