mod constants;
mod components;
mod resources;
mod utils;
mod ai;
mod camera;
mod ui;
mod render;
mod logic;
mod setup;

use bevy::prelude::*;
use bevy::window::PresentMode;
use animals_engine::GameState;

use constants::{GRID_WIDTH, GRID_HEIGHT};
use resources::{
    GameEngine, TickTimer, AppStatus, RenderDirty, PrevPositions, GameSpeed, 
    OverlaySettings, SelectedSnake, ActivationBuffer, AiWorker
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_snakes = 2;
    if let Some(idx) = args.iter().position(|arg| arg == "--snakes") {
        if idx + 1 < args.len() {
            if let Ok(n) = args[idx + 1].parse::<usize>() {
                num_snakes = n;
            }
        }
    }

    let mut num_preys = 1;
    if let Some(idx) = args.iter().position(|arg| arg == "--preys") {
        if idx + 1 < args.len() {
            if let Ok(n) = args[idx + 1].parse::<usize>() {
                num_preys = n;
            }
        }
    }

    let mut num_amphibias = 0;
    if let Some(idx) = args.iter().position(|arg| arg == "--amphibias") {
        if idx + 1 < args.len() {
            if let Ok(n) = args[idx + 1].parse::<usize>() {
                num_amphibias = n;
            }
        }
    }

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Snake".into(),
                        resolution: (800, 800).into(),
                        // Vsync: present one frame per vertical blank for a smooth,
                        // tear-free 60fps on a 60Hz display.
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .insert_resource(ClearColor(Color::srgb(0.09, 0.10, 0.14)))
        .insert_resource(GameEngine(GameState::new(GRID_WIDTH, GRID_HEIGHT, num_snakes, num_preys, num_preys.max(100), num_amphibias, num_amphibias.max(100), false)))
        .insert_resource(TickTimer(Timer::from_seconds(0.033, TimerMode::Repeating)))
        .insert_resource(AiWorker(None))
        .insert_resource(RenderDirty(true))
        .insert_resource(AppStatus::Running)
        .insert_resource(PrevPositions::default())
        .insert_resource(GameSpeed(1.0))
        .insert_resource(OverlaySettings::default())
        .insert_resource(SelectedSnake::default())
        .insert_resource(ActivationBuffer::default())
        .add_systems(Startup, setup::setup)
        .add_systems(
            Update,
            (
                logic::keyboard_input,
                camera::camera_control,
                ai::poll_ai_connection,
                logic::game_tick,
                ui::update_status_text,
                render::render_sync,
                render::update_map,
                render::apply_interpolation,
                render::update_particles,
                ui::toolbar_interaction,
                ui::toolbar_colors,
                ui::speed_control_interaction,
                ui::speed_control_colors,
                render::draw_targets_overlay,
                ui::update_nn_overlay,
            )
                .chain(),
        )
        .run();
}
