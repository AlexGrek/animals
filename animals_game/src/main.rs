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
mod menu;

use bevy::prelude::*;
use bevy::window::PresentMode;
use animals_engine::GameState;

use constants::{GRID_WIDTH, GRID_HEIGHT};
use resources::{
    GameEngine, TickTimer, AppStatus, RenderDirty, PrevPositions, GameSpeed,
    OverlaySettings, SelectedSnake, ActivationBuffer, AiWorker, AppState, MatchConfig, StatsTracker,
    CorpseSpriteIndex
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_ai = args.iter().any(|arg| arg == "--ai");
    let initial_state = if is_ai { AppState::Menu } else { AppState::InGame };

    let num_preys = 24;
    let num_amphibias = 8;
    let num_corpsefags = 10;

    let match_config = MatchConfig {
        is_ai,
        num_preys,
        num_amphibias,
        num_corpsefags,
        snakes: Vec::new(),
        prey_models: Vec::new(),
        amphibia_models: Vec::new(),
        corpsefag_models: Vec::new(),
    };

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
        .insert_resource(match_config)
        .insert_state(initial_state)
        .insert_resource(GameEngine(GameState::new(GRID_WIDTH, GRID_HEIGHT, 2, num_preys, num_preys.max(100), num_amphibias, num_amphibias.max(100), num_corpsefags, false, !is_ai)))
        .insert_resource(TickTimer(Timer::from_seconds(0.033, TimerMode::Repeating)))
        .insert_resource(AiWorker(None))
        .insert_resource(RenderDirty(true))
        .insert_resource(CorpseSpriteIndex::default())
        .insert_resource(AppStatus::Running)
        .insert_resource(PrevPositions::default())
        .insert_resource(GameSpeed(1.0))
        .insert_resource(OverlaySettings::default())
        .insert_resource(SelectedSnake::default())
        .insert_resource(ActivationBuffer::default())
        .insert_resource(StatsTracker::default())
        .add_plugins(menu::MenuPlugin)
        .add_systems(Startup, setup::setup_camera)
        .add_systems(OnEnter(AppState::InGame), setup::in_game_setup)
        .add_systems(
            Update,
            (
                logic::keyboard_input,
                camera::camera_control,
                ai::poll_ai_connection,
                logic::game_tick,
                ui::update_status_text,
                ui::update_stats_text,
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
                ui::update_models_overlay,
            )
                .chain()
                .run_if(in_state(AppState::InGame)),
        )
        .run();
}
