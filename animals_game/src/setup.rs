use bevy::prelude::*;
use crate::components::*;
use crate::constants::*;
use crate::resources::*;
use crate::ai::spawn_ai_server;
use crate::render::spawn_map;
use crate::ui::spawn_nn_overlay;
use crate::utils::{controller_labels, snake_color};

pub fn setup_camera(
    mut commands: Commands,
) {
    // Start zoomed out so a good chunk of the field is framed on load.
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: INITIAL_ZOOM,
            ..OrthographicProjection::default_2d()
        }),
    ));
}

pub fn in_game_setup(
    mut commands: Commands,
    mut engine: ResMut<GameEngine>,
    mut status: ResMut<AppStatus>,
    mut images: ResMut<Assets<Image>>,
    config: Res<MatchConfig>,
) {
    // If it's an AI match configured from the menu, we need to update the GameEngine
    if config.is_ai {
        engine.0 = animals_engine::GameState::new(
            GRID_WIDTH, 
            GRID_HEIGHT, 
            config.snakes.len(), 
            config.num_preys, 
            config.num_preys.max(100), 
            config.num_amphibias,
            config.num_amphibias.max(100),
            10,
            false,
            false, // AI-driven snakes: auto-steer would fight the policy's learned dynamics
        );
    }

    spawn_map(&mut commands, &engine.0, &mut images);

    let is_ai = config.is_ai;
    let num_snakes = engine.0.snakes.len();
    
    // Build dummy args for controller labels to reuse existing logic
    // or just pass a simple boolean. We can construct dummy args since controller_labels expects them.
    let mut args = vec!["animals_game".to_string()];
    if is_ai {
        args.push("--ai".to_string());
    }
    let labels = controller_labels(&args, num_snakes, is_ai);


    // Header: which actor is controlled by what, drawn top-left.
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new(if is_ai {
                    "AI Match"
                } else {
                    "Camera: WASD/Arrows pan, mouse wheel zoom"
                }),
                TextFont { font_size: 20.0, ..default() },
                TextColor(Color::WHITE),
            ));
            parent.spawn((
                ModelNamesOverlay,
                Node {
                    flex_direction: FlexDirection::Column,
                    display: Display::None,
                    ..default()
                }
            )).with_children(|models_parent| {
                for (i, label) in labels.iter().enumerate() {
                    models_parent.spawn((
                        Text::new(format!("Snake {i}  -  {label}")),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(snake_color(i, num_snakes)),
                    ));
                }
            });
            parent.spawn((
                Text::new(""),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::srgb(0.9, 0.9, 0.2)),
                StatusText,
            ));
            parent.spawn((
                Text::new(""),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::srgb(0.7, 0.7, 0.7)),
                StatsText,
            ));
        });

    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            top: Val::Px(200.0),
            left: Val::Px(8.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(10.0),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Button,
                Node {
                    width: Val::Px(40.0),
                    height: Val::Px(40.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                ToolbarButton::ToggleNames,
            )).with_children(|parent| {
                parent.spawn((
                    Text::new("N"),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });

            parent.spawn((
                Button,
                Node {
                    width: Val::Px(40.0),
                    height: Val::Px(40.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                ToolbarButton::ToggleTargets,
            )).with_children(|parent| {
                parent.spawn((
                    Text::new("T"),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });

            parent.spawn((
                Button,
                Node {
                    width: Val::Px(40.0),
                    height: Val::Px(40.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                ToolbarButton::ToggleNn,
            )).with_children(|parent| {
                parent.spawn((
                    Text::new("A"),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });

            parent.spawn((
                Button,
                Node {
                    width: Val::Px(40.0),
                    height: Val::Px(40.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                ToolbarButton::ToggleModels,
            )).with_children(|parent| {
                parent.spawn((
                    Text::new("M"),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });
        });

    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            right: Val::Px(8.0),
            flex_direction: FlexDirection::Row,
            column_gap: Val::Px(5.0),
            ..default()
        })
        .with_children(|parent| {
            let speeds = [
                ("||", 0.0),
                ("0.1x", 0.1),
                ("0.25x", 0.25),
                ("0.5x", 0.5),
                ("1x", 1.0),
                ("2x", 2.0),
                ("5x", 5.0),
                ("MAX", 100.0),
            ];
            for (label, val) in speeds {
                parent.spawn((
                    Button,
                    Node {
                        width: Val::Px(40.0),
                        height: Val::Px(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                    SpeedButton(val),
                )).with_children(|parent| {
                    parent.spawn((
                        Text::new(label),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::WHITE),
                    ));
                });
            }
        });

    spawn_nn_overlay(&mut commands);

    if is_ai {
        spawn_ai_server(&mut commands, &config, num_snakes, &mut status);
    }
}
