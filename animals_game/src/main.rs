use bevy::prelude::*;
use animals_engine::snake::{Direction, GameState, RelativeAction};
use std::io::{Read, Write};
use std::net::TcpStream;
const GRID_WIDTH: i32 = 100;
const GRID_HEIGHT: i32 = 100;
const TILE_SIZE: f32 = 6.0;

#[derive(Resource)]
struct GameEngine(pub GameState);

#[derive(Resource)]
struct TickTimer(pub Timer);

#[derive(Resource)]
struct AiConnection(pub Option<TcpStream>);

#[derive(Component)]
struct SnakeSegment;

#[derive(Component)]
struct Apple;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Snake".into(),
                resolution: (800, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(GameEngine(GameState::new(GRID_WIDTH, GRID_HEIGHT)))
        .insert_resource(TickTimer(Timer::from_seconds(0.1, TimerMode::Repeating)))
        .insert_resource(AiConnection(None))
        .add_systems(Startup, setup)
        .add_systems(Update, (keyboard_input, game_tick, render_sync).chain())
        .run();
}

fn setup(mut commands: Commands, mut ai_conn: ResMut<AiConnection>) {
    commands.spawn(Camera2d);

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--ai") {
        println!("AI Mode enabled! Attempting to connect to inference server at 127.0.0.1:31337...");
        match TcpStream::connect("127.0.0.1:31337") {
            Ok(stream) => {
                println!("Connected to AI inference server!");
                ai_conn.0 = Some(stream);
            }
            Err(e) => {
                eprintln!("Failed to connect to AI server: {}. Ensure you have run 'task run-ai-server' first!", e);
                std::process::exit(1);
            }
        }
    }
}

fn keyboard_input(keyboard_input: Res<ButtonInput<KeyCode>>, mut engine: ResMut<GameEngine>) {
    if keyboard_input.just_pressed(KeyCode::ArrowUp) {
        engine.0.set_direction(Direction::Up);
    } else if keyboard_input.just_pressed(KeyCode::ArrowDown) {
        engine.0.set_direction(Direction::Down);
    } else if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
        engine.0.set_direction(Direction::Left);
    } else if keyboard_input.just_pressed(KeyCode::ArrowRight) {
        engine.0.set_direction(Direction::Right);
    } else if keyboard_input.just_pressed(KeyCode::Space) && engine.0.game_over {
        // Restart the game
        engine.0 = GameState::new(GRID_WIDTH, GRID_HEIGHT);
    }
}

fn game_tick(time: Res<Time>, mut timer: ResMut<TickTimer>, mut engine: ResMut<GameEngine>, mut ai_conn: ResMut<AiConnection>) {
    if timer.0.tick(time.delta()).just_finished() {
        if engine.0.game_over {
            if ai_conn.0.is_some() {
                // Auto restart in AI mode
                engine.0 = GameState::new(GRID_WIDTH, GRID_HEIGHT);
            }
            return;
        }

        if let Some(stream) = &mut ai_conn.0 {
            // 1. Get Observation
            let obs = engine.0.get_relative_observation();
            let mut byte_payload = [0u8; 32];
            for (i, &val) in obs.iter().enumerate() {
                byte_payload[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }

            // 2. Send to Python
            if stream.write_all(&byte_payload).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            // 3. Read Action
            let mut action_bytes = [0u8; 4];
            if stream.read_exact(&mut action_bytes).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            let action_int = i32::from_le_bytes(action_bytes);
            let relative_action = RelativeAction::from_usize(action_int as usize);
            let new_dir = relative_action.to_absolute_direction(engine.0.direction);
            engine.0.set_direction(new_dir);
        }

        engine.0.step();
    }
}

fn render_sync(
    mut commands: Commands,
    engine: Res<GameEngine>,
    segment_query: Query<Entity, With<SnakeSegment>>,
    apple_query: Query<Entity, With<Apple>>,
) {
    // 1. Remove old sprites
    for entity in segment_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in apple_query.iter() {
        commands.entity(entity).despawn();
    }

    let offset_x = (GRID_WIDTH as f32 * TILE_SIZE) / 2.0;
    let offset_y = (GRID_HEIGHT as f32 * TILE_SIZE) / 2.0;

    // 2. Draw Apple
    let apple_pos = engine.0.apple_pos;
    commands.spawn((
        Sprite {
            color: Color::srgb(1.0, 0.0, 0.0),
            custom_size: Some(Vec2::new(TILE_SIZE, TILE_SIZE)),
            ..default()
        },
        Transform::from_xyz(
            apple_pos.0 as f32 * TILE_SIZE - offset_x,
            apple_pos.1 as f32 * TILE_SIZE - offset_y,
            0.0,
        ),
        Apple,
    ));

    // 3. Draw Snake Body
    for (i, pos) in engine.0.snake_body.iter().enumerate() {
        let color = if engine.0.game_over {
            Color::srgb(0.5, 0.5, 0.5) // Gray when dead
        } else if i == 0 {
            Color::srgb(0.0, 0.8, 0.0) // Head is bright green
        } else {
            Color::srgb(0.0, 0.5, 0.0) // Body is darker green
        };

        commands.spawn((
            Sprite {
                color,
                custom_size: Some(Vec2::new(TILE_SIZE, TILE_SIZE)),
                ..default()
            },
            Transform::from_xyz(
                pos.0 as f32 * TILE_SIZE - offset_x,
                pos.1 as f32 * TILE_SIZE - offset_y,
                0.0,
            ),
            SnakeSegment,
        ));
    }
}
