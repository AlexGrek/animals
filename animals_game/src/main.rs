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

#[derive(Resource)]
struct AiServerProcess(std::sync::Mutex<std::process::Child>);

impl Drop for AiServerProcess {
    fn drop(&mut self) {
        if let Ok(mut child) = self.0.lock() {
            println!("Shutting down Python AI server...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

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
        let mut model_path = "models/snake_model".to_string();
        if let Some(idx) = args.iter().position(|arg| arg == "--model") {
            if idx + 1 < args.len() {
                model_path = args[idx + 1].clone();
            }
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        println!("Spawning AI inference server on port {} with model {}...", port, model_path);
        
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let learner_dir = format!("{}/../learner", manifest_dir);

        let child = std::process::Command::new("uv")
            .args(["run", "python", "-m", "learner.play", "--port", &port.to_string(), "--model", &model_path])
            .current_dir(learner_dir)
            .env("PYTHONPATH", "src")
            .spawn()
            .expect("Failed to spawn Python AI server");

        commands.insert_resource(AiServerProcess(std::sync::Mutex::new(child)));

        println!("Waiting for AI server to start...");
        let mut stream = None;
        for _ in 0..50 {
            if let Ok(s) = TcpStream::connect(format!("127.0.0.1:{}", port)) {
                stream = Some(s);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        if let Some(s) = stream {
            println!("Connected to AI inference server!");
            s.set_nodelay(true).unwrap();
            ai_conn.0 = Some(s);
        } else {
            eprintln!("Failed to connect to spawned AI server.");
            std::process::exit(1);
        }
    }
}

fn keyboard_input(keyboard_input: Res<ButtonInput<KeyCode>>, mut engine: ResMut<GameEngine>) {
    if keyboard_input.just_pressed(KeyCode::ArrowUp) {
        engine.0.set_direction(0, Direction::Up);
    } else if keyboard_input.just_pressed(KeyCode::ArrowDown) {
        engine.0.set_direction(0, Direction::Down);
    } else if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
        engine.0.set_direction(0, Direction::Left);
    } else if keyboard_input.just_pressed(KeyCode::ArrowRight) {
        engine.0.set_direction(0, Direction::Right);
    } 

    if keyboard_input.just_pressed(KeyCode::KeyW) {
        engine.0.set_direction(1, Direction::Up);
    } else if keyboard_input.just_pressed(KeyCode::KeyS) {
        engine.0.set_direction(1, Direction::Down);
    } else if keyboard_input.just_pressed(KeyCode::KeyA) {
        engine.0.set_direction(1, Direction::Left);
    } else if keyboard_input.just_pressed(KeyCode::KeyD) {
        engine.0.set_direction(1, Direction::Right);
    }

    if keyboard_input.just_pressed(KeyCode::Space) && engine.0.game_over {
        // Restart the game
        engine.0 = GameState::new(GRID_WIDTH, GRID_HEIGHT);
    }
}

fn game_tick(time: Res<Time>, mut timer: ResMut<TickTimer>, mut engine: ResMut<GameEngine>, mut ai_conn: ResMut<AiConnection>) {
    if timer.0.tick(time.delta()).just_finished() {
        if engine.0.game_over {
            return;
        }

        if let Some(stream) = &mut ai_conn.0 {
            // 1. Get Observation
            let obs0 = engine.0.get_relative_observation(0);
            let obs1 = engine.0.get_relative_observation(1);
            let mut byte_payload = [0u8; 528];
            for (i, &val) in obs0.iter().enumerate() {
                byte_payload[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
            for (i, &val) in obs1.iter().enumerate() {
                byte_payload[(i + 66) * 4..(i + 67) * 4].copy_from_slice(&val.to_le_bytes());
            }

            // 2. Send to Python
            if stream.write_all(&byte_payload).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            // 3. Read Action
            let mut action_bytes = [0u8; 8];
            if stream.read_exact(&mut action_bytes).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            let action0_int = i32::from_le_bytes(action_bytes[0..4].try_into().unwrap());
            let action1_int = i32::from_le_bytes(action_bytes[4..8].try_into().unwrap());

            let relative_action0 = RelativeAction::from_usize(action0_int as usize);
            let new_dir0 = relative_action0.to_absolute_direction(engine.0.snakes[0].direction);
            engine.0.set_direction(0, new_dir0);

            let relative_action1 = RelativeAction::from_usize(action1_int as usize);
            let new_dir1 = relative_action1.to_absolute_direction(engine.0.snakes[1].direction);
            engine.0.set_direction(1, new_dir1);
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

    // 3. Draw Snake Bodies
    for (s_idx, snake) in engine.0.snakes.iter().enumerate() {
        for (i, pos) in snake.body.iter().enumerate() {
            let color = if engine.0.game_over {
                Color::srgb(0.5, 0.5, 0.5) // Gray when dead
            } else if i == 0 {
                if s_idx == 0 { Color::srgb(0.0, 0.8, 0.0) } else { Color::srgb(0.0, 0.0, 0.8) } // Head
            } else {
                if s_idx == 0 { Color::srgb(0.0, 0.5, 0.0) } else { Color::srgb(0.0, 0.0, 0.5) } // Body
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
}
