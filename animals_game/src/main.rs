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
    let args: Vec<String> = std::env::args().collect();
    let mut num_snakes = 2;
    if let Some(idx) = args.iter().position(|arg| arg == "--snakes") {
        if idx + 1 < args.len() {
            if let Ok(n) = args[idx + 1].parse::<usize>() {
                num_snakes = n;
            }
        }
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Snake".into(),
                resolution: (800, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(GameEngine(GameState::new(GRID_WIDTH, GRID_HEIGHT, num_snakes)))
        .insert_resource(TickTimer(Timer::from_seconds(0.1, TimerMode::Repeating)))
        .insert_resource(AiConnection(None))
        .add_systems(Startup, setup)
        .add_systems(Update, (keyboard_input, game_tick, render_sync).chain())
        .run();
}

fn setup(mut commands: Commands, mut ai_conn: ResMut<AiConnection>, engine: Res<GameEngine>) {
    commands.spawn(Camera2d);

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--ai") {
        let mut model_paths = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i] == "--model" && i + 1 < args.len() {
                model_paths.push(args[i + 1].clone());
                i += 1;
            }
            i += 1;
        }
        
        let num_snakes = engine.0.snakes.len();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        println!("Spawning AI inference server on port {} with {} snakes...", port, num_snakes);
        
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let learner_dir = format!("{}/../learner", manifest_dir);

        let mut cmd = std::process::Command::new("uv");
        cmd.args(["run", "python", "-m", "learner.play", "--port", &port.to_string(), "--snakes", &num_snakes.to_string()])
           .current_dir(learner_dir)
           .env("PYTHONPATH", "src");

        for m in model_paths {
            cmd.arg("--model");
            cmd.arg(m);
        }

        let child = cmd.spawn().expect("Failed to spawn Python AI server");

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
    // Only handles first 2 snakes manually for testing
    if engine.0.snakes.len() > 0 {
        if keyboard_input.just_pressed(KeyCode::ArrowUp) {
            engine.0.set_direction(0, Direction::Up);
        } else if keyboard_input.just_pressed(KeyCode::ArrowDown) {
            engine.0.set_direction(0, Direction::Down);
        } else if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
            engine.0.set_direction(0, Direction::Left);
        } else if keyboard_input.just_pressed(KeyCode::ArrowRight) {
            engine.0.set_direction(0, Direction::Right);
        } 
    }

    if engine.0.snakes.len() > 1 {
        if keyboard_input.just_pressed(KeyCode::KeyW) {
            engine.0.set_direction(1, Direction::Up);
        } else if keyboard_input.just_pressed(KeyCode::KeyS) {
            engine.0.set_direction(1, Direction::Down);
        } else if keyboard_input.just_pressed(KeyCode::KeyA) {
            engine.0.set_direction(1, Direction::Left);
        } else if keyboard_input.just_pressed(KeyCode::KeyD) {
            engine.0.set_direction(1, Direction::Right);
        }
    }

    if keyboard_input.just_pressed(KeyCode::Space) && engine.0.game_over {
        // Restart the game
        let num_snakes = engine.0.snakes.len();
        engine.0 = GameState::new(GRID_WIDTH, GRID_HEIGHT, num_snakes);
    }
}

fn game_tick(time: Res<Time>, mut timer: ResMut<TickTimer>, mut engine: ResMut<GameEngine>, mut ai_conn: ResMut<AiConnection>) {
    if timer.0.tick(time.delta()).just_finished() {
        if engine.0.game_over {
            return;
        }

        if let Some(stream) = &mut ai_conn.0 {
            let num_snakes = engine.0.snakes.len();
            let mut byte_payload = vec![0u8; num_snakes * 66 * 4];
            
            // 1. Get Observations
            for s in 0..num_snakes {
                let obs = engine.0.get_relative_observation(s);
                for (i, &val) in obs.iter().enumerate() {
                    let offset = (s * 66 + i) * 4;
                    byte_payload[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                }
            }

            // 2. Send to Python
            if stream.write_all(&byte_payload).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            // 3. Read Actions
            let mut action_bytes = vec![0u8; num_snakes * 4];
            if stream.read_exact(&mut action_bytes).is_err() {
                eprintln!("Lost connection to AI server");
                std::process::exit(1);
            }

            for s in 0..num_snakes {
                let offset = s * 4;
                let action_int = i32::from_le_bytes(action_bytes[offset..offset + 4].try_into().unwrap());
                let relative_action = RelativeAction::from_usize(action_int as usize);
                let new_dir = relative_action.to_absolute_direction(engine.0.snakes[s].direction);
                engine.0.set_direction(s, new_dir);
            }
        }

        engine.0.step();

        // The engine no longer ends the game itself on death (it respawns
        // dead snakes in place so training episodes aren't truncated across
        // the whole game). For the visualizer/manual-play we still want a
        // clear "game over, press Space to restart" moment, so detect any
        // death here and freeze the game ourselves.
        if engine.0.snakes.iter().any(|s| s.is_dead) {
            engine.0.game_over = true;
        }
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
    let num_snakes = engine.0.snakes.len();
    for (s_idx, snake) in engine.0.snakes.iter().enumerate() {
        for (i, pos) in snake.body.iter().enumerate() {
            let color = if engine.0.game_over {
                Color::srgb(0.5, 0.5, 0.5) // Gray when dead
            } else if i == 0 {
                // Determine head color based on snake index
                let hue = (s_idx as f32 / num_snakes as f32) * 360.0;
                Color::hsl(hue, 1.0, 0.5)
            } else {
                // Determine body color based on snake index
                let hue = (s_idx as f32 / num_snakes as f32) * 360.0;
                Color::hsl(hue, 1.0, 0.3)
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
