use bevy::prelude::*;
use bevy::window::PresentMode;
use bevy::input::mouse::MouseWheel;
use animals_engine::{GameState, RelativeAction, PREY_OBS_SIZE, SNAKE_OBS_SIZE};
use animals_engine::species::Species;
use animals_engine::map::Terrain;
use std::io::{Read, Write};
use std::net::TcpStream;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::asset::RenderAssetUsages;
const GRID_WIDTH: i32 = 400;
const GRID_HEIGHT: i32 = 400;
const TILE_SIZE: f32 = 6.0;
/// Camera pan speed in world units/sec at zoom scale 1.0 (scaled by the
/// current projection scale so panning feels constant on screen at any zoom).
const PAN_SPEED: f32 = 500.0;
/// Multiplicative zoom step applied per "notch" of mouse-wheel scroll.
const ZOOM_STEP: f32 = 1.1;
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 8.0;
/// Initial orthographic scale so a good chunk of the 400x400 field is framed
/// on load (field is 2400x2400 world units, window is 800x800).
const INITIAL_ZOOM: f32 = 4.0;

#[derive(Resource)]
struct GameEngine(pub GameState);

#[derive(Resource)]
struct TickTimer(pub Timer);

/// Handle to the background thread that owns the TCP socket to the Python
/// inference server. The blocking `write`/`read` round-trip lives on that
/// thread so it can never stall the render loop; the main thread only does
/// non-blocking channel sends/receives.
struct AiWorkerHandle {
    obs_tx: Sender<Vec<f32>>,
    act_rx: Receiver<Vec<i32>>,
    /// True while a request is in flight and we're waiting for its actions.
    awaiting: bool,
}

#[derive(Resource, Default)]
struct AiWorker(Option<AiWorkerHandle>);

/// Snapshot of every actor's position as of the START of the most recent
/// `step()` call, i.e. where it was a moment before the current tick's
/// positions. Used by `render_sync` to give each sprite an `Interp` so it can
/// glide from its previous position to its new one over the following frames
/// instead of teleporting on every tick.
#[derive(Resource, Default)]
struct PrevPositions {
    snake_bodies: Vec<Vec<(i32, i32)>>,
    prey_pos: Vec<(f32, f32)>,
}

/// Marks a sprite that should be smoothly interpolated between two world
/// positions over the course of the current tick interval, driven by
/// `apply_interpolation` every frame using `TickTimer`'s fraction-elapsed.
#[derive(Component)]
struct Interp {
    from: Vec3,
    to: Vec3,
}

/// Set whenever the game state changes so `render_sync` only rebuilds sprites
/// on ticks that actually moved the game, instead of every frame.
#[derive(Resource)]
struct RenderDirty(bool);

#[derive(Resource)]
struct AiServerProcess {
    child: std::sync::Mutex<std::process::Child>,
    /// Lines read from the child's stderr by a background thread, so we can
    /// surface its last error message if it exits before connecting.
    stderr_rx: Receiver<String>,
}

impl Drop for AiServerProcess {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            println!("Shutting down Python AI server...");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// High-level state of the app, surfaced to the player as on-screen text.
#[derive(Resource, Clone, PartialEq)]
enum AppStatus {
    /// Still getting ready (spawning / connecting to the AI server). The
    /// string is the message shown on screen.
    Loading(String),
    /// The game is live and ticking.
    Running,
    /// Something went wrong; the string is shown on screen and the game stays
    /// frozen instead of silently exiting.
    Failed(String),
}

/// Present only in `--ai` mode while we wait for the Python inference server to
/// finish loading its models and open its TCP port. Removed once connected.
#[derive(Resource)]
struct PendingConnection {
    port: u16,
    retry: Timer,
    elapsed: f32,
    timeout: f32,
    /// Stderr lines collected from the Python process so far, newest last.
    stderr_lines: Vec<String>,
}

/// The on-screen line that reflects [`AppStatus`] (loading / error messages).
#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct SnakeSegment;

#[derive(Component)]
struct Particle {
    velocity: Vec2,
    lifetime: Timer,
}

fn update_particles(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut Transform, &mut Sprite, &mut Particle)>,
) {
    let dt = time.delta();
    let dt_secs = time.delta_secs();
    for (entity, mut transform, mut sprite, mut particle) in query.iter_mut() {
        if particle.lifetime.tick(dt).just_finished() {
            commands.entity(entity).despawn();
        } else {
            transform.translation.x += particle.velocity.x * dt_secs;
            transform.translation.y += particle.velocity.y * dt_secs;
            
            let remaining = particle.lifetime.fraction_remaining();
            sprite.color.set_alpha(remaining * 0.8);
        }
    }
}

#[derive(Component)]
struct Apple;

#[derive(Component)]
struct MapTile;

fn spawn_map(commands: &mut Commands, state: &GameState, images: &mut Assets<Image>) {

    let width = GRID_WIDTH as u32 + 2;
    let height = GRID_HEIGHT as u32 + 2;
    let mut data = vec![0; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let grid_x = x as i32 - 1;
            let grid_y = y as i32 - 1;
            
            let color = if grid_x < 0 || grid_x >= GRID_WIDTH || grid_y < 0 || grid_y >= GRID_HEIGHT {
                [76, 76, 89, 255] // srgb(0.3, 0.3, 0.35) mapped to u8
            } else {
                let terrain = state.map.get_terrain(grid_x, grid_y);
                match terrain {
                    Terrain::Grass => [35, 81, 40, 255], // srgb(0.14, 0.32, 0.16)
                    Terrain::Road => [127, 102, 76, 255], // srgb(0.5, 0.4, 0.3)
                    Terrain::Water => [51, 127, 229, 255], // srgb(0.2, 0.5, 0.9)
                    Terrain::Rock => [102, 102, 102, 255], // srgb(0.4, 0.4, 0.4)
                }
            };
            
            let idx = ((height - 1 - y) * width + x) as usize * 4;
            data[idx] = color[0];
            data[idx + 1] = color[1];
            data[idx + 2] = color[2];
            data[idx + 3] = color[3];
        }
    }

    let image = Image::new(
        Extent3d { width, height, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );

    let image_handle = images.add(image);
    
    commands.spawn((
        Sprite {
            image: image_handle,
            custom_size: Some(Vec2::new(width as f32 * TILE_SIZE, height as f32 * TILE_SIZE)),
            ..default()
        },
        Transform::from_xyz(-TILE_SIZE / 2.0, -TILE_SIZE / 2.0, -1.5),
        MapTile,
    ));
}

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
        .insert_resource(GameEngine(GameState::new(GRID_WIDTH, GRID_HEIGHT, num_snakes, num_preys, num_amphibias)))
        .insert_resource(TickTimer(Timer::from_seconds(0.033, TimerMode::Repeating)))
        .insert_resource(AiWorker(None))
        .insert_resource(RenderDirty(true))
        .insert_resource(AppStatus::Running)
        .insert_resource(PrevPositions::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                keyboard_input,
                camera_control,
                poll_ai_connection,
                game_tick,
                update_status_text,
                render_sync,
                apply_interpolation,
                update_particles,
            )
                .chain(),
        )
        .run();
}

/// Colour used for a snake's head and its header line, so the two are visually
/// linked. Matches the hue formula used when drawing bodies in `render_sync`.
fn snake_color(idx: usize, total: usize) -> Color {
    let hue = (idx as f32 / total.max(1) as f32) * 360.0;
    Color::hsl(hue, 1.0, 0.5)
}

/// Flattens every snake observation followed by every prey observation into one
/// buffer, in the same layout the Python inference server expects.
fn gather_observations(game: &GameState) -> Vec<f32> {
    let mut obs = Vec::with_capacity(game.snakes.len() * SNAKE_OBS_SIZE + game.preys.len() * PREY_OBS_SIZE);
    for s in 0..game.snakes.len() {
        obs.extend_from_slice(&game.get_relative_observation(s));
    }
    for p in 0..game.preys.len() {
        obs.extend_from_slice(&game.get_prey_observation(p));
    }
    obs
}

/// Spawns the background thread that owns `stream` and services inference
/// requests: it blocks on the socket round-trip (write observations, read
/// actions) so the render thread never has to.
fn spawn_ai_worker(mut stream: TcpStream, total_preys: usize) -> AiWorkerHandle {
    let (obs_tx, obs_rx) = crossbeam_channel::unbounded::<Vec<f32>>();
    let (act_tx, act_rx) = crossbeam_channel::unbounded::<Vec<i32>>();

    std::thread::spawn(move || {
        while let Ok(obs) = obs_rx.recv() {
            let num_snakes = (obs.len() - total_preys * PREY_OBS_SIZE) / SNAKE_OBS_SIZE;

            let mut payload = vec![0u8; obs.len() * 4];
            for (i, &val) in obs.iter().enumerate() {
                payload[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
            if stream.write_all(&payload).is_err() {
                break;
            }

            let mut action_bytes = vec![0u8; (num_snakes + total_preys) * 4];
            if stream.read_exact(&mut action_bytes).is_err() {
                break;
            }

            let mut actions = Vec::with_capacity(num_snakes + total_preys);
            for s in 0..(num_snakes + total_preys) {
                let off = s * 4;
                actions.push(i32::from_le_bytes(action_bytes[off..off + 4].try_into().unwrap()));
            }
            if act_tx.send(actions).is_err() {
                break;
            }
        }
    });

    AiWorkerHandle { obs_tx, act_rx, awaiting: false }
}

/// Turns a model path like `models/v1.zip` into the short `v1` shown in the UI.
fn model_display_name(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.strip_suffix(".zip").unwrap_or(file).to_string()
}

/// Builds one label per snake describing who controls it. In `--ai` mode this
/// mirrors the model-to-snake assignment in `learner/play.py` (1 model =
/// replicated to every snake, otherwise one model per snake in order); outside
/// `--ai` mode there is no manual snake control anymore (keyboard/mouse only
/// pan and zoom the camera), so every snake is uncontrolled.
fn controller_labels(args: &[String], num_snakes: usize, is_ai: bool) -> Vec<String> {
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

fn setup(
    mut commands: Commands,
    engine: Res<GameEngine>,
    mut status: ResMut<AppStatus>,
    mut images: ResMut<Assets<Image>>,
) {
    // Start zoomed out so a good chunk of the (now much larger) field is
    // framed on load; the field is centered on the origin so the default
    // transform needs no offset. `camera_control` handles pan/zoom from here.
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: INITIAL_ZOOM,
            ..OrthographicProjection::default_2d()
        }),
    ));

    spawn_map(&mut commands, &engine.0, &mut images);

    let args: Vec<String> = std::env::args().collect();
    let is_ai = args.iter().any(|arg| arg == "--ai");
    let num_snakes = engine.0.snakes.len();
    let labels = controller_labels(&args, num_snakes, is_ai);

    // Header: which actor is controlled by what, drawn top-left. Each snake's
    // line is coloured to match its head so the mapping is unambiguous.
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
            for (i, label) in labels.iter().enumerate() {
                parent.spawn((
                    Text::new(format!("Snake {i}  —  {label}")),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(snake_color(i, num_snakes)),
                ));
            }
            // Status / loading line, updated by `update_status_text`.
            parent.spawn((
                Text::new(""),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::srgb(0.9, 0.9, 0.2)),
                StatusText,
            ));
        });

    if is_ai {
        spawn_ai_server(&mut commands, &args, num_snakes, &mut status);
    }
}

/// Spawns the Python inference server as a child process and wires up the
/// resources `poll_ai_connection` uses to connect to it and (if it dies
/// early) surface its error. Used both at startup and to restart a crashed
/// AI match when the player presses Space.
fn spawn_ai_server(
    commands: &mut Commands,
    args: &[String],
    num_snakes: usize,
    status: &mut AppStatus,
) {
    let mut model_paths = Vec::new();
    let mut prey_model_paths = Vec::new();
    let mut amphibia_model_paths = Vec::new();
    let mut num_preys = 1;
    let mut num_amphibias = 0;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_paths.push(args[i + 1].clone());
            i += 1;
        } else if args[i] == "--prey-model" && i + 1 < args.len() {
            prey_model_paths.push(args[i + 1].clone());
            i += 1;
        } else if args[i] == "--amphibia-model" && i + 1 < args.len() {
            amphibia_model_paths.push(args[i + 1].clone());
            i += 1;
        } else if args[i] == "--preys" && i + 1 < args.len() {
            if let Ok(n) = args[i + 1].parse::<usize>() {
                num_preys = n;
            }
            i += 1;
        } else if args[i] == "--amphibias" && i + 1 < args.len() {
            if let Ok(n) = args[i + 1].parse::<usize>() {
                num_amphibias = n;
            }
            i += 1;
        }
        i += 1;
    }

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    println!("Spawning AI inference server on port {} with {} snakes, {} preys, and {} amphibias...", port, num_snakes, num_preys, num_amphibias);

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let learner_dir = format!("{}/../learner", manifest_dir);

    let mut cmd = std::process::Command::new("uv");
    cmd.args(["run", "python", "-m", "learner.play", "--port", &port.to_string(), "--snakes", &num_snakes.to_string(), "--preys", &num_preys.to_string(), "--amphibias", &num_amphibias.to_string()])
       .current_dir(learner_dir)
       .env("PYTHONPATH", "src");

    for m in model_paths {
        cmd.arg("--model");
        cmd.arg(m);
    }

    for pm in prey_model_paths {
        cmd.arg("--prey-model");
        cmd.arg(pm);
    }
    
    for am in amphibia_model_paths {
        cmd.arg("--amphibia-model");
        cmd.arg(am);
    }

    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().expect("Failed to spawn Python AI server");
    let stderr = child.stderr.take().expect("stderr was piped");
    let (stderr_tx, stderr_rx) = crossbeam_channel::unbounded();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("{line}");
            if stderr_tx.send(line).is_err() {
                break;
            }
        }
    });

    commands.insert_resource(AiServerProcess {
        child: std::sync::Mutex::new(child),
        stderr_rx,
    });

    // Connect asynchronously across frames (see `poll_ai_connection`) so the
    // window keeps rendering the loading text instead of freezing while the
    // Python server imports torch and loads its models.
    commands.insert_resource(PendingConnection {
        port,
        retry: Timer::from_seconds(0.2, TimerMode::Repeating),
        elapsed: 0.0,
        timeout: 60.0,
        stderr_lines: Vec::new(),
    });
    *status = AppStatus::Loading("Starting AI inference server…".to_string());
}

/// While a [`PendingConnection`] exists, retry connecting to the Python server
/// once per timer tick without blocking the render loop, updating the loading
/// message with elapsed time and giving up (→ `Failed`) after the timeout.
fn poll_ai_connection(
    time: Res<Time>,
    pending: Option<ResMut<PendingConnection>>,
    ai_server: Option<Res<AiServerProcess>>,
    mut ai_worker: ResMut<AiWorker>,
    mut status: ResMut<AppStatus>,
    mut commands: Commands,
    engine: Res<GameEngine>,
) {
    let Some(mut pending) = pending else { return };

    if let Some(ai_server) = &ai_server {
        while let Ok(line) = ai_server.stderr_rx.try_recv() {
            pending.stderr_lines.push(line);
        }

        let exited = ai_server.child.lock().ok().and_then(|mut c| c.try_wait().ok().flatten());
        if let Some(exit_status) = exited {
            let detail = pending
                .stderr_lines
                .last()
                .cloned()
                .unwrap_or_else(|| format!("exited with {exit_status}"));
            *status = AppStatus::Failed(format!("AI inference server exited: {detail}"));
            commands.remove_resource::<PendingConnection>();
            return;
        }
    }

    pending.elapsed += time.delta_secs();
    if !pending.retry.tick(time.delta()).just_finished() {
        return;
    }

    match TcpStream::connect(("127.0.0.1", pending.port)) {
        Ok(stream) => {
            stream.set_nodelay(true).ok();
            println!("Connected to AI inference server!");
            let num_preys = engine.0.preys.len();
            ai_worker.0 = Some(spawn_ai_worker(stream, num_preys));
            *status = AppStatus::Running;
            commands.remove_resource::<PendingConnection>();
        }
        Err(_) => {
            if pending.elapsed >= pending.timeout {
                *status = AppStatus::Failed(
                    "Could not connect to AI inference server (timed out)".to_string(),
                );
                commands.remove_resource::<PendingConnection>();
            } else {
                *status = AppStatus::Loading(format!(
                    "Waiting for AI server to load models… {:.0}s",
                    pending.elapsed
                ));
            }
        }
    }
}

fn keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut engine: ResMut<GameEngine>,
    mut dirty: ResMut<RenderDirty>,
    mut commands: Commands,
    map_query: Query<Entity, With<MapTile>>,
    mut status: ResMut<AppStatus>,
    ai_server: Option<Res<AiServerProcess>>,
    mut images: ResMut<Assets<Image>>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        // Restart the game — works whether it's still running, over, or the
        // AI server failed.
        let num_snakes = engine.0.snakes.len();
        let num_amphibias = engine.0.preys.iter().filter(|p| p.species == Species::Amphibia).count();
        let num_preys = engine.0.preys.len() - num_amphibias;
        engine.0 = GameState::new(GRID_WIDTH, GRID_HEIGHT, num_snakes, num_preys, num_amphibias);

        // Despawn old map and spawn new map
        for entity in map_query.iter() {
            commands.entity(entity).despawn();
        }
        spawn_map(&mut commands, &engine.0, &mut images);
        dirty.0 = true;

        let args: Vec<String> = std::env::args().collect();
        let is_ai = args.iter().any(|arg| arg == "--ai");
        let needs_ai_respawn =
            is_ai && (ai_server.is_none() || matches!(*status, AppStatus::Failed(_)));

        if needs_ai_respawn {
            // Drop any previous AI server (kills it via `Drop`) and spawn a
            // fresh one so a crashed match can be retried without relaunching
            // the whole game.
            commands.remove_resource::<AiServerProcess>();
            commands.remove_resource::<PendingConnection>();
            spawn_ai_server(&mut commands, &args, num_snakes, &mut status);
        } else if !is_ai {
            *status = AppStatus::Running;
        }
        // Else: AI server already connected and healthy — leave it running,
        // it'll pick up the freshly reset game state on the next tick.
    }
}

/// Pans and zooms the camera every frame. WASD/Arrow keys pan (scaled by the
/// current zoom so it feels constant on screen at any zoom level); the mouse
/// wheel zooms in/out by adjusting the orthographic projection's `scale`.
fn camera_control(
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

fn spawn_particles_for_dead_preys(commands: &mut Commands, state: &GameState, prev: &PrevPositions) {
    let offset_x = (GRID_WIDTH as f32 * TILE_SIZE) / 2.0;
    let offset_y = (GRID_HEIGHT as f32 * TILE_SIZE) / 2.0;
    for (p_idx, &died) in state.prey_died_this_tick.iter().enumerate() {
        if died {
            if let Some(pos) = prev.prey_pos.get(p_idx) {
                let origin = Vec3::new(pos.0 * TILE_SIZE - offset_x, pos.1 * TILE_SIZE - offset_y, 1.0);
                for _ in 0..15 {
                    let angle = rand::random::<f32>() * std::f32::consts::TAU;
                    let speed = rand::random::<f32>() * 60.0 + 20.0;
                    let velocity = Vec2::new(angle.cos() * speed, angle.sin() * speed);
                    commands.spawn((
                        Sprite {
                            color: Color::srgba(1.0, 0.0, 0.0, 0.8),
                            custom_size: Some(Vec2::new(TILE_SIZE * 0.6, TILE_SIZE * 0.6)),
                            ..default()
                        },
                        Transform::from_translation(origin),
                        Particle {
                            velocity,
                            lifetime: Timer::from_seconds(0.5, TimerMode::Once),
                        },
                    ));
                }
            }
        }
    }
}

fn game_tick(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<TickTimer>,
    mut engine: ResMut<GameEngine>,
    mut ai_worker: ResMut<AiWorker>,
    status: Res<AppStatus>,
    mut dirty: ResMut<RenderDirty>,
    mut prev: ResMut<PrevPositions>,
) {
    // Don't advance the game while still loading or after a fatal error.
    if !matches!(*status, AppStatus::Running) {
        return;
    }

    if engine.0.game_over {
        // Drain any late AI response so a restart begins from a clean state.
        if let Some(worker) = &mut ai_worker.0 {
            while worker.act_rx.try_recv().is_ok() {}
            worker.awaiting = false;
        }
        return;
    }

    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }

    // In AI mode the socket round-trip runs on the worker thread. Here we only
    // poll for its result without blocking: if the actions for this tick aren't
    // ready yet we simply skip stepping this tick and try again next one, so a
    // slow inference slows the game slightly but never drops a render frame.
    if let Some(worker) = &mut ai_worker.0 {
        if !worker.awaiting {
            // Prime the first request; step once its actions come back.
            if worker.obs_tx.send(gather_observations(&engine.0)).is_err() {
                eprintln!("AI worker thread stopped");
                std::process::exit(1);
            }
            worker.awaiting = true;
            return;
        }

        match worker.act_rx.try_recv() {
            Ok(actions) => {
                let num_snakes = engine.0.snakes.len();
                for s in 0..num_snakes {
                    if let Some(&a) = actions.get(s) {
                        let rel = RelativeAction::from_usize(a as usize);
                        let dir = rel.to_absolute_direction(engine.0.snakes[s].direction);
                        engine.0.set_direction(s, dir);
                    }
                }
                let prey_actions: Vec<usize> = actions[num_snakes..]
                    .iter()
                    .map(|&a| a as usize)
                    .collect();
                // Snapshot pre-step positions for `render_sync`'s Interp setup,
                // so sprites can glide from here to their new tick position.
                prev.snake_bodies = engine.0.snakes.iter().map(|s| s.body.clone()).collect();
                prev.prey_pos = engine.0.preys.iter().map(|p| p.pos).collect();
                engine.0.step(1.0, &prey_actions);
                spawn_particles_for_dead_preys(&mut commands, &engine.0, &prev);
                // The engine no longer respawns eaten prey inside `step()`
                // (the trainer captures their terminal observation first); the
                // visualizer just respawns them immediately every tick.
                engine.0.respawn_dead_preys();
                worker.awaiting = false;
            }
            Err(TryRecvError::Empty) => return, // not ready; keep rendering, retry next tick
            Err(TryRecvError::Disconnected) => {
                eprintln!("AI worker thread stopped");
                std::process::exit(1);
            }
        }
    } else {
        let num_preys = engine.0.preys.len();
        let prey_actions = vec![0; num_preys];
        prev.snake_bodies = engine.0.snakes.iter().map(|s| s.body.clone()).collect();
        prev.prey_pos = engine.0.preys.iter().map(|p| p.pos).collect();
        engine.0.step(1.0, &prey_actions);
        spawn_particles_for_dead_preys(&mut commands, &engine.0, &prev);
        engine.0.respawn_dead_preys();
    }

    // The engine no longer ends the game itself on death (it respawns dead
    // snakes in place so training episodes aren't truncated across the whole
    // game). For the visualizer/manual-play we still want a clear "game over,
    // press Space to restart" moment, so detect any death here and freeze.
    let alive_count = engine.0.snakes.iter().filter(|s| !s.is_dead).count();
    let is_over = alive_count == 0;
    if is_over {
        engine.0.game_over = true;
    }
    dirty.0 = true;

    // Queue the next inference request now so it overlaps the render frames
    // until the next tick, keeping the game stepping at the full tick rate.
    if !engine.0.game_over {
        if let Some(worker) = &mut ai_worker.0 {
            if worker.obs_tx.send(gather_observations(&engine.0)).is_err() {
                eprintln!("AI worker thread stopped");
                std::process::exit(1);
            }
            worker.awaiting = true;
        }
    }
}

/// Mirrors [`AppStatus`] onto the on-screen [`StatusText`] line.
fn update_status_text(
    status: Res<AppStatus>,
    mut query: Query<(&mut Text, &mut TextColor), With<StatusText>>,
) {
    if !status.is_changed() {
        return;
    }

    let (msg, color) = match &*status {
        AppStatus::Loading(m) => (m.clone(), Color::srgb(0.9, 0.9, 0.2)),
        AppStatus::Running => (String::new(), Color::srgb(0.9, 0.9, 0.2)),
        AppStatus::Failed(m) => (format!("ERROR: {m}"), Color::srgb(1.0, 0.3, 0.3)),
    };

    for (mut text, mut text_color) in &mut query {
        text.0 = msg.clone();
        text_color.0 = color;
    }
}

fn render_sync(
    mut commands: Commands,
    engine: Res<GameEngine>,
    segment_query: Query<Entity, With<SnakeSegment>>,
    apple_query: Query<Entity, With<Apple>>,
    mut dirty: ResMut<RenderDirty>,
    prev: Res<PrevPositions>,
) {
    // Only rebuild sprites on frames where the game state actually changed
    // (a logic tick or a restart); the sprites persist between those frames,
    // so idle frames do no work and the render loop stays at a smooth 60fps.
    if !dirty.0 {
        return;
    }
    dirty.0 = false;

    // 1. Remove old sprites
    for entity in segment_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in apple_query.iter() {
        commands.entity(entity).despawn();
    }

    let offset_x = (GRID_WIDTH as f32 * TILE_SIZE) / 2.0;
    let offset_y = (GRID_HEIGHT as f32 * TILE_SIZE) / 2.0;

    // 2. Draw Preys
    for (p_idx, prey) in engine.0.preys.iter().enumerate() {
        if !prey.is_dead {
            let prey_pos = prey.pos;
            let color = match prey.species {
                Species::Amphibia => Color::srgb(0.0, 1.0, 1.0), // Cyan for Amphibia
                _ => Color::srgb(1.0, 0.5, 0.0), // Orange for Prey
            };
            let to = Vec3::new(
                prey_pos.0 * TILE_SIZE - offset_x,
                prey_pos.1 * TILE_SIZE - offset_y,
                0.0,
            );
            // Interpolate from the previous tick's position, unless the prey
            // teleported this tick (died and respawned elsewhere) or there's
            // no previous position to speak of — then snap instead of
            // streaking a sprite across the whole map.
            let died_this_tick = engine.0.prey_died_this_tick.get(p_idx).copied().unwrap_or(false);
            let from = prev
                .prey_pos
                .get(p_idx)
                .map(|p| Vec3::new(p.0 * TILE_SIZE - offset_x, p.1 * TILE_SIZE - offset_y, 0.0))
                .filter(|from| !died_this_tick && from.distance(to) <= 2.0 * TILE_SIZE)
                .unwrap_or(to);
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::new(TILE_SIZE, TILE_SIZE)),
                    ..default()
                },
                Transform::from_translation(from),
                Apple,
                Interp { from, to },
            ));
        }
    }

    // 3. Draw Snake Bodies
    let num_snakes = engine.0.snakes.len();
    for (s_idx, snake) in engine.0.snakes.iter().enumerate() {
        for (i, pos) in snake.body.iter().enumerate() {
            let color = if snake.is_dead {
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

            let to = Vec3::new(
                pos.0 as f32 * TILE_SIZE - offset_x,
                pos.1 as f32 * TILE_SIZE - offset_y,
                0.0,
            );
            // Index-aligned with the previous tick's body: a forward-moving
            // segment slides smoothly from its old cell to its new one. A
            // freshly grown tail segment (no previous entry at this index)
            // has no "from" to speak of, so it just appears in place.
            let from = prev
                .snake_bodies
                .get(s_idx)
                .and_then(|body| body.get(i))
                .map(|p| Vec3::new(p.0 as f32 * TILE_SIZE - offset_x, p.1 as f32 * TILE_SIZE - offset_y, 0.0))
                .unwrap_or(to);

            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::new(TILE_SIZE, TILE_SIZE)),
                    ..default()
                },
                Transform::from_translation(from),
                SnakeSegment,
                Interp { from, to },
            ));
        }
    }
}

/// Runs every frame (unlike `render_sync`, which only rebuilds sprites on
/// logic ticks) to glide each `Interp`-tagged sprite from its previous tick's
/// position to its current one, driven by how far we are into the current
/// tick interval. This is purely visual — the engine itself only ever holds
/// discrete per-tick positions.
fn apply_interpolation(timer: Res<TickTimer>, mut query: Query<(&Interp, &mut Transform)>) {
    let a = timer.0.fraction().clamp(0.0, 1.0);
    for (interp, mut transform) in &mut query {
        transform.translation = interp.from.lerp(interp.to, a);
    }
}
