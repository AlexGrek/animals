use bevy::prelude::*;
use animals_engine::GameState;
use crossbeam_channel::Receiver;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum AppState {
    #[default]
    Menu,
    InGame,
}

#[derive(Resource, Default)]
pub struct MatchConfig {
    pub is_ai: bool,
    pub snakes: Vec<String>, // List of model paths for each snake slot
    pub prey_models: Vec<String>,
    pub amphibia_models: Vec<String>,
    pub num_preys: usize,
    pub num_amphibias: usize,
}

#[derive(Resource)]
pub struct GameEngine(pub GameState);

#[derive(Resource)]
pub struct TickTimer(pub Timer);

#[derive(Resource, PartialEq, Clone, Copy)]
pub struct GameSpeed(pub f32);

#[derive(Resource)]
pub struct OverlaySettings {
    pub show_names: bool,
    pub show_targets: bool,
    pub show_nn: bool,
    pub show_models: bool,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self { show_names: false, show_targets: false, show_nn: true, show_models: false }
    }
}

/// Index of the snake whose neural net is shown in the overlay (`None` = hidden).
#[derive(Resource, Default)]
pub struct SelectedSnake(pub Option<usize>);

/// Latest activation vector for the selected snake.
#[derive(Resource, Default)]
pub struct ActivationBuffer(pub Vec<f32>);

/// Snapshot of every actor's position as of the START of the most recent
/// `step()` call.
#[derive(Resource, Default)]
pub struct PrevPositions {
    pub snake_bodies: Vec<Vec<(i32, i32)>>,
    pub prey_pos: Vec<(f32, f32)>,
}

#[derive(Resource)]
pub struct StatsTracker {
    pub frames: u32,
    pub last_fps_update: f32,
    pub client_fps: f32,
    
    pub inference_steps: u32,
    pub last_inference_update: f32,
    pub inference_fps: f32,
}

impl Default for StatsTracker {
    fn default() -> Self {
        Self {
            frames: 0,
            last_fps_update: 0.0,
            client_fps: 0.0,
            inference_steps: 0,
            last_inference_update: 0.0,
            inference_fps: 0.0,
        }
    }
}

/// Set whenever the game state changes so `render_sync` only rebuilds sprites
/// on ticks that actually moved the game, instead of every frame.
#[derive(Resource)]
pub struct RenderDirty(pub bool);

/// High-level state of the app, surfaced to the player as on-screen text.
#[derive(Resource, Clone, PartialEq)]
pub enum AppStatus {
    /// Still getting ready (spawning / connecting to the AI server).
    Loading(String),
    /// The game is live and ticking.
    Running,
    /// Something went wrong.
    Failed(String),
}

#[derive(Resource)]
pub struct AiServerProcess {
    pub child: std::sync::Mutex<std::process::Child>,
    pub stderr_rx: Receiver<String>,
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

/// Present only in `--ai` mode while we wait for the Python inference server to connect.
#[derive(Resource)]
pub struct PendingConnection {
    pub port: u16,
    pub retry: Timer,
    pub elapsed: f32,
    pub timeout: f32,
    pub stderr_lines: Vec<String>,
}

/// One tick's response from the Python worker.
pub struct WorkerReply {
    pub actions: Vec<i32>,
    pub activations: Vec<f32>,
}

pub struct AiWorkerHandle {
    pub obs_tx: crossbeam_channel::Sender<(Vec<f32>, usize, usize, usize, i32, Vec<u32>, Vec<u32>, Vec<u32>)>,
    pub act_rx: crossbeam_channel::Receiver<WorkerReply>,
    pub awaiting: bool,
}

#[derive(Resource, Default)]
pub struct AiWorker(pub Option<AiWorkerHandle>);
