use bevy::prelude::*;
use std::io::{Read, Write};
use std::net::TcpStream;
use animals_engine::species::Species;
use animals_engine::GameState;
use crate::resources::*;
use crate::utils::{gather_observations, selected_i32};

/// Spawns the background thread that owns `stream` and services inference
/// requests: it blocks on the socket round-trip (write observations, read
/// actions) so the render thread never has to.
pub fn spawn_ai_worker(mut stream: TcpStream) -> AiWorkerHandle {
    let (obs_tx, obs_rx) = crossbeam_channel::unbounded::<(Vec<f32>, usize, usize, usize, usize, i32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)>();
    let (act_tx, act_rx) = crossbeam_channel::unbounded::<WorkerReply>();

    std::thread::spawn(move || {
        while let Ok((obs, num_snakes, num_preys, num_amphibias, num_corpsefags, selected, family_ids, prey_family_ids, amphibia_family_ids, corpsefag_family_ids)) = obs_rx.recv() {
            let mut payload = vec![0u8; 20 + (num_snakes + num_preys + num_amphibias + num_corpsefags) * 4 + obs.len() * 4];
            payload[0..4].copy_from_slice(&(num_snakes as i32).to_le_bytes());
            payload[4..8].copy_from_slice(&(num_preys as i32).to_le_bytes());
            payload[8..12].copy_from_slice(&(num_amphibias as i32).to_le_bytes());
            payload[12..16].copy_from_slice(&(num_corpsefags as i32).to_le_bytes());
            payload[16..20].copy_from_slice(&selected.to_le_bytes());

            let mut offset = 20;
            for s in 0..num_snakes {
                payload[offset..offset + 4].copy_from_slice(&(family_ids[s] as i32).to_le_bytes());
                offset += 4;
            }
            for p in 0..num_preys {
                payload[offset..offset + 4].copy_from_slice(&(prey_family_ids[p] as i32).to_le_bytes());
                offset += 4;
            }
            for a in 0..num_amphibias {
                payload[offset..offset + 4].copy_from_slice(&(amphibia_family_ids[a] as i32).to_le_bytes());
                offset += 4;
            }
            for c in 0..num_corpsefags {
                payload[offset..offset + 4].copy_from_slice(&(corpsefag_family_ids[c] as i32).to_le_bytes());
                offset += 4;
            }

            for (i, &val) in obs.iter().enumerate() {
                payload[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
            if stream.write_all(&payload).is_err() {
                break;
            }

            let total_preys_sent = num_preys + num_amphibias + num_corpsefags;
            let mut action_bytes = vec![0u8; (num_snakes + total_preys_sent) * 4];
            if stream.read_exact(&mut action_bytes).is_err() {
                break;
            }

            let mut actions = Vec::with_capacity(num_snakes + total_preys_sent);
            for s in 0..(num_snakes + total_preys_sent) {
                let off = s * 4;
                actions.push(i32::from_le_bytes(action_bytes[off..off + 4].try_into().unwrap()));
            }

            // Length-prefixed activation blob for the selected snake (may be 0).
            let mut count_bytes = [0u8; 4];
            if stream.read_exact(&mut count_bytes).is_err() {
                break;
            }
            let count = i32::from_le_bytes(count_bytes).max(0) as usize;
            let mut activations = Vec::with_capacity(count);
            if count > 0 {
                let mut act_f_bytes = vec![0u8; count * 4];
                if stream.read_exact(&mut act_f_bytes).is_err() {
                    break;
                }
                for k in 0..count {
                    let off = k * 4;
                    activations.push(f32::from_le_bytes(act_f_bytes[off..off + 4].try_into().unwrap()));
                }
            }

            if act_tx.send(WorkerReply { actions, activations }).is_err() {
                break;
            }
        }
    });

    AiWorkerHandle { obs_tx, act_rx, awaiting: false }
}

/// Spawns the Python inference server as a child process and wires up the
/// resources `poll_ai_connection` uses to connect to it.
pub fn spawn_ai_server(
    commands: &mut Commands,
    config: &MatchConfig,
    num_snakes: usize,
    status: &mut AppStatus,
) {
    let model_paths = config.snakes.clone();
    let prey_model_paths = config.prey_models.clone();
    let amphibia_model_paths = config.amphibia_models.clone();
    let num_preys = config.num_preys;
    let num_amphibias = config.num_amphibias;
    let num_corpsefags = config.num_corpsefags;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    println!("Spawning AI inference server on port {} with {} snakes, {} preys, and {} amphibias...", port, num_snakes, num_preys, num_amphibias);

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let learner_dir = format!("{}/../learner", manifest_dir);

    let mut cmd = std::process::Command::new("uv");
    cmd.args(["run", "python", "-m", "learner.play", "--port", &port.to_string(), "--snakes", &num_snakes.to_string(), "--preys", &num_preys.to_string(), "--amphibias", &num_amphibias.to_string(), "--corpsefags", &num_corpsefags.to_string()])
       .current_dir(learner_dir.clone())
       .env("PYTHONPATH", "src");

    for m in model_paths {
        cmd.arg("--model");
        cmd.arg(m);
    }

    if let Some(pm) = prey_model_paths.first() {
        cmd.arg("--prey-model");
        cmd.arg(pm);
    }
    
    if let Some(am) = amphibia_model_paths.first() {
        cmd.arg("--amphibia-model");
        cmd.arg(am);
    }
    
    let corpsefag_model_paths = &config.corpsefag_models;
    if let Some(cm) = corpsefag_model_paths.first() {
        cmd.arg("--corpsefag-model");
        cmd.arg(cm);
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

    commands.insert_resource(PendingConnection {
        port,
        retry: Timer::from_seconds(0.2, TimerMode::Repeating),
        elapsed: 0.0,
        timeout: 60.0,
        stderr_lines: Vec::new(),
    });
    *status = AppStatus::Loading("Starting AI inference server...".to_string());
}

/// While a `PendingConnection` exists, retry connecting to the Python server.
pub fn poll_ai_connection(
    time: Res<Time>,
    pending: Option<ResMut<PendingConnection>>,
    ai_server: Option<Res<AiServerProcess>>,
    mut ai_worker: ResMut<AiWorker>,
    mut status: ResMut<AppStatus>,
    mut commands: Commands,
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
            ai_worker.0 = Some(spawn_ai_worker(stream));
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
                    "Waiting for AI server to load models... {:.0}s",
                    pending.elapsed
                ));
            }
        }
    }
}

pub fn queue_ai_inference(engine: &GameState, worker: &mut AiWorkerHandle, selected: &SelectedSnake) {
    let obs = gather_observations(engine);
    let num_snakes = engine.snakes.len();
    let num_amphibias = engine.preys.iter().filter(|p| p.species == Species::Amphibia).count();
    let num_preys = engine.preys.len() - num_amphibias;
    let sel = selected_i32(selected, num_snakes);
    let family_ids: Vec<u32> = engine.snakes.iter().map(|s| s.family_id).collect();
    let prey_family_ids: Vec<u32> = engine.preys.iter().filter(|p| p.species == Species::Prey).map(|p| p.family_id).collect();
    let amphibia_family_ids: Vec<u32> = engine.preys.iter().filter(|p| p.species == Species::Amphibia).map(|p| p.family_id).collect();
    let num_corpsefags = engine.corpsefags.len();
    let corpsefag_family_ids: Vec<u32> = engine.corpsefags.iter().map(|c| c.family_id).collect();
    if worker.obs_tx.send((obs, num_snakes, num_preys, num_amphibias, num_corpsefags, sel, family_ids, prey_family_ids, amphibia_family_ids, corpsefag_family_ids)).is_err() {
        eprintln!("AI worker thread stopped");
        std::process::exit(1);
    }
    worker.awaiting = true;
}
