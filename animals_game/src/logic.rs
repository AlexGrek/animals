use animals_engine::{GameState, RelativeAction};
use bevy::prelude::*;
use crossbeam_channel::TryRecvError;

use crate::ai::queue_ai_inference;
use crate::components::*;
use crate::constants::*;
use crate::render::{spawn_map, spawn_particles_for_dead_preys, spawn_particles_for_snake_deaths, spawn_particles_for_cf_births, spawn_particles_for_snake_births, spawn_particles_for_cf_eats, spawn_particles_for_egg_eats};
use crate::resources::*;

pub fn keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut engine: ResMut<GameEngine>,
    mut dirty: ResMut<RenderDirty>,
    mut commands: Commands,
    map_query: Query<Entity, With<MapTile>>,
    mut status: ResMut<AppStatus>,
    ai_server: Option<Res<AiServerProcess>>,
    mut images: ResMut<Assets<Image>>,
    mut selected: ResMut<SelectedSnake>,
    config: Res<MatchConfig>,
) {
    let num_snakes = engine.0.snakes.len();
    const DIGIT_KEYS: [(KeyCode, usize); 18] = [
        (KeyCode::Digit1, 0),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3),
        (KeyCode::Digit5, 4),
        (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6),
        (KeyCode::Digit8, 7),
        (KeyCode::Digit9, 8),
        (KeyCode::Numpad1, 0),
        (KeyCode::Numpad2, 1),
        (KeyCode::Numpad3, 2),
        (KeyCode::Numpad4, 3),
        (KeyCode::Numpad5, 4),
        (KeyCode::Numpad6, 5),
        (KeyCode::Numpad7, 6),
        (KeyCode::Numpad8, 7),
        (KeyCode::Numpad9, 8),
    ];
    for (key, idx) in DIGIT_KEYS {
        if keyboard_input.just_pressed(key) && idx < num_snakes {
            selected.0 = Some(idx);
        }
    }
    if keyboard_input.just_pressed(KeyCode::Digit0)
        || keyboard_input.just_pressed(KeyCode::Numpad0)
        || keyboard_input.just_pressed(KeyCode::Escape)
    {
        selected.0 = None;
    }
    if keyboard_input.just_pressed(KeyCode::Tab) && num_snakes > 0 {
        selected.0 = Some(match selected.0 {
            Some(i) => (i + 1) % num_snakes,
            None => 0,
        });
    }

    if keyboard_input.just_pressed(KeyCode::Space) {
        let num_snakes = engine.0.snakes.len();
        let num_preys = config.num_preys;
        let num_amphibias = config.num_amphibias;
        let num_corpsefags = config.num_corpsefags;
        let is_ai = config.is_ai;
        engine.0 = GameState::new(
            GRID_WIDTH,
            GRID_HEIGHT,
            num_snakes,
            num_preys,
            num_preys.max(100),
            num_amphibias,
            num_amphibias.max(100),
            num_corpsefags,
            false,
            !is_ai,
        );

        for entity in map_query.iter() {
            commands.entity(entity).despawn();
        }
        spawn_map(&mut commands, &engine.0, &mut images);
        dirty.0 = true;

        let needs_ai_respawn =
            is_ai && (ai_server.is_none() || matches!(*status, AppStatus::Failed(_)));

        if needs_ai_respawn {
            commands.remove_resource::<AiServerProcess>();
            commands.remove_resource::<PendingConnection>();
            crate::ai::spawn_ai_server(&mut commands, &config, num_snakes, &mut status);
        } else if !is_ai {
            *status = AppStatus::Running;
        }
    }
}

pub fn game_tick(
    mut commands: Commands,
    time: Res<Time>,
    mut timer: ResMut<TickTimer>,
    mut engine: ResMut<GameEngine>,
    mut ai_worker: ResMut<AiWorker>,
    status: Res<AppStatus>,
    mut dirty: ResMut<RenderDirty>,
    mut prev: ResMut<PrevPositions>,
    selected: Res<SelectedSnake>,
    mut act_buffer: ResMut<ActivationBuffer>,
    speed: Res<GameSpeed>,
    mut stats: ResMut<StatsTracker>,
) {
    if matches!(*status, AppStatus::Loading(_) | AppStatus::Failed(_)) {
        return;
    }

    if engine.0.game_over {
        if let Some(worker) = &mut ai_worker.0 {
            while worker.act_rx.try_recv().is_ok() {}
            worker.awaiting = false;
        }
        return;
    }

    if speed.0 == 0.0 {
        return;
    }

    timer.0.tick(time.delta().mul_f32(speed.0));
    let mut ticks = timer.0.times_finished_this_tick();
    if speed.0 >= 100.0 {
        ticks = 100;
    }

    for _ in 0..ticks {
        if let Some(worker) = &mut ai_worker.0 {
            if !worker.awaiting {
                break;
            }

            match worker.act_rx.try_recv() {
                Ok(crate::resources::WorkerReply {
                    actions,
                    activations,
                }) => {
                    act_buffer.0 = activations;
                    let num_snakes = engine.0.snakes.len();
                    for s in 0..num_snakes {
                        if let Some(&a) = actions.get(s) {
                            let rel = RelativeAction::from_usize(a as usize);
                            let dir = rel.to_absolute_direction(engine.0.snakes[s].direction);
                            engine.0.set_direction(s, dir);
                        }
                    }
                    let num_preys = engine.0.preys.len();
                    let prey_actions: Vec<usize> =
                        actions[num_snakes..num_snakes+num_preys].iter().map(|&a| a as usize).collect();
                    let corpsefag_actions: Vec<usize> =
                        actions[num_snakes+num_preys..].iter().map(|&a| a as usize).collect();
                    prev.snake_bodies = engine.0.snakes.iter().map(|s| s.body.clone()).collect();
                    prev.prey_pos = engine.0.preys.iter().map(|p| p.pos).collect();
                    engine.0.step(1.0, &prey_actions, &corpsefag_actions);
                    spawn_particles_for_dead_preys(&mut commands, &engine.0, &prev);
                    spawn_particles_for_snake_deaths(&mut commands, &engine.0);
                    spawn_particles_for_cf_births(&mut commands, &engine.0);
                    spawn_particles_for_snake_births(&mut commands, &engine.0);
                    spawn_particles_for_cf_eats(&mut commands, &engine.0);
                    spawn_particles_for_egg_eats(&mut commands, &engine.0);
                    engine.0.respawn_dead_preys();
                    // Ecosystem model: a dead snake's entity is reaped (no
                    // respawn) but its body is left behind as a static corpse
                    // obstacle. Population is governed by births (mitosis) vs
                    // deaths (hunger/collision), so it self-balances against prey
                    // abundance. game_over below still fires only if the predators
                    // go fully extinct (count hits 0).
                    engine.0.remove_dead_snakes();
                    worker.awaiting = false;
                    stats.inference_steps += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    eprintln!("AI worker thread stopped");
                    std::process::exit(1);
                }
            }
        } else {
            let num_preys = engine.0.preys.len();
            let prey_actions = vec![0; num_preys];
            let num_cf = engine.0.corpsefags.len();
            let cf_actions = vec![0; num_cf];
            prev.snake_bodies = engine.0.snakes.iter().map(|s| s.body.clone()).collect();
            prev.prey_pos = engine.0.preys.iter().map(|p| p.pos).collect();
            engine.0.step(1.0, &prey_actions, &cf_actions);
            spawn_particles_for_dead_preys(&mut commands, &engine.0, &prev);
            spawn_particles_for_snake_deaths(&mut commands, &engine.0);
            spawn_particles_for_cf_births(&mut commands, &engine.0);
            spawn_particles_for_snake_births(&mut commands, &engine.0);
            spawn_particles_for_cf_eats(&mut commands, &engine.0);
            spawn_particles_for_egg_eats(&mut commands, &engine.0);
            engine.0.respawn_dead_preys();
            // Same ecosystem model as AI mode: dead snakes become static corpses
            // and leave the active list (so the player's death still reduces the
            // count to 0 -> game_over, with the corpse left visible on the grid).
            engine.0.remove_dead_snakes();
        }

        let alive_count = engine.0.snakes.iter().filter(|s| !s.is_dead).count();
        if alive_count == 0 {
            engine.0.game_over = true;
            break;
        }

        dirty.0 = true;

        if let Some(worker) = &mut ai_worker.0 {
            if !worker.awaiting && !engine.0.game_over {
                queue_ai_inference(&engine.0, worker, &selected);
            }
        }
    }

    if !engine.0.game_over {
        if let Some(worker) = &mut ai_worker.0 {
            if !worker.awaiting {
                queue_ai_inference(&engine.0, worker, &selected);
            }
        }
    }
}
