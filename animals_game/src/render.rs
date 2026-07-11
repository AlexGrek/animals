use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::asset::RenderAssetUsages;
use animals_engine::GameState;
use animals_engine::map::Terrain;
use animals_engine::species::Species;

use crate::components::*;
use crate::constants::*;
use crate::resources::*;

pub fn spawn_map(commands: &mut Commands, state: &GameState, images: &mut Assets<Image>) {
    let width = GRID_WIDTH as u32;
    let height = GRID_HEIGHT as u32;
    let mut data = vec![0; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let grid_x = x as i32;
            let grid_y = y as i32;
            
            let terrain = state.map.get_terrain(grid_x, grid_y);
            let color = match terrain {
                Terrain::Grass => [35, 81, 40, 255], // srgb(0.14, 0.32, 0.16)
                Terrain::Road => [127, 102, 76, 255], // srgb(0.5, 0.4, 0.3)
                Terrain::Water => [51, 127, 229, 255], // srgb(0.2, 0.5, 0.9)
                Terrain::Rock => [102, 102, 102, 255], // srgb(0.4, 0.4, 0.4)
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

pub fn update_map(
    engine: Res<GameEngine>,
    map_query: Query<&Sprite, With<MapTile>>,
    mut images: ResMut<Assets<Image>>,
) {
    let width = GRID_WIDTH as u32;
    let height = GRID_HEIGHT as u32;

    for sprite in map_query.iter() {
        if let Some(image) = images.get_mut(&sprite.image) {
            for y in 0..height {
                for x in 0..width {
                    let grid_x = x as i32;
                    let grid_y = y as i32;
                    let terrain = engine.0.map.get_terrain(grid_x, grid_y);
                    let color = match terrain {
                        Terrain::Grass => {
                            let health = engine.0.map.grass_health[(grid_y * GRID_WIDTH as i32 + grid_x) as usize];
                            // Health 1.0 -> Green [35, 81, 40], Health 0.0 -> Dirt/Yellow [127, 127, 40]
                            let r = (127.0 - health * (127.0 - 35.0)) as u8;
                            let g = (127.0 - health * (127.0 - 81.0)) as u8;
                            let b = 40;
                            [r, g, b, 255]
                        },
                        Terrain::Road => [127, 102, 76, 255],
                        Terrain::Water => [51, 127, 229, 255],
                        Terrain::Rock => [102, 102, 102, 255],
                    };
                    
                    let idx = ((height - 1 - y) * width + x) as usize * 4;
                    if let Some(data) = &mut image.data {
                        data[idx] = color[0];
                        data[idx + 1] = color[1];
                        data[idx + 2] = color[2];
                        data[idx + 3] = color[3];
                    }
                }
            }
        }
    }
}

pub fn spawn_particles_for_dead_preys(commands: &mut Commands, state: &GameState, prev: &PrevPositions) {
    let offset_x = (GRID_WIDTH as f32 * TILE_SIZE) / 2.0;
    let offset_y = (GRID_HEIGHT as f32 * TILE_SIZE) / 2.0;
    for (p_idx, &died) in state.prey_died_this_tick.iter().enumerate() {
        if died {
            if let Some(pos) = prev.prey_pos.get(p_idx) {
                let is_reproduction = state.preys[p_idx].death_by_reproduction;
                let origin = Vec3::new(pos.0 * TILE_SIZE - offset_x, pos.1 * TILE_SIZE - offset_y, 1.0);
                for i in 0..15 {
                    let angle = rand::random::<f32>() * std::f32::consts::TAU;
                    let speed = rand::random::<f32>() * 150.0 + 50.0;
                    let velocity = Vec2::new(angle.cos() * speed, angle.sin() * speed);
                    let color = if is_reproduction {
                        if i % 2 == 0 { Color::srgba(0.0, 1.0, 0.0, 0.8) } else { Color::srgba(1.0, 1.0, 1.0, 0.8) }
                    } else {
                        Color::srgba(1.0, 0.0, 0.0, 0.8)
                    };
                    commands.spawn((
                        Sprite {
                            color,
                            custom_size: Some(Vec2::new(TILE_SIZE * 0.6, TILE_SIZE * 0.6)),
                            ..default()
                        },
                        Transform::from_translation(origin),
                        Particle {
                            velocity,
                            lifetime: Timer::from_seconds(0.8, TimerMode::Once),
                        },
                    ));
                }
            }
        }
    }
}

pub fn update_particles(
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

pub fn apply_interpolation(timer: Res<TickTimer>, mut query: Query<(&Interp, &mut Transform)>) {
    let a = timer.0.fraction().clamp(0.0, 1.0);
    for (interp, mut transform) in &mut query {
        transform.translation = interp.from.lerp(interp.to, a);
    }
}

pub fn render_sync(
    mut commands: Commands,
    engine: Res<GameEngine>,
    segment_query: Query<Entity, With<SnakeSegment>>,
    apple_query: Query<Entity, With<Apple>>,
    mut dirty: ResMut<RenderDirty>,
    prev: Res<PrevPositions>,
    settings: Res<OverlaySettings>,
) {
    if !dirty.0 {
        return;
    }
    dirty.0 = false;

    for entity in segment_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in apple_query.iter() {
        commands.entity(entity).despawn();
    }

    let offset_x = (GRID_WIDTH as f32 * TILE_SIZE) / 2.0;
    let offset_y = (GRID_HEIGHT as f32 * TILE_SIZE) / 2.0;

    for (p_idx, prey) in engine.0.preys.iter().enumerate() {
        if !prey.is_dead {
            let prey_pos = prey.pos;
            let color = match prey.species {
                Species::Amphibia => Color::srgb(0.0, 0.6, 0.6), // Teal for Amphibia
                _ => Color::srgb(0.5, 0.9, 0.5), // Light Green for Prey
            };
            let to = Vec3::new(
                prey_pos.0 * TILE_SIZE - offset_x,
                prey_pos.1 * TILE_SIZE - offset_y,
                0.0,
            );
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
                Apple { prey_idx: p_idx },
                Interp { from, to },
            ));
        }
    }

    for (s_idx, snake) in engine.0.snakes.iter().enumerate() {
        for (i, pos) in snake.body.iter().enumerate() {
            let color = if snake.is_dead {
                Color::srgb(0.5, 0.5, 0.5) // Gray when dead
            } else if i == 0 {
                Color::srgb(1.0, 0.6, 0.0)
            } else {
                Color::srgb(0.8, 0.4, 0.0)
            };

            let to = Vec3::new(
                pos.0 as f32 * TILE_SIZE - offset_x,
                pos.1 as f32 * TILE_SIZE - offset_y,
                0.0,
            );
            let from = prev
                .snake_bodies
                .get(s_idx)
                .and_then(|body| body.get(i))
                .map(|p| Vec3::new(p.0 as f32 * TILE_SIZE - offset_x, p.1 as f32 * TILE_SIZE - offset_y, 0.0))
                .filter(|from| from.distance(to) <= 2.0 * TILE_SIZE)
                .unwrap_or(to);

            let mut head_cmd = commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::new(TILE_SIZE, TILE_SIZE)),
                    ..default()
                },
                Transform::from_translation(from),
                SnakeSegment,
                Interp { from, to },
            ));
            
            if i == 0 {
                head_cmd.insert(SnakeHead { snake_idx: s_idx });
                if settings.show_names {
                    let short_label = format!("Snake {}", s_idx);
                    head_cmd.with_children(|parent| {
                        parent.spawn((
                            Text2d::new(short_label),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::WHITE),
                            Transform::from_translation(Vec3::new(0.0, TILE_SIZE + 5.0, 1.0)),
                        ));
                    });
                }
            }
        }
    }
}

pub fn draw_targets_overlay(
    mut gizmos: Gizmos,
    settings: Res<OverlaySettings>,
    engine: Res<GameEngine>,
    selected: Res<SelectedSnake>,
    head_query: Query<(&SnakeHead, &Transform)>,
    apple_query: Query<(&Apple, &Transform)>,
) {
    if let Some(sel) = selected.0 {
        for (head, transform) in &head_query {
            if head.snake_idx == sel {
                gizmos.circle_2d(
                    bevy::math::Isometry2d::from_translation(transform.translation.truncate()),
                    TILE_SIZE * 2.0,
                    Color::srgb(1.0, 1.0, 0.0),
                );
            }
        }
    }

    if !settings.show_targets {
        return;
    }

    let mut prey_transforms = std::collections::HashMap::new();
    for (apple, transform) in &apple_query {
        prey_transforms.insert(apple.prey_idx, transform.translation.truncate());
    }

    for (head, transform) in &head_query {
        let snake = &engine.0.snakes[head.snake_idx];
        if snake.is_dead { continue; }
        if let Some(target_idx) = snake.tracked_target {
            if let Some(&prey_pos) = prey_transforms.get(&target_idx) {
                gizmos.line_2d(
                    transform.translation.truncate(),
                    prey_pos,
                    Color::srgba(1.0, 0.0, 0.0, 0.6)
                );
            }
        }
    }
}
