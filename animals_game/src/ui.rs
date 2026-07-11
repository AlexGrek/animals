use bevy::prelude::*;
use crate::components::*;
use crate::constants::*;
use crate::resources::*;

/// Mirrors `AppStatus` onto the on-screen `StatusText` line.
pub fn update_status_text(
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

pub fn toolbar_interaction(
    mut interaction_query: Query<
        (&Interaction, &ToolbarButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut settings: ResMut<OverlaySettings>,
) {
    for (interaction, button) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            match button {
                ToolbarButton::ToggleNames => settings.show_names = !settings.show_names,
                ToolbarButton::ToggleTargets => settings.show_targets = !settings.show_targets,
                ToolbarButton::ToggleNn => settings.show_nn = !settings.show_nn,
                ToolbarButton::ToggleModels => settings.show_models = !settings.show_models,
            }
        }
    }
}

pub fn toolbar_colors(
    mut query: Query<(&Interaction, &mut BackgroundColor, &ToolbarButton), With<Button>>,
    settings: Res<OverlaySettings>,
) {
    for (interaction, mut color, button) in &mut query {
        let is_active = match button {
            ToolbarButton::ToggleNames => settings.show_names,
            ToolbarButton::ToggleTargets => settings.show_targets,
            ToolbarButton::ToggleNn => settings.show_nn,
            ToolbarButton::ToggleModels => settings.show_models,
        };

        match *interaction {
            Interaction::Pressed => *color = BackgroundColor(Color::srgb(0.4, 0.4, 0.4)),
            Interaction::Hovered => *color = BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
            Interaction::None => {
                if is_active {
                    *color = BackgroundColor(Color::srgb(0.25, 0.45, 0.25));
                } else {
                    *color = BackgroundColor(Color::srgb(0.2, 0.2, 0.2));
                }
            }
        }
    }
}

pub fn speed_control_interaction(
    mut interaction_query: Query<
        (&Interaction, &SpeedButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut speed: ResMut<GameSpeed>,
) {
    for (interaction, button) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            speed.0 = button.0;
        }
    }
}

pub fn speed_control_colors(
    mut query: Query<(&Interaction, &mut BackgroundColor, &SpeedButton), With<Button>>,
    speed: Res<GameSpeed>,
) {
    for (interaction, mut color, button) in &mut query {
        let is_active = speed.0 == button.0;

        match *interaction {
            Interaction::Pressed => *color = BackgroundColor(Color::srgb(0.4, 0.4, 0.4)),
            Interaction::Hovered => *color = BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
            Interaction::None => {
                if is_active {
                    *color = BackgroundColor(Color::srgb(0.25, 0.45, 0.25));
                } else {
                    *color = BackgroundColor(Color::srgb(0.2, 0.2, 0.2));
                }
            }
        }
    }
}

/// Specs for each NN-overlay layer: (label, layer id, cell count, columns, cell px).
pub const NN_LAYER_SPECS: [(&str, NnLayer, usize, usize, f32); 7] = [
    ("input: entity 8x8", NnLayer::InputEntity, 64, 8, 12.0),
    ("input: grass 8x8", NnLayer::InputGrass, 64, 8, 12.0),
    ("scalars (smell x/y, dist, hunger, len)", NnLayer::Scalars, 5, 5, 16.0),
    ("cnn features (128)", NnLayer::Features, NN_FEATURES, 16, 9.0),
    ("policy layer 1 - tanh (256)", NnLayer::Pi0, NN_PI0, 16, 8.0),
    ("policy layer 2 - tanh (256)", NnLayer::Pi1, NN_PI1, 16, 8.0),
    ("action logits (straight/right/left)", NnLayer::Action, NN_LOGITS, 3, 22.0),
];

/// Spawns the NN overlay panel once (top-right). Cells are recolored every tick
/// by `update_nn_overlay`; the panel is shown/hidden via its `Node.display`.
pub fn spawn_nn_overlay(commands: &mut Commands) {
    commands
        .spawn((
            NnPanel,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                right: Val::Px(8.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                padding: UiRect::all(Val::Px(6.0)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Neural net"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
            ));
            for (label, layer, count, cols, cell) in NN_LAYER_SPECS {
                parent.spawn((
                    Text::new(label),
                    TextFont { font_size: 10.0, ..default() },
                    TextColor(Color::srgb(0.7, 0.7, 0.7)),
                ));
                parent
                    .spawn(Node {
                        width: Val::Px(cols as f32 * (cell + 1.0) + 2.0),
                        flex_direction: FlexDirection::Row,
                        flex_wrap: FlexWrap::Wrap,
                        ..default()
                    })
                    .with_children(|grid| {
                        for i in 0..count {
                            grid.spawn((
                                Node {
                                    width: Val::Px(cell),
                                    height: Val::Px(cell),
                                    margin: UiRect::all(Val::Px(0.5)),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.05, 0.05, 0.05)),
                                NnCell { layer, index: i },
                            ));
                        }
                    });
            }
            parent.spawn((
                Text::new(""),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgb(1.0, 0.8, 0.2)),
                NnActionText,
            ));
        });
}

/// Recolors the NN overlay cells each frame from the selected snake's local
/// observation (input layers) and the streamed activation buffer (hidden layers
/// + action logits), and toggles the panel's visibility.
pub fn update_nn_overlay(
    selected: Res<SelectedSnake>,
    settings: Res<OverlaySettings>,
    engine: Res<GameEngine>,
    buffer: Res<ActivationBuffer>,
    mut panel_q: Query<&mut Node, With<NnPanel>>,
    mut cell_q: Query<(&NnCell, &mut BackgroundColor)>,
    mut action_text_q: Query<&mut Text, With<NnActionText>>,
) {
    let sel = selected.0.filter(|&i| i < engine.0.snakes.len());
    let show = sel.is_some() && settings.show_nn;
    for mut node in &mut panel_q {
        node.display = if show { Display::Flex } else { Display::None };
    }
    let Some(sel) = sel else { return };
    if !settings.show_nn {
        return;
    }

    // Input layers are recomputed locally (not sent over the wire).
    let obs = engine.0.get_relative_observation(sel);
    let entity = &obs[0..64];
    let scalars = &obs[64..69];
    let grass = &obs[69..133];

    // Hidden-layer activations + logits come from the Python worker.
    let empty: [f32; 0] = [];
    let have = buffer.0.len() >= NN_ACT_LEN;
    let features: &[f32] = if have { &buffer.0[0..NN_FEATURES] } else { &empty[..] };
    let pi0: &[f32] = if have { &buffer.0[NN_FEATURES..NN_FEATURES + NN_PI0] } else { &empty[..] };
    let pi1: &[f32] = if have {
        &buffer.0[NN_FEATURES + NN_PI0..NN_FEATURES + NN_PI0 + NN_PI1]
    } else {
        &empty[..]
    };
    let logits: &[f32] = if have { &buffer.0[NN_FEATURES + NN_PI0 + NN_PI1..NN_ACT_LEN] } else { &empty[..] };

    // Softmax the logits for the action bars / readout.
    let mut probs = [0.0f32; NN_LOGITS];
    if have {
        let m = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for k in 0..NN_LOGITS {
            probs[k] = (logits[k] - m).exp();
            sum += probs[k];
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    use crate::utils::{activation_color, max_abs};

    let ma_entity = max_abs(entity);
    let ma_grass = max_abs(grass);
    let ma_scalars = max_abs(scalars);
    let ma_features = max_abs(features);
    let ma_pi0 = max_abs(pi0);
    let ma_pi1 = max_abs(pi1);

    for (cell, mut bg) in &mut cell_q {
        let v = match cell.layer {
            NnLayer::InputEntity => entity.get(cell.index).copied().unwrap_or(0.0) / ma_entity,
            NnLayer::InputGrass => grass.get(cell.index).copied().unwrap_or(0.0) / ma_grass,
            NnLayer::Scalars => scalars.get(cell.index).copied().unwrap_or(0.0) / ma_scalars,
            NnLayer::Features => features.get(cell.index).copied().unwrap_or(0.0) / ma_features,
            NnLayer::Pi0 => pi0.get(cell.index).copied().unwrap_or(0.0) / ma_pi0,
            NnLayer::Pi1 => pi1.get(cell.index).copied().unwrap_or(0.0) / ma_pi1,
            // Action cells show softmax probability (already 0..1).
            NnLayer::Action => probs.get(cell.index).copied().unwrap_or(0.0),
        };
        *bg = BackgroundColor(activation_color(v));
    }

    let names = ["Straight", "Right", "Left"];
    let text = if have {
        let mut argmax = 0;
        for k in 1..NN_LOGITS {
            if probs[k] > probs[argmax] {
                argmax = k;
            }
        }
        format!("Snake {}: {} ({:.0}%)", sel, names[argmax], probs[argmax] * 100.0)
    } else {
        format!("Snake {}: waiting for inference...", sel)
    };
    for mut t in &mut action_text_q {
        t.0 = text.clone();
    }
}

pub fn update_models_overlay(
    settings: Res<OverlaySettings>,
    mut panel_q: Query<&mut Node, With<ModelNamesOverlay>>,
) {
    if !settings.is_changed() {
        return;
    }
    let show = settings.show_models;
    for mut node in &mut panel_q {
        node.display = if show { Display::Flex } else { Display::None };
    }
}
