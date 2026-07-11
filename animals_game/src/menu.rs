use bevy::prelude::*;
use std::fs;
use crate::resources::{AppState, MatchConfig};
use serde::{Serialize, Deserialize};

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuData>()
           .add_systems(OnEnter(AppState::Menu), setup_menu)
           .add_systems(Update, (button_system, menu_action).run_if(in_state(AppState::Menu)))
           .add_systems(OnExit(AppState::Menu), cleanup_menu);
    }
}

#[derive(Resource)]
pub struct MenuData {
    pub available_models: Vec<String>,
    pub snake_models: Vec<usize>, // index into available_models
    pub num_preys: usize,
    pub num_amphibias: usize,
}

impl Default for MenuData {
    fn default() -> Self {
        Self {
            available_models: vec!["None".to_string()],
            snake_models: vec![0, 0],
            num_preys: 1,
            num_amphibias: 0,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SavedMenuConfig {
    snakes: Vec<String>,
    num_preys: usize,
    num_amphibias: usize,
}

fn config_path() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    format!("{}/../.animals_config.json", manifest_dir)
}

fn load_config() -> Option<SavedMenuConfig> {
    if let Ok(data) = fs::read_to_string(config_path()) {
        serde_json::from_str(&data).ok()
    } else {
        None
    }
}

fn save_config(menu_data: &MenuData) {
    let mut config = SavedMenuConfig {
        snakes: Vec::new(),
        num_preys: menu_data.num_preys,
        num_amphibias: menu_data.num_amphibias,
    };
    for &idx in &menu_data.snake_models {
        config.snakes.push(menu_data.available_models[idx].clone());
    }
    if let Ok(data) = serde_json::to_string_pretty(&config) {
        let _ = fs::write(config_path(), data);
    }
}

#[derive(Component)]
struct MenuUi;

#[derive(Component)]
enum MenuAction {
    AddSnake,
    RemoveSnake,
    ChangeSnakeModel(usize, isize),
    IncPrey,
    DecPrey,
    IncAmphibia,
    DecAmphibia,
    StartGame,
}

fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => *color = Color::srgb(0.3, 0.3, 0.3).into(),
            Interaction::Hovered => *color = Color::srgb(0.4, 0.4, 0.4).into(),
            Interaction::None => *color = Color::srgb(0.15, 0.15, 0.15).into(),
        }
    }
}

fn menu_action(
    mut commands: Commands,
    mut interaction_query: Query<(&Interaction, &MenuAction), (Changed<Interaction>, With<Button>)>,
    mut menu_data: ResMut<MenuData>,
    mut app_state: ResMut<NextState<AppState>>,
    mut match_config: ResMut<MatchConfig>,
    ui_query: Query<Entity, With<MenuUi>>,
) {
    let mut changed = false;
    for (interaction, action) in &mut interaction_query {
        if *interaction == Interaction::Pressed {
            match action {
                MenuAction::AddSnake => {
                    let last = *menu_data.snake_models.last().unwrap_or(&0);
                    menu_data.snake_models.push(last);
                    changed = true;
                }
                MenuAction::RemoveSnake => {
                    if menu_data.snake_models.len() > 1 {
                        menu_data.snake_models.pop();
                        changed = true;
                    }
                }
                MenuAction::ChangeSnakeModel(idx, delta) => {
                    let models_len = menu_data.available_models.len() as isize;
                    if let Some(model_idx) = menu_data.snake_models.get_mut(*idx) {
                        *model_idx = ((*model_idx as isize + *delta + models_len) % models_len) as usize;
                        changed = true;
                    }
                }
                MenuAction::IncPrey => {
                    menu_data.num_preys += 1;
                    changed = true;
                }
                MenuAction::DecPrey => {
                    if menu_data.num_preys > 0 {
                        menu_data.num_preys -= 1;
                        changed = true;
                    }
                }
                MenuAction::IncAmphibia => {
                    menu_data.num_amphibias += 1;
                    changed = true;
                }
                MenuAction::DecAmphibia => {
                    if menu_data.num_amphibias > 0 {
                        menu_data.num_amphibias -= 1;
                        changed = true;
                    }
                }
                MenuAction::StartGame => {
                    match_config.is_ai = true;
                    match_config.num_preys = menu_data.num_preys;
                    match_config.num_amphibias = menu_data.num_amphibias;
                    match_config.snakes.clear();
                    for &idx in &menu_data.snake_models {
                        let model_name = &menu_data.available_models[idx];
                        match_config.snakes.push(model_name.clone());
                    }
                    save_config(&menu_data);
                    app_state.set(AppState::InGame);
                    return;
                }
            }
        }
    }

    if changed {
        save_config(&menu_data);
        for entity in ui_query.iter() {
            commands.entity(entity).despawn();
        }
        build_menu_ui(&mut commands, &menu_data);
    }
}

fn setup_menu(mut commands: Commands, mut menu_data: ResMut<MenuData>) {
    let mut models = Vec::new();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let models_dir = format!("{}/../learner/models", manifest_dir);
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                if name.ends_with(".zip") {
                    models.push(name);
                }
            }
        }
    }
    models.sort();
    if models.is_empty() {
        models.push("None".to_string());
    }
    menu_data.available_models = models;
    
    // Select default snake model if available, otherwise just use 0
    let mut default_idx = 0;
    if let Some(idx) = menu_data.available_models.iter().position(|m| m == "snake_model.zip") {
        default_idx = idx;
    }
    
    if let Some(saved) = load_config() {
        menu_data.snake_models.clear();
        for s in saved.snakes {
            if let Some(idx) = menu_data.available_models.iter().position(|m| m == &s) {
                menu_data.snake_models.push(idx);
            } else {
                menu_data.snake_models.push(default_idx);
            }
        }
        if menu_data.snake_models.is_empty() {
            menu_data.snake_models = vec![default_idx, default_idx];
        }
        menu_data.num_preys = saved.num_preys;
        menu_data.num_amphibias = saved.num_amphibias;
    } else {
        for s in &mut menu_data.snake_models {
            *s = default_idx;
        }
    }

    build_menu_ui(&mut commands, &menu_data);
}

fn cleanup_menu(mut commands: Commands, query: Query<Entity, With<MenuUi>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

macro_rules! spawn_text_button {
    ($parent:expr, $text:expr, $action:expr, $width:expr) => {
        $parent.spawn((
            Button,
            Node {
                width: Val::Px($width),
                height: Val::Px(30.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                margin: UiRect::all(Val::Px(5.0)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
            $action,
        )).with_children(|p| {
            p.spawn((
                Text::new($text),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
    };
}

fn build_menu_ui(commands: &mut Commands, menu_data: &MenuData) {
    commands.spawn((
        MenuUi,
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            flex_direction: FlexDirection::Column,
            ..default()
        },
    )).with_children(|parent| {
        parent.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(20.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.8)),
        )).with_children(|panel| {
            panel.spawn((
                Text::new("AI Match Configuration"),
                TextFont { font_size: 30.0, ..default() },
                TextColor(Color::WHITE),
            ));

            panel.spawn(Node { height: Val::Px(20.0), ..default() });

            // Snakes section
            panel.spawn((
                Text::new(format!("Snakes ({})", menu_data.snake_models.len())),
                TextFont { font_size: 20.0, ..default() },
                TextColor(Color::WHITE),
            ));
            
            for (i, &model_idx) in menu_data.snake_models.iter().enumerate() {
                panel.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    ..default()
                }).with_children(|row| {
                    row.spawn((
                        Text::new(format!("Snake {}: ", i)),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::WHITE),
                    ));
                    
                    spawn_text_button!(row, "<", MenuAction::ChangeSnakeModel(i, -1), 30.0);
                    
                    row.spawn(Node {
                        width: Val::Px(200.0),
                        justify_content: JustifyContent::Center,
                        ..default()
                    }).with_children(|m_row| {
                        m_row.spawn((
                            Text::new(&menu_data.available_models[model_idx]),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::srgb(0.5, 0.8, 1.0)),
                        ));
                    });
                    
                    spawn_text_button!(row, ">", MenuAction::ChangeSnakeModel(i, 1), 30.0);
                });
            }

            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                ..default()
            }).with_children(|row| {
                spawn_text_button!(row, "Add Snake", MenuAction::AddSnake, 120.0);
                if menu_data.snake_models.len() > 1 {
                    spawn_text_button!(row, "Remove Snake", MenuAction::RemoveSnake, 120.0);
                }
            });

            panel.spawn(Node { height: Val::Px(20.0), ..default() });

            // Prey section
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                ..default()
            }).with_children(|row| {
                row.spawn((
                    Text::new(format!("Preys: {}", menu_data.num_preys)),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                ));
                spawn_text_button!(row, "-", MenuAction::DecPrey, 30.0);
                spawn_text_button!(row, "+", MenuAction::IncPrey, 30.0);
            });

            // Amphibia section
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                ..default()
            }).with_children(|row| {
                row.spawn((
                    Text::new(format!("Amphibias: {}", menu_data.num_amphibias)),
                    TextFont { font_size: 20.0, ..default() },
                    TextColor(Color::WHITE),
                ));
                spawn_text_button!(row, "-", MenuAction::DecAmphibia, 30.0);
                spawn_text_button!(row, "+", MenuAction::IncAmphibia, 30.0);
            });

            panel.spawn(Node { height: Val::Px(30.0), ..default() });

            spawn_text_button!(panel, "START SIMULATION", MenuAction::StartGame, 250.0);
        });
    });
}
