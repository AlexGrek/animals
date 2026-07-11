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
    // Prey / amphibia model choice as an index into a *virtual* list where 0 =
    // "Static (no model)" and 1.. maps to `available_models[idx - 1]`. Static
    // means no `--prey-model`/`--amphibia-model` is passed to the inference
    // server, so those actors default to action 0 (Stand) — i.e. static prey.
    pub prey_model: usize,
    pub amphibia_model: usize,
}

impl Default for MenuData {
    fn default() -> Self {
        Self {
            available_models: vec!["None".to_string()],
            snake_models: vec![0, 0],
            num_preys: 1,
            num_amphibias: 0,
            prey_model: 0,
            amphibia_model: 0,
        }
    }
}

/// Number of selectable choices for a prey/amphibia model slot (Static + every
/// available model).
fn model_choice_count(menu_data: &MenuData) -> usize {
    menu_data.available_models.len() + 1
}

/// Display name for a prey/amphibia model choice index (0 = Static).
fn model_choice_name(menu_data: &MenuData, idx: usize) -> &str {
    if idx == 0 {
        "Static (no model)"
    } else {
        menu_data.available_models[idx - 1].as_str()
    }
}

/// Serializable name for a prey/amphibia model choice: the model filename, or
/// the sentinel "Static" for index 0 (round-trips via `model_choice_from_name`).
fn model_choice_save_name(menu_data: &MenuData, idx: usize) -> String {
    if idx == 0 {
        "Static".to_string()
    } else {
        menu_data.available_models[idx - 1].clone()
    }
}

/// Resolve a saved prey/amphibia model name back to a choice index, defaulting
/// to `default_idx` when the name is unknown and to Static (0) for "Static".
fn model_choice_from_name(menu_data: &MenuData, name: &str, default_idx: usize) -> usize {
    match name {
        "" => default_idx,
        "Static" => 0,
        n => menu_data
            .available_models
            .iter()
            .position(|m| m == n)
            .map(|i| i + 1)
            .unwrap_or(default_idx),
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SavedMenuConfig {
    snakes: Vec<String>,
    num_preys: usize,
    num_amphibias: usize,
    // Model name, or "Static" for no model. `#[serde(default)]` keeps configs
    // written before these fields existed loadable (empty => pick the default).
    #[serde(default)]
    prey_model: String,
    #[serde(default)]
    amphibia_model: String,
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
        prey_model: model_choice_save_name(menu_data, menu_data.prey_model),
        amphibia_model: model_choice_save_name(menu_data, menu_data.amphibia_model),
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
    ChangePreyModel(isize),
    ChangeAmphibiaModel(isize),
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
                MenuAction::ChangePreyModel(delta) => {
                    let n = model_choice_count(&menu_data) as isize;
                    menu_data.prey_model =
                        ((menu_data.prey_model as isize + *delta + n) % n) as usize;
                    changed = true;
                }
                MenuAction::ChangeAmphibiaModel(delta) => {
                    let n = model_choice_count(&menu_data) as isize;
                    menu_data.amphibia_model =
                        ((menu_data.amphibia_model as isize + *delta + n) % n) as usize;
                    changed = true;
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
                    // Prey/amphibia model slots. A Static choice (index 0) leaves
                    // the list empty, so no --prey-model/--amphibia-model reaches
                    // the inference server and those actors stay still (action 0).
                    match_config.prey_models.clear();
                    if menu_data.num_preys > 0 && menu_data.prey_model != 0 {
                        let name = menu_data.available_models[menu_data.prey_model - 1].clone();
                        match_config.prey_models = vec![name; menu_data.num_preys];
                    }
                    match_config.amphibia_models.clear();
                    if menu_data.num_amphibias > 0 && menu_data.amphibia_model != 0 {
                        let name =
                            menu_data.available_models[menu_data.amphibia_model - 1].clone();
                        match_config.amphibia_models = vec![name; menu_data.num_amphibias];
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

    // Default prey/amphibia model choices (as virtual-list indices, 0 = Static):
    // the trained model if present, else Static.
    let prey_default = menu_data
        .available_models
        .iter()
        .position(|m| m == "prey_model.zip")
        .map(|i| i + 1)
        .unwrap_or(0);
    let amphibia_default = menu_data
        .available_models
        .iter()
        .position(|m| m == "amphibia_model.zip")
        .map(|i| i + 1)
        .unwrap_or(0);

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
        menu_data.prey_model = model_choice_from_name(&menu_data, &saved.prey_model, prey_default);
        menu_data.amphibia_model =
            model_choice_from_name(&menu_data, &saved.amphibia_model, amphibia_default);
    } else {
        for s in &mut menu_data.snake_models {
            *s = default_idx;
        }
        menu_data.prey_model = prey_default;
        menu_data.amphibia_model = amphibia_default;
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

            // Prey model selector (Static = stand still / no model)
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                ..default()
            }).with_children(|row| {
                row.spawn((
                    Text::new("Prey model: "),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
                spawn_text_button!(row, "<", MenuAction::ChangePreyModel(-1), 30.0);
                row.spawn(Node {
                    width: Val::Px(200.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                }).with_children(|m_row| {
                    m_row.spawn((
                        Text::new(model_choice_name(menu_data, menu_data.prey_model)),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::srgb(0.5, 0.8, 1.0)),
                    ));
                });
                spawn_text_button!(row, ">", MenuAction::ChangePreyModel(1), 30.0);
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

            // Amphibia model selector (Static = stand still / no model)
            panel.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                ..default()
            }).with_children(|row| {
                row.spawn((
                    Text::new("Amphibia model: "),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::WHITE),
                ));
                spawn_text_button!(row, "<", MenuAction::ChangeAmphibiaModel(-1), 30.0);
                row.spawn(Node {
                    width: Val::Px(200.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                }).with_children(|m_row| {
                    m_row.spawn((
                        Text::new(model_choice_name(menu_data, menu_data.amphibia_model)),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::srgb(0.5, 0.8, 1.0)),
                    ));
                });
                spawn_text_button!(row, ">", MenuAction::ChangeAmphibiaModel(1), 30.0);
            });

            panel.spawn(Node { height: Val::Px(30.0), ..default() });

            spawn_text_button!(panel, "START SIMULATION", MenuAction::StartGame, 250.0);
        });
    });
}
