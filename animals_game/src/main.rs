use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    sprite::{Material2d, Material2dPlugin, MaterialMesh2dBundle},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Material2dPlugin::<GradientMaterial>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, update_time)
        .run();
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct GradientMaterial {
    #[uniform(0)]
    pub params: Vec4,
}

impl Material2d for GradientMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/gradient.wgsl".into()
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GradientMaterial>>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    let width = window.resolution.width();
    let height = window.resolution.height();

    commands.spawn(Camera2dBundle::default());

    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes.add(Rectangle::new(width, height)).into(),
        material: materials.add(GradientMaterial {
            params: Vec4::new(0.0, 0.0, 0.0, 0.0),
        }),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });
}

fn update_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<GradientMaterial>>,
) {
    for (_, material) in materials.iter_mut() {
        material.params.x = time.elapsed_seconds();
    }
}
