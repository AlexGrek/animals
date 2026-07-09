use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::AsBindGroup,
    sprite_render::{Material2d, Material2dPlugin},
    shader::ShaderRef,
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
    let window = windows.single().unwrap();
    let width = window.resolution.width();
    let height = window.resolution.height();

    commands.spawn(Camera2d);

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(width, height))),
        MeshMaterial2d(materials.add(GradientMaterial {
            params: Vec4::new(0.0, 0.0, 0.0, 0.0),
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

fn update_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<GradientMaterial>>,
) {
    for (_, material) in materials.iter_mut() {
        material.params.x = time.elapsed_secs();
    }
}
