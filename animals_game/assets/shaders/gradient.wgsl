#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var<uniform> params: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = mesh.uv;
    let time = params.x;
    
    // Flowing green gradient
    let t = time * 2.0;
    
    let wave = sin(uv.x * 10.0 + t) * 0.5 + 0.5;
    let wave2 = cos(uv.y * 8.0 - t) * 0.5 + 0.5;
    let mix_factor = (wave + wave2 + uv.y) / 3.0;
    
    // Dark green to light green
    let color1 = vec4<f32>(0.0, 0.3, 0.0, 1.0);
    let color2 = vec4<f32>(0.2, 1.0, 0.4, 1.0);
    
    return mix(color1, color2, mix_factor);
}
