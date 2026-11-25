#version 450
layout(set = 0, binding = 0) uniform sampler2D ui_texture;
layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 overlay_color;
layout(location = 1) out vec4 coverage;
void main() {
    overlay_color = color * texture(ui_texture, uv);
    coverage = vec4(1.0, 0.0, 0.0, overlay_color.a);
}
