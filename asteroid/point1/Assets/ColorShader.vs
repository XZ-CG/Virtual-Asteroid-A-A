#version 430

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

uniform vec4 color;

in vec3 position;

out vec3 world_position;
out vec4 obj_color;

void main() {
    world_position = vec3(M * vec4(position, 1.0));
    obj_color = color;
    gl_Position = P * V * vec4(world_position, 1.0);
}
