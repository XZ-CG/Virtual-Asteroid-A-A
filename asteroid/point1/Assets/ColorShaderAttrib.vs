#version 430

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 instance_pos;
layout(location = 2) in vec3 color;

out vec3 world_position;
out vec4 obj_color;

flat out int InstanceID;

void main() {
    world_position = vec3(M * vec4(position + instance_pos, 1.0));
    obj_color = vec4(color, 1.0);
    gl_Position = P * V * vec4(world_position, 1.0);
    InstanceID = gl_InstanceID;
}
