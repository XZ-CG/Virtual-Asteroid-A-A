#version 430

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform mat3 NormalMatrix;

uniform bool water_clip;
uniform bool water_reflection_clip;
uniform float clip_height;

in vec3 position;
in vec3 normal;
in float ambient_occlusion;

out vertexData
{
    vec3 position;
    vec3 normal;
    float ambient_occlusion;
} vertex_out;

void main() {
    vec4 world_position = M * vec4(position, 1.0);
    vertex_out.position = vec3(world_position);
    vertex_out.ambient_occlusion = ambient_occlusion;

    if (water_clip) {
        gl_ClipDistance[0] = world_position.y - clip_height;
    } else if (water_reflection_clip) {
        gl_ClipDistance[0] = clip_height - world_position.y;
    }

    vertex_out.normal = NormalMatrix * normal;

    gl_Position = P * V * world_position;
}
