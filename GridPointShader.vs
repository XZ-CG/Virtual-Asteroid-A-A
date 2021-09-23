#version 430

in vec3 position;

void main() {
    gl_Position = vec4(position.x, gl_InstanceID, position.z, 1.0);
}
