#version 430

uniform int total_items;

layout(r32i, binding = 7) uniform iimage3D index_lookup;

layout(std430, binding = 6) buffer Input0 {
    uint z6_y6_x6_edge4[];
} input;

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

void main() {
    int index = int(gl_GlobalInvocationID.x);
    if (index >= total_items) {
        return;
    }

    int packed_info = int(input.z6_y6_x6_edge4[index]);
    int edge_index = packed_info & 0xF;
    int x = (packed_info >> 4) & 0x3F;
    int y = (packed_info >> 10) & 0x3F;
    int z = (packed_info >> 16) & 0x3F;

    if (edge_index == 3) {
        imageStore(index_lookup, ivec3(x * 3 + 0, y, z), ivec4(index));
    } else if (edge_index == 0) {
        imageStore(index_lookup, ivec3(x * 3 + 1, y, z), ivec4(index));
    } else { // 8
        imageStore(index_lookup, ivec3(x * 3 + 2, y, z), ivec4(index));
    }
}
