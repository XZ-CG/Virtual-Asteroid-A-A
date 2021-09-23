#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 15) out;

layout(binding = 7) uniform isampler3D index_lookup;

uniform int texture_size;

in uint z6_y6_x6_case8[];

out int index;

#include "marching_cubes_common.h"

void main()
{
    int packed_info = int(z6_y6_x6_case8[0]);
    int case_index = packed_info & 0xFF;

    int x = (packed_info >> 8) & 0x3F;
    int y = (packed_info >> 14) & 0x3F;
    int z = (packed_info >> 20) & 0x3F;

    int numpolys = case_to_numpolys[case_index];

    for (int i = 0; i < numpolys; i++) {
        ivec3 edge_index = edge_connect_list[case_index][i];

        vec3 edge_coords;

        edge_coords = edge_start[edge_index.x];
        edge_coords = vec3(3 * (x + edge_coords.x) + edge_axis[edge_index.x],
                           y + edge_coords.y,
                           z + edge_coords.z);
        index = texture(index_lookup, edge_coords / texture_size).x;
        EmitVertex();

        edge_coords = edge_start[edge_index.y];
        edge_coords = vec3(3 * (x + edge_coords.x) + edge_axis[edge_index.y],
                           y + edge_coords.y,
                           z + edge_coords.z);
        index = texture(index_lookup, edge_coords / texture_size).x;
        EmitVertex();

        edge_coords = edge_start[edge_index.z];
        edge_coords = vec3(3 * (x + edge_coords.x) + edge_axis[edge_index.z],
                           y + edge_coords.y,
                           z + edge_coords.z);
        index = texture(index_lookup, edge_coords / texture_size).x;
        EmitVertex();

        EndPrimitive();
    }
}
