#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

in uint z6_y6_x6_edge1_edge2_edge3[];

uniform ivec4 block_index;

uniform bool short_range_ambient;
uniform bool long_range_ambient;
uniform float period;
uniform int octaves;
uniform float octaves_decay;
uniform vec2 warp_params;

out vec3 position;
out vec3 normal;
out float ambient_occlusion;

#include "noise_layer.h"
#include "marching_cubes_common.h"
#include "metaball_vertex_common.h"

void createVertex(vec3 vertex)
{
    ambient_occlusion = ambientOcclusion(
        vertex,
        vertex * block_index.w + block_index.xyz * block_size,
        block_index.w,
        short_range_ambient, long_range_ambient);

    // Map vertices to range [0, 1]
    position = vertex / block_size;

    normal = normalAtVertex(vertex);
    EmitVertex();
}

void main() {
    uint packed_triangle = z6_y6_x6_edge1_edge2_edge3[0];
    ivec3 edge_index = ivec3((packed_triangle >> 0) & 0xF,
                             (packed_triangle >> 4) & 0xF,
                             (packed_triangle >> 8) & 0xF);
    ivec3 coords = ivec3((packed_triangle >> 12) & 0x3F,
                         (packed_triangle >> 18) & 0x3F,
                         (packed_triangle >> 24) & 0x3F);

    // the same here to ensure the zero is between positive and negative value 
    // t = d1 / (d1 - d2)
    vec3 d1 = vec3(density(coords + edge_start[edge_index.x]),
                   density(coords + edge_start[edge_index.y]),
                   density(coords + edge_start[edge_index.z]));
    vec3 d2 = vec3(density(coords + edge_end[edge_index.x]),
                   density(coords + edge_end[edge_index.y]),
                   density(coords + edge_end[edge_index.z]));
    vec3 t = d1 / (d1 - d2);

    vec3 v1 = edge_start[edge_index.x] + edge_dir[edge_index.x] * t.x;
    vec3 v2 = edge_start[edge_index.y] + edge_dir[edge_index.y] * t.y;
    vec3 v3 = edge_start[edge_index.z] + edge_dir[edge_index.z] * t.z;

    v1 += coords;
    v2 += coords;
    v3 += coords;

    createVertex(v1);
    createVertex(v2);
    createVertex(v3);

    EndPrimitive();
}
