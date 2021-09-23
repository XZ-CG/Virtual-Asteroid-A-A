#version 430

layout(points) in;
layout(points, max_vertices = 3) out;

in uint z6_y6_x6_case8[];

out uint z6_y6_x6_edge4;

#include "marching_cubes_common.h"

void main()
{
    uint packed_info = z6_y6_x6_case8[0];
    uint case_index = packed_info & 0xFF;

    uint z6_y6_x6_null4 = (packed_info & uint(~0xFF)) >> 4;

    bool need_0 = false;
    bool need_3 = false;
    bool need_8 = false;

    int numpolys = case_to_numpolys[case_index];
    for (int i = 0; i < numpolys; i++) {
        ivec3 edge_index = edge_connect_list[case_index][i];

        if (edge_index.x == 0 || edge_index.y == 0 || edge_index.z == 0)
            need_0 = true;
        if (edge_index.x == 3 || edge_index.y == 3 || edge_index.z == 3)
            need_3 = true;
        if (edge_index.x == 8 || edge_index.y == 8 || edge_index.z == 8)
            need_8 = true;
    }

    if (need_0) {
        z6_y6_x6_edge4 = z6_y6_x6_null4 | 0;
        EmitVertex();
    }

    if (need_3) {
        z6_y6_x6_edge4 = z6_y6_x6_null4 | 3;
        EmitVertex();
    }

    if (need_8) {
        z6_y6_x6_edge4 = z6_y6_x6_null4 | 8;
        EmitVertex();
    }
}
