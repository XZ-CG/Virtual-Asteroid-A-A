#version 430

in vertexData
{
    vec3 position;
    vec3 normal;
    float ambient_occlusion;
} vertex_in;

out vec4 fragColor;

layout(binding = 0) uniform sampler3D density_map;
layout(binding = 1) uniform sampler2D x_texture;
layout(binding = 2) uniform sampler2D y_texture;
layout(binding = 3) uniform sampler2D z_texture;
layout(binding = 4) uniform sampler2D x_normal_map;
layout(binding = 5) uniform sampler2D y_normal_map;
layout(binding = 6) uniform sampler2D z_normal_map;

uniform bool triplanar_colors;
uniform bool show_ambient_occlusion;
uniform bool use_ambient;
uniform bool use_normal_map;
uniform bool debug_flag;
uniform vec3 eye_position;

uniform vec3 light_position;
uniform vec3 light_ambient;
uniform vec3 light_diffuse;
uniform vec3 light_specular;

uniform vec3 fog_params;

uniform float alpha;

vec3 calculateBlendWeights()
{
    vec3 blend_weights = abs(normalize(vertex_in.normal));
    // only blend in the neighborhood
    // of 45 degree angles.
    blend_weights = blend_weights - 0.45;
    blend_weights = max(blend_weights, 0.0);
    // ensure weights sum to one.
    blend_weights = blend_weights / (blend_weights.x + blend_weights.y + blend_weights.z);

    return blend_weights;
}

vec3 calculateNormalMap(vec3 texture_normal)
{
    // http://learnopengl.com/#!Advanced-Lighting/Normal-Mapping

    // Need the normal to be in the range [-1, 1];
    texture_normal = normalize(texture_normal * 2 - 1.0);

    return texture_normal;
}

vec3 calculateLight(vec3 normal)
{
    float shininess = 2;

    vec3 L = normalize(light_position - vertex_in.position);
    // position of eye coordinate is (0, 0, 0) is view space
    vec3 E = normalize(eye_position - vertex_in.position);
    vec3 R = normalize(-reflect(L, normal));

    vec3 ambient = light_ambient;

    vec3 diffuse = light_diffuse * max(dot(normal, L), 0.0);
    diffuse = clamp(diffuse, 0.0, 1.0);

    vec3 specular = light_specular * pow(max(dot(R, E), 0.0), shininess);
    specular = clamp(specular, 0.0, 1.0);

    return ambient + diffuse + specular;
}

vec3 normalMapValue(sampler2D map, vec2 pos)
{
    vec3 input = texture(map, pos).rgb;
    return normalize(input - vec3(0.5));
}

void main() {
    // normalize the lengths.
    vec3 normal = normalize(vertex_in.normal);

    vec3 blend_weights = calculateBlendWeights();

    float ambient_occlusion = 1.0;
    if (use_ambient) {
        ambient_occlusion = vertex_in.ambient_occlusion;
    }

    if (debug_flag) {
        ambient_occlusion = ambient_occlusion * 1.00001;
    }

    if (triplanar_colors) {
        fragColor = vec4(blend_weights * ambient_occlusion, 1.0);
    } else {
        mat3 transform;

        vec3 light1, light2, light3;

        vec2 x_uv = vec2(-vertex_in.position.z, -vertex_in.position.y);
        vec2 y_uv = vec2(vertex_in.position.x, vertex_in.position.z);
        vec2 z_uv = vec2(vertex_in.position.x, -vertex_in.position.y);

        if (use_normal_map) {
            {
                vec3 tangent = vec3(0.0, 0.0, -1.0);
                vec3 bitangent = normalize(cross(normal, tangent));
                tangent = normalize(cross(bitangent, normal));
                mat3 TBN = mat3(tangent, bitangent, normal);

                light1 = calculateLight(TBN * normalMapValue(x_normal_map, x_uv).xyz);
            }
            {
                vec3 tangent = vec3(1.0, 0.0, 0.0);
                vec3 bitangent = normalize(cross(tangent, normal));
                tangent = normalize(cross(normal, bitangent));
                mat3 TBN = mat3(tangent, bitangent, normal);

                light2 = calculateLight(TBN * normalMapValue(y_normal_map, y_uv).xyz);
            }
            {
                vec3 tangent = vec3(1.0, 0.0, 0.0);
                vec3 bitangent = normalize(cross(normal, tangent));
                tangent = normalize(cross(bitangent, normal));
                mat3 TBN = mat3(tangent, bitangent, normal);

                light3 = calculateLight(TBN * normalMapValue(z_normal_map, z_uv).xyz);
            }
        } else {
            mat3 transform = mat3(1.0);
            light1 = calculateLight(normal);
            light2 = light1;
            light3 = light1;
        }

        float vertex_distance = length(eye_position - vertex_in.position);
        float fog_falloff = clamp(fog_params.x * vertex_distance / fog_params.y - fog_params.z, 0.0, 1.0);

        vec3 color_x = texture(x_texture, x_uv).xyz * light1;
        vec3 color_y = texture(y_texture, y_uv).xyz * light2;
        vec3 color_z = texture(z_texture, z_uv).xyz * light3;
        vec3 base_color = (color_x * blend_weights.x +
                           color_y * blend_weights.y +
                           color_z * blend_weights.z);
        vec3 occluded = mix(base_color * ambient_occlusion, vec3(0.5, 0.5, 0.5), fog_falloff);

        if (show_ambient_occlusion) {
            fragColor = vec4(occluded * 0.0001 + vec3(ambient_occlusion), alpha);
        } else {
            fragColor = vec4(occluded, alpha);
        }
    }
}
