uniform vec4 ambient_occlusion_param;

// 32 random rays on sphere with Poisson distribution, table from Ryan Geiss.
vec3 random_rays[32] = {
    vec3( 0.286582,  0.257763, -0.922729),
    vec3(-0.171812, -0.888079,  0.426375),
    vec3( 0.440764, -0.502089, -0.744066),
    vec3(-0.841007, -0.428818, -0.329882),
    vec3(-0.380213, -0.588038, -0.713898),
    vec3(-0.055393, -0.207160, -0.976738),
    vec3(-0.901510, -0.077811,  0.425706),
    vec3(-0.974593,  0.123830, -0.186643),
    vec3( 0.208042, -0.524280,  0.825741),
    vec3( 0.258429, -0.898570, -0.354663),
    vec3(-0.262118,  0.574475, -0.775418),
    vec3( 0.735212,  0.551820,  0.393646),
    vec3( 0.828700, -0.523923, -0.196877),
    vec3( 0.788742,  0.005727, -0.614698),
    vec3(-0.696885,  0.649338, -0.304486),
    vec3(-0.625313,  0.082413, -0.776010),
    vec3( 0.358696,  0.928723,  0.093864),
    vec3( 0.188264,  0.628978,  0.754283),
    vec3(-0.495193,  0.294596,  0.817311),
    vec3( 0.818889,  0.508670, -0.265851),
    vec3( 0.027189,  0.057757,  0.997960),
    vec3(-0.188421,  0.961802, -0.198582),
    vec3( 0.995439,  0.019982,  0.093282),
    vec3(-0.315254, -0.925345, -0.210596),
    vec3( 0.411992, -0.877706,  0.244733),
    vec3( 0.625857,  0.080059,  0.775818),
    vec3(-0.243839,  0.866185,  0.436194),
    vec3(-0.725464, -0.643645,  0.243768),
    vec3( 0.766785, -0.430702,  0.475959),
    vec3(-0.446376, -0.391664,  0.804580),
    vec3(-0.761557,  0.562508,  0.321895),
    vec3( 0.344460,  0.753223, -0.560359)
};

vec3 normalAtVertex(vec3 vertex)
{
    float d = 1.0;
    vec3 gradient = vec3(
        density(vertex + vec3(d, 0, 0)) - density(vertex - vec3(d, 0, 0)),
        density(vertex + vec3(0, d, 0)) - density(vertex - vec3(0, d, 0)),
        density(vertex + vec3(0, 0, d)) - density(vertex - vec3(0, 0, d)));
    return -normalize(gradient);
}

// Returns the visibility.
float ambientOcclusion(vec3 vertex, vec3 world_position, int block_124,
                       bool short_range_ambient, bool long_range_ambient)
{
    float occlusion = 0.0;
    for (int i = 0; i < 32; i++) {
        vec3 ray = random_rays[i];
        float ray_visibility = 1.0;

        // Short-range samples
        // Don't use multiplication! Adding is faster.
        // Start some (large) epsilon away.
        if (short_range_ambient) {
            vec3 short_ray = vertex + ray;
            vec3 delta = ray / 4 / block_124;
            for (int j = 0; j < 16; j++) {
                short_ray += delta;
                float d = density(short_ray);
                ray_visibility *= (1.0 - clamp(d * ambient_occlusion_param.y, 0.0, 1.0) * ambient_occlusion_param.x);
            }
        }

        // Long-range samples, only look at those pointing up.
        if (long_range_ambient && ray.y > 0) {
            for (int j = 0; j < 5; j++) {
                float distance = pow((j + 3) / 5.0, 1.8) * 20;
                float d = terrainDensity(world_position + distance * ray, block_size, period,
                                         min(3, octaves), octaves_decay);
                ray_visibility *= (1.0 - clamp(d * ambient_occlusion_param.w, 0.0, 1.0) * ambient_occlusion_param.z);
            }
        }

        occlusion += (1.0 - ray_visibility);
    }

    // Use this to make sure the ambient occlusion is continuous.
    // return terrainDensity(vertex, block_size, 10, 3, 2);

    return (1.0 - occlusion / 32.0);
}
