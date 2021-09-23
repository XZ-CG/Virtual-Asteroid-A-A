// not normalize, the range of the perlin noise will be in range [-2, 2].
vec3 perlinVectors[12] = {
    vec3(1,1,0),vec3(-1,1,0),vec3(1,-1,0),vec3(-1,-1,0),
    vec3(1,0,1),vec3(-1,0,1),vec3(1,0,-1),vec3(-1,0,-1),
    vec3(0,1,1),vec3(0,-1,1),vec3(0,1,-1),vec3(0,-1,-1)
};

unsigned int hash(ivec3 lowerCorner) {
    unsigned int x = lowerCorner.x * 256 * 256 + lowerCorner.y * 256 + lowerCorner.z;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

vec3 gradientAtCoordinate(ivec3 gridCoords)
{
    return perlinVectors[hash(gridCoords) % 12];
}

float influenceAtCoordinate(ivec3 lowerCorner, ivec3 offset, vec3 innerCoords)
{
    return dot(gradientAtCoordinate(lowerCorner + offset), innerCoords - offset);
}

// Perlin interpolant easing function
float ease(float t)
{
    float t3 = t * t * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    return 6 * t5 - 15 * t4 + 10 * t3;
}

// perlin noise generation
float perlinNoise(vec3 coords, float frequency)
{
    vec3 scaledCoords = vec3(coords) * frequency;
    vec3 innerCoords = vec3(mod(scaledCoords, 1.0));

    ivec3 lowerCorner = ivec3(floor(scaledCoords));

    // For swizzling.
    ivec2 offset = ivec2(0, 1);

    float xInterpolant = ease(innerCoords.x);
    float yInterpolant = ease(innerCoords.y);
    float zInterpolant = ease(innerCoords.z);

    // Calculate and store the influence at each corner from the gradients.
    vec4 face1 = vec4(influenceAtCoordinate(lowerCorner, offset.xxx, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.xyx, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.yxx, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.yyx, innerCoords));
    vec4 face2 = vec4(influenceAtCoordinate(lowerCorner, offset.xxy, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.xyy, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.yxy, innerCoords),
                      influenceAtCoordinate(lowerCorner, offset.yyy, innerCoords));
    vec4 zInterp = mix(face1, face2, zInterpolant);
    vec2 yInterp = mix(zInterp.xz, zInterp.yw, yInterpolant);
    float xInterp = mix(yInterp.x, yInterp.y, xInterpolant);
    return xInterp;
}

float hash(float n)
{
	return fract(sin(n)*753.5453123);
}

float nrand( vec2 n )
{
	return fract(sin(dot(n.xy, vec2(12.9898, 78.233)))* 43758.5453);
}

float hash13(vec3 p3)
{
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float fbm(vec3 coords, int octaves, float octaves_decay, float frequency)
{
	float noise = 0.0;
	for (int i = 1; i <= octaves; i++){
		noise += abs(perlinNoise(coords, frequency) / pow(i, octaves_decay));
		frequency *= 1.95;
	}
	return noise;
}

// --------------- other noise function --------------- 
// value noise
float value_noise(vec3 coords, float frequency)
{
	vec3 x = vec3(coords) * frequency;
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f*f*(3.0 - 2.0 * f);
	float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(mix(mix( hash(n +  0.0), hash(n +  1.0), f.x),
                 mix( hash(n + 157.0), hash(n + 158.0), f.x), f.y),
             mix(mix( hash(n + 113.0), hash(n + 114.0), f.x),
                 mix( hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

float fbm3(vec3 p, int octaves, float octaves_decay, float frequency)
{
	float n = 0.0;
	n = value_noise(p, frequency);
	float a = 0.5;
	for(int i = 0;i < octaves;i++){
		n += a * value_noise(p, frequency);
		frequency *= octaves_decay;
		a = a * 0.5;
	}
	return n;
}

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 hash33(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+19.19);
    return fract((p3.xxy + p3.yxx)*p3.zyx);

}

//simplex noise
float snoise(vec3 coords, float frequency)
{ 
	vec3 v = vec3(coords) * frequency;
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

float noise3D(vec3 p)
{
	return fract(sin(dot(p, vec3(12.9898, 78.233, 126.7235))) * 43758.5453);
}

float worley3D(vec3 p)
{
	float r = 3.0;
	vec3 f = floor(p);
	vec3 x = fract(p);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			for (int k = -1; k <= 1; k++)
			{
				vec3 q = vec3(float(i), float(j), float(k));
				vec3 v = q + vec3(noise3D((q + f)*1.11), noise3D((q + f)*1.14), noise3D((q + f)*1.17)) - x;
				float d = dot(v, v);
				r = min(r, d);
			}
		}
	}
	return sqrt(r);
}

// --------------- SDF modeling ---------------------
struct IMS 
{
	vec3 normal;
	float potential;
};

// scaling factor in Section 3.1, which will change the size of cells and craters
float scale[5] = {26.95, 13.85, 6.85, 3.65, 45.337};
// 0.87 means 87% of cells not exist craters. Influecen the number of crater in this layer
float outer_probability[5] = {0.87, 0.93, 0.9, 0.9, 0.99};

// see Eq: 3.1
float ellipsoid(vec3 coords, vec3 shape_radius)
{
	float m1 = length(coords / shape_radius);
	float m2 = length(coords / (shape_radius*shape_radius));
	return m1*(m1 - 1.0)/m2;    
}

IMS ellipsoid_normal(vec3 coords, vec3 shape_radius)
{ 
	float res = 0.0;
	vec3 res_xyz0 = vec3(0.0);
	vec3 res_xyz1 = vec3(0.0);
	vec3 normal;
	vec3 eps = vec3(0.01,0.,0.);
	
    vec3 x0 = vec3(coords) + eps.xyy;
    vec3 x1 = vec3(coords) - eps.xyy;
    vec3 y0 = vec3(coords) + eps.yxy;
    vec3 y1 = vec3(coords) - eps.yxy;
    vec3 z0 = vec3(coords) + eps.yyx;
    vec3 z1 = vec3(coords) - eps.yyx;
    
    float blend_x0 = ellipsoid(x0, shape_radius);
    float blend_x1 = ellipsoid(x1, shape_radius);
    float blend_y0 = ellipsoid(y0, shape_radius);
    float blend_y1 = ellipsoid(y1, shape_radius);
    float blend_z0 = ellipsoid(z0, shape_radius);
    float blend_z1 = ellipsoid(z1, shape_radius);

    res += ellipsoid(coords, shape_radius);
    res_xyz0.x += blend_x0;
    res_xyz0.y += blend_y0;
    res_xyz0.z += blend_z0;
    res_xyz1.x += blend_x1;
    res_xyz1.y += blend_y1;
    res_xyz1.z += blend_z1;
    
	normal = normalize(-vec3( res_xyz1.x - res_xyz0.x, res_xyz1.y - res_xyz0.y, res_xyz1.z - res_xyz0.z));
    return IMS(normal, res);    	  
}

// Spot noise part
// -----------------------------------------------------------------------------
#define M_PI 3.1415926536
#define number_of_bandwidths_ 7
#define seed_new_ 0u
#define BLEND_SIZE 0.3
const uint  LRPN_GLOBAL_SEED = 12345678u;
// -----------------------------------------------------------------------------
struct noise_prng {
	uint x_;
};

void noise_prng_srand(inout noise_prng this_, const in uint s) {
	this_.x_ = s;
}

uint noise_prng_rand(inout noise_prng this_) {
	return this_.x_ *= 3039177861u;
}

uint noise_prng_myrand(inout noise_prng this_)
{
	uint p = this_.x_;
	p ^= (p << 13u);
	p ^= (p >> 17u);
	p ^= (p << 5u);
	return p;
}

float noise_prng_uniform_0_1(inout noise_prng this_) {
	return float(noise_prng_myrand(this_)) / float(4294967295u);
}

float noise_prng_uniform(inout noise_prng this_, const in float min_, const in float max_) {
	return min_ + (noise_prng_uniform_0_1(this_)*(max_ - min_));
}

float noise_prng_gaussian_0_1(inout noise_prng this_) {
	return sqrt(-2.0 * log(noise_prng_uniform_0_1(this_))) * cos(2.0*M_PI*noise_prng_uniform_0_1(this_));
}

uint noise_prng_poisson(inout noise_prng this_, const in float mean){
	return uint(floor(mean + 0.5 + (sqrt(mean) * noise_prng_gaussian_0_1(this_))));
}

uint cell_seed(const in ivec3 c, const in uint offset)
{
	const uint period = 1024u;
	uint s = (((uint(c.z) % period)*period + uint(c.y) % period)*period + (uint(c.x) % period))*period + offset;
	if (s == 0u) {
		s = 1u;
	}
	return s;
}

void wang_hash(inout noise_prng this_, const in uint s)
{
	uint seed = 0u;
	seed = (s ^ 61u) ^ (s >> 16u);
	seed *= 9u;
	seed = seed ^ (seed >> 4u);
	seed *= 668265261u;
	seed = seed ^ (seed >> 15u);
	this_.x_ = seed;
}

uint hash1D(uint x){
    x += x << 11u;
    x ^= x >> 7u;
    x += (x << 15u);
    x ^= (x >> 5u);
    x += (x << 12u);
    x += (x << 9u);
    return x;
}

uint hash3D(uvec3 v){	
	v.x += v.x >> 11;
    v.x ^= v.x << 7;
    v.x += v.y;
    v.x += v.z^(v.x >> 14);
    v.x ^= v.x << 6;
    v.x += v.x >> 15;
    v.x ^= v.x << 5;
    v.x += v.x >> 12;
    v.x ^= v.x << 9;
    return v.x;
}

float random(uvec3 v){
    const uint mantissaMask = 0x007FFFFFu;
    const uint one          = 0x3F800000u;
    
    uint h = hash3D(v);
    h &= mantissaMask;
    h |= one;

    float r2 = uintBitsToFloat(h);
    return r2 - 1.0;
}

// kernel 1: see Eq:3.3, i = 1, normal crater 1
float spot_kernel_1(const in float a, const in float b, const in vec3 x)
{
	float g = 0.0;
	g += (- b * exp(-a * pow(dot(x, x), 2)));
	return g ;
}

// kernel 2: see Eq:3.3, i = 2, normal crater 2
float spot_kernel_2(const in float a, const in float b, const in vec3 x)
{
	float g = 0.0;
	g += (- b * exp(-a * pow(dot(x, x), 2)));
	return g ;
}

// kernel 3: see Eq:3.6, complex crater
float spot_kernel_3(const in float height, const in float diameter, const in float center_height, const in vec3 x)
{
	float g = 0.0;
	g += (( center_height * exp(-5.5 * pow(dot(x, x), 0.4)))+height*exp(-16.0*(pow(log(diameter*dot(x, x)), 2)))); 
	return g ;
}

// kernel 4: see Eq:3.12, fresh crater
float spot_kernel_4(const in float a, const in float b, const in float height, const in float diameter, const in vec3 x)
{
	float g = 0.0;
	g += ( (- b * exp(-a * dot(x, x))) + height*exp(-10.0*(pow(log(diameter*dot(x, x)), 2))) );
	return g ;
}

float spot_kernel_5(const in float a, const in float b, const in vec3 x)
{
	float g = 0.0;
	g += (- b * exp(-a * pow(dot(x, x), 0.8)));
	return g ;
}


float noise_cell(inout noise_prng prng, const in vec3 x_c, const in int order, const in ivec3 int_coords)
{
	float random_number = max(0.0, noise_prng_uniform(prng, 0.0, 1.0));
	float number_of_impulses = max(0.0, ( random_number - outer_probability[order]) );
	float sum = 0.0;
	
    vec3 x_i_c = vec3(random(int_coords), random(int_coords), random(int_coords));
	vec3 x_k_i = x_c - x_i_c/scale[order];
	
	for (uint i = 0u; i < number_of_impulses; ++i){
		//float ratio = noise_prng_uniform(prng, 0.18, 0.22); // random(int_coords)*0.04 + 0.15
		float x_low = 0.0;
		float x_high = 1.0;
		float power_law = 2.0;
		// see Eq:4.5 here we must scale the diameter crater to match the size of our ellipsoid
		float diameter_crater = pow((pow(x_high, power_law+1.0) - pow(x_low, power_law+1.0))*random(int_coords) + pow(x_low, power_law+1.0), 1.0/(power_law+1.0));	

        // degrade_simple_crater, see Section3.3:ratio = depth/Diameter
        float ratio_simple = random(int_coords)*0.15 + 0.15;
		float a_0 = max(0.7, diameter_crater*20.0);
		float b_0 = min(1.5, 15.0 * ratio_simple * sqrt(0.5 / a_0)); //1.03
		
		// degrade_simple_crater, see Section3.3:ratio = depth/Diameter
		float a_1 = max(0.8, diameter_crater*10.0);
		float b_1 = min(1.3, 12.0 * ratio_simple * sqrt(0.5 / a_1)); //0.62;
		
		// degrade_complex_crater(center_height, height, diameter), see Section3.3
		//float height = max(0.18, random(int_coords)*0.45);
		//float diameter = 1.0 + random(int_coords)*0.4;
		//float center_height = 0.9; // a
		float a_2 = max(0.9, diameter_crater*30.0);//11.0 * sqrt(M_PI) * ratio / (2.0*sqrt(2.0));
		float b_2 = min(1.2, 15.0 * ratio_simple * sqrt(0.5 / a_2)); //sqrt(11.0 * ratio / (2.0*sqrt(2.0*M_PI)));
		
		// degrade_fresh_crater(b_3, fresh_height, fresh_diameter), see Section3.3
		float a_3 = max(1.0, diameter_crater*30.0);
		float b_3 = min(1.0, 12.0 * ratio_simple * sqrt(0.5 / a_3)); //0.33;
		//float ratio_3 = 0.12 + random(int_coords) * 0.05;
        //float a_3 = max(0.7, diameter_crater*20.0); //3.0 * sqrt(2.0 * M_PI) * ratio_3;
        //float b_3 = sqrt(3.0 * sqrt(2) * ratio_3 / sqrt(M_PI));// fresh a
		//float fresh_height = -0.1*log(random(int_coords)*0.35); // fresh k
		//float fresh_diameter = 1.0;

        // degrade_simple_crater, see Section3.3:ratio = depth/Diameter
		float ratio_4 = hash13(int_coords) * 0.4 + 0.4;
		float a_4 = 3.0 * sqrt(2.0 * M_PI) * ratio_4;
		float b_4 = sqrt(3.0 * sqrt(2) * ratio_4 / sqrt(M_PI)) + 0.5;
		
		if (dot(x_k_i, x_k_i) < (1.0))//2.65713
		{
            if(order == 0){
                sum += spot_kernel_1(a_0, b_0, x_k_i);
            }
            else if(order == 1){
                sum += spot_kernel_2(a_1, b_1, x_k_i);
            }
            else if(order == 2){
                // here this ellipsoid asteroid actually not exist complex craters
                //sum += spot_kernel_3(height, diameter, center_height, x_k_i);
				sum += spot_kernel_2(a_2, b_2, x_k_i);
            }
            else if(order == 3){
                //sum += spot_kernel_4(a_3, b_3, fresh_height, fresh_diameter, x_k_i);
				sum += spot_kernel_2(a_3, b_3, x_k_i);
            }
            else{
                sum += spot_kernel_5(a_4, b_4, x_k_i);
            }		
		}
	}
	return sum;
}

float noise_grid(const in vec3 x_g,const in int order,const in vec3 normal)
{
	vec3 int_x_g = floor(x_g);
	vec3 x_c = x_g - int_x_g;
	ivec3 c = ivec3(int_x_g);
	
	vec3 cp = vec3(1.0, 0.0, 0.0);
	vec3 cu = normalize(cross(normal,cp));
	vec3 cv = cross(cu,normal);

	float sum = 0.0;
	ivec3 i;
	uint seed;
	noise_prng prng;
	for (i[2] = -1; i[2] <= +1; ++i[2])
	{
		for (i[1] = -1; i[1] <= +1; ++i[1])
		{
			for (i[0] = -1; i[0] <= +1; ++i[0])
			{
				ivec3 c_i = ivec3(c) + i + ivec3(order);
				seed = cell_seed(c_i, LRPN_GLOBAL_SEED);
				wang_hash(prng, seed);
				vec3 x_c_i = vec3(x_c) - i;
				vec3 x_c_inverse = inverse(mat3(cu,cv,normal))*x_c_i;
				sum += noise_cell(prng, x_c_inverse, order, c_i);
			}
		}
	}
    return sum / sqrt(0.00039472);//0.00039472
}

float noise_evaluate(const in vec3 x, const in int order, const in vec3 normal)
{
	float sum = 0.0f;
	vec3 x_g = vec3(x / scale[order]);
	sum += noise_grid(x_g, order, normal);
	return sum;
}

// -----------------------------------------------------------------------------
float terrainDensity(vec3 coords, float block_size, float period, int octaves, float octaves_decay)
{
    float noise0 = 0.0; 
	float noise1 = 0.0;
	float noise5 = 0.0;
    float noise6 = 0.0;
    float noise7 = 0.0;
    float noise_extra = 0.0;
    
	float noise = 0.0;
	float frequency = 1.0 / period;

    float max_blocks_y = 2.0;

	noise += fbm(coords, octaves, octaves_decay, warp_params.x*frequency);
    noise_extra += 0.603978*fbm3(coords, octaves, octaves_decay, frequency*17.01211) + 0.495284*snoise(coords, frequency*11.11231) - 1.618231*worley3D(coords*frequency*24.89812*0.133892*worley3D(coords*frequency*24.89812)) - 3.5*worley3D(vec3(coords.x*0.195445, coords.y*0.104271, coords.z*0.327721)*frequency*worley3D(vec3(coords.x*1.866895, coords.y*1.847469, coords.z*3.921171)*frequency*2.719572));

    float min = -0.3;
    float max = 0.2;
    float height_gradient = max - (max - min) * (coords.y / max_blocks_y) / block_size;

	ivec3 origin = ivec3(0, 0, 0);

	vec3 shape = vec3(164.0, 126.0, 105.0); 
	IMS ellipsoid_1 = ellipsoid_normal(coords, shape);
	vec3 coords_zyx = vec3(coords.z, coords.y, coords.x);
	vec3 coords_zxy = vec3(coords.z, coords.x, coords.y);
	vec3 coords_xzy = vec3(coords.x, coords.z, coords.y);
    
	noise0 = noise_evaluate(coords, 0, ellipsoid_1.normal);//simple 1
	noise1 = noise_evaluate(coords, 1, ellipsoid_1.normal);//simple 2
	noise5 = noise_evaluate(coords.zyx, 2, ellipsoid_1.normal);//complex
    noise6 = noise_evaluate(coords.zxy, 3, ellipsoid_1.normal);//mixture simple/fresh
    noise7 = noise_evaluate(coords_xzy, 4, ellipsoid_1.normal);

	float density = height_gradient*0.01 - 4.68*noise0 - 3.64*noise1- 3.12*noise5- 2.6*noise6 - 5.2*noise7 + 0.01*noise + noise_extra*24.96;

	density += ellipsoid_1.potential;

    return density;
}

