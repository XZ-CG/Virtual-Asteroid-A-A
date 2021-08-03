// Represent the twelve vectors of the edges of a cube.
// Note that since these are not normalize, the range of the perlin
// noise will be in [-2, 2]
vec3 perlinVectors[12] = {
    vec3(1,1,0),vec3(-1,1,0),vec3(1,-1,0),vec3(-1,-1,0),
    vec3(1,0,1),vec3(-1,0,1),vec3(1,0,-1),vec3(-1,0,-1),
    vec3(0,1,1),vec3(0,-1,1),vec3(0,1,-1),vec3(0,-1,-1)
};

// Perlin interpolant easing function that has first and second derivatives
// equal to zero at the endpoints.
float ease(float t)
{
    float t3 = t * t * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    return 6 * t5 - 15 * t4 + 10 * t3;
}

float perlinNoise(vec3 coords, float frequency)
{
    vec3 scaledCoords = vec3(coords) * frequency;
    vec3 innerCoords = vec3(mod(scaledCoords, 1.0));

    // Need to use floor first to truncate consistently towards negative
    // infinity. Otherwise, there will be symmetry around 0.
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

float fbm(vec3 coords, int octaves, float octaves_decay, float frequency)
{
	float noise = 0.0;
	for (int i = 1; i <= octaves; i++){
		noise += abs(perlinNoise(coords, frequency) / pow(i, octaves_decay));
		frequency *= 1.95;
	}
	return noise;
}

//other noise function
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

//float snoise(vec3 v)
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
    vec3 x2 = x0 - i2 + C.yyy; 
    vec3 x3 = x0 - D.yyy;      

    // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));


    float n_ = 0.142857142857; 
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    

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

struct IMS 
{
	vec3 normal;
	float potential;
};

float scale[3] = {5.15, 14.35, 42.357};
float outer_probability[3] = {0.965, 0.96, 0.95};

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

// -----------------------------------------------------------------------------
#define M_PI 3.1415926536
#define number_of_bandwidths_ 7
//#define seed_new_ 1234u
#define seed_new_ 0u
#define BLEND_SIZE 0.3
const uint  LRPN_GLOBAL_SEED = 0u;
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


float spot_noise_kernel(const in float a, const in float b, const in vec3 x, const in int order)
{
	// see Eqn. 1
	//float g = K * exp(-M_PI * (a * a) * dot(x, x));
	//float g = - exp(-M_PI * (a * a) * dot(x*x*x, x*x*x));
	float g = 0.0;
	if (order == 0){
		g += (- b * exp(-a * pow(dot(x, x), 1)));
	}
	else if(order == 1){
		//g += ((- b * exp(-a * dot(x, x)))+0.5*exp(-10.0*(pow(log(dot(x, x)), 2))));
		g += (- b * exp(-a * pow(dot(x, x), 2)));
	}
	else{
		g += (- b * exp(-a * pow(dot(x, x), 2)));
	}
	//float g = - 1.2 * exp(-M_PI * (a * a) * dot(x, x));
	//float g = - b * exp(-a * pow(dot(x, x), 0.4));
	//return w * g * h;
	return g ;
}

float noise_cell(inout noise_prng prng, const in vec3 x_c, const in int order)
{

	float number_of_impulses = max(0.0, ( noise_prng_uniform(prng, 0.0, 1.0) - outer_probability[order]) );
	float sum = 0.0;
	for (uint i = 0u; i < number_of_impulses; ++i){
		vec3 x_k_i = x_c * (noise_prng_uniform(prng, 0.1, 1.0));
		//vec3 x_k_i = x_c ;
		float ratio = 0.0;
		
		if(order == 0){
			//ratio += (-0.95*log(noise_prng_uniform(prng, 0.35, 0.5)));//old
			ratio += noise_prng_uniform(prng, 0.05, 0.12);//old
			//float ratio = noise_prng_uniform(prng, 0.19, 0.21);//young

		}
		else if(order == 1){
			ratio += noise_prng_uniform(prng, 0.16, 0.25);			
		}
		else{
			ratio += noise_prng_uniform(prng, 0.5, 0.9);
			//float ratio = noise_prng_uniform(prng, 0.19, 0.21);//young
		}	
		
		float a_i = 3.0 * sqrt(2.0 * M_PI) * ratio;
		float b_i = sqrt(3.0 * sqrt(2) * ratio / sqrt(M_PI));
		
		if (dot(x_k_i, x_k_i) < (1.167))//2.65713
		{
			sum += spot_noise_kernel(a_i, b_i, x_k_i, order);
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
				ivec3 c_i = ivec3(c) + i;
				seed = cell_seed(c_i, LRPN_GLOBAL_SEED);
				wang_hash(prng, seed);
				vec3 x_c_i = vec3(x_c) - i;
				vec3 x_c_inverse = inverse(mat3(cu,cv,normal))*x_c_i;
				sum += noise_cell(prng, x_c_inverse, order);
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

	vec3 shape = vec3(164.0, 126.0, 105.0); // the lenght of ellipsoid
	IMS ellipsoid_1 = ellipsoid_normal(coords, shape);
    
	noise0 = noise_evaluate(coords, 0, ellipsoid_1.normal);
	noise1 = noise_evaluate(coords, 1, ellipsoid_1.normal);
	noise5 = noise_evaluate(coords, 2, ellipsoid_1.normal);

	float density = height_gradient*0.01 - 0.06*noise0 - 0.08*noise1 - 0.12*noise5  + 0.01*noise + noise_extra*0.48;

	density += ellipsoid_1.potential;

    return density;
}
