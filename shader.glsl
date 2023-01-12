#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
struct Material
{
    vec3 	diffuseAlbedo;
    vec3 	specularAlbedo;
    float 	specularPower;
};

#define AA 1
#define NUM_OCTAVES 20

// dist funcs
// https://iquilezles.org/articles/distfunctions
float dot2(in vec3 v ) { return dot(v,v); }
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;
    
    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

float sdCappedTorus(in vec3 p, in vec2 sc, in float ra, in float rb)
{
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

// http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
float det( vec2 a, vec2 b ) { return a.x*b.y-b.x*a.y; }
vec3 getClosest( vec2 b0, vec2 b1, vec2 b2 ) 
{
    float a =     det(b0,b2);
    float b = 2.0*det(b1,b0);
    float d = 2.0*det(b2,b1);
    float f = b*d - a*a;
    vec2  d21 = b2-b1;
    vec2  d10 = b1-b0;
    vec2  d20 = b2-b0;
    vec2  gf = 2.0*(b*d21+d*d10+a*d20); gf = vec2(gf.y,-gf.x);
    vec2  pp = -f*gf/dot(gf,gf);
    vec2  d0p = b0-pp;
    float ap = det(d0p,d20);
    float bp = 2.0*det(d10,d0p);
    float t = clamp( (ap+bp)/(2.0*a+b+d), 0.0 ,1.0 );
    return vec3( mix(mix(b0,b1,t), mix(b1,b2,t),t), t );
}
vec4 sdBezier( vec3 a, vec3 b, vec3 c, vec3 p )
{
	vec3 w = normalize( cross( c-b, a-b ) );
	vec3 u = normalize( c-b );
	vec3 v =          ( cross( w, u ) );

	vec2 a2 = vec2( dot(a-b,u), dot(a-b,v) );
	vec2 b2 = vec2( 0.0 );
	vec2 c2 = vec2( dot(c-b,u), dot(c-b,v) );
	vec3 p3 = vec3( dot(p-b,u), dot(p-b,v), dot(p-b,w) );

	vec3 cp = getClosest( a2-p3.xy, b2-p3.xy, c2-p3.xy );

	return vec4( sqrt(dot(cp.xy,cp.xy)+p3.z*p3.z), cp.z, length(cp.xy), p3.z );
}

float sdEllipsoid( in vec3 p, in vec3 c, in vec3 r )
{
#if 1
    return (length( (p-c)/r ) - 1.0) * min(min(r.x,r.y),r.z);
#else
    float k0 = length((p-c)/r);
    float k1 = length((p-c)/(r*r));
    return k0*(k0-1.0)/k1;
#endif    
}

vec2 sdSegment( vec3 p, vec3 a, vec3 b )
{
	vec3 pa = p-a, ba = b-a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return vec2( length( pa - ba*h ), h );
}

// https://iquilezles.org/articles/smin
float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}

float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

vec3 smax( vec3 a, vec3 b, float k )
{
    vec3 h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}
// ------------------------------------------------------------------------
vec2 map( in vec3 pos )
{
    float objID = 0.0;
    
    // body
    vec3  pa = vec3(-0.180,-0.160,0.100);
    vec3  pb = vec3(0.400,0.293,0.316);
    float ra = 0.3;
    float rb = 0.1;
    float d = sdRoundCone(pos, pa, pb, ra, rb );
    vec2 res = vec2(d, objID);
    
    // back
    pa = vec3(0.32,0.20,0.108);
    pb = vec3(0.5,-0.,0.050);
    ra = 0.02;
    rb = 0.12;
    float d1 = sdRoundCone(pos+vec3(0.4, -0.1, 0.0), pa, pb, ra, rb );
    d = smin(d, d1, 0.3);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // throat
    pa = vec3(0.520,0.015,0.152);
    pb = vec3(0.820,0.08,0.3);
    ra = 0.1;
    rb = 0.03;
    d1 = sdRoundCone(pos+vec3(0.4, -0.1, 0.0), pa, pb, ra, rb );
    d = smin(d, d1, 0.06);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // mouth
    float an = 2.5*(0.428+0.164*sin(1.1+3.0));
    vec2 c = vec2(sin(an),cos(an));
    float theta1 = -0.968; 
    float theta2 = 0.056;
    float theta3 = -1.164;
    mat3 rot_X = mat3(1.0,0.,0.,
                 0., cos(theta1),-sin(theta1),
                 0.0, sin(theta1),cos(theta1) );
    mat3 rot_Y = mat3(cos(theta2),0.,sin(theta2),
                 0., 1.0,0.,
                 -sin(theta2), 0.,cos(theta2) );
    mat3 rot_Z = mat3(cos(theta3),-sin(theta3),0.,
                 sin(theta3), cos(theta3),0.,
                 0.0, 0.,1. );
    d1 = sdCappedTorus(rot_Z*rot_Y*rot_X*(pos+vec3(-0.280,-0.300,-0.380)), c, 0.208, 0.016 );
    d = smin(d, d1, 0.06);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // eyes
    // eyeframe
    d1 = sdSphere( pos+vec3(-0.360,-0.300,-0.4200), 0.08);
    d = smin(d, d1, 0.02);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    d1 = sdSphere( pos+vec3(-0.440,-0.530,0.200), 0.1);
    d = smin(d, d1, 0.02); 
    mat3 scale = mat3(0.2,0.,0.,
                 0., 0.224,0.,
                 0., 0.,0.2);
    d1 = sdEllipsoid(pos+vec3(-0.36,-0.3,-0.420),vec3(0.0,0.00,0.00),vec3(0.046,0.030,0.30)  );
    d = smax(d, -d1, 0.02);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // eye
    mat3 rot = mat3(0.8,-0.6,0.0,
                    0.6, 0.8,0.0,
                    0.0, 0.0,1.0 );
    d1 = sdEllipsoid(rot*(pos+vec3(-0.36,-0.31,-0.430)),vec3(0.0,0.00,0.00),vec3(0.100,0.099,0.099)*0.3  );
    d = smin(d, d1, 0.02);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // legs
    // front left
    pa = vec3(0.22,0.22,0.16);
    pb = vec3(0.400,0.023,0.081);
    ra = 0.08;
    rb = 0.1;
    d1 = sdRoundCone(pos+vec3(0.4, 0.36, -0.24), pa, pb, ra, rb );
    d = smin(d, d1, 0.05);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    pa = vec3(0.220,0.200,0.156);
    pb = vec3(0.360,-0.100,0.150);
    ra = 0.09;
    rb = 0.03;
    d1 = sdRoundCone(pos+vec3(0.4, 0.36, -0.24), pa, pb, ra, rb );
    d = smin(d, d1, 0.01);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // front right
    vec4 d2 = sdBezier( vec3(0.520,-0.200,0.145), vec3(0.432,-0.100+0.1,0.14), vec3(0.460,0.248,0.15), pos +vec3(0.220,0.140,-0.200));
    d1 = d2.x - smoothstep(-0.512,0.360,d2.y)*(0.163 - 0.074 *smoothstep(1.8,1.0,d2.y));
    d = smin(d, d1, 0.03);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // back right
    d2 = sdBezier( vec3(0.540,-0.200,0.050), vec3(0.432,-0.052+0.1,0.14), vec3(0.460,0.248,0.15), pos +vec3(0.1,0.1,0.20));
    d1 = d2.x - smoothstep(-0.624,0.280,d2.y)*(0.187 - 0.106 *smoothstep(1.8,1.0,d2.y));
    d = smin(d, d1, 0.03);
    if( d1<res.x ) res = vec2(d1,objID++);
    
    // paw
    vec3 offset = vec3(-0.270,0.360,-0.850);
    d2 = sdBezier( vec3(0.159,-0.082,-0.700), vec3(0.120,-0.180+0.1,-0.6), vec3(0.040,0.008,-0.5), pos + offset);
    float dd = d2.x - smoothstep(-0.352,0.47,d2.y)*(0.12 - 0.09 *smoothstep(1.8,1.0,d2.y));
    
    d2 = sdBezier( vec3(0.179,-0.102,-0.500), vec3(0.112,-0.212+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.440,0.470,d2.y)*(0.115 - 0.09 *smoothstep(1.8,1.0,d2.y));
    dd = smin( dd, d1, 0.04 );
    
    d2 = sdBezier( vec3(0.019,-0.162,-0.500), vec3(0.080,-0.116+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.224,0.200,d2.y)*(0.115 - 0.082 *smoothstep(1.8,1.0,d2.y));
    dd = smin( dd, d1, 0.04 );

    d1 = sdSphere(pos+vec3(-0.420,0.430,-0.200), 0.03);
    dd = smin(dd, d1, 0.03);
    d1 = sdSphere(pos+vec3(-0.450,0.480,-0.350), 0.02);
    dd = smin(dd, d1, 0.03);
    d1 = sdSphere(pos+vec3(-0.290,0.520,-0.380), 0.02);
    dd = smin(dd, d1, 0.03);
    d = smin(d, dd, 0.02);
    if( dd<res.x ) res = vec2(dd,objID++);
    
    offset = vec3(-0.40,0.28,-0.40);
    d2 = sdBezier( vec3(0.119,-0.182,-0.700), vec3(0.056,-0.244+0.1,-0.6), vec3(0.040,0.008,-0.5), pos + offset);
    dd = d2.x - smoothstep(-0.352,0.47,d2.y)*(0.12 - 0.09 *smoothstep(1.8,1.0,d2.y));
    
    d2 = sdBezier( vec3(0.039,-0.122,-0.300), vec3(0.040,-0.212+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.696,0.470,d2.y)*(0.115 - 0.09 *smoothstep(1.8,1.0,d2.y));
    dd = smin( dd, d1, 0.06 );
    
    d2 = sdBezier( vec3(0.219,-0.162,-0.500), vec3(0.136,-0.212+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.440,0.470,d2.y)*(0.115 - 0.09 *smoothstep(1.512,1.040,d2.y));
    dd = smin( dd, d1, 0.06 );

    d1 = sdSphere(pos+vec3(-0.14,0.55,-0.40), 0.026);
    dd = smin(dd, d1, 0.03);
    d1 = sdSphere(pos+vec3(0.03,0.610,-0.40), 0.02);
    dd = smin(dd, d1, 0.03);
    d1 = sdSphere(pos+vec3(-0.140,0.620,-0.200), 0.03);
    dd = smin(dd, d1, 0.03);
    d = smin(d, dd, 0.02);
    if( dd<res.x ) res = vec2(dd,objID++);
    
    
    offset = vec3(0.0660,0.450,-0.8800);
    d2 = sdBezier( vec3(0.199,-0.162,-0.700), vec3(0.136,-0.268+0.1,-0.6), vec3(0.040,0.008,-0.5), pos + offset);
    dd = d2.x - smoothstep(-0.352,0.47,d2.y)*(0.12 - 0.09 *smoothstep(1.8,1.0,d2.y));
    
    d2 = sdBezier( vec3(0.219,-0.102,-0.500), vec3(0.112,-0.212+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.440,0.470,d2.y)*(0.115 - 0.09 *smoothstep(1.8,1.0,d2.y));
    dd = smin( dd, d1, 0.06 );
    
    d2 = sdBezier( vec3(0.019,-0.162,-0.500), vec3(0.040,-0.172+0.1,-0.5), vec3(0.040,0.008,-0.5), pos + offset);
    d1 = d2.x - smoothstep(-0.440,0.08,d2.y)*(0.115 - 0.09 *smoothstep(1.8,1.0,d2.y));
    dd = smin( dd, d1, 0.06 );

    d1 = sdSphere(pos+vec3(-0.620,0.420,0.100), 0.02);
    dd = smin(dd, d1, 0.03);
    d1 = sdSphere(pos+vec3(-0.440,0.40,-0.080), 0.02);
    dd = smin(dd, d1, 0.03);
    d = smin(d, dd, 0.02);
    if( dd<res.x ) res = vec2(dd,objID++);
    
    return vec2(d, objID);
}

// https://iquilezles.org/articles/normalsSDF
vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773;
    const float eps = 0.0005;
    return normalize( e.xyy*map( pos + e.xyy*eps ).x + 
					  e.yyx*map( pos + e.yyx*eps ).x + 
					  e.yxy*map( pos + e.yxy*eps ).x + 
					  e.xxx*map( pos + e.xxx*eps ).x );
}

// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20201014 (stegu)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
}
// 2D
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
{
  const vec2  C = vec2(1.0/6.0,1.0/3.0);
  const vec4  D = vec4(0.211324865405187,
                        // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,
                        // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,
                        // -1.0 + 2.0 * C.x
                        0.024390243902439);
                        // 1.0 / 41.0

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
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

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
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
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 130.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,vec2(12.9898,78.633)))*43758.5453123);
}

float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * f * f *(3.0 - 2.0 * f);  //+2.0*sin(u_time/10.0)

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float fbm ( in vec2 _st, in vec2 _mo) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5),
                   -sin(0.5), cos(0.5));
  
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * (4.5) + shift;
        a *= 0.45;
    }
    return v;
}
// ------------------------------------------------------------------------
// Translucency: https://www.shadertoy.com/view/3lK3RR
vec3 calculateTransmittance(vec3 ro, vec3 rd, float tmin, float tmax, float atten)
{
    const int MAX_DEPTH = 4;
    float hitPoints[MAX_DEPTH];
    int depth = 0;
    
    for (float t = tmin; t < tmax;)
    {
        float h = abs(map(ro + t * rd).x);
        if (h < 1e-5) { hitPoints[depth++] = t; t += 0.01; };
        if (depth >= MAX_DEPTH) break;
        t += h;
    }
    
    float thickness = 0.0;
    for (int i = 0; i < depth - 1; i += 2) thickness += hitPoints[i+1] - hitPoints[i];
    
    return vec3(1.0) * exp(-atten * thickness * thickness);
}

// ------------------------------------------------------------------------
void main() {    
    // camera movement	
	float an = 0.7;
	vec3 ro = vec3( 0.5*cos(an), 0.0, 2.0*sin(an) );
    vec3 ta = vec3( 0.0, 0.0, 0.0 );
    // camera matrix
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    
    // render
    vec3 tot = vec3(0.9);
    
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        // pixel coordinates 
        vec2 offset = vec2(float(m)+0.5,float(n)+0.5) / float(AA) - 0.5;
        vec2 p = (-u_resolution.xy + 2.0*(gl_FragCoord.xy+offset))/u_resolution.y;

	    // create view ray
        vec3 rd = normalize( p.x*uu + p.y*vv + 1.5*ww );

        // raymarch
        const float tmax = 2.0;
        float t = 0.0;
        for( int i=0; i<256; i++ )
        {
            vec3 pos = ro + t*rd;
            float h = map(pos).x;
            if( h<0.0001 || t>tmax ) break;
            t += h;
        }
        
    
        // shading/lighting	
        vec3 col = vec3(0.0);
        if( t<tmax )
        {
            vec3 pos = ro + t*rd;
            vec3 nor = calcNormal(pos);
            
            vec3 lightDir = normalize(vec3(1.0, 1.5, -1.0));
        	vec3 lightColor = vec3(1.0);
        	Material mat = Material(vec3(0.6,0.08,0.0), vec3(0.3), 8.0);
            
            float dif = clamp( dot(nor,vec3(0.57703)), 0.0, 1.0 ); // diffusion
            float t = clamp(0.5, 0.2, 1.0);
      	vec3 light = t * lightColor * calculateTransmittance(pos+nor*vec3(0.01), lightDir, 0.01, 10.0, 2.0);
            light += (1.0 - t) * calculateTransmittance(pos+nor*vec3(0.01), rd, 0.01, 10.0, 0.5);
        	col +=  light * mat.diffuseAlbedo + mat.specularAlbedo * pow(max(0.0, dot(reflect(lightDir,nor),rd)), 4.0);
            
            // float amb = 0.5 + 0.5*dot(nor,vec3(0.0,1.0,0.0));
            // col = vec3(0.2,0.3,0.4)*amb + vec3(0.8,0.7,0.5)*dif;
        }
        
        // apply texture
        float DF = 0.0;

        // Add a random position
        DF += snoise(9.0*p)*.25+.25;

        // Add a random position
        float a = snoise(p*0.1)*3.1415;
        vec2 vel = vec2(cos(a),sin(a));
        DF += snoise(12.0*p+vel)*.25+.25;
        // float n = texture(p * 4.0); // 2D
        
    	if( length((p-vec2(0.3, -0.3)).xy) > 0.32 && length(p.xy) < 0.33) { col *= 1.0-smoothstep(.7,.75,fract(DF)); }
    	else { col*=1.0; }

        // gamma        
        col = sqrt( col );
	    tot *= col;
    }
    tot /= float(AA*AA);
    
    vec2 p = (-u_resolution.xy + 2.0*gl_FragCoord.xy)/u_resolution.y;
    vec2 m = u_mouse.xy/u_resolution.xy;
    
    vec2 q = vec2(0.);
    q.x = fbm( p, m);
    q.y = fbm( p + vec2(1.0), m);

    vec2 r = vec2(0.);
    r.x = fbm( p + 1.0*q + vec2(1.7,9.2), m);
    r.y = fbm( p + 1.0*q + vec2(8.3,2.8), m);

    float f = fbm(p+r, m);
    vec2 g = vec2(f);
    
	vec3 color = vec3(0.0);
    color = mix(vec3(0.681,0.858,0.920),
                vec3(0.967,0.156,0.573),
                clamp((f*f)*4.312,0.992,1.0));

    color = mix(color,
                vec3(0.300,0.034,0.134),
                clamp(length(q),0.0,1.0));

    color = mix(color,
                vec3(1.000,0.700,0.315),
                clamp(length(r.x),0.0,1.0));
    
    tot *= vec3((f*f*f+0.7*f*f*f*f+3.068*f*f)*color*5.0);
    tot = pow(tot, vec3(0.4545));
    colour_out = vec4(tot, 1.0);
}