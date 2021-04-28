#version 430 core 

in float CLIMATE_VALS_VAR;
out vec4 out_Color;
uniform float dMax;

const float scalars[8] = {
0,
0.142857143,
0.285714286,
0.428571429,
0.571428571,
0.714285714,
0.857142857,
1
};
const float RGB_r[8] = {
0,
0.022690282,
0.014712149,
0.189397609,
0.891586422,
0.978214237,
0.88402603,
1
};
const float RGB_g[8] = {
0,
0.049634618,
0.304056764,
0.454376494,
0.389697938,
0.544449995,
0.807586693,
1
};
const float RGB_b[8] = {
0,
0.469850365,
0.208741431,
0.0216376,
0.042785206,
0.791623071,
0.990677803,
1
};

void main(){
	float scalar = CLIMATE_VALS_VAR / dMax;
	float r, g, b;
	for (int i = 0; i < 7; i++){
		if (scalar > scalars[i] && scalar < scalars[i + 1]){
			r = RGB_r[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_r[i + 1] -  RGB_r[i]);
			g = RGB_g[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_g[i + 1] -  RGB_g[i]);
			b = RGB_b[i] + (scalar - scalars[i]) / (scalars[i + 1] - scalars[i]) * (RGB_b[i + 1] -  RGB_b[i]);
			break;
		}
	}
	out_Color = vec4(vec3(r, g, b), 1.0);
}