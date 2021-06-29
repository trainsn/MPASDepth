#version 430 core 

in float CLIMATE_VALS_VAR;
out vec4 out_Color;
uniform float dMax;

void main(){
	float scalar = CLIMATE_VALS_VAR / dMax;
	out_Color = vec4(vec3(scalar), 1.0);
}