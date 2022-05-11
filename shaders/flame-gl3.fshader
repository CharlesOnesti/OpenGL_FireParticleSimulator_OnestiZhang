#version 150

uniform sampler2D uTexColor;
uniform float uLifespan;

in vec2 vTexCoord;
in vec3 vPosition;

out vec4 fragColor;

void main() {
  vec3 color = texture(uTexColor, vTexCoord).xyz;
  fragColor = vec4(1,(uLifespan/600)* .8,0, color.x*1.2);
}
