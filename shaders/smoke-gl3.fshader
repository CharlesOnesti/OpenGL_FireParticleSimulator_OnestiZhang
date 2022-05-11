#version 150

uniform sampler2D uTexColor;
uniform vec3 uColor;

in vec2 vTexCoord;
in vec3 vPosition;

out vec4 fragColor;

void main() {
  vec3 color = texture(uTexColor, vTexCoord).xyz;
  fragColor = vec4(.2,.2,.2, color.x);
}
