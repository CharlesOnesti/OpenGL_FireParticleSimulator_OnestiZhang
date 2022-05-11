#version 150

uniform sampler2D uTexColor;
uniform float uLifespan;

in vec2 vTexCoord;
in vec3 vPosition;

out vec4 fragColor;

void main() {
  vec3 color = texture(uTexColor, vTexCoord).xyz;
  if (uLifespan > 200) {
    fragColor = vec4(.2,.2,.2, color.x / (uLifespan/600));

  } else {
    fragColor = vec4(.2,.2,.2, color.x * (uLifespan/600));
  }

}
