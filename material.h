#ifndef MATERIAL_H
#define MATERIAL_H

#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cvec.h"
#include "geometry.h"
#include "glsupport.h"
#include "matrix4.h"
#include "renderstates.h"
#include "uniforms.h"

struct GlProgramDesc;

class Material {
  public:
    Material(const std::string &vsFilename, const std::string &fsFilename);

    void draw(Geometry &geometry, const Uniforms &extraUniforms);

    Uniforms &getUniforms() { return uniforms_; }
    const Uniforms &getUniforms() const { return uniforms_; }

    RenderStates &getRenderStates() { return renderStates_; }
    const RenderStates &getRenderStates() const { return renderStates_; }

    /* These allow you to provide GLSL sources inline. */
    static void addInlineSource(const std::string &filename, int len,
                                const char *content);
    static void removeInlineSource(const std::string &filename);

  protected:
    std::shared_ptr<GlProgramDesc> programDesc_;

    Uniforms uniforms_;

    RenderStates renderStates_;
};

#endif
