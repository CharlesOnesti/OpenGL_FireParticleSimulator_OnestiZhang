#ifndef PICKER_H
#define PICKER_H

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "asstcommon.h"
#include "cvec.h"
#include "drawer.h"
#include "ppm.h"
#include "scenegraph.h"

class Picker : public SgNodeVisitor {
    std::vector<shared_ptr<SgNode>> nodeStack_;

    typedef std::map<int, shared_ptr<SgRbtNode>> IdToRbtNodeMap;
    IdToRbtNodeMap idToRbtNode_;

    int idCounter_;
    bool srgbFrameBuffer_;

    Drawer drawer_;

    void addToMap(int id, shared_ptr<SgRbtNode> node);
    shared_ptr<SgRbtNode> find(int id);

    Cvec3 idToColor(int id);
    int colorToId(const PackedPixel &p);

  public:
    Picker(const RigTForm &initialRbt, Uniforms &uniforms);

    virtual bool visit(SgTransformNode &node);
    virtual bool postVisit(SgTransformNode &node);
    virtual bool visit(SgShapeNode &node);
    virtual bool postVisit(SgShapeNode &node);

    shared_ptr<SgRbtNode> getRbtNodeAtXY(int x, int y);
};

#endif
