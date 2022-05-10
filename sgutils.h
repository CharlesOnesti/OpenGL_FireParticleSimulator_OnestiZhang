#ifndef SGUTILS_H
#define SGUTILS_H

#include <memory>
#include <vector>

#include "scenegraph.h"

using namespace std;

struct RbtNodesScanner : public SgNodeVisitor {
    typedef std::vector<shared_ptr<SgRbtNode> > SgRbtNodes;

    SgRbtNodes &nodes_;

    RbtNodesScanner(SgRbtNodes &nodes) : nodes_(nodes) {}

    virtual bool visit(SgTransformNode &node) {
        shared_ptr<SgRbtNode> rbtPtr =
            dynamic_pointer_cast<SgRbtNode>(node.shared_from_this());
        if (rbtPtr)
            nodes_.push_back(rbtPtr);
        return true;
    }
};

inline void dumpSgRbtNodes(shared_ptr<SgNode> root,
                           std::vector<shared_ptr<SgRbtNode> > &rbtNodes) {
    RbtNodesScanner scanner(rbtNodes);
    root->accept(scanner);
}

#endif
