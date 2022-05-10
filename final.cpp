////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sstream>
#include <unordered_map>


#define GLEW_STATIC
#include "GL/glew.h"
#include "GL/glfw3.h"

#include "arcball.h"
#include "cvec.h"
#include "geometrymaker.h"
#include "glsupport.h"
#include "matrix4.h"
#include "ppm.h"
#include "rigtform.h"


//insert additional headers
#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
#include "mesh.h"

//insert utils
#include "sgutils.h"

using namespace std;

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
// const bool g_Gl2Compatible = false;
extern const bool g_Gl2Compatible = false;

static const float g_frustMinFov = 60.0; // A minimal of 60 degree field of view
static float g_frustFovY =
    g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;  // near plane
static const float g_frustFar = -50.0;  // far plane
static const float g_groundY = -2.0;    // y coordinate of the ground
static const float g_groundSize = 10.0; // half the ground length

static GLFWwindow *g_window;

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static double g_wScale = 1;
static double g_hScale = 1;


static int g_framesPerSecond = 60; // Frames to render per second
static int g_msBetweenKeyFrames = 2000; // 2 seconds between keyframes
static bool g_playingAnimation = false; // Is the animation playing?
static int g_animateTime = 0; // Time since last key frame

enum ObjId { SKY = 0, OBJECT0 = 1, OBJECT1 = 2 };
enum SkyMode { WORLD_SKY = 0, SKY_SKY = 1 };

static const char *const g_objNames[] = {"Sky", "Object 0", "Object 1"};

static bool g_mouseClickDown = false; // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false; // space state, for middle mouse emulation
static double g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static bool g_globalPick = false;

static ObjId g_activeObject = SKY;
static ObjId g_activeEye = SKY;
static SkyMode g_activeCameraFrame = WORLD_SKY;

static bool g_displayArcball = true;
static double g_arcballScreenRadius = 100; // number of pixels
static double g_arcballScale = 1;

// --------- Materials
// This should replace all the contents in the Shaders section, e.g., g_numShaders, g_shaderFiles, and so on
static shared_ptr<Material> g_redDiffuseMat,
                            g_blueDiffuseMat,
                            g_bumpFloorMat,
                            g_arcballMat,
                            g_pickingMat,
                            g_lightMat,
                            g_eyeMat;

shared_ptr<Material> g_overridingMaterial;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;


//
static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node, g_light1Node, g_light2Node, g_eye1Node, g_eye2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking
// PSET 7
static vector<shared_ptr<SgRbtNode> > dump;
static list<vector<RigTForm> > keyFrames;
static list<vector<RigTForm> >::iterator keyFramesIter = keyFrames.begin();
static list<vector<RigTForm> >::iterator prevFrame = keyFrames.begin();
static list<vector<RigTForm> >::iterator nextFrame = keyFrames.begin();
static list<vector<RigTForm> >::iterator nextNextFrame = keyFrames.begin();
//

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;


// --------- Scene
// For Simulation
static double g_lastFrameClock;


///////////////// END OF G L O B A L S
/////////////////////////////////////////////////////

static void initGround() {
    int ibLen, vbLen;
    getPlaneVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makePlane(g_groundSize*2, vtx.begin(), idx.begin());
    g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
    int ibLen, vbLen;
    getCubeVbIbLen(vbLen, ibLen);

    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);

    makeCube(1, vtx.begin(), idx.begin());
    g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
    int ibLen, vbLen;
    getSphereVbIbLen(20, 10, vbLen, ibLen);

    // Temporary storage for sphere Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makeSphere(1, 20, 10, vtx.begin(), idx.begin());
    g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
    uniforms.put("uProjMatrix", projMatrix);
}

static shared_ptr<SgRbtNode> getNodeFromObjId(ObjId objId) {
    shared_ptr<SgRbtNode> nodesArray[] = {g_skyNode, g_robot1Node, g_robot2Node};
    return nodesArray[objId];
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
    if (g_windowWidth >= g_windowHeight)
        g_frustFovY = g_frustMinFov;
    else {
        const double RAD_PER_DEG = 0.5 * CS175_PI / 180;
        g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight /
                                g_windowWidth,
                            cos(g_frustMinFov * RAD_PER_DEG)) /
                      RAD_PER_DEG;
    }
}

static Matrix4 makeProjectionMatrix() {
    return Matrix4::makeProjection(
        g_frustFovY, g_windowWidth / static_cast<double>(g_windowHeight),
        g_frustNear, g_frustFar);
}

enum ManipMode { ARCBALL_ON_PICKED, ARCBALL_ON_SKY, EGO_MOTION };

static ManipMode getManipMode() {
    if (g_currentPickedRbtNode == g_skyNode) {
        if (g_activeEye == SKY && g_activeCameraFrame == WORLD_SKY)
            return ARCBALL_ON_SKY;
        else
            return EGO_MOTION;
    } else
        return ARCBALL_ON_PICKED;
}

static bool shouldUseArcball() {

    return !(g_currentPickedRbtNode == g_skyNode);
    //  return getManipMode() != EGO_MOTION && (!(g_activeEye != SKY &&
    //  g_activeObject == SKY));
}

// The translation part of the aux frame either comes from the current
// active object, or is the identity matrix when
static RigTForm getArcballRbt() {
    switch (getManipMode()) {
    case ARCBALL_ON_PICKED:
        return getPathAccumRbt(g_world, g_currentPickedRbtNode);
    case ARCBALL_ON_SKY:
        return RigTForm();
    case EGO_MOTION:
        return getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye));
    default:
        throw runtime_error("Invalid ManipMode");
    }
}

static void updateArcballScale() {
    RigTForm arcballEye = inv(getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye))) * getArcballRbt();
    double depth = arcballEye.getTranslation()[2];
    if (depth > -CS175_EPS)
        g_arcballScale = 0.02;
    else
        g_arcballScale =
            getScreenToEyeScale(depth, g_frustFovY, g_windowHeight);
}

static void drawArcBall(Uniforms& uniforms) {
    // switch to wire frame mode
    RigTForm arcballEye = inv(getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye))) * getArcballRbt();
    Matrix4 MVM = rigTFormToMatrix(arcballEye) *
                  Matrix4::makeScale(Cvec3(1, 1, 1) * g_arcballScale *
                                     g_arcballScreenRadius);
    // Use uniforms as opposed to curSS
    sendModelViewNormalMatrix(uniforms, MVM, normalMatrix(MVM));

    // safe_glUniform3f(uniforms.h_uColor, 0.27, 0.82, 0.35); // set color

    g_arcballMat->draw(*g_sphere, uniforms);

}

static void drawStuff(bool picking) {
    // Declare an empty uniforms
    Uniforms uniforms;

    // if we are not translating, update arcball scale
    if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) ||
          (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown)))
        updateArcballScale();


    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    // send proj. matrix to be stored by uniforms,
    // as opposed to the current vtx shader
    sendProjectionMatrix(uniforms, projmat);

    RigTForm eyeRbt = getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye));

    const RigTForm invEyeRbt = inv(eyeRbt);

    // get world space coordinates of the light
    Cvec3 light1 = getPathAccumRbt(g_world, g_light1Node).getTranslation();
    Cvec3 light2 = getPathAccumRbt(g_world, g_light2Node).getTranslation();

    const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(light1, 1));
    const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(light2, 1));
    // send the eye space coordinates of lights to uniforms
    uniforms.put("uLight", eyeLight1);
    uniforms.put("uLight2", eyeLight2);
    
    if (!picking) {
        // initialize the drawer with our uniforms, as opposed to curSS
        Drawer drawer(invEyeRbt, uniforms);
        g_world->accept(drawer);

        // draw arcball as part of asst3
        if (g_displayArcball && shouldUseArcball()) {
            drawArcBall(uniforms);
        }
    } else {
        Picker picker(invEyeRbt, uniforms);
        // set overiding material to our picking material
        g_overridingMaterial = g_pickingMat;
        g_world->accept(picker);
        // unset the overriding material
        g_overridingMaterial.reset();
        glFlush();
        // The OpenGL framebuffer uses pixel units, but it reads mouse coordinates
        // using point units. Most of the time these match, but on some hi-res
        // screens there can be a scaling factor.
        g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX * g_wScale,
                                                    g_mouseClickY * g_hScale);
        if (g_currentPickedRbtNode == g_groundNode || !g_currentPickedRbtNode)
            g_currentPickedRbtNode = g_skyNode;   // set to NULL
    }
}

static void pick() {
    // We need to set the clear color to black, for pick rendering.
    // so let's save the clear color
    GLdouble clearColor[4];
    glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // No more glUseProgram
    drawStuff(true); // no more curSS
    //Now set back the clear color
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    checkGlErrors();
}

static void display() {
    // No more glUseProgram
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawStuff(false); // no more curSS
    glfwSwapBuffers(g_window);
    checkGlErrors();
}

// Given t in the range [0, n], perform interpolation for the
// particular t. Returns true if we are at the end of the animation playback, or
// false otherwise.
bool interpolate(double t) {
    double a = t-floor(t);
    if (t < 0 || t >= 1) {
        ++prevFrame;
        ++keyFramesIter;
        ++nextFrame;
        ++nextNextFrame;
        g_animateTime = 0;
        if (nextNextFrame == keyFrames.end()) {
            return true;
        }
    }
    for (int i = 0; i < dump.size(); i++) {
        Cvec3 TTemp;
        for (int j = 0; j < 3; j++) { // translation
            float c_i = (*keyFramesIter)[i].getTranslation()[j];
            float c_iplus1 = (*nextFrame)[i].getTranslation()[j];
            float c_iminus1 = (*prevFrame)[i].getTranslation()[j];
            float c_iplus2 = (*nextNextFrame)[i].getTranslation()[j];
            float d = ((c_iplus1 - c_iminus1)/6) + c_i;
            float e = (-(c_iplus2 - c_i)/6) + c_iplus1;
            TTemp[j] = c_i*pow(1-a, 3) + 3*d*a*pow(1-a,2) + 3*e*pow(a, 2)*(1-a) + c_iplus1*pow(a, 3);

        }
        Quat QTemp;
        Quat c_i = (*keyFramesIter)[i].getRotation();
        Quat c_iplus1 = (*nextFrame)[i].getRotation(); 
        Quat c_iminus1 = (*prevFrame)[i].getRotation();
        Quat c_iplus2 = (*nextNextFrame)[i].getRotation();

        Quat d = power(cn(c_iplus1 * inv(c_iminus1)), 1.0/6) * c_i;
        Quat e = power(cn(c_iplus2 * inv(c_i)), -1.0/6) * c_iplus1;

        Quat f = slerp(c_i, d, a);
        Quat g = slerp(d, e, a);
        Quat h = slerp(e, c_iplus1, a);
        Quat m = slerp(f, g, a);
        Quat n = slerp(g, h, a);
        QTemp = slerp(m, n, a);
        
        (*dump[i]).setRbt(RigTForm(TTemp, QTemp));
    }
    return false;
}

static void copy() {
    if (!keyFrames.empty()) {
        for (int i = 0; i < (*keyFramesIter).size(); i++) {
            (*dump[i]).setRbt((*keyFramesIter)[i]);
        }
    }
}

// Call every frame to advance the animation
void animationUpdate() {
    if (g_playingAnimation) {
        bool endReached = interpolate((float) g_animateTime / g_msBetweenKeyFrames);
        if (!endReached)
            g_animateTime += 1000./g_framesPerSecond;
        else {
            // finish and clean up
            g_playingAnimation = false;
            g_animateTime = 0;
        }
    }
}


static void reshape(GLFWwindow * window, const int w, const int h) {
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height); 
    glViewport(0, 0, width, height);
    
    g_windowWidth = w;
    g_windowHeight = h;
    cerr << "Size of window is now " << g_windowWidth << "x" << g_windowHeight << endl;
    g_arcballScreenRadius = max(1.0, min(h, w) * 0.25);
    updateFrustFovY();
}

static Cvec3 getArcballDirection(const Cvec2 &p, const double r) {
    double n2 = norm2(p);
    if (n2 >= r * r)
        return normalize(Cvec3(p, 0));
    else
        return normalize(Cvec3(p, sqrt(r * r - n2)));
}

static RigTForm moveArcball(const Cvec2 &p0, const Cvec2 &p1) {
    const Matrix4 projMatrix = makeProjectionMatrix();
    const RigTForm eyeInverse = inv(getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye)));
    const Cvec3 arcballCenter = getArcballRbt().getTranslation();
    const Cvec3 arcballCenter_ec = Cvec3(eyeInverse * Cvec4(arcballCenter, 1));

    if (arcballCenter_ec[2] > -CS175_EPS)
        return RigTForm();

    Cvec2 ballScreenCenter =
        getScreenSpaceCoord(arcballCenter_ec, projMatrix, g_frustNear,
                            g_frustFovY, g_windowWidth, g_windowHeight);
    const Cvec3 v0 =
        getArcballDirection(p0 - ballScreenCenter, g_arcballScreenRadius);
    const Cvec3 v1 =
        getArcballDirection(p1 - ballScreenCenter, g_arcballScreenRadius);

    return RigTForm(Quat(0.0, v1[0], v1[1], v1[2]) *
                    Quat(0.0, -v0[0], -v0[1], -v0[2]));
}

static RigTForm doMtoOwrtA(const RigTForm &M, const RigTForm &O,
                           const RigTForm &A) {
    return A * M * inv(A) * O;
}

static RigTForm getMRbt(const double dx, const double dy) {
    RigTForm M;

    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) {
        if (shouldUseArcball())
            M = moveArcball(Cvec2(g_mouseClickX, g_mouseClickY),
                            Cvec2(g_mouseClickX + dx, g_mouseClickY + dy));
        else
            M = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
    } else {
        double movementScale =
            getManipMode() == EGO_MOTION ? 0.02 : g_arcballScale;
        if (g_mouseRClickButton && !g_mouseLClickButton) {
            M = RigTForm(Cvec3(dx, dy, 0) * movementScale);
        } else if (g_mouseMClickButton ||
                   (g_mouseLClickButton && g_mouseRClickButton) ||
                   (g_mouseLClickButton && g_spaceDown)) {
            M = RigTForm(Cvec3(0, 0, -dy) * movementScale);
        }
    }

    switch (getManipMode()) {
    case ARCBALL_ON_PICKED:
        break;
    case ARCBALL_ON_SKY:
        M = inv(M);
        break;
    case EGO_MOTION:
        //    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown)
        //    // only invert rotation
        M = inv(M);
        break;
    }
    return M;
}

static RigTForm makeMixedFrame(const RigTForm &objRbt, const RigTForm &eyeRbt) {
    return transFact(objRbt) * linFact(eyeRbt);
}

static void motion(GLFWwindow *window, double x, double y) {
    if (!g_mouseClickDown)
        return;
    if (g_activeObject == SKY && g_activeEye != SKY)
        return; // we do not edit the sky when viewed from the objects

    const double dx = x - g_mouseClickX;
    const double dy = g_windowHeight - y - 1 - g_mouseClickY;

    const RigTForm M = getMRbt(dx, dy); // the "action" matrix

    // the matrix for the auxiliary frame (the w.r.t.)    
    // const RigTForm A =
    //     makeMixedFrame(getArcballRbt(), getPathAccumRbt(g_world, getNodeFromObjId(g_activeEye)));
    const RigTForm A =
        makeMixedFrame(getPathAccumRbt(g_world, g_currentPickedRbtNode), getPathAccumRbt(g_world, g_skyNode));
    const RigTForm As = inv(getPathAccumRbt(g_world, g_currentPickedRbtNode, 1)) * A;
    RigTForm O = doMtoOwrtA(M, (*g_currentPickedRbtNode).getRbt(), As);

    if ((g_mouseLClickButton && !g_mouseRClickButton &&
         !g_spaceDown) // rotating
        && g_currentPickedRbtNode == g_skyNode) {
        RigTForm My = getMRbt(dx, 0);
        RigTForm Mx = getMRbt(0, dy);
        RigTForm B = makeMixedFrame(getArcballRbt(), RigTForm());
        O = doMtoOwrtA(Mx, (*g_currentPickedRbtNode).getRbt(), makeMixedFrame(getArcballRbt(), (*g_currentPickedRbtNode).getRbt()));
        O = doMtoOwrtA(My, O, B);
    }
    (*g_currentPickedRbtNode).setRbt(O);

    g_mouseClickX += dx;
    g_mouseClickY += dy;
}

static void mouse(GLFWwindow *window, int button, int state, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    // if in p-mode, run pick function and disable p-mode

    g_mouseClickX = x;
    g_mouseClickY =
        g_windowHeight - y - 1; // conversion from GLUT window-coordinate-system
                                // to OpenGL window-coordinate-system

    g_mouseLClickButton |= (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS);
    g_mouseRClickButton |= (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS);
    g_mouseMClickButton |= (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS);

    g_mouseLClickButton &= !(button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_RELEASE);
    g_mouseRClickButton &= !(button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_RELEASE);
    g_mouseMClickButton &= !(button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_RELEASE);

    g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
    if (g_mouseLClickButton && g_globalPick) {
        g_globalPick = false;
        pick();
    }
}

static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        cout << key << endl;
        switch (key) {
            case GLFW_KEY_ESCAPE:
                exit(0);
            case GLFW_KEY_H:
                cout << " ============== H E L P ==============\n\n"
                    << "h\t\thelp menu\n"
                    << "s\t\tsave screenshot\n"
                    << "f\t\tToggle flat shading on/off.\n"
                    << "o\t\tCycle object to edit\n"
                    << "v\t\tCycle view\n"
                    << "drag left mouse to rotate\n"
                    << endl;
                break;
            case GLFW_KEY_S:
                glFlush();
                writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
                break;
            case GLFW_KEY_V:
                g_activeEye = ObjId((g_activeEye + 1) % 3);
                cerr << "Active eye is " << g_objNames[g_activeEye] << endl;
                break;
            case GLFW_KEY_O:
                g_activeObject = ObjId((g_activeObject + 1) % 3);
                cerr << "Active object is " << g_objNames[g_activeObject] << endl;
                break;
            case GLFW_KEY_M:
                g_activeCameraFrame = SkyMode((g_activeCameraFrame + 1) % 2);
                cerr << "Editing sky eye w.r.t. "
                    << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n"
                                                        : "sky-sky frame\n")
                    << endl;
                break;
            case GLFW_KEY_P:
                g_globalPick = !g_globalPick;
                break;
            case GLFW_KEY_SPACE:
                g_spaceDown = true;
                break;
            case GLFW_KEY_C:
                cout << "C!" << endl;
                copy();
                break;
            case GLFW_KEY_U: {
                static vector<RigTForm> temp;
                temp.resize(dump.size());
                for (int i = 0; i < dump.size(); i++) {
                    temp[i] = (*dump[i]).getRbt();
                }
                if (keyFrames.empty()) {
                    keyFramesIter = keyFrames.insert(keyFramesIter, temp);
                } else {
                    *keyFramesIter = temp;
                }
                
                break;
            }
            case GLFW_KEY_N: {
                static vector<RigTForm> temp;
                temp.resize(dump.size());
                for (int i = 0; i < dump.size(); i++) {
                    temp[i] = (*dump[i]).getRbt();
                }
                if (keyFrames.empty()) {
                    keyFramesIter = keyFrames.insert(keyFramesIter, temp);
                } else {
                    ++keyFramesIter;
                    keyFrames.insert(keyFramesIter, temp);
                    --keyFramesIter;
                }
                break;
            }
            case GLFW_KEY_PERIOD: // >
                if (!(mods & GLFW_MOD_SHIFT)) break;
                if (!keyFrames.empty() && keyFramesIter != --(keyFrames.end())) {
                    ++keyFramesIter;
                    for (int i = 0; i < (*keyFramesIter).size(); i++) {
                        (*dump[i]).setRbt((*keyFramesIter)[i]);
                    }
                }
                break;
            case GLFW_KEY_COMMA: // <
                if (!(mods & GLFW_MOD_SHIFT)) break;
                if (!keyFrames.empty() && keyFramesIter != keyFrames.begin()) {
                    --keyFramesIter;
                    for (int i = 0; i < (*keyFramesIter).size(); i++) {
                        (*dump[i]).setRbt((*keyFramesIter)[i]);
                    }
                }
                break;
            case GLFW_KEY_D:
                if (!keyFrames.empty()) {
                    keyFramesIter = keyFrames.erase(keyFramesIter);
                }
                if (!keyFrames.empty()) {
                    if (keyFramesIter != keyFrames.begin()) --keyFramesIter;
                    for (int i = 0; i < (*keyFramesIter).size(); i++) {
                        (*dump[i]).setRbt((*keyFramesIter)[i]);
                    }
                }
                break;
            case GLFW_KEY_I: {
                ifstream myfile ("animation.txt");
                if (myfile.is_open()) {
                    string line, token;
                    int numFrames = 0, frameSize;
                    keyFrames.clear();
                    getline(myfile, line); istringstream stream(line); // get first line
                    getline(stream, token, ' '); // get first token
                    numFrames = stoi(token); // set the num of frames from first token
                    getline(stream, token, ' '); // get second token
                    frameSize = stoi(token); // set the frame size from second token
                    for (int i = 0; i < numFrames; i++) { // looping over frames
                        vector<RigTForm> temp;
                        for (int j = 0; j < frameSize; j++) { // looping over RBT's
                            getline(myfile, line); stream =  istringstream(line);// get next line
                            getline(stream, token, ' '); const float tx = stof(token);
                            getline(stream, token, ' '); const float ty = stof(token);
                            getline(stream, token, ' '); const float tz = stof(token);
                            getline(stream, token, ' '); const float qw = stof(token);
                            getline(stream, token, ' '); const  float qx = stof(token);
                            getline(stream, token, ' '); const  float qy = stof(token);
                            getline(stream, token, ' '); const  float qz = stof(token);
                            const RigTForm rbt(Cvec3(tx, ty, tz), Quat(qw, qx, qy, qz));
                            temp.push_back(rbt);
                        }
                        keyFrames.push_back(temp);    
                    }
                    myfile.close();
                    keyFramesIter = keyFrames.begin();
                    copy();
                }
                else cout << "Unable to open file"; 
                break;
            }
            case GLFW_KEY_W: {
                if (keyFrames.empty()) break;
                ofstream myfile ("animation.txt");
                if (myfile.is_open()) {
                    myfile << keyFrames.size() << " " << (*keyFramesIter).size() << endl;
                    for (list<vector<RigTForm> >::iterator i = keyFrames.begin(); i != keyFrames.end(); ++i) {
                        for (int j = 0; j < (*i).size(); j++) {
                            myfile << (*i)[j].getTranslation()[0] << " " << (*i)[j].getTranslation()[1] << " " << (*i)[j].getTranslation()[2] << " " << (*i)[j].getRotation()[0] << " " << (*i)[j].getRotation()[1] << " " << (*i)[j].getRotation()[2] << " " << (*i)[j].getRotation()[3] << endl;
                        }
                    }
                    myfile.close();
                }
                else cout << "Unable to open file";
                break;
            }
            case GLFW_KEY_Y: {
                if (!g_playingAnimation) {
                    if (keyFrames.size() < 4) {
                        cout << "Not Enough Keyframes For Animation" << endl;
                        break;
                    }
                    prevFrame = keyFrames.begin();
                    keyFramesIter = prevFrame;
                    ++keyFramesIter;
                    nextFrame = keyFramesIter;
                    ++nextFrame;
                    nextNextFrame = nextFrame;
                    ++nextNextFrame;

                    g_playingAnimation = true;
                } else {
                    g_playingAnimation = false;
                } break;
            }
            case 45:
                g_msBetweenKeyFrames += 250;
                break;
            case 61:
                g_msBetweenKeyFrames = max(g_msBetweenKeyFrames - 250, 1);
                break;
        }

    } else {
        switch(key) {
        case GLFW_KEY_SPACE:
            g_spaceDown = false;
            break;
        }
    }
}

void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

static void initGlfwState() {
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    g_window = glfwCreateWindow(g_windowWidth, g_windowHeight,
                                "final", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "Failed to create GLFW window or OpenGL context\n");
        exit(1);
    }
    glfwMakeContextCurrent(g_window);
    glewInit();

    glfwSwapInterval(1);

    glfwSetErrorCallback(error_callback);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetKeyCallback(g_window, keyboard);

    int screen_width, screen_height;
    glfwGetWindowSize(g_window, &screen_width, &screen_height);
    int pixel_width, pixel_height;
    glfwGetFramebufferSize(g_window, &pixel_width, &pixel_height);

    cout << screen_width << " " << screen_height << endl;
    cout << pixel_width << " " << pixel_width << endl;

    g_wScale = pixel_width / screen_width;
    g_hScale = pixel_height / screen_height;
}

static void initGLState() {
    glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);
    if (!g_Gl2Compatible)
        glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

    // copy diffuse prototype and set red color
    g_redDiffuseMat.reset(new Material(diffuse));
    g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

    // copy diffuse prototype and set blue color
    g_blueDiffuseMat.reset(new Material(diffuse));
    g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

    // normal mapping material
    g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
    g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
    g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

    // copy solid prototype, and set to wireframed rendering
    g_arcballMat.reset(new Material(solid));
    g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
    g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // copy solid prototype, and set to color white
    g_lightMat.reset(new Material(solid));
    g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

    g_eyeMat.reset(new Material(diffuse));
    g_eyeMat->getUniforms().put("uColor", Cvec3f(0, 0, 0));

    // pick shader
    g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));

};

static void initGeometry() {
    initGround();
    initCubes();
    initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {

    const float ARM_LEN = 0.7,
                ARM_THICK = 0.25,
                LEG_LEN = 0.9,
                LEG_THICK = 0.3,
                TORSO_LEN = 1.5,
                TORSO_THICK = 0.25,
                TORSO_WIDTH = 1,
                HEAD_SIZE = 0.25;
    const int NUM_JOINTS = 10,
              NUM_SHAPES = 10;

    struct JointDesc {
        int parent;
        float x, y, z;
    };

    JointDesc jointDesc[NUM_JOINTS] = {
        {-1}, // torso
        {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
        {1,  ARM_LEN, 0, 0}, // lower right arm
        {0,  -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
        {3,  -ARM_LEN, 0, 0}, // lower left arm
        {0,  TORSO_WIDTH/2, -TORSO_LEN/2, 0}, // upper right leg
        {5,  0, -LEG_LEN, 0}, // lower right leg
        {0,  -TORSO_WIDTH/2, -TORSO_LEN/2, 0}, // upper left leg
        {7,  0, -LEG_LEN, 0}, // lower left leg
        {0,  0, TORSO_LEN/2, 0}, // head

    };

    struct ShapeDesc {
        int parentJointId;
        float x, y, z, sx, sy, sz;
        shared_ptr<Geometry> geometry;
    };

    ShapeDesc shapeDesc[NUM_SHAPES] = {
        {0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
        {1, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // upper right arm
        {2, ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // lower right arm
        {3, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // upper left arm
        {4, -ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // lower left arm
        {5, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // upper right leg
        {6, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // lower right leg
        {7, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // upper left leg
        {8, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // lower left leg
        {9, 0, HEAD_SIZE, 0, HEAD_SIZE, HEAD_SIZE, HEAD_SIZE, g_sphere}, // head    
    };

    shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (jointDesc[i].parent == -1)
            jointNodes[i] = base;
        else {
            jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
            jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
        }
    }
    // The new MyShapeNode takes in a material as opposed to color
    for (int i = 0; i < NUM_SHAPES; ++i) {
      shared_ptr<SgGeometryShapeNode> shape(
        new MyShapeNode(shapeDesc[i].geometry,
                        material, // USE MATERIAL as opposed to color
                        Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                        Cvec3(0, 0, 0),
                        Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
      jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
    }
}

static void initScene() {
    g_world.reset(new SgRootNode());

    g_light1Node.reset(new SgRbtNode(RigTForm(Cvec3(1.5, 0.25, -8.0))));
    g_light2Node.reset(new SgRbtNode(RigTForm(Cvec3(-1.5, 0.25, -8.0))));
    g_light1Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat, Cvec3())));
    g_light2Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat, Cvec3())));

    g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));
    g_currentPickedRbtNode = g_skyNode;
    g_groundNode.reset(new SgRbtNode());
    g_groundNode->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));

    g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
    g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

    constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
    constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot
    
    

    g_world->addChild(g_skyNode);
    g_world->addChild(g_groundNode);
    g_world->addChild(g_light1Node);
    g_world->addChild(g_light2Node);
    // g_world->addChild(g_robot1Node);
    // g_world->addChild(g_robot2Node);

    dumpSgRbtNodes(g_world, dump);
}

static void glfwLoop() {
    g_lastFrameClock = glfwGetTime();
    while (!glfwWindowShouldClose(g_window)) {
        double thisTime = glfwGetTime();
        if( thisTime - g_lastFrameClock >= 1. / g_framesPerSecond) {
            animationUpdate();
            display();
            g_lastFrameClock = thisTime;
        }
        glfwPollEvents();
    }
}

int main(int argc, char *argv[]) {
    try {
        initGlfwState();

        // on Mac, we shouldn't use GLEW.
#ifndef __MAC__
        glewInit(); // load the OpenGL extensions

        if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.3");
        else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
            throw runtime_error("Error: card/driver does not support OpenGL "
                                "Shading Language v1.0");
#endif

        cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0"
                                 : "Will use OpenGL 3.x / GLSL 1.5")
             << endl;

        initGLState();
        initMaterials();
        initGeometry();
        initScene();

        glfwLoop();

        return 0;
    } catch (const runtime_error &e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
