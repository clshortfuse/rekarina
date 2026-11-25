// rekarina_dlssg.dll: DLSS Frame Generation through NVIDIA Streamline, behind the C interface of nx2_sl.h. Built with
// the project's clang (llvm-mingw) against the Streamline SDK's headers (tools/build_dlssg.py; the SDK stays outside
// the repository). sl.interposer.dll and its plugins are loaded from the game's folder at run time and their functions
// taken by name; this DLL links none of them. Streamline's DLLs are built with MSVC: what crosses between the two is
// structures passed by pointer or reference and enums, which both compilers lay out alike for the x64 Windows calling
// convention (no bit fields, no by-value classes, and the one class with a virtual function, the frame token, is
// only handed back). The layouts are checked below against the offsets Streamline's own sources assert.
//
// Manual hooking (the SDK's ProgrammingGuideManualHooking): the game creates its instance and device through the
// interposer's vkCreateInstance and vkCreateDevice, which add what DLSS-G needs, and calls the interposer's
// swapchain functions; everything else goes to the Vulkan loader as before. Each frame of the game sleeps in Reflex
// where it starts and gets the PCL latency markers, the camera constants and the depth and motion vector tags (valid
// until the present), then the present goes through the interposer, which inserts the generated frame.
#define NX2SL_BUILD
#include <windows.h>

#include <array>
#include <stdio.h>
#include <string.h>

#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include "nx2_sl.h"

#include <stddef.h>

// The offsets Streamline's sl.api/sl.cpp asserts for its MSVC build (slInit, slSetTag, slSetConstants), and those of
// the options and state of DLSS-G and Reflex worked out from the SDK's headers by the rules both compilers share.
static_assert(offsetof(sl::Preferences, renderAPI) == 136, "sl::Preferences differs from Streamline's");
static_assert(offsetof(sl::ResourceTag, extent) == 48, "sl::ResourceTag differs from Streamline's");
static_assert(offsetof(sl::Resource, internalFlags) == 104 && offsetof(sl::Resource, reserved) == 106,
              "sl::Resource differs from Streamline's");
static_assert(offsetof(sl::Constants, motionVectorsJittered) == 450, "sl::Constants differs from Streamline's");
static_assert(sizeof(sl::BaseStructure) == 32 && offsetof(sl::DLSSGOptions, mode) == 32 &&
              offsetof(sl::DLSSGOptions, flags) == 40 && offsetof(sl::DLSSGOptions, onErrorCallback) == 96 &&
              offsetof(sl::DLSSGOptions, queueParallelismMode) == 108 && sizeof(sl::DLSSGOptions) == 120,
              "sl::DLSSGOptions differs from the SDK's layout");
static_assert(offsetof(sl::DLSSGState, numFramesActuallyPresented) == 48, "sl::DLSSGState differs from the SDK's layout");
static_assert(offsetof(sl::ReflexOptions, mode) == 32, "sl::ReflexOptions differs from the SDK's layout");
static_assert(offsetof(sl::DLSSOptions, mode) == 32, "sl::DLSSOptions differs from the SDK's layout");

namespace {

HMODULE g_mod;
Nx2SlLog g_log;
PFun_slInit *p_init;
PFun_slEvaluateFeature *p_evaluate;
PFun_slDLSSSetOptions *p_sr_options;
bool g_initialized, g_sr_available, g_sr_resources;
unsigned g_sr_errors;
PFun_slShutdown *p_shutdown;
PFun_slIsFeatureLoaded *p_loaded;
PFun_slSetTagForFrame *p_tag;
PFun_slSetConstants *p_constants;
PFun_slGetNewFrameToken *p_token;
PFun_slGetFeatureFunction *p_feature;
PFun_slDLSSGSetOptions *p_dlssg_options;
PFun_slDLSSGGetState *p_dlssg_state;
PFun_slPCLSetMarker *p_marker;
PFun_slReflexSetOptions *p_reflex;
PFun_slReflexSleep *p_sleep;
PFN_vkGetInstanceProcAddr p_gipa;
PFN_vkGetDeviceProcAddr p_gdpa;
int g_available = -1, g_mode = -1;
using RecompositionMetadata = std::array<uint32_t, 8>;
RecompositionMetadata g_recomposition_metadata{};
bool g_recomposition;
uint64_t g_presented, g_shown;

void say(const char *fmt, ...) {
    if (!g_log) return;
    char s[512];
    va_list a;
    va_start(a, fmt);
    vsnprintf(s, sizeof s, fmt, a);
    va_end(a);
    g_log(s);
}

void sl_log(sl::LogType type, const char *msg) {
    if (type == sl::LogType::eInfo) return;   // only warnings and errors reach the game's log
    static int n;
    if (n++ < 200) say("dlssg: streamline %s: %s", type == sl::LogType::eError ? "error" : "warning", msg);
}

template <class T> T *proc(const char *name) { return reinterpret_cast<T *>(GetProcAddress(g_mod, name)); }

// The token of frame n. Tokens come from a small ring Streamline reuses, so each call takes its own right before use.
sl::FrameToken *token(uint32_t frame) {
    sl::FrameToken *t = nullptr;
    return frame && p_token(t, &frame) == sl::Result::eOk ? t : nullptr;
}

void matrix(sl::float4x4 &m, const float *v) {
    for (int r = 0; r < 4; r++) m[r] = sl::float4(v[r * 4], v[r * 4 + 1], v[r * 4 + 2], v[r * 4 + 3]);
}

sl::Resource resource(const Nx2SlImage *i) {
    sl::Resource r(sl::ResourceType::eTex2d, (void *)i->image, nullptr, (void *)i->view, (uint32_t)i->layout);
    r.width = i->width;
    r.height = i->height;
    r.nativeFormat = (uint32_t)i->format;
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.usage = (uint32_t)i->usage;
    return r;
}

}  // namespace

extern "C" NX2SL_API int nx2sl_version(void) { return NX2SL_VERSION; }

extern "C" NX2SL_API int nx2sl_init(const wchar_t *dir, Nx2SlLog log, int sr, int fg) {
    g_log = log;
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%ls\\sl.interposer.dll", dir);
    g_mod = LoadLibraryW(path);
    if (!g_mod) return say("dlssg: sl.interposer.dll not found next to the game: Streamline off"), 0;
    p_init = proc<PFun_slInit>("slInit");
    p_shutdown = proc<PFun_slShutdown>("slShutdown");
    p_loaded = proc<PFun_slIsFeatureLoaded>("slIsFeatureLoaded");
    p_tag = proc<PFun_slSetTagForFrame>("slSetTagForFrame");
    p_constants = proc<PFun_slSetConstants>("slSetConstants");
    p_token = proc<PFun_slGetNewFrameToken>("slGetNewFrameToken");
    p_feature = proc<PFun_slGetFeatureFunction>("slGetFeatureFunction");
    p_evaluate = proc<PFun_slEvaluateFeature>("slEvaluateFeature");
    p_gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(g_mod, "vkGetInstanceProcAddr"));
    p_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(g_mod, "vkGetDeviceProcAddr"));
    if (!p_init || !p_tag || !p_constants || !p_token || !p_feature || !p_gipa || !p_gdpa)
        return say("dlssg: sl.interposer.dll lacks its entry points"), 0;
    sl::Feature features[4];
    uint32_t count = 0;
    if (fg) {
        features[count++] = sl::kFeatureDLSS_G;
        features[count++] = sl::kFeatureReflex;
        features[count++] = sl::kFeaturePCL;
    }
    if (sr) features[count++] = sl::kFeatureDLSS;
    const wchar_t *paths[] = {dir};
    sl::Preferences p{};
    p.showConsole = false;
    p.logLevel = sl::LogLevel::eDefault;
    p.pathsToPlugins = paths;
    p.numPathsToPlugins = 1;
    p.pathToLogsAndData = dir;
    p.logMessageCallback = sl_log;
    // the game tracks its own command buffer state; no plugin downloads at run time. The depth and motion vectors are
    // tagged per frame (slSetTagForFrame, nx2sl_frame), which Streamline refuses without the frame-based flag.
    p.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
              sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    p.featuresToLoad = features;
    p.numFeaturesToLoad = count;
    p.applicationId = 231313132;   // the id NVIDIA's samples use for builds without an issued one (as vk_dlaa.c's NGX)
    p.engine = sl::EngineType::eCustom;
    p.renderAPI = sl::RenderAPI::eVulkan;
    sl::Result r = p_init(p, sl::kSDKVersion);
    if (r != sl::Result::eOk) return say("dlssg: slInit failed (%d): Streamline off", (int)r), 0;
    g_initialized = true;
    say("dlssg: Streamline up (sr=%d fg=%d)", sr != 0, fg != 0);
    return 1;
}

extern "C" NX2SL_API PFN_vkGetInstanceProcAddr nx2sl_instance_proc(void) { return p_gipa; }
extern "C" NX2SL_API PFN_vkGetDeviceProcAddr nx2sl_device_proc(void) { return p_gdpa; }

extern "C" NX2SL_API int nx2sl_available(void) {
    if (g_available >= 0) return g_available;
    bool loaded = false;
    g_available = p_loaded && p_loaded(sl::kFeatureDLSS_G, loaded) == sl::Result::eOk && loaded;
    void *f = nullptr;
    if (g_available && p_feature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", f) == sl::Result::eOk) p_dlssg_options = (PFun_slDLSSGSetOptions *)f;
    if (g_available && p_feature(sl::kFeatureDLSS_G, "slDLSSGGetState", f) == sl::Result::eOk) p_dlssg_state = (PFun_slDLSSGGetState *)f;
    if (p_feature(sl::kFeaturePCL, "slPCLSetMarker", f) == sl::Result::eOk) p_marker = (PFun_slPCLSetMarker *)f;
    if (p_feature(sl::kFeatureReflex, "slReflexSetOptions", f) == sl::Result::eOk) p_reflex = (PFun_slReflexSetOptions *)f;
    if (p_feature(sl::kFeatureReflex, "slReflexSleep", f) == sl::Result::eOk) p_sleep = (PFun_slReflexSleep *)f;
    if (!p_dlssg_options || !p_marker) g_available = 0;
    say("dlssg: DLSS Frame Generation %s", g_available ? "available" : "not available on this GPU or driver");
    return g_available;
}

// Streamline sets Reflex's mode on the swapchain it was told of last (VK_NV_low_latency2) and drops it while there is
// none, so it is set again after each swapchain.
extern "C" NX2SL_API void nx2sl_reflex(int mode) {
    if (!p_reflex) return;
    sl::ReflexOptions o{};
    o.mode = mode == 2 ? sl::ReflexMode::eLowLatencyWithBoost : mode ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
    sl::Result r = p_reflex(o);
    static int n;
    if (n++ < 8) say("dlssg: Reflex low-latency mode %s (%d)", mode == 2 ? "on with boost" : mode ? "on" : "off", (int)r);
}

// The game's simulation and its recording of the frame's commands are one span here: the guest's threads cannot be
// told apart, so the simulation markers sit at the frame's start and the render submission runs until the frame's
// submissions went out (nx2sl_submit_end).
extern "C" NX2SL_API void nx2sl_sleep(uint32_t frame, int hold) {
    if (!p_marker) return;
    sl::FrameToken *t = token(frame);
    if (hold && t && p_sleep) p_sleep(*t);
    if ((t = token(frame))) {
        p_marker(sl::PCLMarker::eSimulationStart, *t);
        p_marker(sl::PCLMarker::eSimulationEnd, *t);
        p_marker(sl::PCLMarker::eRenderSubmitStart, *t);
    }
}

extern "C" NX2SL_API void nx2sl_submit_end(uint32_t frame) {
    sl::FrameToken *t = p_marker ? token(frame) : nullptr;
    if (t) p_marker(sl::PCLMarker::eRenderSubmitEnd, *t);
}

extern "C" NX2SL_API int nx2sl_frame_with_interface(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv,
    const Nx2SlImage *hudless, const Nx2SlImage *interface_coverage, const Nx2SlCamera *cam,
    uint32_t out_w, uint32_t out_h, int enable, uint32_t frame) {
    if (g_available != 1) return 0;
    sl::FrameToken *t = token(frame);
    if (!t) return 0;
    sl::ViewportHandle vp(0);
    const bool valid_inputs = enable && depth && mv && cam;
    bool recompose = valid_inputs && hudless && interface_coverage &&
        interface_coverage->width == out_w && interface_coverage->height == out_h &&
        hudless->width == out_w && hudless->height == out_h;
    RecompositionMetadata metadata{};
    if (recompose) {
        metadata = {out_w, out_h, mv->width, mv->height, uint32_t(hudless->format), uint32_t(mv->format),
                    uint32_t(depth->format), uint32_t(interface_coverage->format)};
    }
    const bool metadata_changed = recompose && metadata != g_recomposition_metadata;
    if (valid_inputs) {
        sl::Constants c{};
        matrix(c.cameraViewToClip, cam->view_to_clip);
        matrix(c.clipToCameraView, cam->clip_to_view);
        matrix(c.clipToPrevClip, cam->clip_to_prev_clip);
        matrix(c.prevClipToClip, cam->prev_clip_to_clip);
        c.jitterOffset = sl::float2(cam->jitter[0], cam->jitter[1]);
        c.mvecScale = sl::float2(cam->mv_scale[0], cam->mv_scale[1]);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos = sl::float3(cam->pos[0], cam->pos[1], cam->pos[2]);
        c.cameraUp = sl::float3(cam->up[0], cam->up[1], cam->up[2]);
        c.cameraRight = sl::float3(cam->right[0], cam->right[1], cam->right[2]);
        c.cameraFwd = sl::float3(cam->fwd[0], cam->fwd[1], cam->fwd[2]);
        c.cameraNear = cam->znear;
        c.cameraFar = cam->zfar;
        c.cameraFOV = cam->fov_y;
        c.cameraAspectRatio = cam->aspect;
        c.depthInverted = sl::Boolean::eFalse;
        c.cameraMotionIncluded = sl::Boolean::eTrue;
        c.motionVectors3D = sl::Boolean::eFalse;
        c.reset = cam->reset || g_recomposition != recompose || metadata_changed ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        p_constants(c, *t, vp);
    }
    sl::Resource depth_resource = valid_inputs ? resource(depth) : sl::Resource{};
    sl::Resource motion_resource = valid_inputs ? resource(mv) : sl::Resource{};
    sl::Resource hudless_resource = valid_inputs && hudless ? resource(hudless) : sl::Resource{};
    sl::Resource coverage_resource = recompose ? resource(interface_coverage) : sl::Resource{};
    sl::Extent depth_extent{0, 0, depth ? depth->width : 0, depth ? depth->height : 0};
    sl::Extent motion_extent{0, 0, mv ? mv->width : 0, mv ? mv->height : 0};
    sl::Extent presentation_extent{0, 0, out_w, out_h};
    constexpr auto lifetime = sl::ResourceLifecycle::eValidUntilPresent;
    sl::ResourceTag tags[] = {
        {valid_inputs ? &depth_resource : nullptr, sl::kBufferTypeDepth, lifetime, valid_inputs ? &depth_extent : nullptr},
        {valid_inputs ? &motion_resource : nullptr, sl::kBufferTypeMotionVectors, lifetime, valid_inputs ? &motion_extent : nullptr},
        {valid_inputs && hudless ? &hudless_resource : nullptr, sl::kBufferTypeHUDLessColor, lifetime,
         valid_inputs && hudless ? &presentation_extent : nullptr},
        {recompose ? &coverage_resource : nullptr, sl::kBufferTypeUIAlpha, lifetime, recompose ? &presentation_extent : nullptr}};
    sl::Result tagging = p_tag(*t, vp, tags, 4, (sl::CommandBuffer *)cb);
    if (tagging != sl::Result::eOk && recompose) {
        recompose = false;
        tags[2] = sl::ResourceTag(nullptr, sl::kBufferTypeHUDLessColor, lifetime);
        tags[3] = sl::ResourceTag(nullptr, sl::kBufferTypeUIAlpha, lifetime);
        tagging = p_tag(*t, vp, tags, 4, (sl::CommandBuffer *)cb);
    }
    if (tagging != sl::Result::eOk) enable = 0;
    // Off keeps the resources (the SDK's guide for pause menus): the game turns generation off in every frame without
    // a camera, which the pause screens are, and on again when the camera is back. Freeing them at each switch cost
    // 400-750 ms a frame for four or five frames when they were made again; kept on instead, generation blended whole
    // pause screens, their tabs and our overlay with the frames before them.
    if ((enable != 0) != g_mode || recompose != g_recomposition || (recompose && metadata_changed)) {
        sl::DLSSGOptions o{};
        o.mode = enable ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        o.numFramesToGenerate = 1;
        o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        o.enableUserInterfaceRecomposition = recompose ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        if (recompose) {
            o.colorWidth = out_w;
            o.colorHeight = out_h;
            o.colorBufferFormat = hudless->format;
            o.mvecDepthWidth = mv->width;
            o.mvecDepthHeight = mv->height;
            o.mvecBufferFormat = mv->format;
            o.depthBufferFormat = depth->format;
            o.hudLessBufferFormat = hudless->format;
            o.uiBufferFormat = interface_coverage->format;
        }
        sl::Result r = p_dlssg_options(vp, o);
        static int said;
        if (said < 16 || r != sl::Result::eOk) {
            said++;
            say("dlssg: frame generation %s (%d), output %ux%u", enable ? "on" : "off (resources kept)", (int)r, out_w,
                out_h);
        }
        if (r != sl::Result::eOk && recompose) {
            // Ordinary FG can still use the accepted HUD-less and UI Alpha tags.
            o.enableUserInterfaceRecomposition = sl::Boolean::eFalse;
            recompose = false;
            r = p_dlssg_options(vp, o);
            say("dlssg: frame generation retry without interface recomposition (%d), keeping UI inputs", (int)r);
        }
        if (r != sl::Result::eOk) return 0;
        if (recompose != g_recomposition) say("dlssg: interface recomposition %s", recompose ? "on" : "off");
        g_recomposition = recompose;
        g_recomposition_metadata = recompose ? metadata : RecompositionMetadata{};
        g_mode = enable != 0;
    }
    return 1;
}

extern "C" NX2SL_API int nx2sl_frame(VkCommandBuffer command_buffer, const Nx2SlImage *depth, const Nx2SlImage *motion,
    const Nx2SlImage *hudless, const Nx2SlCamera *camera, uint32_t width, uint32_t height, int enable, uint32_t frame) {
    return nx2sl_frame_with_interface(command_buffer, depth, motion, hudless, nullptr, camera, width, height, enable, frame);
}

extern "C" NX2SL_API void nx2sl_present_marker(int after, uint32_t frame) {
    sl::FrameToken *t = p_marker ? token(frame) : nullptr;
    if (!t) return;
    p_marker(after ? sl::PCLMarker::ePresentEnd : sl::PCLMarker::ePresentStart, *t);
    if (!after || g_available != 1) return;
    g_presented++;
    // Asked after each present, numFramesActuallyPresented is what that present showed: 2 with a generated frame.
    sl::DLSSGState s{};
    if (!p_dlssg_state || p_dlssg_state(sl::ViewportHandle(0), s, nullptr) != sl::Result::eOk) return;
    g_shown += s.numFramesActuallyPresented;
    if (g_presented == 600 || g_presented % 3600 == 0)
        say("dlssg: %llu frames, %u presented per frame, status %u", (unsigned long long)g_presented,
            s.numFramesActuallyPresented, (unsigned)s.status);
}

extern "C" NX2SL_API uint64_t nx2sl_shown(void) { return g_shown; }

extern "C" NX2SL_API void nx2sl_shutdown(void) {
    if (p_shutdown && g_initialized) p_shutdown();
    g_initialized = g_sr_available = g_sr_resources = false;
    g_available = g_mode = -1;
    g_recomposition = false;
    g_recomposition_metadata = {};
}

extern "C" NX2SL_API int nx2sl_sr_available(void) {
    bool loaded = false;
    void *fn = nullptr;
    g_sr_available = g_initialized && p_loaded && p_evaluate &&
        p_loaded(sl::kFeatureDLSS, loaded) == sl::Result::eOk && loaded &&
        p_feature(sl::kFeatureDLSS, "slDLSSSetOptions", fn) == sl::Result::eOk && fn;
    p_sr_options = reinterpret_cast<PFun_slDLSSSetOptions *>(fn);
    return g_sr_available;
}

static bool sr_mode(Nx2SlSrMode mode, sl::DLSSMode &out) {
    switch (mode) {
    case NX2SL_SR_DLAA: out = sl::DLSSMode::eDLAA; return true;
    case NX2SL_SR_QUALITY: out = sl::DLSSMode::eMaxQuality; return true;
    case NX2SL_SR_BALANCED: out = sl::DLSSMode::eBalanced; return true;
    case NX2SL_SR_PERFORMANCE: out = sl::DLSSMode::eMaxPerformance; return true;
    case NX2SL_SR_ULTRA_PERFORMANCE: out = sl::DLSSMode::eUltraPerformance; return true;
    default: return false;
    }
}

extern "C" NX2SL_API int nx2sl_sr_evaluate_mode(VkCommandBuffer cb, const Nx2SlImage *color, const Nx2SlImage *depth,
                                                const Nx2SlImage *mv, const Nx2SlImage *out, const Nx2SlCamera *cam,
                                                uint32_t frame, int preset, int depth_inverted, Nx2SlSrMode mode) {
    if (!g_initialized || !g_sr_available) return 0;
    const Nx2SlImage *images[] = {color, depth, mv, out};
    sl::DLSSMode selected{};
    bool valid = cb && cam && color && out && sr_mode(mode, selected) &&
        (preset == 0 || (preset >= 5 && preset < (int)sl::DLSSPreset::eCount));
    for (const Nx2SlImage *i : images) {
        valid = valid && i && i->image && i->view && i->width && i->height &&
                i->layout == VK_IMAGE_LAYOUT_GENERAL &&
                (i == out || (i->width == color->width && i->height == color->height));
    }
    valid = valid && (mode != NX2SL_SR_DLAA || (out->width == color->width && out->height == color->height));
    valid = valid && out->width >= color->width && out->height >= color->height;
    if (!valid || out->format != VK_FORMAT_R16G16B16A16_SFLOAT || color->image == out->image) {
        say("dlss-sr: invalid SR contract: mode=%d input=%ux%u output=%ux%u preset=%d",
            (int)mode, color ? color->width : 0, color ? color->height : 0,
            out ? out->width : 0, out ? out->height : 0, preset);
        return 0;
    }
    sl::FrameToken *t = nullptr;
    sl::Result r = p_token(t, &frame);
    if (r != sl::Result::eOk || !t) {
        if (g_sr_errors++ < 16) say("dlss-sr: frame token failed (%d)", (int)r);
        return 0;
    }
    sl::ViewportHandle vp(1);
    sl::Constants c{};
    matrix(c.cameraViewToClip, cam->view_to_clip);
    matrix(c.clipToCameraView, cam->clip_to_view);
    matrix(c.clipToPrevClip, cam->clip_to_prev_clip);
    matrix(c.prevClipToClip, cam->prev_clip_to_clip);
    c.jitterOffset = sl::float2(cam->jitter[0], cam->jitter[1]);
    c.mvecScale = sl::float2(cam->mv_scale[0], cam->mv_scale[1]);
    c.cameraPinholeOffset = sl::float2(0, 0);
    c.cameraPos = sl::float3(cam->pos[0], cam->pos[1], cam->pos[2]);
    c.cameraUp = sl::float3(cam->up[0], cam->up[1], cam->up[2]);
    c.cameraRight = sl::float3(cam->right[0], cam->right[1], cam->right[2]);
    c.cameraFwd = sl::float3(cam->fwd[0], cam->fwd[1], cam->fwd[2]);
    c.cameraNear = cam->znear;
    c.cameraFar = cam->zfar;
    c.cameraFOV = cam->fov_y;
    c.cameraAspectRatio = cam->aspect;
    c.depthInverted = depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    c.reset = (cam->reset || !g_sr_resources) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    r = p_constants(c, *t, vp);
    const char *stage = "constants";
    if (r == sl::Result::eOk) {
        sl::DLSSOptions o{};
        o.mode = selected;
        o.outputWidth = out->width;
        o.outputHeight = out->height;
        o.colorBuffersHDR = sl::Boolean::eTrue;
        o.useAutoExposure = sl::Boolean::eTrue;
        o.dlaaPreset = static_cast<sl::DLSSPreset>(preset);
        o.qualityPreset = o.balancedPreset = o.performancePreset = o.ultraPerformancePreset = o.dlaaPreset;
        g_sr_resources = true;
        r = p_sr_options(vp, o);
        stage = "options";
    }
    sl::Resource rc = resource(color), rd = resource(depth), rm = resource(mv), ro = resource(out);
    sl::Extent extent{0, 0, color->width, color->height};
    sl::Extent output_extent{0, 0, out->width, out->height};
    sl::ResourceTag tags[] = {
        sl::ResourceTag(&rc, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &extent),
        sl::ResourceTag(&ro, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &output_extent),
        sl::ResourceTag(&rd, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &extent),
        sl::ResourceTag(&rm, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &extent)};
    if (r == sl::Result::eOk) {
        r = p_tag(*t, vp, tags, 4, reinterpret_cast<sl::CommandBuffer *>(cb));
        stage = "tags";
    }
    if (r == sl::Result::eOk) {
        const sl::BaseStructure *inputs[] = {&vp};
        r = p_evaluate(sl::kFeatureDLSS, *t, inputs, 1, reinterpret_cast<sl::CommandBuffer *>(cb));
        stage = "evaluate";
    }
    if (r != sl::Result::eOk && g_sr_errors++ < 16)
        say("dlss-sr: %s failed: %s (%d)", stage, sl::getResultAsStr(r), (int)r);
    return r == sl::Result::eOk;
}
