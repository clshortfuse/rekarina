// AMD FSR through its API library, loaded at run time from beside the executable: temporal upscaling here, in the
// place of DLSS in vk_dlaa.c, and frame generation (vk_fg.cpp). The API's headers are in third_party/ffx_api (MIT, FSR
// SDK 2.4.0); they are C++ in places, hence this file. Nothing of it runs unless the upscaler or frame generation asks
// for FSR.
//
// This runtime uses FSR SDK 2 (amd_fidelityfx_loader_vk.dll with amd_fidelityfx_upscaler_vk.dll and
// amd_fidelityfx_framegeneration_vk.dll): FSR Upscaling 4.1 (the machine-learning one) on the Radeon cards it supports,
// FSR 3.1.6 on every other card, and the player may name the version (REKARINA_FSR_VERSION, fsr_version).
//
// FSR takes the same inputs vk_dlaa.c makes for DLSS: the scene's light target (linear HDR), its D32F depth
// (OpenGL-style, an infinite far plane, not inverted) and the camera motion vectors in render pixels from the current
// frame to the previous one, the convention both share, so the vectors go in unscaled.
#ifdef _WIN32
#include <windows.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#ifndef _WIN32
// ffx_api.h marks its entry points __declspec(dllexport), a Windows attribute; the declarations are only read here (the
// library is loaded at run time), so on other hosts the attribute is dropped rather than the header changed
#define __declspec(x)
#endif
#include "third_party/ffx_api/ffx_api.h"
#include "third_party/ffx_api/ffx_upscale.h"
#include "third_party/ffx_api/vk/ffx_api_vk.h"

extern "C" {
#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_ffx.h"
}

namespace {

struct Api {
    void *lib = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    int tried = 0, ok = 0;
} A;

void message(uint32_t type, const wchar_t *msg) {
    char s[512];
    size_t n = 0;
    for (; msg && msg[n] && n < sizeof s - 1; n++) s[n] = msg[n] < 128 ? (char)msg[n] : '?';
    s[n] = 0;
    static int told;
    if (told++ < 40) rt_log("ffx: %s %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error:" : "warning:", s);
}

// One upscaling context per render and output size (the menu switches the mode live, so a few may exist); never
// destroyed before shutdown, so none is released while the GPU may still run its work.
struct Up {
    ffxContext ctx = nullptr;
    uint32_t w = 0, h = 0, ow = 0, oh = 0;
    uint64_t ver = 0;   // the version it was asked for (0: the library's choice)
} U[12];

// The upscaler versions the library offers on this device, in the order it lists them (SDK 2 only).
struct {
    int asked = 0, n = 0;
    uint64_t id[16];
    char name[16][32];
    char running[32];   // the provider of the newest context (vk_ffx_running)
} V;

void versions_query() {
    if (V.asked) return;
    V.asked = 1;
    const VkCtx *c = vk_ctx();
    VkDeviceContext dc = {c->device, c->phys, vkGetDeviceProcAddr};
    ffxQueryDescGetVersions q = {};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device = &dc;
    uint64_t n = 0;
    q.outputCount = &n;
    if (A.query(nullptr, &q.header) != FFX_API_RETURN_OK || !n) {
        rt_log("ffx: the library lists no upscaler versions for this device");
        return;
    }
    if (n > 16) n = 16;
    const char *names[16] = {};
    q.versionIds = V.id, q.versionNames = names;
    if (A.query(nullptr, &q.header) != FFX_API_RETURN_OK) return;
    char line[512];
    size_t o = 0;
    line[0] = 0;
    for (uint64_t i = 0; i < n; i++) {   // the names are copied: the library may write over them at a later query
        snprintf(V.name[i], sizeof V.name[i], "%s", names[i] ? names[i] : "?");
        if (o < sizeof line) o += (size_t)snprintf(line + o, sizeof line - o, "%s%s", i ? ", " : "", V.name[i]);
    }
    V.n = (int)n;
    rt_log("ffx: upscaler versions on this device: %s", line);
}

// The first listed upscaler version whose name starts with want ("4", "3", "3.1.6"), 0 when none does.
uint64_t listed(const char *want) {
    size_t len = strlen(want);
    for (int i = 0; i < V.n; i++)
        if (!strncmp(V.name[i], want, len) && (V.name[i][len] == 0 || V.name[i][len] == '.')) return V.id[i];
    return 0;
}

// REKARINA_FSR_VERSION (applies to the next context; the menu changes it live): auto leaves the choice to the library,
// which takes FSR 4 where the card runs it; a number takes the first listed version of that generation ("4", "3"), a
// full name that one ("3.1.6"). 0: no override (auto or a version the device does not list, said once).
// FSR 4 needs two device extensions the library cannot enable itself (vk_device.c, fsr4_ready): where the library lists
// FSR 4 but the device was made without them, FSR 3.1 runs in its place, for auto and for "4" alike.
uint64_t fsr_version() {
    const char *want = cfg_value("REKARINA_FSR_VERSION");
    int is_auto = !want || !*want || !strcmp(want, "auto");
    if (!vk_ctx()->fsr4_ready && (is_auto || want[0] == '4')) {
        versions_query();
        uint64_t v3 = listed("3");
        if (listed("4") && v3) {
            static int said;
            if (!said++)
                rt_log("ffx: the library offers FSR 4, which needs VK_KHR_cooperative_matrix and VK_EXT_shader_float8 on "
                       "the device; it was made without them: FSR 3.1 runs");
            return v3;
        }
    }
    if (is_auto) return 0;
    versions_query();
    if (uint64_t v = listed(want)) return v;
    static char told[16];
    if (strncmp(told, want, sizeof told - 1)) {
        snprintf(told, sizeof told, "%s", want);
        rt_log("ffx: this card has no FSR %s upscaler: the library chooses", want);
    }
    return 0;
}

FfxApiResource image(const VkFfxImage *i, uint32_t state, uint32_t usage) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = i->format;
    ci.extent = {i->width, i->height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.usage = i->storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0;
    return ffxApiGetResourceVK((void *)i->image, ffxApiGetImageResourceDescriptionVK(i->image, ci, usage), state);
}

bool debug_view() {
    static const bool enabled = rt_knob("FSR_DEBUG_VIEW", "0") != 0;
    return enabled;
}

Up *context_for(uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    uint64_t ver = fsr_version();
    for (Up &u : U)
        if (u.ctx && u.w == w && u.h == h && u.ow == ow && u.oh == oh && u.ver == ver) return &u;
    Up *slot = nullptr;
    for (Up &u : U)
        if (!u.ctx) { slot = &u; break; }
    if (!slot) return nullptr;
    ffxCreateContextDescUpscale d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    d.header.pNext = vk_ffx_backend();
    d.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    if (debug_view()) d.flags |= FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION;
    d.maxRenderSize = {w, h};
    d.maxUpscaleSize = {ow, oh};
    d.fpMessage = message;
    // SDK 2: the version of the API these structures are (without it the newer providers refuse them), and the
    // version of the upscaler the player chose
    ffxCreateContextDescUpscaleVersion api = {};
    api.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    api.version = FFX_UPSCALER_VERSION;
    ffxOverrideVersion ov = {};
    ov.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    ov.versionId = ver;
    api.header.pNext = d.header.pNext, d.header.pNext = &api.header;
    if (ver) ov.header.pNext = d.header.pNext, d.header.pNext = &ov.header;
    // said before the library runs: a crash inside it then names the context it was making
    rt_log("ffx: making an FSR context at %ux%u -> %ux%u (%s)", w, h, ow, oh,
           ver ? "the version chosen here" : "the library's choice");
    ffxReturnCode_t r = A.create(&slot->ctx, &d.header, nullptr);
    if (r != FFX_API_RETURN_OK) {
        rt_log("ffx: FSR context creation at %ux%u -> %ux%u failed (%u)", w, h, ow, oh, (unsigned)r);
        slot->ctx = nullptr;
        return nullptr;
    }
    slot->w = w, slot->h = h, slot->ow = ow, slot->oh = oh, slot->ver = ver;
    ffxQueryGetProviderVersion v = {};
    v.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    A.query(&slot->ctx, &v.header);
    snprintf(V.running, sizeof V.running, "%s", v.versionName ? v.versionName : "");
    rt_log("ffx: FSR upscaling at %ux%u -> %ux%u (%s%s)", w, h, ow, oh, v.versionName ? v.versionName : "provider unknown",
           ver ? ", the version asked for" : "");
    return slot;
}

}  // namespace

// The descriptor that names the device, for the end of a creation chain (its pNext is NULL). One static per form:
// the contexts are made on the replay thread, one at a time.
extern "C" ffxCreateContextDescHeader *vk_ffx_backend(void) {
    const VkCtx *c = vk_ctx();
    static ffxCreateBackendVKDesc be;
    be = {};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    be.deviceContext = {c->device, c->phys, vkGetDeviceProcAddr};
    return &be.header;
}

// The menu's note beside the FSR rows: the upscaler versions this card offers ("4.1.2, 3.1.6, 2.3.4"), or the version
// running when the list is not known. Empty before the library is asked.
extern "C" const char *vk_ffx_versions(void) {
    static char line[256];
    if (!A.ok) return "";
    if (!V.n) return V.running;
    size_t o = 0;
    line[0] = 0;
    for (int i = 0; i < V.n && o < sizeof line; i++)
        o += (size_t)snprintf(line + o, sizeof line - o, "%s%s", i ? ", " : "", V.name[i]);
    return line;
}

extern "C" const char *vk_ffx_running(void) { return V.running; }

extern "C" int vk_ffx_load(void) {
    if (A.tried) {
        if (A.ok && vk_ctx() && vk_ctx()->device) versions_query();
        return A.ok;
    }
    A.tried = 1;
    const char *lib = "amd_fidelityfx_loader_vk.dll";
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, path, MAX_PATH);
    while (n && path[n - 1] != '\\') n--;
    snprintf(path + n, sizeof path - n, "%s", lib);
    HMODULE m = LoadLibraryA(path);
    if (!m) {
        rt_log("ffx: cannot load %s (%lu); install the AMD FSR SDK 2.4 Vulkan components", lib, (unsigned long)GetLastError());
        return 0;
    }
    A.lib = m;
    A.create = (PfnFfxCreateContext)(void *)GetProcAddress(m, "ffxCreateContext");
    A.destroy = (PfnFfxDestroyContext)(void *)GetProcAddress(m, "ffxDestroyContext");
    A.configure = (PfnFfxConfigure)(void *)GetProcAddress(m, "ffxConfigure");
    A.query = (PfnFfxQuery)(void *)GetProcAddress(m, "ffxQuery");
    A.dispatch = (PfnFfxDispatch)(void *)GetProcAddress(m, "ffxDispatch");
#endif
    A.ok = A.create && A.destroy && A.configure && A.query && A.dispatch;
    rt_log("ffx: %s %s", lib, A.ok ? "loaded (FSR SDK 2)" : "lacks the API entry points: FSR off");
    if (A.ok && vk_ctx() && vk_ctx()->device) versions_query();
    return A.ok;
}

extern "C" int vk_ffx_upscale(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                              const VkFfxImage *out, float jx, float jy, int reset, float dt_ms, float fov_y) {
    if (!vk_ffx_load()) return 0;
    Up *u = context_for(color->width, color->height, out->width, out->height);
    if (!u) return 0;
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = cb;
    // Every image is in GENERAL, which the backend reads as its unordered-access state, and is left there
    d.color = image(color, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, 0);
    d.depth = image(depth, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, 0);
    d.motionVectors = image(mv, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, 0);
    d.output = image(out, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {1.0f, 1.0f};
    d.renderSize = {color->width, color->height};
    d.upscaleSize = {out->width, out->height};
    if (debug_view()) d.flags |= FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW;
    // The sharpness slider (0-100) drives FSR's own RCAS here, as it does for FSR 1
    int sharp = cfg_int("REKARINA_SHARPNESS");
    d.enableSharpening = sharp > 0;
    d.sharpness = sharp > 100 ? 1.0f : sharp / 100.0f;
    d.frameTimeDelta = dt_ms;
    d.preExposure = 1.0f;
    d.reset = reset != 0;
    d.cameraNear = 0.1f;   // OoT's projection (vk_dlaa.c ocam_vp): near 0.1, far at infinity
    d.cameraFar = 1e8f;
    d.cameraFovAngleVertical = fov_y > 0 ? fov_y : 1.0f;
    d.viewSpaceToMetersFactor = 0.01f;   // the game's units are centimetres
    ffxReturnCode_t r = A.dispatch(&u->ctx, &d.header);
    if (r != FFX_API_RETURN_OK) {
        static int told;
        if (told++ < 8) rt_log("ffx: FSR dispatch failed (%u)", (unsigned)r);
        return 0;
    }
    return 1;
}

extern "C" ffxReturnCode_t vk_ffx_create(ffxContext *ctx, ffxCreateContextDescHeader *desc) {
    return vk_ffx_load() ? A.create(ctx, desc, nullptr) : FFX_API_RETURN_ERROR;
}
extern "C" ffxReturnCode_t vk_ffx_destroy(ffxContext *ctx) { return A.ok ? A.destroy(ctx, nullptr) : FFX_API_RETURN_ERROR; }
extern "C" ffxReturnCode_t vk_ffx_configure(ffxContext *ctx, const ffxConfigureDescHeader *desc) {
    return A.ok ? A.configure(ctx, desc) : FFX_API_RETURN_ERROR;
}
extern "C" ffxReturnCode_t vk_ffx_query(ffxContext *ctx, ffxQueryDescHeader *desc) {
    return A.ok ? A.query(ctx, desc) : FFX_API_RETURN_ERROR;
}
extern "C" ffxReturnCode_t vk_ffx_dispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc) {
    return A.ok ? A.dispatch(ctx, desc) : FFX_API_RETURN_ERROR;
}

extern "C" void vk_ffx_release(void) {
    if (!A.ok) return;
    for (Up &u : U)
        if (u.ctx) A.destroy(&u.ctx, nullptr), u.ctx = nullptr;
}
