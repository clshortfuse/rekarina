// AMD FSR (vk_ffx.cpp): temporal upscaling through AMD's API library, loaded at run time: FSR 4.1 or 3.1 with the
// libraries of FSR SDK 2.
#pragma once
#include <vulkan/vulkan.h>

typedef struct {
    VkImage image;   // level 0, layer 0, in VK_IMAGE_LAYOUT_GENERAL
    VkFormat format;
    uint32_t width, height;
    int storage;     // created with VK_IMAGE_USAGE_STORAGE_BIT (FSR writes it)
} VkFfxImage;

int vk_ffx_load(void);   // 1 when the library is there with its entry points (loaded once)
int vk_ffx_upscale_available(void);
int vk_ffx_fsr4_available(void);
const char *vk_ffx_versions(void);   // the upscaler versions this card offers, for the menu ("" when not known)
const char *vk_ffx_running(void);    // the version of the newest upscaling context ("" before the first)
// Records FSR into cb: color (the scene light, render size), depth (D32F), mv (R16G16F, render pixels, current to
// previous), out (the output size, RGBA16F, storage). jx, jy: the jitter the scene took, render pixels. fov_y in
// radians (0: unknown). 0 on failure (the caller falls back as it does when DLSS fails).
int vk_ffx_upscale(VkCommandBuffer cb, const VkFfxImage *color, const VkFfxImage *depth, const VkFfxImage *mv,
                   const VkFfxImage *out, float jx, float jy, int reset, float dt_ms, float fov_y);
void vk_ffx_release(void);   // before the device goes away

#ifdef __cplusplus
// The API's calls themselves, for vk_fg.cpp (0 = FFX_API_RETURN_OK): the library loaded first (vk_ffx_load).
#include "third_party/ffx_api/ffx_api.h"
extern "C" {
ffxCreateContextDescHeader *vk_ffx_backend(void);   // the device's descriptor for the end of a creation chain
ffxReturnCode_t vk_ffx_create(ffxContext *ctx, ffxCreateContextDescHeader *desc);
ffxReturnCode_t vk_ffx_destroy(ffxContext *ctx);
ffxReturnCode_t vk_ffx_configure(ffxContext *ctx, const ffxConfigureDescHeader *desc);
ffxReturnCode_t vk_ffx_query(ffxContext *ctx, ffxQueryDescHeader *desc);
ffxReturnCode_t vk_ffx_dispatch(ffxContext *ctx, const ffxDispatchDescHeader *desc);
}
#endif
