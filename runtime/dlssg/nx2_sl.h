// The C interface of rekarina_dlssg.dll (runtime/dlssg/nx2_sl.cpp), which runtime/vk_dlssg.c loads for DLSS Frame
// Generation. The DLL is built against NVIDIA's Streamline SDK (tools/build_dlssg.py); nothing of the SDK is in this
// repository, and the game itself is built without it.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

// 4: frame generation that is off keeps its resources (nx2sl_frame), so the game turns it off in every frame without
// a camera. A library of version 3 is still taken: the game then keeps generation on through such frames, as it did.
// 5: Windows Streamline DLSS Super Resolution (nx2sl_sr_available, nx2sl_sr_evaluate_mode) and 4-arg nx2sl_init(dir, log, sr, fg)
// so DLSS SR and live DLSS Frame Generation share one Streamline interposer session.
// 6: optional per-frame interface coverage and recomposition.
#define NX2SL_VERSION 6
#define NX2SL_VERSION_OLDEST 3

typedef struct {
    VkImage image;
    VkImageView view;
    VkFormat format;
    uint32_t width, height;
    VkImageLayout layout;      // the layout it is in when Streamline reads it (at the present)
    VkImageUsageFlags usage;
} Nx2SlImage;

typedef struct {
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16];   // row-major
    float pos[3], up[3], right[3], fwd[3];
    float znear, zfar, fov_y, aspect;
    float jitter[2];           // render pixels
    float mv_scale[2];         // what makes the motion vectors fractions of the render size
    int reset;
} Nx2SlCamera;

typedef void (*Nx2SlLog)(const char *line);

#ifdef __cplusplus
extern "C" {
#endif
#ifdef NX2SL_BUILD
#define NX2SL_API __declspec(dllexport)
#else
#define NX2SL_API
#endif
typedef int (*PFN_nx2sl_version)(void);
typedef enum {
    NX2SL_SR_DLAA,
    NX2SL_SR_QUALITY,
    NX2SL_SR_BALANCED,
    NX2SL_SR_PERFORMANCE,
    NX2SL_SR_ULTRA_PERFORMANCE
} Nx2SlSrMode;
// slInit for Vulkan with optional DLSS Super Resolution (sr) and DLSS-G/Reflex/PCL (fg); dir: the folder of sl.interposer.dll and the plugins
typedef int (*PFN_nx2sl_init)(const wchar_t *dir, Nx2SlLog log, int sr, int fg);
typedef int (*Nx2SlInitV4)(const wchar_t *dir, Nx2SlLog log);
typedef int (*PFN_nx2sl_sr_available)(void);
typedef int (*PFN_nx2sl_sr_evaluate_mode)(VkCommandBuffer cb, const Nx2SlImage *color, const Nx2SlImage *depth,
                                          const Nx2SlImage *mv, const Nx2SlImage *out, const Nx2SlCamera *cam,
                                          uint32_t frame, int preset, int depth_inverted, Nx2SlSrMode mode);
NX2SL_API int nx2sl_sr_available(void);
NX2SL_API int nx2sl_sr_evaluate_mode(VkCommandBuffer cb, const Nx2SlImage *color, const Nx2SlImage *depth,
                                     const Nx2SlImage *mv, const Nx2SlImage *out, const Nx2SlCamera *cam,
                                     uint32_t frame, int preset, int depth_inverted, Nx2SlSrMode mode);
// The interposer's proc address functions: vkCreateInstance and vkCreateDevice through them add what DLSS-G needs
// (extensions, features, queues) and tell Streamline the device; the swapchain functions through them are its hooks.
typedef PFN_vkGetInstanceProcAddr (*PFN_nx2sl_instance_proc)(void);
typedef PFN_vkGetDeviceProcAddr (*PFN_nx2sl_device_proc)(void);
// 1 when DLSS-G loaded and supports this GPU (after the device exists)
typedef int (*PFN_nx2sl_available)(void);
// Reflex's low-latency mode, 0 off, 1 on, 2 on with boost: after the device, and after each swapchain the game creates
// through the interposer (the mode is the swapchain's)
typedef void (*PFN_nx2sl_reflex)(int mode);
// Frames are numbered by the game's presents: frame n ends with its n-th present (vk_dlssg.c vk_dlssg_game_present).
// Where the game starts a frame: with hold, Reflex's sleep, which holds the start until the time the driver computes
// from the measured render and queue times; then the frame's simulation and render-submit-start markers.
typedef void (*PFN_nx2sl_sleep)(uint32_t frame, int hold);
// The frame's submissions went out (at the game's present; with asynchronous replay, when the replay reaches it): its
// render-submit-end marker.
typedef void (*PFN_nx2sl_submit_end)(uint32_t frame);
// Each host present, before the present call: the frame's constants and tags, and frame generation on or off (off
// keeps its resources from version 4 on, so on again costs nothing). Then nx2sl_present_marker(0) before and (1) after
// the present.
typedef int (*PFN_nx2sl_frame)(VkCommandBuffer cb, const Nx2SlImage *depth, const Nx2SlImage *mv, const Nx2SlImage *hudless,
                               const Nx2SlCamera *cam, uint32_t out_w, uint32_t out_h, int enable, uint32_t frame);
typedef int (*PFN_nx2sl_frame_with_interface)(VkCommandBuffer command_buffer, const Nx2SlImage *depth,
    const Nx2SlImage *motion, const Nx2SlImage *hudless, const Nx2SlImage *interface_coverage,
    const Nx2SlCamera *camera, uint32_t width, uint32_t height, int enable, uint32_t frame);
NX2SL_API int nx2sl_frame_with_interface(VkCommandBuffer command_buffer, const Nx2SlImage *depth,
    const Nx2SlImage *motion, const Nx2SlImage *hudless, const Nx2SlImage *interface_coverage,
    const Nx2SlCamera *camera, uint32_t width, uint32_t height, int enable, uint32_t frame);
typedef void (*PFN_nx2sl_present_marker)(int after, uint32_t frame);
// The frames the display got, generated ones included, counted after each present (on the present thread)
typedef uint64_t (*PFN_nx2sl_shown)(void);
typedef void (*PFN_nx2sl_shutdown)(void);
#ifdef __cplusplus
}
#endif
