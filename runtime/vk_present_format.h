#pragma once

#include <vulkan/vulkan.h>

static inline int vk_present_surface_format(const VkSurfaceFormatKHR *formats, uint32_t count,
                                            VkFormat preferred, VkColorSpaceKHR encoding,
                                            VkSurfaceFormatKHR *out) {
    const VkFormat pq[] = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32};
    const VkFormat sdr[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    const int hdr = encoding == VK_COLOR_SPACE_HDR10_ST2084_EXT;
    const VkFormat *fallback = hdr ? pq : sdr;
    VkFormat choices[] = {preferred, fallback[0], fallback[1]};
    if (preferred != pq[0] && preferred != pq[1] && (hdr || (preferred != sdr[0] && preferred != sdr[1])))
        choices[0] = VK_FORMAT_UNDEFINED;
    for (unsigned i = 0; i < sizeof choices / sizeof choices[0]; i++) {
        if (choices[i] == VK_FORMAT_UNDEFINED) continue;
        for (uint32_t j = 0; j < count; j++) {
            if (formats[j].colorSpace != encoding ||
                (formats[j].format != choices[i] && !(count == 1 && formats[j].format == VK_FORMAT_UNDEFINED))) continue;
            *out = (VkSurfaceFormatKHR){choices[i], encoding};
            return 1;
        }
    }
    return 0;
}

static inline int vk_present_amd_fg_compatible(VkSurfaceFormatKHR output) {
    if (output.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
        return output.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    return output.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
        (output.format == VK_FORMAT_R8G8B8A8_UNORM || output.format == VK_FORMAT_B8G8R8A8_UNORM);
}
