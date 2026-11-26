#include <assert.h>
#include "vk_present_format.h"

int main(void) {
    const VkFormat packed = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    const VkColorSpaceKHR pq = VK_COLOR_SPACE_HDR10_ST2084_EXT;
    const VkColorSpaceKHR sdr = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    const VkSurfaceFormatKHR both[] = {{packed, pq}, {VK_FORMAT_B8G8R8A8_UNORM, sdr}, {packed, sdr}};
    VkSurfaceFormatKHR out;

    assert(vk_present_surface_format(both, 3, packed, pq, &out));
    assert(out.format == packed && out.colorSpace == pq);
    assert(vk_present_surface_format(both, 3, packed, sdr, &out));
    assert(out.format == packed && out.colorSpace == sdr);
    assert(vk_present_surface_format(both, 2, packed, sdr, &out));
    assert(out.format == VK_FORMAT_B8G8R8A8_UNORM && out.colorSpace == sdr);
    assert(!vk_present_surface_format(both + 1, 2, packed, pq, &out));
    assert(!vk_present_surface_format(both, 1, packed, sdr, &out));

    // An sRGB attachment would encode the already encoded guest values a second time.
    const VkSurfaceFormatKHR srgb[] = {{VK_FORMAT_R8G8B8A8_SRGB, sdr}, {VK_FORMAT_R8G8B8A8_UNORM, sdr}};
    assert(vk_present_surface_format(srgb, 2, VK_FORMAT_R8G8B8A8_SRGB, sdr, &out));
    assert(out.format == VK_FORMAT_R8G8B8A8_UNORM && out.colorSpace == sdr);
    assert(!vk_present_surface_format(srgb, 1, VK_FORMAT_R8G8B8A8_SRGB, sdr, &out));
    return 0;
}
