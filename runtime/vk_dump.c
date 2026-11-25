// REKARINA_DUMP=<frame>: a frame dump for the Vulkan back end, which RenderDoc cannot capture (it hides
// VK_EXT_external_memory_host). Between presents frame-1 and frame, the replay's command buffers also copy, after each
// render pass, its colour and depth targets and, after each dispatch, the storage images it writes into host buffers;
// the completion thread turns them into 8-bit PNGs once the submission is done. 3D textures (LUTs, volumes) sampled or
// written in the frame are copied once each, slices side by side. Files in REKARINA_DUMP_DIR (default "dump"):
//   pass_<seq>_t<i>_<w>x<h>_<vkformat>_<name>.png   colour target i (td: depth; img<slot>: storage image; lut<id>: 3D;
//                                                    a BC target drawn through a block view: its blocks as that view)
//   ..._tm.png                                       float formats: x/(1+x) of the same image (the plain one clamps)
//   passes.txt   per pass or dispatch: targets, draws, each distinct pipeline + texture set with its shaders' SPIR-V
//                hashes (sh_<hash>.spv written once) and the bound textures with their samplers
//   images.txt   per image: channel min/max, distinct texel values over the image and over its top quarter
// Sampled 8-bit 2D textures up to TEX2D_CAP texels are copied once each as pass_<seq>_tex<id>_... at the end of the pass
// that samples them; texture lines carry the storage address, mirror state, upload count and the image's and slot's
// swizzle. Copies and texture clears are logged as "copy/clear after pass N". REKARINA_DUMP_LOG=<from> also logs the
// frames from..frame-1 to passes.txt, with PNGs only of targets up to 64K texels. REKARINA_DUMP_BUF=<addr>:<size>[:...]
// snapshots one GPU buffer range after every logged pass (buf_<pass>.bin and bufs.txt, see buf_snap).
// PNGs are opaque (alpha dropped). Integer formats are scaled by the image maximum, depth by its min..max range, SNORM
// as x/2+1/2, UNORM and sRGB bytes written as stored. Caps: MAX_PASSES passes copied, PNG_CAP bytes of PNGs.
// REKARINA_DUMP_TEX=<id>|<w>x<h>[,...]: those texture slots, or 2D textures of that size, are copied where first sampled,
// whatever their format.
// REKARINA_DUMP_UBO=<n>: each listed pipeline also prints the first n vec4 of its user uniform blocks (c3 and up; 1 means 16,
// the cap is 4096), read from guest memory when the entry is listed. A block the submission rewrites with UpdateUniformBuffer shows its value from
// before the submission (vk_exec_draw.c upd_note).
// REKARINA_DUMP_SKIP_RUNS=1: a render pass with the same targets as the render pass before it is logged but not copied
// (the terrain decals draw one pass each into the G-buffer and would use up both caps).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define MAX_PASSES 1200
#define PNG_CAP (2ull << 30)
#define INFLIGHT_CAP (3ull << 30)   // host buffers waiting for their submission
#define MAX_ENTS 48                 // distinct pipeline + texture sets listed per pass
#define MAX_LUTS 256                // 3D textures and small sampled 8-bit 2D textures dumped per frame
#define TEX2D_CAP (1u << 20)        // texels of a sampled 2D texture that is dumped

int vk_dump_armed;
static uint64_t dump_tick;   // REKARINA_DUMP=t<ticks>: dump_frame is set once game time reaches it
static uint64_t dump_frame, log_from;   // log_from: REKARINA_DUMP_LOG, first frame logged to passes.txt without images
static char dir[400];
static FILE *passes_f, *images_f;
static int seq, finished, copied;      // copied: passes and dispatches of the dump frame whose images were copied
static int skip_runs, dump_ubo;        // REKARINA_DUMP_SKIP_RUNS, REKARINA_DUMP_UBO (vec4 per block, 0 for none)
static uint32_t want_tex[16][2];       // REKARINA_DUMP_TEX: slot id and 0, or width and height
static int nwant_tex;
static uint64_t last_targets;          // targets_sig of the last render pass; 0 after a dispatch
static const char *pass_skip;          // why the current pass copies nothing, or NULL
static volatile int64_t png_bytes, inflight;
static PlatMutex rec_lock = PLAT_MUTEX_INIT, img_lock = PLAT_MUTEX_INIT;

// ---- formats

enum { K_UNORM, K_SNORM, K_UINT, K_SINT, K_FLOAT, K_RGB10A2, K_BGR10A2, K_RG11B10, K_E5, K_R5G6B5, K_D16, K_D24, K_D32F };
enum { M_UNORM, M_SNORM, M_INT, M_FLOAT, M_DEPTH };
typedef struct { VkFormat vk; const char *name; uint8_t bytes, nc, bits, kind, bgr; } Fmt;

#define F(v, n, by, nc, bi, k) {VK_FORMAT_##v, n, by, nc, bi, k, 0}
static const Fmt fmts[] = {
    F(R8_UNORM, "r8", 1, 1, 8, K_UNORM), F(R8_SNORM, "r8sn", 1, 1, 8, K_SNORM), F(R8_UINT, "r8ui", 1, 1, 8, K_UINT),
    F(R8_SINT, "r8i", 1, 1, 8, K_SINT), F(R8_SRGB, "r8srgb", 1, 1, 8, K_UNORM),
    F(R8G8_UNORM, "rg8", 2, 2, 8, K_UNORM), F(R8G8_SNORM, "rg8sn", 2, 2, 8, K_SNORM),
    F(R8G8_UINT, "rg8ui", 2, 2, 8, K_UINT), F(R8G8_SINT, "rg8i", 2, 2, 8, K_SINT),
    F(R8G8B8A8_UNORM, "rgba8", 4, 4, 8, K_UNORM), F(R8G8B8A8_SNORM, "rgba8sn", 4, 4, 8, K_SNORM),
    F(R8G8B8A8_UINT, "rgba8ui", 4, 4, 8, K_UINT), F(R8G8B8A8_SINT, "rgba8i", 4, 4, 8, K_SINT),
    F(R8G8B8A8_SRGB, "rgba8srgb", 4, 4, 8, K_UNORM),
    {VK_FORMAT_B8G8R8A8_UNORM, "bgra8", 4, 4, 8, K_UNORM, 1}, {VK_FORMAT_B8G8R8A8_SRGB, "bgra8srgb", 4, 4, 8, K_UNORM, 1},
    F(A2B10G10R10_UNORM_PACK32, "rgb10a2", 4, 4, 10, K_RGB10A2), F(A2R10G10B10_UNORM_PACK32, "bgr10a2", 4, 4, 10, K_BGR10A2),
    F(R5G6B5_UNORM_PACK16, "r5g6b5", 2, 3, 5, K_R5G6B5),
    F(R16_UNORM, "r16", 2, 1, 16, K_UNORM), F(R16_SNORM, "r16sn", 2, 1, 16, K_SNORM), F(R16_UINT, "r16ui", 2, 1, 16, K_UINT),
    F(R16_SINT, "r16i", 2, 1, 16, K_SINT), F(R16_SFLOAT, "r16f", 2, 1, 16, K_FLOAT),
    F(R16G16_UNORM, "rg16", 4, 2, 16, K_UNORM), F(R16G16_SNORM, "rg16sn", 4, 2, 16, K_SNORM),
    F(R16G16_UINT, "rg16ui", 4, 2, 16, K_UINT), F(R16G16_SINT, "rg16i", 4, 2, 16, K_SINT),
    F(R16G16_SFLOAT, "rg16f", 4, 2, 16, K_FLOAT),
    F(R16G16B16A16_UNORM, "rgba16", 8, 4, 16, K_UNORM), F(R16G16B16A16_SNORM, "rgba16sn", 8, 4, 16, K_SNORM),
    F(R16G16B16A16_UINT, "rgba16ui", 8, 4, 16, K_UINT), F(R16G16B16A16_SINT, "rgba16i", 8, 4, 16, K_SINT),
    F(R16G16B16A16_SFLOAT, "rgba16f", 8, 4, 16, K_FLOAT),
    F(R32_UINT, "r32ui", 4, 1, 32, K_UINT), F(R32_SINT, "r32i", 4, 1, 32, K_SINT), F(R32_SFLOAT, "r32f", 4, 1, 32, K_FLOAT),
    F(R32G32_UINT, "rg32ui", 8, 2, 32, K_UINT), F(R32G32_SINT, "rg32i", 8, 2, 32, K_SINT),
    F(R32G32_SFLOAT, "rg32f", 8, 2, 32, K_FLOAT),
    F(R32G32B32A32_UINT, "rgba32ui", 16, 4, 32, K_UINT), F(R32G32B32A32_SINT, "rgba32i", 16, 4, 32, K_SINT),
    F(R32G32B32A32_SFLOAT, "rgba32f", 16, 4, 32, K_FLOAT),
    F(B10G11R11_UFLOAT_PACK32, "r11g11b10f", 4, 3, 11, K_RG11B10), F(E5B9G9R9_UFLOAT_PACK32, "rgb9e5", 4, 3, 9, K_E5),
    // depth aspect of depth formats, as vkCmdCopyImageToBuffer writes it
    F(D16_UNORM, "d16", 2, 1, 16, K_D16), F(D16_UNORM_S8_UINT, "d16s8", 2, 1, 16, K_D16),
    F(X8_D24_UNORM_PACK32, "d24", 4, 1, 24, K_D24), F(D24_UNORM_S8_UINT, "d24s8", 4, 1, 24, K_D24),
    F(D32_SFLOAT, "d32f", 4, 1, 32, K_D32F), F(D32_SFLOAT_S8_UINT, "d32fs8", 4, 1, 32, K_D32F),
};
#undef F

static const Fmt *fmt_of(VkFormat f) {
    for (size_t i = 0; i < sizeof fmts / sizeof fmts[0]; i++)
        if (fmts[i].vk == f) return &fmts[i];
    return NULL;
}

static const char *fmt_name(VkFormat f, char tmp[16]) {
    const Fmt *d = fmt_of(f);
    if (d) return d->name;
    snprintf(tmp, 16, "vk%d", (int)f);
    return tmp;
}

static int mode_of(const Fmt *f) {
    switch (f->kind) {
    case K_SNORM: return M_SNORM;
    case K_UINT: case K_SINT: return M_INT;
    case K_FLOAT: case K_RG11B10: case K_E5: return M_FLOAT;
    case K_D16: case K_D24: case K_D32F: return M_DEPTH;
    default: return M_UNORM;
    }
}

// Unsigned small float with a 5-bit exponent and mbits of mantissa (half: 10, R11G11B10: 6 and 5).
static float ufloat(uint32_t v, int mbits) {
    uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return ldexpf((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf((float)(m | 1u << mbits), (int)e - 15 - mbits);
}

static int32_t sext(uint32_t v, int bits) { return bits >= 32 ? (int32_t)v : (int32_t)(v << (32 - bits)) >> (32 - bits); }

static void decode(const Fmt *f, const uint8_t *p, float o[4]) {
    uint32_t u = 0;
    o[0] = o[1] = o[2] = 0, o[3] = 1;
    switch (f->kind) {
    case K_RGB10A2: case K_BGR10A2:
        memcpy(&u, p, 4);
        o[0] = (u & 1023) / 1023.f, o[1] = (u >> 10 & 1023) / 1023.f, o[2] = (u >> 20 & 1023) / 1023.f, o[3] = (u >> 30) / 3.f;
        if (f->kind == K_BGR10A2) { float t = o[0]; o[0] = o[2], o[2] = t; }
        return;
    case K_RG11B10:
        memcpy(&u, p, 4);
        o[0] = ufloat(u & 2047, 6), o[1] = ufloat(u >> 11 & 2047, 6), o[2] = ufloat(u >> 22 & 1023, 5);
        return;
    case K_E5: {
        memcpy(&u, p, 4);
        float s = ldexpf(1.f, (int)(u >> 27) - 24);
        o[0] = (u & 511) * s, o[1] = (u >> 9 & 511) * s, o[2] = (u >> 18 & 511) * s;
        return;
    }
    case K_R5G6B5:
        memcpy(&u, p, 2);
        o[0] = (u >> 11 & 31) / 31.f, o[1] = (u >> 5 & 63) / 63.f, o[2] = (u & 31) / 31.f;
        return;
    case K_D16: memcpy(&u, p, 2); o[0] = u / 65535.f; return;
    case K_D24: memcpy(&u, p, 4); o[0] = (u & 0xffffff) / 16777215.f; return;
    case K_D32F: memcpy(&o[0], p, 4); return;
    }
    int cb = f->bits / 8;
    for (int c = 0; c < f->nc; c++) {
        u = 0;
        memcpy(&u, p + c * cb, (size_t)cb);
        float x;
        switch (f->kind) {
        case K_UNORM: x = (float)u / (float)((1ull << f->bits) - 1); break;
        case K_SNORM: x = fmaxf((float)sext(u, f->bits) / (float)((1u << (f->bits - 1)) - 1), -1.f); break;
        case K_UINT: x = (float)u; break;
        case K_SINT: x = (float)sext(u, f->bits); break;
        default:
            if (f->bits == 16) x = (u & 0x8000 ? -1.f : 1.f) * ufloat(u & 0x7fff, 10);
            else memcpy(&x, &u, 4);
        }
        o[c] = x;
    }
    if (f->bgr) { float t = o[0]; o[0] = o[2], o[2] = t; }
}

// ---- conversion on the completion thread

typedef struct {
    VkBuffer buf;
    VmaAllocation alloc;
    const uint8_t *map;
    uint64_t bytes;
    const Fmt *f;
    uint32_t w, h, d;   // d: 3D slices, laid side by side in the PNG
    char name[160];
} Job;

// Distinct raw texel values among rows [y0, y1) of every slice.
static uint64_t distinct(const Job *j, uint32_t y0, uint32_t y1) {
    uint64_t n = (uint64_t)j->w * (y1 - y0) * j->d, cap = 1024;
    while (cap < 2 * n) cap *= 2;
    uint64_t *set = calloc(cap, sizeof *set), count = 0;
    if (!set) return 0;
    for (uint32_t z = 0; z < j->d; z++)
        for (uint32_t y = y0; y < y1; y++)
            for (uint32_t x = 0; x < j->w; x++) {
                const uint8_t *p = j->map + (((uint64_t)z * j->h + y) * j->w + x) * j->f->bytes;
                uint64_t a = 0, b = 0;
                memcpy(&a, p, j->f->bytes < 8 ? j->f->bytes : 8);
                if (j->f->bytes > 8) memcpy(&b, p + 8, j->f->bytes - 8u);
                uint64_t k = (a ^ (b * 0x9E3779B97F4A7C15ull)) + 1;
                if (!k) k = 1;
                uint64_t i = (k * 0xff51afd7ed558ccdull) >> 17;
                for (;; i++) {
                    uint64_t *s = &set[i & (cap - 1)];
                    if (*s == k) break;
                    if (!*s) { *s = k, count++; break; }
                }
            }
    free(set);
    return count;
}

static uint8_t u8(float x) { return x > 0 ? x < 1 ? (uint8_t)(x * 255.f + .5f) : 255 : 0; }

static void write_images(const Job *j) {
    uint32_t pw = j->w * j->d, ph = j->h;
    uint64_t n = (uint64_t)pw * ph;
    float *px = malloc(n * 16);
    uint8_t *out = malloc(n * 4);
    if (!px || !out) rt_fatal("vk_dump: out of memory for %s", j->name);
    float mn[4] = {INFINITY, INFINITY, INFINITY, INFINITY}, mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint64_t nans = 0;
    for (uint32_t z = 0; z < j->d; z++)
        for (uint32_t y = 0; y < j->h; y++)
            for (uint32_t x = 0; x < j->w; x++) {
                float *o = px + 4 * ((uint64_t)y * pw + (uint64_t)z * j->w + x);
                decode(j->f, j->map + (((uint64_t)z * j->h + y) * j->w + x) * j->f->bytes, o);
                for (int c = 0; c < 4; c++) {
                    if (isnan(o[c])) { nans++, o[c] = 0; continue; }
                    if (o[c] < mn[c]) mn[c] = o[c];
                    if (o[c] > mx[c]) mx[c] = o[c];
                }
            }
    int mode = mode_of(j->f), grey = j->f->nc == 1;
    if (strstr(j->name, "_blocks")) {   // DEBUG: a BC texture's raw blocks, for decoding elsewhere
        char rp[600];
        snprintf(rp, sizeof rp, "%s/%s.bin", dir, j->name);
        FILE *rf = fopen(rp, "wb");
        if (rf) fwrite(j->map, 1, j->bytes, rf), fclose(rf);
    }
    if (mode == M_DEPTH && cfg_env("REKARINA_DUMP_RAW")) {   // DEBUG-PROBE: the depth values as floats, row by row
        char rp[600];
        snprintf(rp, sizeof rp, "%s/%s.f32", dir, j->name);
        FILE *rf = fopen(rp, "wb");
        if (rf) {
            for (uint64_t i = 0; i < n; i++) fwrite(px + 4 * i, 4, 1, rf);
            fclose(rf);
        }
    } else if (cfg_env("REKARINA_DUMP_RAW4")) {   // DEBUG: any image's decoded texels, four floats each, row by row
        char rp[600];
        snprintf(rp, sizeof rp, "%s/%s.rgba32f", dir, j->name);
        FILE *rf = fopen(rp, "wb");
        if (rf) fwrite(px, 16, n, rf), fclose(rf);
    }
    float scale = 1, lo = mn[0], range = mx[0] - mn[0];
    if (mode == M_INT) {
        float m = 1;
        for (int c = 0; c < j->f->nc && c < 3; c++) m = fmaxf(m, fmaxf(fabsf(mx[c]), fabsf(mn[c])));
        scale = 1.f / m;
    }
    for (int pass = 0; pass < (mode == M_FLOAT ? 2 : 1); pass++) {
        for (uint64_t i = 0; i < n; i++) {
            const float *o = px + 4 * i;
            for (int c = 0; c < 3; c++) {
                float x = o[grey ? 0 : c];
                if (mode == M_SNORM) x = x * .5f + .5f;
                else if (mode == M_INT) x *= scale;
                else if (mode == M_DEPTH) x = range > 0 ? (x - lo) / range : 0;
                else if (mode == M_FLOAT && pass) x = x > 0 ? x / (1 + x) : 0;
                out[4 * i + c] = u8(x);
            }
            out[4 * i + 3] = 255;
        }
        char path[600];
        snprintf(path, sizeof path, "%s/%s%s.png", dir, j->name, pass ? "_tm" : "");
        rt_write_png(path, out, pw, ph);
    }
    uint64_t all = distinct(j, 0, j->h), top = distinct(j, 0, j->h / 4 ? j->h / 4 : 1);
    plat_mutex_lock(&img_lock);
    if (images_f) {
        fprintf(images_f, "%s %ux%ux%u min %g %g %g %g max %g %g %g %g distinct %llu top-quarter %llu nan %llu\n", j->name,
                j->w, j->h, j->d, mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3], (unsigned long long)all,
                (unsigned long long)top, (unsigned long long)nans);
        fflush(images_f);
    }
    plat_mutex_unlock(&img_lock);
    free(out);
    free(px);
}

static void job_done(void *arg) {
    Job *j = arg;
    VmaAllocator vma = vk_ctx()->vma;
    vmaInvalidateAllocation(vma, j->alloc, 0, VK_WHOLE_SIZE);
    write_images(j);
    vmaDestroyBuffer(vma, j->buf, j->alloc);
    plat_add64(&inflight, -(int64_t)j->bytes);
    free(j);
}

// ---- recording

// Records the copy of one subresource into a host buffer and queues its PNGs. Returns NULL, or why it was skipped.
// The buffer is w x h x d texels of vf; the image region is `ext` image texels at depth plane z (differs for the
// blocks of a BC image read as an uncompressed format of the block size, and for a 3D slice).
static const char *copy_region(VkExec *ex, VkImage image, VkFormat vf, VkImageAspectFlags aspect, uint32_t level,
                               uint32_t layer, uint32_t z, VkExtent3D ext, uint32_t w, uint32_t h, uint32_t d,
                               const char *name) {
    const Fmt *f = fmt_of(vf);
    if (!f) return "format not decoded";
    // DEBUG: REKARINA_DUMP_NOTARGETS=1 copies only the textures asked for (REKARINA_DUMP_TEX), not the passes' targets
    if (cfg_env("REKARINA_DUMP_NOTARGETS") && !strstr(name, "_tex")) return "targets off";
    const char *only = cfg_env("REKARINA_DUMP_ONLY");   // DEBUG: copy only images whose name holds one of these (comma list)
    if (only && *only) {
        int hit = 0;
        for (const char *s = only; *s && !hit;) {
            size_t k = strcspn(s, ",");
            char part[64];
            if (k && k < sizeof part) memcpy(part, s, k), part[k] = 0, hit = strstr(name, part) != NULL;
            s += k + (s[k] == ',');
        }
        if (!hit) return "filtered";
    }
    if (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1 != dump_frame && (uint64_t)w * h * d > 65536)
        return "logged frame, large image";
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) return "stencil only";
    uint64_t bytes = (uint64_t)w * h * d * f->bytes;
    uint64_t est = ((uint64_t)h * (1 + 4ull * w * d) + 128) * (mode_of(f) == M_FLOAT ? 2 : 1);
    if ((uint64_t)png_bytes + est > PNG_CAP) return "PNG cap reached";
    if ((uint64_t)inflight + bytes > INFLIGHT_CAP) return "too many copies in flight";
    Job *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk_dump: out of memory");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        return "host buffer allocation failed";
    }
    j->map = info.pMappedData, j->bytes = bytes, j->f = f, j->w = w, j->h = h, j->d = d;
    snprintf(j->name, sizeof j->name, "%s", name);
    VkMemoryBarrier2 pre = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                             .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &pre};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferImageCopy r = {.imageSubresource = {aspect, level, layer, 1}, .imageOffset = {0, 0, (int32_t)z},
                           .imageExtent = ext};   // packed rows: a BC row of ext.width texels is w blocks
    vkCmdCopyImageToBuffer(ex->cb, image, VK_IMAGE_LAYOUT_GENERAL, j->buf, 1, &r);
    di.pMemoryBarriers = &post;
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    plat_add64(&png_bytes, (int64_t)est);
    plat_add64(&inflight, (int64_t)bytes);
    vk_completion_add(ex->fence, job_done, j);
    return NULL;
}

static const char *copy_out(VkExec *ex, VkImage image, VkFormat vf, VkImageAspectFlags aspect, uint32_t level,
                            uint32_t layer, uint32_t w, uint32_t h, uint32_t d, const char *name) {
    return copy_region(ex, image, vf, aspect, level, layer, 0, (VkExtent3D){w, h, d}, w, h, d, name);
}

// ---- REKARINA_DUMP_BUF=<addr>:<size>[:<off>:<stride>:<w>:<h>]: after every pass and dispatch that is logged, the GPU
// buffer range is copied to a host buffer; when it differs from the previous copy it goes to buf_<pass>.bin, and
// bufs.txt gets the changed bytes' extent plus, with the grid arguments, one map per dword of the w x h records of
// `stride` bytes at `off` (each record a letter: its value's index in the table under the map).

static uint64_t bd_addr, bd_size, bd_off, bd_stride;
static uint32_t bd_w, bd_h;
static uint8_t *bd_prev;
static FILE *bufs_f;

typedef struct { VkBuffer buf; VmaAllocation alloc; const uint8_t *map; int pass; } BufJob;

static void buf_grid(const uint8_t *m) {
    for (uint32_t dw = 0; dw < bd_stride / 4; dw++) {
        uint32_t vals[52];
        int nv = 0, over = 0;
        fprintf(bufs_f, "  dword %u:\n", dw);
        for (uint32_t y = 0; y < bd_h; y++) {
            char line[260];
            uint32_t x;
            for (x = 0; x < bd_w && x < 256; x++) {
                uint64_t o = bd_off + ((uint64_t)y * bd_w + x) * bd_stride + dw * 4;
                if (o + 4 > bd_size) { line[x] = ' '; continue; }
                uint32_t v;
                memcpy(&v, m + o, 4);
                int k = 0;
                while (k < nv && vals[k] != v) k++;
                if (k == nv) {
                    if (nv < 52) vals[nv++] = v;
                    else { over = 1; line[x] = '?'; continue; }
                }
                line[x] = (char)(k < 26 ? 'a' + k : 'A' + k - 26);
            }
            line[x] = 0;
            fprintf(bufs_f, "    %2u %s\n", y, line);
        }
        for (int k = 0; k < nv; k++) fprintf(bufs_f, "%s%c=%08x", k % 8 ? " " : "    ", k < 26 ? 'a' + k : 'A' + k - 26, vals[k]);
        fprintf(bufs_f, "%s\n", over ? " (more: ?)" : "");
    }
}

static void buf_done(void *arg) {
    BufJob *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
    plat_mutex_lock(&img_lock);
    if (bufs_f) {
        uint64_t lo = bd_size, hi = 0;
        for (uint64_t i = 0; i < bd_size; i++)
            if (!bd_prev || j->map[i] != bd_prev[i]) { if (i < lo) lo = i; hi = i + 1; }
        if (hi) {
            char path[600];
            snprintf(path, sizeof path, "%s/buf_%04d.bin", dir, j->pass);
            FILE *f = fopen(path, "wb");
            if (f) fwrite(j->map, 1, bd_size, f), fclose(f);
            fprintf(bufs_f, "after pass %04d: bytes %#llx..%#llx changed\n", j->pass, (unsigned long long)lo,
                    (unsigned long long)hi);
            if (bd_stride) buf_grid(j->map);
            fflush(bufs_f);
            if (!bd_prev) bd_prev = malloc(bd_size);
            if (bd_prev) memcpy(bd_prev, j->map, bd_size);
        }
    }
    plat_mutex_unlock(&img_lock);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static void buf_snap(VkExec *ex, int pass) {
    if (!bd_size) return;
    VkMemSpan s;
    if (!vk_mem_resolve(bd_addr, bd_size, &s) || s.avail < bd_size) return;
    VK_MEM_PICK(s);   // the GPU's view of the buffer
    BufJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk_dump: out of memory");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bd_size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        return;
    }
    j->map = info.pMappedData, j->pass = pass;
    VkMemoryBarrier2 mb[2] = {
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb[0]};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferCopy c = {s.offset, 0, bd_size};
    vkCmdCopyBuffer(ex->cb, s.buffer, j->buf, 1, &c);
    di.pMemoryBarriers = &mb[1];
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
    vk_completion_add(ex->fence, buf_done, j);
}

static int active(void) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    return !finished && (f == dump_frame || (log_from && f >= log_from && f < dump_frame));
}
static uint64_t frame_now(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1; }
int vk_dump_active_dbg(void) { return active(); }   // DEBUG-DRAWDBG

typedef struct { char *s; size_t n, cap; } Str;

__attribute__((format(printf, 2, 3))) static void sp(Str *b, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    int k = vsnprintf(NULL, 0, fmt, a);
    va_end(a);
    if (k < 0) return;
    if (b->n + (size_t)k + 1 > b->cap) {
        b->cap = (b->n + (size_t)k + 1) * 2;
        b->s = realloc(b->s, b->cap);
        if (!b->s) rt_fatal("vk_dump: out of memory");
    }
    va_start(a, fmt);
    vsnprintf(b->s + b->n, (size_t)k + 1, fmt, a);
    va_end(a);
    b->n += (size_t)k;
}

// SPIR-V hash of a shader; the first time, its SPIR-V goes to sh_<hash>.spv.
static uint64_t shader_hash(const GfxShader *s) {
    static struct { const GfxShader *s; uint64_t h; } tab[1024];
    static int n;
    if (!s) return 0;
    for (int i = 0; i < n; i++)
        if (tab[i].s == s) return tab[i].h;
    uint64_t words = 0, h = 0xcbf29ce484222325ull;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    const uint8_t *b = (const uint8_t *)w;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    char path[600];
    snprintf(path, sizeof path, "%s/sh_%016llx.spv", dir, (unsigned long long)h);
    FILE *f = fopen(path, "wb");
    if (f) fwrite(w, 4, (size_t)words, f), fclose(f);
    if (n < 1024) tab[n].s = s, tab[n].h = h, n++;
    return h;
}

// Per pass: distinct pipeline + texture sets, draws, 3D textures to copy at its end.
typedef struct { VkPipeline p; uint64_t th; int draws; char *text; } Ent;
static Ent ents[MAX_ENTS];
static int nents, ents_over, pass_draws;
typedef struct { VkImage image; VkFormat format; uint32_t w, h, d; int id, is2d; uint32_t levels; } Lut;
static Lut luts[MAX_LUTS];          // queued for the next pass end or dispatch
static int nluts;
static VkImage luts_done[MAX_LUTS];
static int nluts_done;

static void queue_lut(const VkTexImage *t, int id) {
    if (frame_now() != dump_frame) return;
    for (int i = 0; i < nluts_done; i++)
        if (luts_done[i] == t->image) return;
    if (nluts_done == MAX_LUTS) return;
    luts_done[nluts_done++] = t->image;
    luts[nluts++] = (Lut){t->image, t->format, t->width, t->height, t->type == VK_IMAGE_TYPE_3D ? t->depth : 1, id,
                          t->type != VK_IMAGE_TYPE_3D, t->levels};
}

// The uncompressed format of a BC format's block size, UNDEFINED for other formats: a BC texture is dumped as its raw
// blocks (one texel per block), which shows at least whether it was ever written.
static VkFormat block_format(VkFormat f) {
    if (f == VK_FORMAT_BC4_UNORM_BLOCK || f == VK_FORMAT_BC4_SNORM_BLOCK ||
        (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK))
        return VK_FORMAT_R32G32_UINT;
    if (f >= VK_FORMAT_BC2_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) return VK_FORMAT_R32G32B32A32_UINT;
    return VK_FORMAT_UNDEFINED;
}

static void flush_luts(VkExec *ex, int n) {
    char fn[16];
    for (int i = 0; i < nluts; i++) {
        const Lut *l = &luts[i];
        char name[160];
        VkFormat bf = block_format(l->format);
        snprintf(name, sizeof name, "pass_%04d_%s%d_%ux%ux%u_%d_%s%s", n, l->is2d ? "tex" : "lut", l->id, l->w, l->h, l->d,
                 (int)l->format, fmt_name(l->format, fn), bf ? "_blocks" : "");
        const char *why = bf ? copy_region(ex, l->image, bf, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0,
                                           (VkExtent3D){l->w, l->h, l->d}, (l->w + 3) / 4, (l->h + 3) / 4, l->d, name)
                             : copy_out(ex, l->image, l->format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, l->w, l->h, l->d, name);
        for (uint32_t lv = 1; bf && l->is2d && lv < l->levels && lv < 8 && cfg_env("REKARINA_DUMP_TEX_LEVELS"); lv++) {
            // DEBUG: the smaller levels of a BC texture too (_L<level>_blocks)
            uint32_t lw = l->w >> lv ? l->w >> lv : 1, lh = l->h >> lv ? l->h >> lv : 1;
            char ln[180];
            snprintf(ln, sizeof ln, "pass_%04d_tex%d_L%u_%ux%u_%d_blocks", n, l->id, lv, lw, lh, (int)l->format);
            copy_region(ex, l->image, bf, VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, 0, (VkExtent3D){lw, lh, 1}, (lw + 3) / 4,
                        (lh + 3) / 4, 1, ln);
        }
        fprintf(passes_f, "  %s tex %d -> %s.png%s%s\n", l->is2d ? "2D" : "3D", l->id, name, why ? " skipped: " : "",
                why ? why : "");
    }
    nluts = 0;
}

// The NVN slot table a stage's bindings are kept under (vk_exec_draw.c nvn_stage_of).
static int nvn_stage(int gfx_stage) {
#ifdef GAME_NS2
    if (gfx_stage == GFX_STAGE_GS) return 2;
    if (gfx_stage == GFX_STAGE_TS) return 3;   // NVN2_SLOT_TASK, NVN2_SLOT_MESH (nvn_ctl.h)
    if (gfx_stage == GFX_STAGE_MS) return 4;
#endif
    return gfx_stage == GFX_STAGE_VS ? 0 : gfx_stage == GFX_STAGE_FS ? 1 : 5;
}

static const char *decl_type(uint16_t t, char tmp[24]) {
    static const char *base[] = {"?", "1D", "buffer", "2D", "3D", "cube"};
    snprintf(tmp, 24, "%s%s%s%s", base[(t & 0xff) < 6 ? t & 0xff : 0], t & 0x100 ? "Array" : "", t & 0x200 ? "MS" : "",
             t & 0x400 ? "Shadow" : "");
    return tmp;
}

static const char *wrap_name(int w) {
    static const char *n[] = {"clamp", "repeat", "mirror_clamp", "mirror_clamp_edge", "mirror_clamp_border",
                              "clamp_border", "mirrored_repeat", "clamp_edge"};
    return w >= 0 && w < 8 ? n[w] : "?";
}

static uint64_t handle_of(const VkExec *ex, int st, int slot, int img) {
    if (img) return slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
    return slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
}

// The textures, images and samplers of info's bindings; 3D textures are queued for a copy.
static void describe_bindings(Str *b, const VkExec *ex, const VkPipeInfo *info) {
#ifdef GAME_NS2
    static const char *stn[6] = {"vs", "fs", "gs", "ts", "ms", "cs"};
#else
    static const char *stn[6] = {"vs", "fs", "?", "?", "?", "cs"};
#endif
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        const GfxShaderRes *r = bd->res;
        if (r && bd->kind == GFX_RES_SSBO) {   // where the address lives (c0: the stage's NVN slot table) and the stage's slots
            int st = nvn_stage(bd->stage);
            sp(b, "    %s ssbo at c%d+%#x%s, bound:", stn[st], r->cbuf_slot, r->handle * 4, r->flags & 1 ? " written" : "");
            for (int k = 0; k < 16; k++)
                if (ex->ssbo[st][k].addr)
                    sp(b, " %d=%#llx+%#llx", k, (unsigned long long)ex->ssbo[st][k].addr, (unsigned long long)ex->ssbo[st][k].size);
            sp(b, "\n");
            continue;
        }
        if (r && bd->kind == GFX_RES_UBO && dump_ubo && r->slot >= 3 && r->slot - 3 < 18) {
            int st = nvn_stage(bd->stage);
            const p_bind_buf *u = &ex->ubo[st][r->slot - 3];
            VkMemSpan ms;
            sp(b, "    %s ubo c%d addr %#llx size %#llx", stn[st], r->slot, (unsigned long long)u->addr, (unsigned long long)u->size);
            if (u->addr && vk_mem_resolve(u->addr, 1, &ms)) {
                uint64_t n = u->size && u->size < ms.avail ? u->size : ms.avail;
                const float *f = (const float *)(uintptr_t)u->addr;
                for (uint64_t k = 0; k < (uint64_t)dump_ubo && (k + 1) * 16 <= n; k++)
                    sp(b, "%s[%llu] %g %g %g %g", k % 4 ? " " : "\n      ", (unsigned long long)k, f[4 * k], f[4 * k + 1],
                       f[4 * k + 2], f[4 * k + 3]);
            } else {
                sp(b, " (not mapped)");
            }
            sp(b, "\n");
            continue;
        }
        if (!r || (bd->kind != GFX_RES_TEX && bd->kind != GFX_RES_IMG)) continue;
        int st = nvn_stage(bd->stage), img = bd->kind == GFX_RES_IMG;
        char dt[24], fn[16];
        if (bd->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || bd->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
            sp(b, "    %s %s slot %d: buffer texture (zero view bound)\n", stn[st], img ? "img" : "tex", r->slot);
            continue;
        }
        for (uint32_t k = 0; k < (bd->count ? bd->count : 1u); k++) {
            int slot = r->slot + (int)k;
            uint64_t h = handle_of(ex, st, slot, img);
            sp(b, "    %s %s slot %d decl %s%s: ", stn[st], img ? "img" : "tex", slot, decl_type(r->type, dt),
               img && (r->flags & 4) ? " store" : "");
            if (!h) { sp(b, "no handle (dummy)\n"); continue; }
            view_t v;
            int hv = 0;
            const VkTexImage *t = vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv);
            if (!t) { sp(b, "tex %d unresolved (dummy)\n", NVN_HANDLE_TEX(h)); continue; }
            static const char *vt[] = {"1D", "2D", "3D", "cube", "1DArray", "2DArray", "cubeArray"};
            sp(b, "tex %d %s(%d) %ux%ux%u levels %u layers %u image %s", NVN_HANDLE_TEX(h), fmt_name(t->format, fn),
               (int)t->format, t->width, t->height, t->depth, t->levels, t->layers,
               t->view_type <= 6 ? vt[t->view_type] : "?");
            sp(b, " vk %#llx s%u", (unsigned long long)(uintptr_t)t->image, t->serial);   // which image of the storage's entries
            if (hv && v.set_format) sp(b, " view-nvn-fmt %#x", v.format);
            if (hv && v.set_levels) sp(b, " view-levels %u+%u", v.base_level, v.nlevels);
            if (hv && v.set_layers) sp(b, " view-layers %u+%u", v.min_layer, v.nlayers);
            samp_par s;
            if (img) {
                // storage image: no sampler
            } else if (r->separate) {
                sp(b, " | separate sampler (not replayed: default sampler)");
            } else if (vk_sampler_params(NVN_HANDLE_SAMP(h), &s)) {
                static const char *minn[] = {"N", "L", "N_mipN", "L_mipN", "N_mipL", "L_mipL"};
                int mf = s.minf;
                sp(b, " | samp %d: mag %s min %s (vk min %s mip %s) wrap %s/%s/%s aniso %g lod %g..%g bias %g%s",
                   NVN_HANDLE_SAMP(h), s.magf == 1 ? "L" : "N", mf < 6 ? minn[mf] : "?",
                   mf == 1 || mf == 3 || mf == 5 ? "L" : "N", mf <= 1 ? "none" : mf >= 4 ? "L" : "N",
                   wrap_name(s.ws), wrap_name(s.wt), wrap_name(s.wr), s.aniso, s.lod_min, s.lod_max, s.lod_bias,
                   s.cmp_mode ? " compare" : "");
            } else {
                sp(b, " | samp %d empty (default sampler)", NVN_HANDLE_SAMP(h));
            }
            uint32_t ups;
            int last, wr;
            vk_texture_debug(t, &ups, &last, &wr);
            sp(b, " | addr %#llx mirrored %d gpu_owned %d bound %d uploads %u last-check %d written-since %d",
               (unsigned long long)t->addr, t->mirrored, t->gpu_owned, t->bound, ups, last, wr);
            uint8_t ssw[4] = {0};
            vk_texture_slot_swz(NVN_HANDLE_TEX(h), ssw);
            sp(b, " | swz image %u%u%u%u slot %u%u%u%u", t->swz[0], t->swz[1], t->swz[2], t->swz[3], ssw[0], ssw[1], ssw[2],
               ssw[3]);
            if (hv && v.set_swizzle) sp(b, " view %u%u%u%u", v.swz[0], v.swz[1], v.swz[2], v.swz[3]);
            sp(b, "\n");
            int wanted = 0;
            for (int k = 0; k < nwant_tex; k++)
                wanted |= want_tex[k][1] ? want_tex[k][0] == t->width && want_tex[k][1] == t->height
                                         : want_tex[k][0] == (uint32_t)NVN_HANDLE_TEX(h);
            if (t->type == VK_IMAGE_TYPE_3D) queue_lut(t, NVN_HANDLE_TEX(h));
            else if (wanted && t->type == VK_IMAGE_TYPE_2D && t->samples == 1) queue_lut(t, NVN_HANDLE_TEX(h));
            else if (!img && t->type == VK_IMAGE_TYPE_2D && t->samples == 1 && (uint64_t)t->width * t->height <= TEX2D_CAP &&
                     (t->format == VK_FORMAT_R8_UNORM || t->format == VK_FORMAT_R8G8_UNORM ||
                      t->format == VK_FORMAT_R8G8B8A8_UNORM || t->format == VK_FORMAT_R8G8B8A8_SRGB ||
                      t->format == VK_FORMAT_B8G8R8A8_UNORM || t->format == VK_FORMAT_B8G8R8A8_SRGB))
                queue_lut(t, NVN_HANDLE_TEX(h));
        }
    }
}

static uint64_t bindings_hash(const VkExec *ex, const VkPipeInfo *info) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        if (bd->res && bd->kind == GFX_RES_SSBO) {
            int st = nvn_stage(bd->stage), slot = bd->res->slot;
            if (slot >= 0 && slot < 16) h = (h ^ ex->ssbo[st][slot].addr) * 0x100000001b3ull;
            continue;
        }
        if (!bd->res || (bd->kind != GFX_RES_TEX && bd->kind != GFX_RES_IMG)) continue;
        for (uint32_t k = 0; k < (bd->count ? bd->count : 1u); k++)
            h = (h ^ handle_of(ex, nvn_stage(bd->stage), bd->res->slot + (int)k, bd->kind == GFX_RES_IMG)) * 0x100000001b3ull;
    }
    return h;
}

void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info) {
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    pass_draws++;
    // The depth state is dynamic, not part of the pipeline: draws that differ in it are listed apart.
    uint64_t th = bindings_hash(ex, info) ^ ((uint64_t)s->ds.dt << 61 | (uint64_t)s->ds.dw << 60 | (uint64_t)s->ds.df << 56);
#ifdef GAME_NS2   // conditional draws are listed apart from the same draws without a condition
    if (ex->cond_active) th ^= ex->render_cond.addr * 31 + (uint64_t)ex->render_cond.mode;
#endif
    for (int i = 0; i < nents; i++)
        if (ents[i].p == p && ents[i].th == th) {
            ents[i].draws++;
            goto out;
        }
    if (nents == MAX_ENTS) {
        ents_over++;
        goto out;
    }
    Str b = {0};
    sp(&b, "vs %016llx fs %016llx prog %llu prim %d depth test %d write %d func %d blend %#x t0 src %u dst %u eq %u "
           "a %u/%u/%u wmask %#x\n",
       (unsigned long long)shader_hash(s->vs), (unsigned long long)shader_hash(s->fs),
       (unsigned long long)(ex->prog ? ex->prog->serial : 0), s->prim, (int)s->ds.dt, (int)s->ds.dw, (int)s->ds.df,
       (unsigned)s->color.blend_en, (unsigned)s->blend[0].src_rgb, (unsigned)s->blend[0].dst_rgb, (unsigned)s->blend[0].eq_rgb,
       (unsigned)s->blend[0].src_a, (unsigned)s->blend[0].dst_a, (unsigned)s->blend[0].eq_a, (unsigned)s->chmask.mask);
#ifdef GAME_NS2
    if (ex->cond_active)
        sp(&b, "    conditional on %#llx mode %d gpu %d cpu %d\n", (unsigned long long)ex->render_cond.addr,
           (int)ex->render_cond.mode, ex->gpu_cond, ex->cond_cpu);
#endif
    if (rt_knob("DUMP_SHADER_IDS", "0")) {
        const uint8_t *ids[] = {s->vs ? gfx_shader_input_id(s->vs) : NULL,
                                s->fs ? gfx_shader_input_id(s->fs) : NULL};
        for (unsigned stage = 0; stage < 2; ++stage) {
            if (!ids[stage]) continue;
            sp(&b, "    %s input id ", stage ? "fs" : "vs");
            for (unsigned j = 0; j < 32; ++j) sp(&b, "%02x", (unsigned)ids[stage][j]);
            sp(&b, "\n");
        }
    }
    describe_bindings(&b, ex, info);
    ents[nents++] = (Ent){p, th, 1, b.s};
out:
    plat_mutex_unlock(&rec_lock);
}

static void target_line(VkExec *ex, const VkExecTarget *t, const char *tag, int n) {
    char fn[16], name[160];
    if (!t->image) {
        fprintf(passes_f, "  %s: nvn %#x target %d %ux%u: not renderable, left out\n", tag, t->tex->d.format,
                t->tex->d.target, t->tex->d.width, t->tex->d.height);
        return;
    }
    snprintf(name, sizeof name, "pass_%04d_%s_%ux%u_%d_%s", n, tag, t->width, t->height, (int)t->format,
             fmt_name(t->format, fn));
    // a block view: the plane's blocks, read as the view format (one texel per block)
    uint32_t tw = t->tex->d.width >> t->level ? t->tex->d.width >> t->level : 1;
    uint32_t th = t->tex->d.height >> t->level ? t->tex->d.height >> t->level : 1;
    (void)n;
    const char *why = pass_skip ? pass_skip : t->samples > 1 ? "multisampled" :
                      t->blocks ? copy_region(ex, t->image, t->format, t->aspect, t->level, t->is3d ? 0 : t->base_layer,
                                              t->is3d ? t->base_layer : 0, (VkExtent3D){tw, th, 1}, t->width,
                                              t->height, 1, name) :
                      t->is3d ? "3D slice (the volume is dumped as lut<id> where it is sampled)" :
                      copy_out(ex, t->image, t->format, t->aspect, t->level, t->base_layer, t->width, t->height, 1, name);
    fprintf(passes_f, "  %s: nvn %#x %s(%d) %ux%u level %u layers %u+%u samples %u addr %#llx serial %u -> %s.png%s%s\n", tag,
            t->nvn_format, fmt_name(t->format, fn), (int)t->format, t->width, t->height, t->level, t->base_layer,
            t->layers, t->samples, (unsigned long long)(uintptr_t)t->tex->mem, t->img ? t->img->serial : 0,
            name, why ? " skipped: " : "", why ? why : "");
}

static uint64_t targets_sig(const VkExec *ex) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        uint64_t v[3] = {(uint64_t)(uintptr_t)t->image, t->level, t->base_layer};
        for (int k = 0; k < 3; k++) h = (h ^ v[k]) * 1099511628211ull;
    }
    return h;
}

// Sets pass_skip for the pass or dispatch being logged and counts it when it copies.
static void pass_decide(int same_targets) {
    if (frame_now() != dump_frame) copied = 0;
    pass_skip = copied >= MAX_PASSES ? "pass cap" : same_targets ? "same targets as the previous pass" : NULL;
    if (!pass_skip) copied++;
}

void vk_dump_pass_end(VkExec *ex) {
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    int n = seq++;
    uint64_t sig = targets_sig(ex);
    pass_decide(skip_runs && sig == last_targets);
    last_targets = sig;
    fprintf(passes_f, "pass %04d: render pass, draws %d, area %ux%u layers %u, fence %llu, frame %llu\n", n, pass_draws,
            ex->pass_w, ex->pass_h, ex->pass_layers, (unsigned long long)ex->fence, (unsigned long long)frame_now());
    for (int i = 0; i < 8; i++) {
        if (!ex->color[i].tex) continue;
        char tag[8];
        snprintf(tag, sizeof tag, "t%d", i);
        target_line(ex, &ex->color[i], tag, n);
    }
    if (ex->depth.tex) target_line(ex, &ex->depth, "td", n);
    for (int i = 0; i < nents; i++) {
        fprintf(passes_f, "  pipe %d (draws %d): %s", i, ents[i].draws, ents[i].text);
        free(ents[i].text);
    }
    if (ents_over) fprintf(passes_f, "  ... %d more draws with other pipeline/texture sets\n", ents_over);
    if (!pass_skip) flush_luts(ex, n);
    else nluts = 0;
    buf_snap(ex, n);
    fflush(passes_f);
    nents = ents_over = pass_draws = 0;
out:
    plat_mutex_unlock(&rec_lock);
}

void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d) {
    (void)p;
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    int n = seq++;
    pass_decide(0);
    last_targets = 0;
    Str b = {0};
    describe_bindings(&b, ex, info);
    fprintf(passes_f, "pass %04d: dispatch %dx%dx%d cs %016llx prog %llu caller %#llx, fence %llu\n%s", n, d->x, d->y,
            d->z, (unsigned long long)shader_hash(cs), (unsigned long long)(ex->prog ? ex->prog->serial : 0),
            (unsigned long long)d->caller, (unsigned long long)ex->fence, b.s ? b.s : "");
    free(b.s);
    for (uint32_t i = 0; i < info->n_bindings && !pass_skip; i++) {   // storage images the dispatch writes
        const VkPipeBinding *bd = &info->bindings[i];
        if (!bd->res || bd->kind != GFX_RES_IMG || !(bd->res->flags & 4) || bd->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            continue;
        uint64_t h = handle_of(ex, 5, bd->res->slot, 1);
        view_t v;
        int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (!t) continue;
        uint32_t lv = hv && v.set_levels ? v.base_level : 0, layer = hv && v.set_layers ? v.min_layer : 0;
        if (lv >= t->levels || layer >= t->layers) continue;
        uint32_t w = t->width >> lv ? t->width >> lv : 1, hh = t->height >> lv ? t->height >> lv : 1;
        uint32_t dd = t->type == VK_IMAGE_TYPE_3D ? (t->depth >> lv ? t->depth >> lv : 1) : 1;
        char fn[16], name[160];
        snprintf(name, sizeof name, "pass_%04d_img%d_%ux%u%s_%d_%s", n, bd->res->slot, w, hh, dd > 1 ? "x3d" : "",
                 (int)t->format, fmt_name(t->format, fn));
        if (t->type == VK_IMAGE_TYPE_3D) {   // a written volume: this copy replaces the sampled-3D one
            for (int k = 0; k < nluts_done; k++)
                if (luts_done[k] == t->image) luts_done[k] = VK_NULL_HANDLE;
        }
        const char *why = t->samples > 1 ? "multisampled" :
                          copy_out(ex, t->image, t->format, t->aspect, lv, layer, w, hh, dd, name);
        fprintf(passes_f, "  writes img slot %d level %u layer %u depth %u -> %s.png%s%s\n", bd->res->slot, lv, layer,
                dd, name, why ? " skipped: " : "", why ? why : "");
    }
    if (!pass_skip) flush_luts(ex, n);
    else nluts = 0;
    buf_snap(ex, n);
    fflush(passes_f);
out:
    plat_mutex_unlock(&rec_lock);
}

static const char *tex_desc(const tex_t *t, char *buf, size_t n) {
    if (!t) return "null";
    snprintf(buf, n, "addr %#llx nvn-fmt %#x %ux%ux%u", (unsigned long long)(uintptr_t)t->mem, t->d.format, t->d.width,
             t->d.height, t->d.depth);
    return buf;
}

static void region_desc(const NVNcopyRegion *r, char *buf, size_t n) {
    snprintf(buf, n, "(%d,%d,%d %dx%dx%d)", r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth);
}

// Native pre-bloom DLAA's HUD target right after the tone-mapped picture was copied into it, before the HUD draws.
void vk_dump_dlaa_copy(VkExec *ex, const VkTexImage *target) {
    plat_mutex_lock(&rec_lock);
    if (active()) {
        const char *why = target->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            ? copy_out(ex, target->image, target->format, target->aspect, 0, 0, target->width, target->height, 1,
                       "dlaa_hud_destination")
            : "image has no transfer-source usage";
        fprintf(passes_f, "DLAA native copy after pass %d: dlaa_hud_destination%s%s\n", seq - 1, why ? " skipped: " : "",
                why ? why : "");
    }
    plat_mutex_unlock(&rec_lock);
}

void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex;
    plat_mutex_lock(&rec_lock);
    if (!active()) goto out;
    char a[96], b[96], ra[64], rb[64];
    switch (op) {
    case OP_COPY_B2B: {
        const p_copy_b2b *p = payload;
        fprintf(passes_f, "copy after pass %d: buffer %#llx -> buffer %#llx, %llu bytes\n", seq - 1,
                (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
        break;
    }
    case OP_COPY_B2T: case OP_COPY_T2B: {
        const p_copy_b2t *p = payload;
        region_desc(&p->r, ra, sizeof ra);
        fprintf(passes_f, "copy after pass %d: %s buffer %#llx %s texture %s region %s flags %#x\n", seq - 1,
                op == OP_COPY_B2T ? "buffer->texture" : "texture->buffer", (unsigned long long)p->src,
                op == OP_COPY_B2T ? "->" : "<-", tex_desc(p->tex, a, sizeof a), ra, p->flags);
        break;
    }
    case OP_COPY_T2T: {
        const p_copy_t2t *p = payload;
        region_desc(&p->sr, ra, sizeof ra);
        region_desc(&p->dr, rb, sizeof rb);
        fprintf(passes_f, "copy after pass %d: texture %s %s -> texture %s %s flags %#x\n", seq - 1,
                tex_desc(p->src, a, sizeof a), ra, tex_desc(p->dst, b, sizeof b), rb, p->flags);
        break;
    }
    case OP_CLEAR_TEX: {
        const p_clear_tex *p = payload;
        region_desc(&p->r, ra, sizeof ra);
        fprintf(passes_f, "clear after pass %d: texture %s region %s mask %#x\n", seq - 1, tex_desc(p->tex, a, sizeof a),
                ra, p->mask);
        break;
    }
    default: break;
    }
out:
    plat_mutex_unlock(&rec_lock);
}

static void make_dirs(char *path) { plat_file_mkdir_p(path); }

void vk_dump_init(void) {
    const char *e = cfg_get("REKARINA_DUMP");
    if (e && *e == 't') {   // REKARINA_DUMP=t<ticks>: the frame after game time reaches that many 60 Hz ticks
        dump_tick = strtoull(e + 1, NULL, 10);   // (frame_step_ticks): the same scene at any frame rate
        dump_frame = dump_tick ? UINT64_MAX : 0;
        if (!dump_frame) return;
    } else if (!e || !*e || !(dump_frame = strtoull(e, NULL, 10))) {
        return;
    }
    const char *lf = cfg_get("REKARINA_DUMP_LOG");
    if (lf && *lf) log_from = strtoull(lf, NULL, 10);
    const char *sr = cfg_get("REKARINA_DUMP_SKIP_RUNS");
    skip_runs = sr && *sr && *sr != '0';
    const char *du = cfg_get("REKARINA_DUMP_UBO");
    dump_ubo = du && *du ? atoi(du) : 0;
    if (dump_ubo == 1) dump_ubo = 16;
    if (dump_ubo > 4096) dump_ubo = 4096;
    if (dump_ubo < 0) dump_ubo = 0;
    const char *wt = cfg_get("REKARINA_DUMP_TEX");
    for (char *q = (char *)wt; q && *q && nwant_tex < 16; q += *q == ',') {
        uint32_t *w = want_tex[nwant_tex++];
        w[0] = (uint32_t)strtoul(q, &q, 10);
        w[1] = *q == 'x' ? (uint32_t)strtoul(q + 1, &q, 10) : 0;
    }
    const char *d = cfg_get("REKARINA_DUMP_DIR");
    snprintf(dir, sizeof dir, "%s", d && *d ? d : "dump");
    make_dirs(dir);
    char path[600];
    snprintf(path, sizeof path, "%s/passes.txt", dir);
    passes_f = fopen(path, "w");
    snprintf(path, sizeof path, "%s/images.txt", dir);
    images_f = fopen(path, "w");
    if (!passes_f || !images_f) rt_fatal("vk_dump: cannot write to %s", dir);
    const char *bf = cfg_get("REKARINA_DUMP_BUF");
    if (bf && *bf) {
        char *q;
        bd_addr = strtoull(bf, &q, 0);
        if (*q == ':') bd_size = strtoull(q + 1, &q, 0);
        if (*q == ':') bd_off = strtoull(q + 1, &q, 0);
        if (*q == ':') bd_stride = strtoull(q + 1, &q, 0) & ~3ull;
        if (*q == ':') bd_w = (uint32_t)strtoul(q + 1, &q, 0);
        if (*q == ':') bd_h = (uint32_t)strtoul(q + 1, &q, 0);
        if (!bd_w || !bd_h) bd_stride = 0;
        snprintf(path, sizeof path, "%s/bufs.txt", dir);
        if (bd_size && !(bufs_f = fopen(path, "w"))) bd_size = 0;
    }
    vk_dump_armed = 1;
    rt_log("vk_dump: frame %llu goes to %s", (unsigned long long)dump_frame, dir);
}

// DEBUG-LUMLOG: with REKARINA_DUMP=t<ticks never reached>, dump the frame after the next present (nvn_api_cmd.c arms it at
// the first copy of OoT's luminance read-back)
void vk_dump_arm_soon(void) {
    if (dump_tick && dump_frame == UINT64_MAX) {
        const char *o = cfg_env("REKARINA_DUMP_ARM_OFS");   // frames after the present count at arming (default 2)
        dump_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + (uint64_t)(o && *o ? atoi(o) : 2);
        rt_log("vk_dump: armed: frame %llu is dumped", (unsigned long long)dump_frame);
    }
}

// DEBUG-DUMPCLEARS: REKARINA_DUMP_CLEARS=<n> with REKARINA_DUMP=t<ticks never reached>: the frame of the first submission
// that holds n or more clears is dumped (from that submission on). OoT's frames have 23; the ones with 45
// are those in which the sun's light drops out for a frame or two while the camera turns (docs/oot-notes.md).
static int count_clears(const op_hdr *h, int depth) {
    int n = 0;
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_ || h->op == OP_END) return n;
        const void *a = nvn_op_payload(h);
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < 8) n += count_clears(nvn_op_first(((const p_jump *)a)->target), depth + 1);
        n += h->op == OP_CLEAR_COLOR || h->op == OP_CLEAR_DS;
        h = nvn_op_next(h);
    }
}
// DEBUG-DUMPDARK: REKARINA_DUMP_ON_DARK=1 with REKARINA_DUMP=t<ticks never reached>: the first dark frame the flight
// recorder's flash watch sees past frame 3200 (REKARINA_FLIGHT_FLASH_DARK sets how dark) arms a dump of the frame being
// recorded: a run of dark frames is still going on four presents after its start, when the watch sees it.
void vk_dump_arm_dark(uint64_t dark_frame) {
    static int on = -1;
    if (on < 0) on = cfg_get("REKARINA_DUMP_ON_DARK") != NULL;
    if (!on || !dump_tick || dump_frame != UINT64_MAX || dark_frame < 3200) return;
    dump_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    rt_log("vk_dump: frame %llu was dark: frame %llu is dumped", (unsigned long long)dark_frame, (unsigned long long)dump_frame);
}
// DEBUG-DUMPEMPTY: REKARINA_DUMP_EMPTY=1 with REKARINA_DUMP=t<ticks never reached>: the frame being recorded is dumped from
// here on (vk_texture.c calls it when a new 640x360 depth image at an address with no content appears after frame
// 3200: the frame in which OoT's transient targets move to other addresses).
void vk_dump_arm_now(const char *why) {
    static int on = -1;
    if (on < 0) on = cfg_get("REKARINA_DUMP_EMPTY") != NULL;
    if (!on || !dump_tick || dump_frame != UINT64_MAX) return;
    if (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) < 3200) return;
    dump_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    rt_log("vk_dump: %s: frame %llu is dumped from here on", why, (unsigned long long)dump_frame);
}
// DEBUG-DUMPRETURN: REKARINA_DUMP_RETURN=<n>[:<from frame>] with REKARINA_DUMP=t<ticks never reached> and
// REKARINA_FLIGHT=1: the frame n presents after the scene came back from a dark spell (the pause menu closing; flight.c's
// flash watch sees it) is dumped, the first time past <from frame> only (the boot's loading screens are dark spells too).
void vk_dump_arm_return(uint64_t returned) {
    static int ofs = -2;
    static uint64_t from;
    if (ofs == -2) {
        const char *e = cfg_get("REKARINA_DUMP_RETURN");
        ofs = e && *e ? atoi(e) : -1;
        const char *c = e ? strchr(e, ':') : NULL;
        from = c ? strtoull(c + 1, NULL, 10) : 0;
    }
    if (ofs < 0 || returned < from || !dump_tick || dump_frame != UINT64_MAX) return;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), at = returned + (uint64_t)ofs;
    dump_frame = at > now ? at : now + 1;
    rt_log("vk_dump: the scene came back at frame %llu: frame %llu is dumped", (unsigned long long)returned,
           (unsigned long long)dump_frame);
}
// DEBUG-DUMPAFTER: REKARINA_DUMP_AFTER_CLEARS=<n> with REKARINA_DUMP=t<ticks never reached>: the frame after the first frame
// with n clears or more (past frame 3200, the frame before it with fewer) is dumped whole (vk_exec.c pl_frame).
void vk_dump_arm_after(uint64_t frame, uint64_t clears, uint64_t before) {
    static int min = -1;
    if (min < 0) { const char *e = cfg_get("REKARINA_DUMP_AFTER_CLEARS"); min = e && atoi(e) > 0 ? atoi(e) : 0; }
    if (!min || !dump_tick || dump_frame != UINT64_MAX || frame < 3200 || clears < (uint64_t)min || before >= (uint64_t)min) return;
    const char *o = cfg_env("REKARINA_DUMP_ARM_OFS");
    dump_frame = frame + (uint64_t)(o && *o ? atoi(o) : 1);
    rt_log("vk_dump: frame %llu had %llu clears (the one before %llu): frame %llu is dumped", (unsigned long long)frame,
           (unsigned long long)clears, (unsigned long long)before, (unsigned long long)dump_frame);
}
void vk_dump_arm_clears(const uint64_t *handles, int n) {
    static int min = -1;
    if (min < 0) { const char *e = cfg_get("REKARINA_DUMP_CLEARS"); min = e && atoi(e) > 0 ? atoi(e) : 0; }
    if (!min || !dump_tick || dump_frame != UINT64_MAX) return;
    if (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) < 3000) return;   // past the load
    int c = 0;
    for (int i = 0; i < n; i++)
        if (handles[i]) c += count_clears(nvn_op_first(handles[i]), 0);
    if (c < min) return;
    dump_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    rt_log("vk_dump: a submission with %d clears: frame %llu is dumped from here on", c, (unsigned long long)dump_frame);
}

void vk_dump_present(uint64_t frame) {
    if (dump_tick && dump_frame == UINT64_MAX && frame_step_ticks() >= dump_tick) {
        dump_frame = frame + 2;   // recording starts with the next frame's commands
        rt_log("vk_dump: tick %llu reached at frame %llu: frame %llu is dumped", (unsigned long long)frame_step_ticks(),
               (unsigned long long)frame, (unsigned long long)dump_frame);
    }
    if (frame != dump_frame) return;
    plat_mutex_lock(&rec_lock);
    finished = 1;
    plat_mutex_unlock(&rec_lock);
    vk_sync_finish();   // the frame's copies are converted and written
    plat_mutex_lock(&img_lock);
    fclose(images_f);
    images_f = NULL;
    if (bufs_f) fclose(bufs_f), bufs_f = NULL;
    plat_mutex_unlock(&img_lock);
    fclose(passes_f);
    passes_f = NULL;
    rt_log("vk_dump: frame %llu: %d passes and dispatches, about %llu MB of PNGs in %s", (unsigned long long)frame, seq,
           (unsigned long long)(png_bytes >> 20), dir);
}

// ---- DEBUG-PROBE: REKARINA_PASS_PROBE=1 (used with REKARINA_FLIGHT=1)
// After every render pass the centre texel of colour target 0 (of the depth target when there is no colour) and a
// PR_GX x PR_GY grid of texels over it are copied into a host-visible ring of PR_FRAMES frames. A flight dump
// snapshots the ring (vk_probe_snap) and writes probes.tsv: frame, pass number in the frame, target address, format,
// size, the decoded centre texel and how many grid texels hold a NaN or an infinity (with the first one's place), so a
// frame that goes wrong on screen can be compared pass by pass with its neighbours and the first pass that makes a NaN
// stands out. Frame = gfx_counters.frames + 1 at record time.
#define PR_FRAMES 40
#define PR_PASSES 400
#define PR_GX 16
#define PR_GY 9
#define PR_BX 12                       // and a PR_BX x PR_BX grid over the box below (where the hero stands on the porch)
#define PR_BOX_X0 0.40f
#define PR_BOX_X1 0.60f
#define PR_BOX_Y0 0.58f
#define PR_BOX_Y1 0.86f
#define PR_PTS (1 + PR_GX * PR_GY + PR_BX * PR_BX)
#define PR_SLOT (PR_PTS * 16)
typedef struct { uint64_t addr; const Fmt *f; uint32_t w, h, dispatches; } ProbeMeta;
typedef struct VkProbeSnap VkProbeSnap;
// Per draw into a full-size R11G11B10F target (REKARINA_PROBE_SKIP=<addr> leaves one target out): every bound uniform and
// storage block of the vertex and fragment stages, as guest memory holds it at the recording (at most 16 KiB of each):
// non-finite floats in it, a hash, and whether the replay binds a submit-time copy (vk_usnap_find).
#define PD_DRAWS 64
#define PD_BINDS 40
typedef struct { uint64_t addr, size, hash; uint32_t bad; uint8_t st, kind, slot, snap; } ProbeBind;
typedef struct { uint64_t target, vs, fs, sh_addr; int32_t sh_idx; uint32_t n, count, inst; int8_t cond, gpu_cond, cond_cpu; ProbeBind b[PD_BINDS]; } ProbeDraw;
struct VkProbeSnap {
    uint64_t tag[PR_FRAMES];
    uint32_t n[PR_FRAMES];
    ProbeMeta meta[PR_FRAMES][PR_PASSES];
    uint64_t dtag[PR_FRAMES];
    uint32_t nd[PR_FRAMES];
    ProbeDraw d[PR_FRAMES][PD_DRAWS];
    uint8_t sh[PR_FRAMES][PD_DRAWS][144];
    uint8_t ub[PR_FRAMES][2][5][4096];   // [frame][vs, fs][c3..c7]
    uint8_t gub[PR_FRAMES][11][4096];    // the same as the GPU reads it at the pass end, and [10]: the support buffer
    uint64_t ub_addr[PR_FRAMES][2][5];
    uint8_t px[PR_FRAMES][PR_PASSES][PR_SLOT];
};
static VkBuffer pr_buf;
static VmaAllocation pr_alloc;
static uint8_t *pr_map;
static VkProbeSnap *pr;   // live tags, counts and metadata (px unused: the texels live in pr_map)
static int pr_on = -1, pr_light;
static PlatMutex pr_lock = PLAT_MUTEX_INIT;
static struct { VkImage image; VkFormat format; uint32_t w, h, levels; uint64_t addr; } cube[2];
static int cube_n;
static struct { VkImage image; VkFormat format; VkImageAspectFlags aspect; uint32_t w, h; uint64_t addr; } vsp[8];
static int nvsp;
// The first full-size R11G11B10F target of each frame other than REKARINA_PROBE_SKIP (OoT's character layer) is also
// copied whole in a crop around the porch's the hero (CR_W x CR_H from CR_X0, CR_Y0), PR_FRAMES frames kept.
#define CR_X0 640
#define CR_Y0 500
#define CR_W 640
#define CR_H 560
#define CR_FRAMES 96
static VkBuffer cr_buf;
static VmaAllocation cr_alloc;
static uint8_t *cr_map;
static uint64_t cr_tag[CR_FRAMES];
static VkBuffer gub_buf;
static VmaAllocation gub_alloc;
static uint8_t *gub_map;
static VkDescriptorBufferInfo gub_src[PR_FRAMES][11];
static uint64_t gub_frame[PR_FRAMES];
static int gub_pending = -1;   // the frame slot whose first draw's buffers the open pass copies at its end
// Page check of the first probed draw's fragment storage slots 0 and 1 (OoT's skin: light lists and lights): the CPU's
// bytes at the draw (a hash per 4 KiB page) against the GPU's copy at the pass end, compared on the completion thread
// and logged per frame ("PAGECHK").
#define PC_SLOTS 4
#define PC_MAX (16ull << 20)
typedef struct { uint64_t frame, addr[2], size[2]; uint64_t *hash[2]; int slot; } PcJob;
static VkBuffer pc_buf;
static VmaAllocation pc_alloc;
static uint8_t *pc_map;
static PcJob pc_job[PC_SLOTS];
static int pc_next, pc_pending = -1;
static VkMemSpan pc_span[2];
static uint64_t pc_hash(const uint8_t *p, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    return h;
}
static uint64_t *pc_prev;   // the GPU's page hashes of storage slot 0 in the previous checked frame
static uint64_t pc_prev_n;
static void pc_done(void *arg) {   // completion thread
    PcJob *j = arg;
    const uint8_t *g = pc_map + (uint64_t)j->slot * PC_MAX;
    if (j->hash[0]) {   // pages of slot 0 whose GPU bytes changed since the previous checked frame
        uint64_t pages = (j->size[0] + 4095) / 4096, ch = 0, offs[6];
        uint64_t *cur = malloc(pages * 8);
        for (uint64_t p = 0; cur && p < pages; p++) {
            uint64_t n = j->size[0] - p * 4096 < 4096 ? j->size[0] - p * 4096 : 4096;
            cur[p] = pc_hash(g + p * 4096, n);
            if (pc_prev && p < pc_prev_n && cur[p] != pc_prev[p]) {
                if (ch < 6) offs[ch] = p * 4096;
                ch++;
            }
        }
        if (cur) {
            rt_log("PAGEGPU frame %llu: %llu pages of slot 0 changed on the GPU (%#llx %#llx %#llx %#llx %#llx %#llx)",
                   (unsigned long long)j->frame, (unsigned long long)ch, (unsigned long long)(ch > 0 ? offs[0] : 0),
                   (unsigned long long)(ch > 1 ? offs[1] : 0), (unsigned long long)(ch > 2 ? offs[2] : 0),
                   (unsigned long long)(ch > 3 ? offs[3] : 0), (unsigned long long)(ch > 4 ? offs[4] : 0),
                   (unsigned long long)(ch > 5 ? offs[5] : 0));
            free(pc_prev), pc_prev = cur, pc_prev_n = pages;
        }
    }
    {   // DEBUG: REKARINA_PAGECHK_SAVE=<dir>: the GPU's bytes of storage slot 1 (OoT skin: the light structs) per frame
        static const char *sd = (const char *)1;
        if (sd == (const char *)1) sd = cfg_get("REKARINA_PAGECHK_SAVE");
        if (sd && j->hash[0] && j->frame >= 815 && j->frame <= 875) {   // and slot 0 (the light lists) in a short window
            char fp0[1100];
            snprintf(fp0, sizeof fp0, "%s/lists_%06llu_%llx.bin", sd, (unsigned long long)j->frame, (unsigned long long)j->addr[0]);
            FILE *f0 = fopen(fp0, "wb");
            if (f0) fwrite(g, 1, (size_t)j->size[0], f0), fclose(f0);
        }
        if (sd && j->hash[1] && j->frame >= 780 && j->frame <= 1000) {
            char fp_[1100];
            snprintf(fp_, sizeof fp_, "%s/lights_%06llu_%llx.bin", sd, (unsigned long long)j->frame, (unsigned long long)j->addr[1]);
            FILE *fl = fopen(fp_, "wb");
            uint64_t a0 = (j->size[0] + 4095) & ~4095ull;
            if (fl) fwrite(g + a0, 1, (size_t)j->size[1], fl), fclose(fl);
        }
    }
    char line[600];
    int o = snprintf(line, sizeof line, "PAGECHK frame %llu", (unsigned long long)j->frame);
    uint64_t at = 0;
    for (int b = 0; b < 2; b++) {
        if (!j->hash[b]) continue;
        uint64_t pages = (j->size[b] + 4095) / 4096, bad = 0, first = UINT64_MAX;
        for (uint64_t p = 0; p < pages; p++) {
            uint64_t n = j->size[b] - p * 4096 < 4096 ? j->size[b] - p * 4096 : 4096;
            if (pc_hash(g + at + p * 4096, n) != j->hash[b][p]) {
                bad++;
                if (first == UINT64_MAX) first = p;
            }
        }
        o += snprintf(line + o, sizeof line - (size_t)o, " | s%d %#llx: %llu of %llu pages differ (first +%#llx)", b,
                      (unsigned long long)j->addr[b], (unsigned long long)bad, (unsigned long long)pages,
                      first == UINT64_MAX ? 0ull : (unsigned long long)(first * 4096));
        at += (j->size[b] + 4095) & ~4095ull;
        free(j->hash[b]), j->hash[b] = NULL;
    }
    rt_log("%s", line);
}

static VkBuffer sh_buf;
static VmaAllocation sh_alloc;
static uint8_t *sh_map;
static uint32_t sh_pass_first[PR_FRAMES], sh_pass_tag_ok;   // the draw index the open pass started at
static uint64_t sh_pass_frame;

static int probe_init(void) {
    if (pr_on >= 0) return pr_on;
    plat_mutex_lock(&pr_lock);
    if (pr_on < 0) {
        const char *e = cfg_get("REKARINA_PASS_PROBE");
        int on = e && (*e == '1' || *e == '2');
        pr_light = e && *e == '2';   // 2: crops and page checks only, no texel grids (they load the GPU)
        if (on) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                     .size = (VkDeviceSize)PR_FRAMES * PR_PASSES * PR_SLOT,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                   VMA_ALLOCATION_CREATE_MAPPED_BIT};
            VmaAllocationInfo info;
            pr = calloc(1, sizeof *pr);
            if (!pr || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pr_buf, &pr_alloc, &info) != VK_SUCCESS ||
                !info.pMappedData)
                on = 0, rt_log("probe: host buffer allocation failed");
            else
                pr_map = info.pMappedData, memset(pr_map, 0, (size_t)PR_FRAMES * PR_PASSES * PR_SLOT),
                bi.size = (VkDeviceSize)PC_SLOTS * PC_MAX,
                pc_map = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pc_buf, &pc_alloc, &info) == VK_SUCCESS ? info.pMappedData : NULL,
                bi.size = (VkDeviceSize)PR_FRAMES * 11 * 4096,
                gub_map = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &gub_buf, &gub_alloc, &info) == VK_SUCCESS ? info.pMappedData : NULL,
                bi.size = (VkDeviceSize)PR_FRAMES * PD_DRAWS * 144,
                sh_map = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &sh_buf, &sh_alloc, &info) == VK_SUCCESS ? info.pMappedData : NULL,
                bi.size = (VkDeviceSize)CR_FRAMES * CR_W * CR_H * 4,
                cr_map = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &cr_buf, &cr_alloc, &info) == VK_SUCCESS ? info.pMappedData : NULL,
                rt_log("probe: centre texel and a %dx%d grid of every render pass, %d frames kept", PR_GX, PR_GY,
                       PR_FRAMES);
        }
        __atomic_store_n(&pr_on, on, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&pr_lock);
    return pr_on;
}

static void probe_image(VkExec *ex, VkImage image, VkFormat format, VkImageAspectFlags aspect, uint32_t level,
                        uint32_t layer, uint32_t z, uint32_t width, uint32_t height, uint64_t addr);

void vk_probe_pass_end(VkExec *ex) {
    if (!probe_init()) return;
    const VkExecTarget *t = ex->color[0].image ? &ex->color[0] : ex->depth.image ? &ex->depth : NULL;
    if (!t || t->is3d || t->blocks || t->samples > 1 || !t->width || !t->height) return;
    // REKARINA_PASS_PROBE_MINW=<pixels>: only passes into targets that wide. Every pass probed costs the GPU a stop (the
    // forest village fell to 18 fps and its flashes stayed away); the 30 or so large passes of a frame do not.
    static int minw = -1;
    if (minw < 0) { const char *e = cfg_get("REKARINA_PASS_PROBE_MINW"); minw = e && *e ? atoi(e) : 0; }
    if ((int)t->width < minw) return;
    probe_image(ex, t->image, t->format, t == &ex->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                t->base_layer, 0, t->width, t->height, t->img ? t->img->addr : 0);
    for (int i = 0; i < cube_n; i++)   // cube textures of the first probed draw: faces x levels, addr | 2
        for (uint32_t lv = 0; lv < cube[i].levels && lv < 8; lv++)
            for (uint32_t face = 0; face < 6; face++) {
                uint32_t w = cube[i].w >> lv ? cube[i].w >> lv : 1, h = cube[i].h >> lv ? cube[i].h >> lv : 1;
                probe_image(ex, cube[i].image, cube[i].format, VK_IMAGE_ASPECT_COLOR_BIT, lv, face, 0, w, h,
                            cube[i].addr | 2 | ((uint64_t)face << 2) | ((uint64_t)lv << 5));
            }
    cube_n = 0;
    for (int i = 0; i < nvsp; i++)   // REKARINA_PROBE_VS: the vertex stage's textures of the pass's draws
        probe_image(ex, vsp[i].image, vsp[i].format, vsp[i].aspect, 0, 0, 0, vsp[i].w, vsp[i].h, vsp[i].addr | 1);
    nvsp = 0;
    if (pc_pending >= 0) {   // the GPU's bytes of the page-checked buffers at the pass end
        PcJob *j = &pc_job[pc_pending];
        pc_pending = -1;
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &di);
        uint64_t at = (uint64_t)j->slot * PC_MAX;
        for (int b = 0; b < 2; b++) {
            if (!j->hash[b]) continue;
            VkBufferCopy c = {pc_span[b].offset, at, j->size[b]};
            vkCmdCopyBuffer(ex->cb, pc_span[b].buffer, pc_buf, 1, &c);
            at += (j->size[b] + 4095) & ~4095ull;
        }
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded += 4;
        vk_completion_add(ex->fence, pc_done, j);
    }
    if (gub_pending >= 0 && gub_map) {   // the first probed draw's buffers, as bound
        int si = gub_pending;
        gub_pending = -1;
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &di);
        for (int j = 0; j < 11; j++) {
            const VkDescriptorBufferInfo *q = &gub_src[si][j];
            if (!q->buffer) continue;
            VkDeviceSize n = q->range == VK_WHOLE_SIZE || q->range > 4096 ? 4096 : q->range;
            VkBufferCopy c = {q->offset, ((VkDeviceSize)si * 11 + (VkDeviceSize)j) * 4096, n};
            vkCmdCopyBuffer(ex->cb, q->buffer, gub_buf, 1, &c);
        }
        ex->recorded += 12;
    }
    {
        uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
        int si = (int)(fr % PR_FRAMES);
        if (sh_map && pr->dtag[si] == fr && sh_pass_frame == fr) {
            for (uint32_t k = sh_pass_first[si]; k < pr->nd[si] && k < PD_DRAWS; k++) {
                const ProbeDraw *d = &pr->d[si][k];
                if (!d->sh_addr || d->sh_idx < 0 || d->sh_idx > 100000) continue;
                VkMemSpan sp;
                uint64_t a = d->sh_addr + (uint64_t)d->sh_idx * 144;
                if (!vk_mem_resolve(a, 144, &sp)) continue;
                vk_mem_pick(&sp);
                VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                       .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
                VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
                vkCmdPipelineBarrier2(ex->cb, &di);
                VkBufferCopy c = {sp.offset, ((VkDeviceSize)si * PD_DRAWS + k) * 144, 144};
                vkCmdCopyBuffer(ex->cb, sp.buffer, sh_buf, 1, &c);
                ex->recorded += 2;
            }
            sh_pass_first[si] = pr->nd[si];
        }
    }
    static uint64_t skip = 1;
    if (skip == 1) { const char *e = cfg_get("REKARINA_PROBE_SKIP"); skip = e ? strtoull(e, NULL, 0) : 0; }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int s = (int)(frame % CR_FRAMES);
    if (cr_map && t == &ex->color[0] && t->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && t->width >= CR_X0 + CR_W &&
        t->height >= CR_Y0 + CR_H && t->img && t->img->addr != skip && cr_tag[s] != frame) {
        cr_tag[s] = frame;
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(ex->cb, &di);
        VkBufferImageCopy r = {.bufferOffset = (VkDeviceSize)s * CR_W * CR_H * 4,
                               .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, t->base_layer, 1},
                               .imageOffset = {CR_X0, CR_Y0, 0}, .imageExtent = {CR_W, CR_H, 1}};
        vkCmdCopyImageToBuffer(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, cr_buf, 1, &r);
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier2(ex->cb, &di);
        ex->recorded += 3;
    }
}

// REKARINA_PROBE_CS=<hash>: the first dispatch of that compute shader each frame: its uniform blocks c3..c18 and storage
// slots 0..15 (first 4 KiB each) as guest memory (or the submit-time copy) holds them, and as the GPU sees them right
// after the dispatch. Written to dispatch.bin with the flight dump.
#define DC_SLOTS 32
static VkBuffer dc_buf;
static VmaAllocation dc_alloc;
static uint8_t *dc_map;
static uint64_t dc_frame[PR_FRAMES], dc_addr[PR_FRAMES][DC_SLOTS];
static uint8_t dc_cpu[PR_FRAMES][DC_SLOTS][4096];
static void probe_image(VkExec *ex, VkImage image, VkFormat format, VkImageAspectFlags aspect, uint32_t level,
                        uint32_t layer, uint32_t z, uint32_t width, uint32_t height, uint64_t addr);
static void dispatch_capture(VkExec *ex, const GfxShader *cs, const VkPipeInfo *info) {
    static uint64_t want = 1;
    if (want == 1) { const char *e = cfg_get("REKARINA_PROBE_CS"); want = e ? strtoull(e, NULL, 16) : 0; }
    if (!want || !cs || shader_hash(cs) != want) return;
    for (uint32_t i = 0; info && i < info->n_bindings; i++) {   // its sampled 2D textures of 2048 and more: grid-probed
        const VkPipeBinding *bd = &info->bindings[i];
        if (!bd->res || bd->kind != GFX_RES_TEX) continue;
        uint64_t h = handle_of(ex, 5, bd->res->slot, 0);
        view_t v;
        int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        {
            static int nl;
            if (nl++ < 12) rt_log("PROBECS tex slot %d handle %#llx -> %p %ux%u type %d fmt %d", bd->res->slot,
                                  (unsigned long long)h, (const void *)t, t ? t->width : 0, t ? t->height : 0,
                                  t ? (int)t->type : -1, t ? (int)t->format : -1);
        }
        if (!t || t->type != VK_IMAGE_TYPE_2D || t->width < 2048 || t->samples > 1) continue;
        probe_image(ex, t->image, t->format, t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 0, 0, t->width, t->height, t->addr);
    }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int s = (int)(frame % PR_FRAMES);
    if (dc_frame[s] == frame) return;
    if (!dc_map) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)PR_FRAMES * DC_SLOTS * 4096,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &dc_buf, &dc_alloc, &info) != VK_SUCCESS) return;
        dc_map = info.pMappedData;
    }
    dc_frame[s] = frame;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    for (int j = 0; j < DC_SLOTS; j++) {
        const p_bind_buf *u = j < 16 ? &ex->ubo[5][j] : &ex->ssbo[5][j - 16];
        dc_addr[s][j] = u->addr;
        memset(dc_cpu[s][j], 0, 4096);
        if (!u->addr || !u->size) continue;
        uint64_t n = u->size < 4096 ? u->size : 4096;
        VkMemSpan sp;
        if (!vk_mem_resolve(u->addr, n, &sp)) continue;
        const uint8_t *sn = j < 16 ? vk_usnap_bytes(u->addr, n) : NULL;
        memcpy(dc_cpu[s][j], sn ? sn : (const uint8_t *)(uintptr_t)u->addr, (size_t)n);
        VkDescriptorBufferInfo gi = j < 16 ? vk_exec_probe_ubo(ex, 5, j + 3) : (VkDescriptorBufferInfo){0};
        if (!gi.buffer) {
            vk_mem_pick(&sp);
            gi = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        }
        VkBufferCopy c = {gi.offset, ((VkDeviceSize)s * DC_SLOTS + (VkDeviceSize)j) * 4096, n};
        vkCmdCopyBuffer(ex->cb, gi.buffer, dc_buf, 1, &c);
    }
    ex->recorded += 1 + DC_SLOTS;
}

// After a dispatch: each storage image it writes (layer of the view, middle slice of a volume), as a pass.
void vk_probe_dispatch(VkExec *ex, const GfxShader *cs, const VkPipeInfo *info) {
    {   // DEBUG-LISTLOG: dispatches that bind the list pool: the grid words (18, 576, 0, 4) in their uniform buffers
        uint64_t vk_listlog_addr(void), vk_listlog_hash(void);
        double vk_listlog_ms(void);
        uint64_t la = vk_listlog_addr();
        int binds = 0;
        for (int k = 0; la && k < 16; k++) binds |= ex->ssbo[5][k].addr == la;
        if (binds) {
            char line[700];
            int o = snprintf(line, sizeof line, "LISTLOG dispatch fence %llu cs %016llx at %.3f:", (unsigned long long)ex->fence,
                             (unsigned long long)(cs ? shader_hash(cs) : 0), vk_listlog_ms());
            for (int k = 0; k < 18 && o < 600; k++) {
                const p_bind_buf *u = &ex->ubo[5][k];
                if (!u->addr || u->size < 32) continue;
                const int32_t *w = (const int32_t *)(uintptr_t)u->addr;
                uint64_t n = (u->size < 65536 ? u->size : 65536) / 4;
                for (uint64_t i = 0; i + 8 <= n; i += 4)
                    if (w[i] == 18 && w[i + 1] == 576 && w[i + 3] == 4)
                        o += snprintf(line + o, sizeof line - (size_t)o, " ubo%d+%#llx (%d %d %d %d) (%d %d %d %d)", k,
                                      (unsigned long long)(i * 4), w[i], w[i + 1], w[i + 2], w[i + 3], w[i + 4], w[i + 5],
                                      w[i + 6], w[i + 7]);
            }
            rt_log("%s", line);
        }
    }
    if (!probe_init()) return;
    dispatch_capture(ex, cs, info);
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        if (!bd->res || bd->kind != GFX_RES_IMG || !(bd->res->flags & 4) || bd->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            continue;
        uint64_t h = handle_of(ex, 5, bd->res->slot, 1);
        view_t v;
        int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (!t || t->samples > 1) continue;
        uint32_t lv = hv && v.set_levels ? v.base_level : 0, layer = hv && v.set_layers ? v.min_layer : 0;
        if (lv >= t->levels || layer >= t->layers) continue;
        uint32_t w = t->width >> lv ? t->width >> lv : 1, hh = t->height >> lv ? t->height >> lv : 1;
        uint32_t dd = t->type == VK_IMAGE_TYPE_3D ? (t->depth >> lv ? t->depth >> lv : 1) : 1;
        probe_image(ex, t->image, t->format, VK_IMAGE_ASPECT_COLOR_BIT, lv, t->type == VK_IMAGE_TYPE_3D ? 0 : layer,
                    dd / 2, w, hh, t->addr);
    }
}

static void probe_image(VkExec *ex, VkImage image, VkFormat format, VkImageAspectFlags aspect, uint32_t level,
                        uint32_t layer, uint32_t z, uint32_t width, uint32_t height, uint64_t addr) {
    if (pr_light) return;
    const Fmt *f = fmt_of(format);
    if (!f || f->bytes > 16 || !width || !height) return;
    struct { uint32_t width, height; } tt = {width, height}, *t = &tt;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int s = (int)(frame % PR_FRAMES);
    uint32_t idx;
    plat_mutex_lock(&pr_lock);
    if (pr->tag[s] != frame) pr->tag[s] = frame, pr->n[s] = 0;
    idx = pr->n[s] < PR_PASSES ? pr->n[s]++ : PR_PASSES;
    if (idx < PR_PASSES)
        pr->meta[s][idx] = (ProbeMeta){addr, f, t->width, t->height, (uint32_t)vk_exec_frame.dispatches};
    plat_mutex_unlock(&pr_lock);
    if (idx >= PR_PASSES) return;
    VkMemoryBarrier2 pre = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkMemoryBarrier2 post = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                             .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                             .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT |
                                              VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &pre};
    vkCmdPipelineBarrier2(ex->cb, &di);
    VkBufferImageCopy r[PR_PTS];
    VkDeviceSize base = ((VkDeviceSize)s * PR_PASSES + idx) * PR_SLOT;
    for (int k = 0; k < PR_PTS; k++) {
        uint32_t x = t->width / 2, y = t->height / 2;
        if (k && k <= PR_GX * PR_GY)
            x = (uint32_t)(((uint64_t)((k - 1) % PR_GX) * 2 + 1) * t->width / (2 * PR_GX)),
            y = (uint32_t)(((uint64_t)((k - 1) / PR_GX) * 2 + 1) * t->height / (2 * PR_GY));
        else if (k) {
            int q = k - 1 - PR_GX * PR_GY;
            float fx = PR_BOX_X0 + (PR_BOX_X1 - PR_BOX_X0) * ((float)(q % PR_BX) + 0.5f) / PR_BX;
            float fy = PR_BOX_Y0 + (PR_BOX_Y1 - PR_BOX_Y0) * ((float)(q / PR_BX) + 0.5f) / PR_BX;
            x = (uint32_t)(fx * (float)t->width), y = (uint32_t)(fy * (float)t->height);
            if (x >= t->width) x = t->width - 1;
            if (y >= t->height) y = t->height - 1;
        }
        r[k] = (VkBufferImageCopy){.bufferOffset = base + (VkDeviceSize)k * 16,
                                   .imageSubresource = {aspect, level, layer, 1},
                                   .imageOffset = {(int32_t)x, (int32_t)y, (int32_t)z}, .imageExtent = {1, 1, 1}};
    }
    vkCmdCopyImageToBuffer(ex->cb, image, VK_IMAGE_LAYOUT_GENERAL, pr_buf, PR_PTS, r);
    di.pMemoryBarriers = &post;
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += 3;
}

// REKARINA_PROBE_VS=<hash>: the 2D textures the vertex stage of draws with that vertex shader samples are grid-probed at
// the pass end (as passes of their own in probes.tsv / box.tsv).
static void probe_vs_textures(VkExec *ex) {
    static uint64_t want = 1;
    if (want == 1) { const char *e = cfg_get("REKARINA_PROBE_VS"); want = e ? strtoull(e, NULL, 16) : 0; }
    if (!want) return;
    const GfxShader *vs, *fs;
    vk_exec_draw_shaders(&vs, &fs);
    if (!vs || shader_hash(vs) != want) return;
    for (int slot = 0; slot < 8 && nvsp < 8; slot++) {
        uint64_t h = handle_of(ex, 0, slot, 0);
        view_t v;
        int hv = 0;
        const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
        if (!t || t->type != VK_IMAGE_TYPE_2D || t->samples > 1) continue;
        vsp[nvsp++] = (typeof(vsp[0])){t->image, t->format,
                                       t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                                       t->width, t->height, t->addr};
        rt_log("PROBEVS frame %llu slot %d image %p %ux%u fmt %d addr %#llx", (unsigned long long)(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1),
               slot, (void *)t->image, t->width, t->height, (int)t->format, (unsigned long long)t->addr);
    }
}

void vk_probe_draw(VkExec *ex, uint32_t count, uint32_t instances) {
    {   // DEBUG-LISTLOG: the first draw of each submission that binds the list pool, and the first skin draw
        uint64_t vk_listlog_addr(void), vk_listlog_hash(void);
        double vk_listlog_ms(void);
        static uint64_t last_any, last_skin;
        uint64_t la = vk_listlog_addr();
        if (la) {
            const GfxShader *vs, *fs;
            vk_exec_draw_shaders(&vs, &fs);
            int binds = 0;
            for (int b = 0; b < 4; b++) binds |= ex->ssbo[1][b].addr == la;
            uint64_t fh = fs ? shader_hash(fs) : 0;
            int skin = fh == 0xb983152a0a705a26ull || fh == 0x56aaae40b2842a72ull || fh == 0xd3eaffaa82d7f0ffull ||
                       fh == 0x6c7a58ad6c1c3763ull || fh == 0xc790367add79bfe2ull;
            if (skin) {   // every distinct binding set and grid of the skin draws of a submission
                static uint64_t sig_fence, sigs[64];
                static int nsig;
                const int32_t *g = ex->ubo[1][0].addr && ex->ubo[1][0].size >= 150 * 16
                                       ? (const int32_t *)(uintptr_t)(ex->ubo[1][0].addr + 148 * 16) : NULL;
                uint64_t sig = 1469598103934665603ull;
                for (int q = 0; q < 4; q++) sig = (sig ^ ex->ssbo[1][q].addr) * 1099511628211ull, sig = (sig ^ ex->ssbo[1][q].size) * 1099511628211ull;
                for (int q = 0; g && q < 8; q++) sig = (sig ^ (uint32_t)g[q]) * 1099511628211ull;
                sig ^= fh;
                if (sig_fence != ex->fence) sig_fence = ex->fence, nsig = 0;
                int seen = 0;
                for (int q = 0; q < nsig; q++) seen |= sigs[q] == sig;
                if (!seen && nsig < 64) {
                    sigs[nsig++] = sig;
                    rt_log("LISTLOG skinset fence %llu fs %016llx s0 %#llx+%#llx s1 %#llx+%#llx s2 %#llx+%#llx s3 %#llx+%#llx c3[148] %d %d %d %d c3[149] %d %d %d %d ubo3 %#llx",
                           (unsigned long long)ex->fence, (unsigned long long)fh,
                           (unsigned long long)ex->ssbo[1][0].addr, (unsigned long long)ex->ssbo[1][0].size,
                           (unsigned long long)ex->ssbo[1][1].addr, (unsigned long long)ex->ssbo[1][1].size,
                           (unsigned long long)ex->ssbo[1][2].addr, (unsigned long long)ex->ssbo[1][2].size,
                           (unsigned long long)ex->ssbo[1][3].addr, (unsigned long long)ex->ssbo[1][3].size,
                           g ? g[0] : -9, g ? g[1] : -9, g ? g[2] : -9, g ? g[3] : -9, g ? g[4] : -9, g ? g[5] : -9,
                           g ? g[6] : -9, g ? g[7] : -9, (unsigned long long)ex->ubo[1][0].addr);
                }
            }
            if (binds && (last_any != ex->fence || (skin && last_skin != ex->fence))) {
                const int32_t *c3 = ex->ubo[1][0].addr && ex->ubo[1][0].size >= 150 * 16
                                        ? (const int32_t *)(uintptr_t)(ex->ubo[1][0].addr + 148 * 16) : NULL;
                rt_log("LISTLOG draw fence %llu fs %016llx%s hash %016llx at %.3f c3[148] %d %d %d %d c3[149] %d %d %d %d",
                       (unsigned long long)ex->fence, (unsigned long long)fh, skin ? " skin" : "",
                       (unsigned long long)vk_listlog_hash(), vk_listlog_ms(), c3 ? c3[0] : -9, c3 ? c3[1] : -9,
                       c3 ? c3[2] : -9, c3 ? c3[3] : -9, c3 ? c3[4] : -9, c3 ? c3[5] : -9, c3 ? c3[6] : -9, c3 ? c3[7] : -9);
                last_any = ex->fence;
                if (skin) last_skin = ex->fence;
            }
        }
    }
    {   // DEBUG: REKARINA_PROBE_FS=<hash>: logs fs c4[2] of that shader's draws, once a frame (OoT's sky LUT: sun intensity)
        static uint64_t want = 1, last;
        if (want == 1) { const char *e = cfg_get("REKARINA_PROBE_FS"); want = e ? strtoull(e, NULL, 16) : 0; }
        if (want) {
            const GfxShader *vs, *fs;
            vk_exec_draw_shaders(&vs, &fs);
            uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
            if (fs && shader_hash(fs) == want && fr != last && ex->ubo[1][1].addr) {
                last = fr;
                {   // DEBUG-TRAP: REKARINA_EXPTRAP=1: after the sky draw of frames 1400-1406, the page of its c4[2].w is closed
                    static int tr = -1;
                    if (tr < 0) tr = cfg_get("REKARINA_EXPTRAP") != NULL;
                    if (tr && fr >= 1400 && fr <= 1406) {
                        void mem_trap_page(uint64_t addr);
                        rt_log("EXPTRAP frame %llu c4 at %#llx (c4[2].w %g)", (unsigned long long)fr,
                               (unsigned long long)ex->ubo[1][1].addr, ((const float *)(uintptr_t)ex->ubo[1][1].addr)[11]);
                        mem_trap_page(ex->ubo[1][1].addr + 44);
                    }
                }
                {   // DEBUG-SKYDUMP: at frames 1400 and 1401, every uniform block (first 64 floats, as the GPU reads them) and
                    // texture of that draw, to compare two runs
                    if (fr == 1400 || fr == 1401) {
                        for (int s = 0; s < 8; s++) {
                            const p_bind_buf *ub = &ex->ubo[1][s];
                            if (!ub->addr || !ub->size) continue;
                            uint64_t nn = ub->size < 256 ? ub->size : 256;
                            const uint8_t *snb = vk_usnap_bytes(ub->addr, nn);
                            const float *q = (const float *)(snb ? snb : (const uint8_t *)(uintptr_t)ub->addr);
                            char ln[3000];
                            int o2 = snprintf(ln, sizeof ln, "SKYDUMP frame %llu c%d %#llx+%#llx%s:", (unsigned long long)fr, s + 3,
                                              (unsigned long long)ub->addr, (unsigned long long)ub->size, snb ? " snap" : "");
                            for (uint64_t w = 0; w < nn / 4 && o2 < 2900; w++) o2 += snprintf(ln + o2, sizeof ln - (size_t)o2, " %.6g", q[w]);
                            rt_log("%s", ln);
                        }
                        for (int s = 0; s < 32; s++) {
                            uint64_t h = handle_of(ex, 1, s, 0);
                            if (!h) continue;
                            view_t v;
                            int hv = 0;
                            const VkTexImage *t = vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv);
                            if (t) rt_log("SKYDUMP frame %llu tex slot %d addr %#llx %ux%ux%u fmt %d levels %u", (unsigned long long)fr, s,
                                          (unsigned long long)t->addr, t->width, t->height, t->depth, (int)t->format, t->levels);
                        }
                    }
                }
                const float *u = (const float *)(uintptr_t)ex->ubo[1][1].addr;
                const int32_t *c3 = ex->ubo[1][0].addr && ex->ubo[1][0].size >= 152 * 16
                                        ? (const int32_t *)(uintptr_t)(ex->ubo[1][0].addr + 148 * 16) : NULL;
                rt_log("SUNVAL frame %llu c4[2] %g %g %g %g c4[3] %g %g %g c3[148] %d %d %d %d c3[149] %d %d %d %d", (unsigned long long)fr,
                       u[8], u[9], u[10], u[11], u[12], u[13], u[14], c3 ? c3[0] : -9, c3 ? c3[1] : -9, c3 ? c3[2] : -9,
                       c3 ? c3[3] : -9, c3 ? c3[4] : -9, c3 ? c3[5] : -9, c3 ? c3[6] : -9, c3 ? c3[7] : -9);
            }
        }
    }
    if (pr_on == 0 || !probe_init()) return;
    probe_vs_textures(ex);
    const VkExecTarget *t = &ex->color[0];
    if (!t->image || t->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || t->width < 1280 || !t->img) return;
    static uint64_t skip = 1;
    if (skip == 1) { const char *e = cfg_get("REKARINA_PROBE_SKIP"); skip = e ? strtoull(e, NULL, 0) : 0; }
    if (t->img->addr == skip) return;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int s = (int)(frame % PR_FRAMES);
    plat_mutex_lock(&pr_lock);
    if (pr->dtag[s] != frame) pr->dtag[s] = frame, pr->nd[s] = 0;
    uint32_t k = pr->nd[s] < PD_DRAWS ? pr->nd[s]++ : PD_DRAWS;
    plat_mutex_unlock(&pr_lock);
    if (k >= PD_DRAWS) return;
    ProbeDraw *d = &pr->d[s][k];
    d->target = t->img->addr, d->n = 0, d->count = count, d->inst = instances;
    const GfxShader *vs, *fs;
    vk_exec_draw_shaders(&vs, &fs);
    d->vs = vs ? shader_hash(vs) : 0, d->fs = fs ? shader_hash(fs) : 0;
    d->cond = (int8_t)ex->cond_active, d->gpu_cond = (int8_t)ex->gpu_cond, d->cond_cpu = (int8_t)ex->cond_cpu;
    d->sh_addr = ex->ssbo[1][3].addr, d->sh_idx = -1;
    if (k == 0 && cube_n == 0)   // the first probed draw's cube textures: every face and level probed at the pass end
        for (int slot = 0; slot < 32 && cube_n < 2; slot++) {
            uint64_t h = handle_of(ex, 1, slot, 0);
            view_t v;
            int hv = 0;
            const VkTexImage *t = h ? vk_texture_by_id(NVN_HANDLE_TEX(h), &v, &hv) : NULL;
            if (!t || t->layers != 6 || t->type != VK_IMAGE_TYPE_2D) continue;
            cube[cube_n++] = (typeof(cube[0])){t->image, t->format, t->width, t->height, t->levels, t->addr};
        }
    if (k == 0 && pc_map && pc_pending < 0) {   // hash the CPU's pages of fs storage slots 0 and 1 now
        PcJob *j = &pc_job[pc_next];
        if (!j->hash[0] && !j->hash[1]) {
            j->frame = frame, j->slot = pc_next;
            uint64_t total = 0;
            for (int b = 0; b < 2; b++) {
                const p_bind_buf *u = &ex->ssbo[1][b];
                j->addr[b] = u->addr, j->size[b] = 0;
                if (!u->addr || !u->size || total + u->size > PC_MAX) continue;
                VkMemSpan sp;
                if (!vk_mem_resolve(u->addr, u->size, &sp)) continue;
                vk_mem_pick(&sp);
                pc_span[b] = sp;
                j->size[b] = u->size;
                uint64_t pages = (u->size + 4095) / 4096;
                j->hash[b] = malloc(pages * 8);
                const uint8_t *src = (const uint8_t *)(uintptr_t)u->addr;
                for (uint64_t p = 0; j->hash[b] && p < pages && !cfg_env("REKARINA_PAGECHK_SAVE"); p++)
                    j->hash[b][p] = pc_hash(src + p * 4096, u->size - p * 4096 < 4096 ? u->size - p * 4096 : 4096);
                total += (u->size + 4095) & ~4095ull;
            }
            pc_pending = pc_next;
            pc_next = (pc_next + 1) % PC_SLOTS;
        }
    }
    if (k == 0 && gub_map) {
        for (int st = 0; st < 2; st++)
            for (int c = 0; c < 5; c++) gub_src[s][st * 5 + c] = vk_exec_probe_ubo(ex, st, c + 3);
        gub_src[s][10] = vk_exec_probe_support();
        gub_frame[s] = frame, gub_pending = s;
    }
    if (k == 0)
        for (int st = 0; st < 2; st++)
            for (int c = 0; c < 5; c++) {
                const p_bind_buf *u = &ex->ubo[st][c];
                pr->ub_addr[s][st][c] = u->addr;
                uint64_t n = u->size < 4096 ? u->size : 4096;
                VkMemSpan ms;
                if (u->addr && vk_mem_resolve(u->addr, 1, &ms) && n <= ms.avail) {
                    const uint8_t *src = (const uint8_t *)(uintptr_t)u->addr;
                    const uint8_t *sn = vk_usnap_bytes(u->addr, n);   // the submit-time copy the GPU reads, when there is one
                    memcpy(pr->ub[s][st][c], sn ? sn : src, (size_t)n);
                    if (n < 4096) memset(pr->ub[s][st][c] + n, 0, (size_t)(4096 - n));
                } else {
                    memset(pr->ub[s][st][c], 0, 4096);
                }
            }
    if (ex->ubo[1][1].addr && ex->ubo[1][1].size >= 228) memcpy(&d->sh_idx, (const void *)(uintptr_t)(ex->ubo[1][1].addr + 224), 4);
    if (sh_pass_frame != frame) sh_pass_frame = frame, sh_pass_first[s] = k;
    static const int sts[2] = {0, 1};   // NVN stages: vertex, fragment
    for (int si = 0; si < 2; si++)
        for (int kind = 0; kind < 2; kind++)
            for (int slot = 0; slot < (kind ? 16 : 18) && d->n < PD_BINDS; slot++) {
                const p_bind_buf *u = kind ? &ex->ssbo[sts[si]][slot] : &ex->ubo[sts[si]][slot];
                if (!u->addr || !u->size) continue;
                uint64_t n = u->size < 16384 ? u->size : 16384;
                VkMemSpan ms;
                if (!vk_mem_resolve(u->addr, 1, &ms)) continue;
                if (n > ms.avail) n = ms.avail;
                const uint32_t *w = (const uint32_t *)(uintptr_t)u->addr;
                uint64_t h = 1469598103934665603ull;
                uint32_t bad = 0;
                for (uint64_t i = 0; i < n / 4; i++) {
                    h = (h ^ w[i]) * 1099511628211ull;
                    if ((w[i] & 0x7f800000u) == 0x7f800000u) {
                        if (k == 0 && si == 1 && !kind && slot == 0 && bad < 8)   // DEBUG: fs c3's non-finite words
                            rt_log("NONFINITE frame %llu fs c3 %#llx word %llu (vec4 %llu.%c) = %#x", (unsigned long long)frame,
                                   (unsigned long long)u->addr, (unsigned long long)i, (unsigned long long)(i / 4),
                                   "xyzw"[i % 4], w[i]);
                        bad++;
                    }
                }
                VkDescriptorBufferInfo bi;
                int snap = !kind && vk_usnap_find(u->addr, n, 1, &bi);
                d->b[d->n++] = (ProbeBind){u->addr, u->size, h, bad, (uint8_t)sts[si], (uint8_t)kind,
                                           (uint8_t)(kind ? slot : slot + 3), (uint8_t)snap};
            }
}

static void crop_write(const char *dir) {
    if (!cr_map) return;
    const Fmt *f = fmt_of(VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    uint8_t *rgb = malloc((size_t)CR_W * CR_H * 3);
    if (!rgb) return;
    for (int i = 0; i < CR_FRAMES; i++) {
        if (!cr_tag[i]) continue;
        const uint8_t *src = cr_map + (size_t)i * CR_W * CR_H * 4;
        int bad = 0;
        for (int p = 0; p < CR_W * CR_H; p++) {
            float o[4];
            decode(f, src + (size_t)p * 4, o);
            int nf = !isfinite(o[0]) || !isfinite(o[1]) || !isfinite(o[2]);
            bad += nf;
            for (int c = 0; c < 3; c++)
                rgb[(size_t)p * 3 + c] = nf ? (uint8_t)(c == 1 ? 0 : 255) : (uint8_t)(o[c] / (1.f + o[c]) * 255.f + .5f);
        }
        {   // DEBUG: texels a NaN-traced shader replaced (scratchpad me/nantrace.py): vec4(idx % 64, idx / 64, 1, 1)
            static uint32_t hist[4096];
            memset(hist, 0, sizeof hist);
            int any = 0;
            for (int p = 0; p < CR_W * CR_H; p++) {
                float o[4];
                decode(f, src + (size_t)p * 4, o);
                if (o[2] != 1.0f || o[0] < 0 || o[0] > 63 || o[1] < 0 || o[1] > 63 || o[0] != (float)(int)o[0] ||
                    o[1] != (float)(int)o[1])
                    continue;
                hist[(int)o[0] + 64 * (int)o[1]]++, any = 1;
            }
            if (any) {
                char hp[1100];
                snprintf(hp, sizeof hp, "%s/crop_%06llu_idx.txt", dir, (unsigned long long)cr_tag[i]);
                FILE *fh = fopen(hp, "w");
                for (int k = 0; fh && k < 4096; k++)
                    if (hist[k]) fprintf(fh, "%d %u\n", k, hist[k]);
                if (fh) fclose(fh);
            }
        }
        char path[1100];
        snprintf(path, sizeof path, "%s/crop_%06llu_%s.ppm", dir, (unsigned long long)cr_tag[i], bad ? "nan" : "ok");
        FILE *fp = fopen(path, "wb");
        if (fp) fprintf(fp, "P6\n%d %d\n255\n", CR_W, CR_H), fwrite(rgb, 1, (size_t)CR_W * CR_H * 3, fp), fclose(fp);
    }
    free(rgb);
}

VkProbeSnap *vk_probe_snap(void) {
    if (pr_on != 1) return NULL;
    VkProbeSnap *s = malloc(sizeof *s);
    if (!s) return NULL;
    plat_mutex_lock(&pr_lock);
    memcpy(s->tag, pr->tag, sizeof s->tag), memcpy(s->n, pr->n, sizeof s->n), memcpy(s->meta, pr->meta, sizeof s->meta);
    memcpy(s->dtag, pr->dtag, sizeof s->dtag), memcpy(s->nd, pr->nd, sizeof s->nd), memcpy(s->d, pr->d, sizeof s->d);
    if (sh_map) memcpy(s->sh, sh_map, sizeof s->sh);
    memcpy(s->ub, pr->ub, sizeof s->ub), memcpy(s->ub_addr, pr->ub_addr, sizeof s->ub_addr);
    if (gub_map) memcpy(s->gub, gub_map, sizeof s->gub);
    plat_mutex_unlock(&pr_lock);
    memcpy(s->px, pr_map, sizeof s->px);
    return s;
}

void vk_probe_write(VkProbeSnap *s, const char *path) {
    if (!s) return;
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "frame\tpass\taddr\tformat\tw\th\tdispatches\tr\tg\tb\ta\tbad\tbad_at\tboxbad\tbox_at\n");
        uint64_t lo = UINT64_MAX;
        for (int i = 0; i < PR_FRAMES; i++)
            if (s->tag[i] && s->tag[i] < lo) lo = s->tag[i];
        for (uint64_t fr = lo; lo != UINT64_MAX && fr < lo + PR_FRAMES; fr++) {
            int i = (int)(fr % PR_FRAMES);
            if (s->tag[i] != fr) continue;
            for (uint32_t k = 0; k < s->n[i] && k < PR_PASSES; k++) {
                const ProbeMeta *m = &s->meta[i][k];
                float o[4], g[4];
                decode(m->f, s->px[i][k], o);
                int bad = 0, first = -1, boxbad = 0, boxfirst = -1;
                for (int p = 1; p < PR_PTS; p++) {
                    decode(m->f, s->px[i][k] + p * 16, g);
                    int b = 0;
                    for (int c = 0; c < 4; c++) b |= !isfinite(g[c]);
                    if (p <= PR_GX * PR_GY) {
                        if (b && first < 0) first = p - 1;
                        bad += b;
                    } else {
                        if (b && boxfirst < 0) boxfirst = p - 1 - PR_GX * PR_GY;
                        boxbad += b;
                    }
                }
                fprintf(f, "%llu\t%u\t%#llx\t%s\t%u\t%u\t%u\t%.5g\t%.5g\t%.5g\t%.5g\t%d\t%d,%d\t%d\t%d,%d\n", (unsigned long long)fr, k,
                        (unsigned long long)m->addr, m->f->name, m->w, m->h, m->dispatches, o[0], o[1], o[2], o[3], bad,
                        first < 0 ? -1 : first % PR_GX, first < 0 ? -1 : first / PR_GX, boxbad,
                        boxfirst < 0 ? -1 : boxfirst % PR_BX, boxfirst < 0 ? -1 : boxfirst / PR_BX);
            }
        }
        fclose(f);
    }
    {
        char cd[1100];
        snprintf(cd, sizeof cd, "%.*s", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        crop_write(cd);
    }
    {
        char sp_[1100];
        snprintf(sp_, sizeof sp_, "%.*s/sh.tsv", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        FILE *fs_ = fopen(sp_, "w");
        for (int i = 0; fs_ && i < PR_FRAMES; i++)
            for (uint32_t k = 0; s->dtag[i] && k < s->nd[i] && k < PD_DRAWS; k++) {
                const ProbeDraw *d = &s->d[i][k];
                if (!d->sh_addr || d->sh_idx < 0) continue;
                const float *v = (const float *)s->sh[i][k];
                int bad = 0;
                for (int j = 0; j < 36; j++) bad += !isfinite(v[j]);
                fprintf(fs_, "%llu\t%u\t%#llx\t%d\t%d", (unsigned long long)s->dtag[i], k, (unsigned long long)d->sh_addr,
                        d->sh_idx, bad);
                for (int j = 0; j < 36; j++) fprintf(fs_, "\t%.4g", v[j]);
                fprintf(fs_, "\n");
            }
        if (fs_) fclose(fs_);
    }
    {   // means.tsv: per pass, the mean of each channel over the full grid (non-finite texels left out)
        char mp[1100];
        snprintf(mp, sizeof mp, "%.*s/means.tsv", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        FILE *fm = fopen(mp, "w");
        for (int i = 0; fm && i < PR_FRAMES; i++)
            for (uint32_t k = 0; s->tag[i] && k < s->n[i] && k < PR_PASSES; k++) {
                const ProbeMeta *m = &s->meta[i][k];
                double sum[4] = {0};
                int n = 0;
                for (int p = 1; p <= PR_GX * PR_GY; p++) {
                    float g[4];
                    decode(m->f, s->px[i][k] + p * 16, g);
                    if (!isfinite(g[0]) || !isfinite(g[1]) || !isfinite(g[2]) || !isfinite(g[3])) continue;
                    for (int c = 0; c < 4; c++) sum[c] += g[c];
                    n++;
                }
                if (!n) n = 1;
                fprintf(fm, "%llu\t%u\t%#llx\t%s\t%ux%u\t%.5g\t%.5g\t%.5g\t%.5g\n", (unsigned long long)s->tag[i], k,
                        (unsigned long long)m->addr, m->f->name, m->w, m->h, sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n);
            }
        if (fm) fclose(fm);
    }
    {   // box.tsv: the box grid's decoded texels of every pass into a 32-bit float target (RGBA32F, RG32F, R32F)
        char bp[1100];
        snprintf(bp, sizeof bp, "%.*s/box.tsv", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        FILE *fb = fopen(bp, "w");
        for (int i = 0; fb && i < PR_FRAMES; i++)
            for (uint32_t k = 0; s->tag[i] && k < s->n[i] && k < PR_PASSES; k++) {
                const ProbeMeta *m = &s->meta[i][k];
                int depth = m->f->kind == K_D32F || m->f->kind == K_D16 || m->f->kind == K_D24;
                if ((m->f->kind != K_FLOAT || m->f->bits != 32) && !depth) continue;
                fprintf(fb, "%llu\t%u\t%#llx\t%s\t%ux%u", (unsigned long long)s->tag[i], k, (unsigned long long)m->addr,
                        m->f->name, m->w, m->h);
                for (int p = depth ? 1 : 1 + PR_GX * PR_GY; p < PR_PTS; p++) {   // depth: the whole grid too
                    float g[4];
                    decode(m->f, s->px[i][k] + p * 16, g);
                    fprintf(fb, "\t%.5g,%.5g,%.5g,%.5g", g[0], g[1], g[2], g[3]);
                }
                fprintf(fb, "\n");
            }
        if (fb) fclose(fb);
    }
    if (dc_map) {   // dispatch.bin: per frame: u64 frame, u64 addr[32], u8 cpu[32][4096], u8 gpu[32][4096]
        char dpth[1100];
        snprintf(dpth, sizeof dpth, "%.*s/dispatch.bin", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        FILE *fd = fopen(dpth, "wb");
        for (int i = 0; fd && i < PR_FRAMES; i++) {
            if (!dc_frame[i]) continue;
            fwrite(&dc_frame[i], 8, 1, fd), fwrite(dc_addr[i], 8, DC_SLOTS, fd), fwrite(dc_cpu[i], 1, sizeof dc_cpu[i], fd);
            fwrite(dc_map + (size_t)i * DC_SLOTS * 4096, 1, (size_t)DC_SLOTS * 4096, fd);
        }
        if (fd) fclose(fd);
    }
    {
        char up[1100];
        snprintf(up, sizeof up, "%.*s/ubo.bin", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
        FILE *fu = fopen(up, "wb");   // per frame: u64 frame, u64 addr[2][5], u8 bytes[2][5][4096]
        for (int i = 0; fu && i < PR_FRAMES; i++) {
            if (!s->dtag[i]) continue;
            fwrite(&s->dtag[i], 8, 1, fu), fwrite(s->ub_addr[i], 8, 10, fu), fwrite(s->ub[i], 1, sizeof s->ub[i], fu);
            fwrite(s->gub[i], 1, sizeof s->gub[i], fu);
        }
        if (fu) fclose(fu);
    }
    char dp[1100];
    snprintf(dp, sizeof dp, "%.*s/draws.tsv", (int)(strrchr(path, '/') ? strrchr(path, '/') - path : 0), path);
    if ((f = fopen(dp, "w"))) {
        fprintf(f, "frame\tdraw\ttarget\tvs\tfs\tcount\tinst\tcond\tgpu_cond\tcond_cpu\tstage\tkind\tslot\taddr\tsize\tbad\thash\tsnap\n");
        for (int i = 0; i < PR_FRAMES; i++)
            for (uint32_t k = 0; s->dtag[i] && k < s->nd[i] && k < PD_DRAWS; k++)
                for (uint32_t j = 0; j < s->d[i][k].n; j++) {
                    const ProbeBind *q = &s->d[i][k].b[j];
                    const ProbeDraw *pd = &s->d[i][k];
                    fprintf(f, "%llu\t%u\t%#llx\t%016llx\t%016llx\t%u\t%u\t%d\t%d\t%d\t%s\t%s\t%u\t%#llx\t%#llx\t%u\t%016llx\t%u\n",
                            (unsigned long long)s->dtag[i], k, (unsigned long long)pd->target, (unsigned long long)pd->vs,
                            (unsigned long long)pd->fs, pd->count, pd->inst, pd->cond, pd->gpu_cond, pd->cond_cpu, q->st ? "fs" : "vs",
                            q->kind ? "ssbo" : "ubo", q->slot, (unsigned long long)q->addr, (unsigned long long)q->size,
                            q->bad, (unsigned long long)q->hash, q->snap);
                }
        fclose(f);
    }
    free(s);
}
