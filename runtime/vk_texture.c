// Vulkan textures of the graphics layer: a VkImage cache keyed by storage
// GPU address plus descriptor, uploads through staging buffers with block-linear deswizzle and ASTC decode (vk_astc.c),
// re-uploads when page write generations (vk_memory.c) show the storage written, ReadTexels / WriteTexels, the texture
// and sampler descriptor slots, and VkSamplers.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn @ b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile): which textures mirror their pool memory (nvn_mtl.m tex_mirrored, 3347-3362), the
// "GPU-owned" rule that stops re-uploads once the GPU rendered into a mirrored texture (nvn_mtl.m tex_mtl, 3366-3372),
// the live texture at a storage address that a slot of a finalized texture follows (hle_nvn.c:373-393), and
// ReadTexels / WriteTexels (nvn_mtl.m:10105-10195). Image creation, upload regions and samplers are ported from LAS
// Recomp runtime/nvn/vulkan/textures.c:235-330, 778-844 @ da35b83a (MIT, Copyright (c) 2026 Peiikai).
//
// Every image lives in VK_IMAGE_LAYOUT_GENERAL. Uploads are recorded into the upload batch of vk_memory.c; the replay
// submits it (vk_upload_flush) before a submission that samples the textures.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "platform/platform.h"

typedef struct {
    uint64_t addr;
    nvn_texdesc d;
    uint8_t has_layout, layout[8], pad[7];
} TexKey;

typedef struct Tex {
    VkTexImage pub;               // first: vk_texture_get hands out &pub
    TexKey key;
    tex_t t;                      // the newest texture object seen with this key
    TexVkFormat f;
    VmaAllocation alloc;          // own memory; NULL when bound into pool memory
    uint64_t bytes;               // the size of alloc (0 without own memory)
    uint32_t use_round;           // look_round when a lookup last reached it (touch)
    uint64_t use_frame;           // gfx_counters.frames at that lookup
    uint32_t draw_round;          // draw_round when a draw's binding last named it (drawn), or when it was made
    uint64_t draw_frame;          // gfx_counters.frames then
    uint8_t ever_drawn;           // a draw's binding named it at least once: what uses it shows in bindings (evictable)
    VmaAllocation bound_to;       // the pool memory it is bound into
    uint32_t wgen;                // page generation of the last upload check
    uint32_t checked;             // submit_epoch of the last check from a draw binding
    int uploaded, released;
    uint32_t n_up;                // uploads of this entry (vk_texture_debug)
    int last_ch;                  // last vk_mem_changed result of make_fresh: 1 written, 0 clean, -1 not tracked
    struct Tex *next_at;          // other entries at the same storage address
    uint8_t gpu_dirty;            // mirrored, and the GPU wrote the image after the last write-back to the storage
    uint32_t gpu_gen;             // page generation when the GPU write that set gpu_dirty was recorded
    uint64_t gpu_seq;             // order of that GPU write among all of them (gpu_writes)
    uint64_t gpu_fence;           // fence of the submission that made that write (0: not recorded in one)
    uint64_t rb_hold;             // readback: its write-back waits until gfx_counters.frames reaches this
    uint8_t lagged;               // a small target whose bytes reach storage a frame late (frame-latency readbacks)
    uint64_t run_frame;           // mirrored: the frame of the newest GPU write
    uint8_t run;                  // frames in a row up to run_frame with a GPU write, counted up to WB_RUN
    uint64_t wr_fence;            // any entry: newest submission recorded to write its image (render, copy, clear,
                                  // storage binding); ReadTexels waits for it instead of draining the replay queue
    uint8_t reused_now;           // its image came from a released entry the GPU may still be reading (recycle_take,
                                  // scale > 1): its first barrier waits for the earlier work on the queue (init_image)
    uint8_t listed;               // in owned[]
    uint8_t *snap;                // a texture that fills no page: its storage bytes when the GPU write was recorded
    uint32_t wt_n;                // WriteTexels into the storage since the last upload
    struct Tex *rel_prev, *rel_next;   // while released: the list of released entries, oldest first
    uint64_t rel_frame;           // while released: the frame it was released in (sweep_pressure's age rule)
    uint64_t wr_serial;           // order of the newest GPU write recorded into its image among all entries (0: none)
    uint64_t wr_old[4];           // earlier values of wr_serial, newest first ("lookups as of a place" below)
    VkImageCreateFlags ci_flags;  // the image's creation flags (recycle_take)
    uint8_t ci_type;              // the image's VkImageType, format and usage as created (compaction makes it again;
    VkFormat ci_format;           // a block-view image's format and usage differ from pub's)
    VkImageUsageFlags ci_usage;
    uint64_t chash;               // has_chash: content_hash of the storage bytes its image was first filled from
    uint64_t cprint;              // has_chash: content_print of those bytes (adopt_released, "moved storage")
    uint64_t texq_base[3];        // "texture quality": the storage addresses of the textures this smaller one stands for (0: none)
    uint8_t has_chash;            // cleared by a later upload (adopt_released)
    uint8_t rh;                   // released and counted in the stream's set of adoptable hashes (rh_hash: the value)
    uint64_t rh_hash;
    uint8_t *edge;                // its bytes in the storage's first and last page when it was last uploaded (edge_keep)
    uint8_t bc1_redo;             // a BC1 image whose content cannot come from its cache file: made again (bc1_redo)
    uint16_t born_gen;            // map_gen when it was made (census: what survives a map change)
    uint64_t follow_frame;        // REKARINA_TEX_FOLLOW: 1 + the frame follow_away last moved it in; 0 never
    const char *img_how;          // DEBUG-IMGWATCH: how it got its present image (NULL: made for it) and at which frame
    uint64_t img_frame;
} Tex;
static uint16_t map_gen;          // loading screens so far (no scene for COLD_LOAD frames; vk_texture_new_submit)

static void destroy_tex(Tex *e);
static void iw_drain(void);   // DEBUG-IMGWATCH
static void iw_defer_check(VkImage img, const struct Tex *owner, int line);
static void edge_keep(Tex *e);
static VkImage recycle_take(const VkImageCreateInfo *ci, int scale, VmaAllocation *out);
struct Tex;
static int released_in_use(const struct Tex *e, uint64_t now);   // above sweep_released
static int adopt_released(Tex *e, const tex_t *t);
static int park_image(Tex *e);
static void park_clear(void);

// A slot keeps what vk_texture_by_id resolved for it, so a draw binding skips the key lookup and the address chain: the
// entry (ce), valid while the generation of its address (agen) is still cgen, the view and has_view handed out with it,
// and the submit_epoch of the last freshness check through the slot (cepoch). Hot fields first: a lookup that hits
// touches one or two cache lines.
// seq and creg serve vk_texture_by_id_fast, which reads a slot without the texture lock: seq is odd while a writer
// (who holds the lock) changes ce, cgen, cepoch, depoch, cview, chas_view, v or has_view; creg is the registration serial
// (sinfo) ce was resolved under.
typedef struct {
    struct Tex *ce;               // NULL: nothing cached (registered since, or never resolved)
    uint32_t cgen;                // agen of the slot's address when ce was resolved
    uint32_t cepoch;
    uint32_t depoch;              // submit_epoch of the last lookup through the slot that a draw made (drawn)
    volatile uint32_t seq;
    uint32_t creg;
    view_t cview;
    uint8_t chas_view, used, has_view;
    view_t v;
    tex_t t;
} TexSlot;
static inline void slot_wbegin(TexSlot *s) { __atomic_store_n(&s->seq, s->seq + 1, __ATOMIC_RELAXED); __atomic_thread_fence(__ATOMIC_RELEASE); }
static inline void slot_wend(TexSlot *s) { __atomic_store_n(&s->seq, s->seq + 1, __ATOMIC_RELEASE); }
typedef struct { samp_par p; VkSampler s, sp, sc; uint8_t used; } SampSlot;   // s: vk_sampler_for(&p) once looked up; sp:
                                                                           // its point-sampled form (vk_sampler_point_by_id_locked)

static PlatRMutex tex_lock;
static PlatOnce tex_once = PLAT_ONCE_INIT;
static nvn_hmap by_key = {.klen = sizeof(TexKey)};
static nvn_hmap by_addr = {.klen = 8};
static TexSlot *tslots;
static int ntslots;
static SampSlot *sslots;
static int nsslots;
static nvn_hmap samplers = {.klen = sizeof(samp_par)};
static int nreleased;
static Tex *rel_head, *rel_tail;   // released entries in release order (sweep_released drops from the head)
static uint64_t n_uploads, n_reuploads;
// Re-uploads since the last perf line (tex_lock): all, and those of textures a WriteTexels wrote since their last upload.
static uint64_t ru_n, ru_bytes, ru_ns, ru_wt_n, ru_wt_bytes, ru_wt_ns;
// REKARINA_TEX_ALIAS (default on; =0: a new image over live storage starts empty): a new descriptor over the storage of a
// live image that the GPU wrote gets that image's content (alias_copy). Counts since the last log line.
static int tex_alias = -1;
static uint64_t n_alias_copied, n_alias_upload, n_alias_uninit;
static uint64_t n_made_total;   // entries made (entry_of): vk_texture_made_total
uint64_t vk_texture_made_total(void) { return __atomic_load_n(&n_made_total, __ATOMIC_RELAXED); }
static uint32_t submit_epoch = 1;   // bumped per queue submit: draw bindings check a texture's storage once per submit
// Mirrored textures the GPU wrote into, sorted by address (vk_texture_cpu_stale, vk_texture_write_back). Changed under
// tex_lock and owned_lock; read under either.
static PlatMutex owned_lock = PLAT_MUTEX_INIT;
static Tex **owned;
static int nowned, cowned;
static uint64_t max_owned;   // largest storage size in owned[]
static uint64_t gpu_writes;  // GPU writes into mirrored textures so far: Tex.gpu_seq
static uint64_t wr_serials;  // GPU writes recorded into any entry so far: Tex.wr_serial (record order)
static void note_writer(struct Tex *e);
// Bumped whenever the entry a slot resolves to can change: an entry created or destroyed, released or taken back into
// use (the slot lookup and the follow of a finalized slot read only by_key, by_addr and the released flags).
static uint32_t tex_gen = 1;
// Bumped with tex_gen and whenever a slot is registered: what vk_texture_by_id returns for any slot may have changed.
static uint32_t slots_gen = 1;
static uint32_t samp_gen = 1;   // bumped whenever a sampler slot is registered: vk_sampler_by_id may answer otherwise
static uint64_t n_checks;    // draw-binding storage checks (make_fresh_snap) since the last perf line
// tex_gen per storage address (hashed): a slot resolves through the entries at its own address only (slot_resolve), so
// a slot's resolution stays valid while its address's generation does. TotK finalizes and initializes transient
// textures every frame: about 90 tex_gen and 160 slots_gen bumps per frame in the monsters fight, one every 16 draws,
// which emptied every cache keyed on the global generations.
#define AGEN_N 4096
static uint32_t agen[AGEN_N];
// Per slot id, for vk_texture_slot_token without the lock: the registration serial (bits 16-47; 0: never registered)
// and the address generation index (bits 0-15), one word so a reader gets a consistent pair. Fixed-size (slot ids are
// 20 bits), unlike tslots, which moves when it grows.
static uint64_t *sinfo;
static uint32_t reg_serial;

static uint32_t agen_idx(uint64_t addr) { return (uint32_t)(addr * 0x9E3779B97F4A7C15ull >> 52); }

// ---- lookups as of a place in the command stream (REKARINA_CHUNK_ASOF=0 turns it off)
// A slot whose own texture is released, or was never rendered, follows the entry at its storage that the GPU wrote
// last (slot_resolve, "content follows the memory"), and a new image over live storage takes that entry's content
// (entry_of). "Last" is the order in which the replay thread records writes. The threads that record the chunks of a
// draw stretch (vk_exec.c) look textures up while the replay thread is already recording later passes: a chunk then
// found a target that a later pass renders at the same storage, an image the GPU has not written at the chunk's place
// in the queue, and the objects it drew read nothing: black for as long as the game kept that layout of its transient
// targets (the forest while the camera turns, after a frame in which every transient target moves: 1 to 19 dark
// frames, 10 of 49 such frames; none with the stretches off).
// A stretch notes the write count when it is handed out (vk_texture_asof_mark) and its chunks look up under that limit
// (vk_texture_asof): writes recorded later do not count. An entry keeps the serials of its earlier writes that a
// stretch still out may need (wr_old; those not above asof_floor, the newest stretch's mark), so an entry written
// before the stretch and again after it still counts with its earlier write. What a limited lookup resolves is not
// kept in the slot (the replay thread is past it), and the caller keeps it in no memo (vk_texture_asof_clean).
static uint64_t asof_floor;            // wr_serials when the newest draw stretch was handed out
static _Thread_local uint64_t asof;    // this thread's lookups count GPU writes up to this serial; 0: all of them
static uint64_t awr[AGEN_N];           // per address bucket (agen_idx): the serial of the newest GPU write there
static uint64_t n_asof;                // lookups resolved under a limit so far

// p's newest GPU write as this thread's lookups count it: under a limit, the newest one not after it. 0: none.
static inline uint64_t wr_at(const Tex *p) {
    if (__builtin_expect(!asof, 1) || p->wr_serial <= asof) return p->wr_serial;
    for (int i = 0; i < 4; i++)
        if (p->wr_old[i] <= asof) return p->wr_old[i];
    return p->wr_serial;   // written more often since the limit than is kept: as without one
}

static void tex_gen_bump(uint64_t addr) {
    if (!++tex_gen) tex_gen = 1;
    slots_gen++;
    uint32_t *g = &agen[agen_idx(addr)];
    __atomic_store_n(g, *g + 1, __ATOMIC_RELEASE);
}
// The same for an entry that went, was released or was taken back. A slot's kept resolution and its token stand on the
// generation of the slot's own storage address. Under "texture quality" the slot resolves to a smaller texture whose
// storage starts at a later level, another address: when that entry went, nothing told the slot, which kept a pointer
// to the dead record, and the pool update made a view of its destroyed image (the driver never returned from
// vkCreateImageView: quarter quality hung at the entrance of the Market, 2026-11-13). Such an entry carries the
// slot's address and both generations move.
static void tex_gen_bump_entry(uint64_t addr, const uint64_t texq_base[3]) {
    tex_gen_bump(addr);
    for (int i = 0; i < 3; i++)
        if (texq_base[i]) tex_gen_bump(texq_base[i]);
}

static volatile int32_t tex_ready;
static void tex_init(void *p) {
    (void)p;
    __atomic_store_n(&tex_ready, 1, __ATOMIC_RELEASE);
    return;
}
static uint32_t image_serial;   // vk_texture_image_serial
uint32_t vk_texture_image_serial(void) {
    uint32_t s = __atomic_add_fetch(&image_serial, 1, __ATOMIC_RELAXED);
    return s ? s : __atomic_add_fetch(&image_serial, 1, __ATOMIC_RELAXED);
}

// DEBUG-LOCKHOLD (REKARINA_SECTIONS): a hold of the texture lock of LH_MS or more is logged with the thread and the
// function that took it, so a long wait of the replay
// thread for the lock (TEXREG_LOCK, RL_TEX: 11 ms on average in the field's slowest frames) names the holder.
#define LH_MS 2
static _Thread_local int lh_depth;
static _Thread_local uint64_t lh_t0;
static _Thread_local const char *lh_site;
static void lh_enter(const char *site) { if (!lh_depth++) lh_t0 = plat_now_ns(), lh_site = site; }
static void lh_leave(void) {
    if (--lh_depth) return;
    uint64_t dt = plat_now_ns() - lh_t0;
    if (dt < LH_MS * 1000000ull) return;
    static volatile int32_t n;
    if (plat_inc32(&n) <= 400)
        rt_log("LOCKHOLD frame %llu: the texture lock held %.1f ms by %s thread %u, taken in %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), dt / 1e6,
               sec_replay_thread() ? "the replay" : "another", plat_thread_id(), lh_site ? lh_site : "?");
}
#define lock() lock_in(__func__)
static void lock_in(const char *site) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    if (__builtin_expect(sec_on, 0)) {   // RL_TEX: how long the replay thread waited for another thread's hold
        if (plat_rmutex_trylock(&tex_lock)) { lh_enter(site); return; }
        uint64_t t0 = sec_clock();
        plat_rmutex_lock(&tex_lock);
        if (sec_replay_thread()) sec_add(SEC_RL_TEX, sec_clock() - t0);
        lh_enter(site);
        return;
    }
    plat_rmutex_lock(&tex_lock);
}
static void unlock(void) {
    if (__builtin_expect(sec_on, 0) && lh_depth) lh_leave();
    plat_rmutex_unlock(&tex_lock);
}
// The front end's registrations and releases come from game threads that hold a virtual core and no other host lock:
// a wait for the lock is made without the core (kernel.c kern_lock_off_core). The texture pool's update holds the lock
// for milliseconds during a submission's replay, and a registrar that kept its core meanwhile made the game's job
// workers pinned to that core wait for it (REKARINA_CORE_DIAG, live play 2026-11-17).
static int tex_try(void *m) {
    if (!__atomic_load_n(&tex_ready, __ATOMIC_ACQUIRE)) plat_once(&tex_once, tex_init, NULL);
    return plat_rmutex_trylock(m);
}
static void tex_wait_free(void *m) {
    plat_rmutex_lock(m);
    plat_rmutex_unlock(m);
}
#define lock_off_core() lock_off_core_in(__func__)
static void lock_off_core_in(const char *site) {
    if (!kern_lock_off_core(tex_try, tex_wait_free, &tex_lock)) {
        lock_in(site);
        return;
    }
    if (__builtin_expect(sec_on, 0)) lh_enter(site);
}

// ---- uploads ahead of first use (REKARINA_TEX_PREP; "up_main" below)
// up_worker: this thread is the one that uploads ahead. Inside upload() it fills staging memory of its own with the
// texture lock released (fill_lock held instead: vk_texture_range_gone waits on it before a pool's memory goes), so
// the replay thread is not held up by the copy.
static _Thread_local int up_worker;
// The worker hashed this storage with the texture lock released (up_main): adopt_released takes the values.
static _Thread_local struct { uint64_t addr, size, cprint, chash; int valid; } up_pre;
static PlatMutex fill_lock = PLAT_MUTEX_INIT;
typedef struct { VkBuffer buf; VmaAllocation alloc; } OwnedStaging;
static void owned_staging_free(void *arg) {
    OwnedStaging *o = arg;
    VkCtx *c = vk_ctx();
    if (c) vmaDestroyBuffer(c->vma, o->buf, o->alloc);
    free(o);
}
static void owned_staging_later(VkBuffer buf, VmaAllocation alloc) {   // once the GPU is past the batch that copies from it
    OwnedStaging *o = malloc(sizeof *o);
    if (!o) rt_fatal("vk: out of memory");
    o->buf = buf, o->alloc = alloc;
    vk_defer_call(owned_staging_free, o);
}
// The worker's fill of staging memory from an entry's storage, with the texture lock released.
static void fill_enter(void) {
    plat_mutex_lock(&fill_lock);
    unlock();
}
static void up_range_gone(void);
static void up_queue(int id, uint32_t serial);
// the texture stream (above upload)
static void ts_ask(const tex_t *t);
static int ts_take_hash(const tex_t *t, uint64_t *cprint, uint64_t *chash, uint32_t *now);
static void ts_range_gone(uint64_t lo, uint64_t size);
static void ts_tick(uint64_t frame, int log);
static void ts_forget(const tex_t *t);
static void ts_resident(const tex_t *t);
// BC1 for opaque ASTC textures (below the texture cache): entry_of's choice for the image create_tex makes next, the
// storage hash it took for that (adopt_released takes it), and whether an entry's image is BC1.
static _Thread_local int bc1_next;
static _Thread_local struct { uint64_t addr, size, cprint, chash; uint32_t gen; int valid; } bc1_pre;
static int bc1_choose(const tex_t *t);
static int is_bc1(const struct Tex *e);
static void bc1_redo(struct Tex *e, const char *why);
#define HOLD_RING 32768
static uint64_t hold_ring[HOLD_RING], hold_n;   // DEBUG-HOLDCOUNT: the time each pool-bound image went to the hold (bit 0: depth)
static uint64_t bc1_failed[64];   // keys whose file could not fill a BC1 image (texture lock)
static uint32_t bc1_nfailed, bc1_pending;
static uint64_t bc1_made, bc1_made_bytes, bc1_redone;   // texture lock
static volatile int64_t bc1_written, bc1_refilled;      // the reaper thread; BC1 images filled again from another file

static uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
static uint32_t mip(uint32_t v, unsigned lv) { return umax(v >> lv, 1); }

static void make_key(const tex_t *t, TexKey *k) {
    memset(k, 0, sizeof *k);
    k->addr = (uintptr_t)t->mem;
    k->d = t->d;
    k->has_layout = t->has_layout;
    memcpy(k->layout, t->layout, 8);
}

static int texq_mode;                                   // "texture quality", above slot_resolve
static int texq_shape(const tex_t *t, tex_t *out);

// Keys equal apart from the descriptor's swizzle selectors: the same storage, format and geometry.
static int same_but_swizzle(const TexKey *a, const TexKey *b) {
    TexKey x = *a, y = *b;
    x.d.swizzle_r = x.d.swizzle_g = x.d.swizzle_b = x.d.swizzle_a = 0;
    y.d.swizzle_r = y.d.swizzle_g = y.d.swizzle_b = y.d.swizzle_a = 0;
    return !memcmp(&x, &y, sizeof x);
}

// MNG nvn_mtl.m tex_mirrored: textures whose content the CPU provides in pool memory. Display, compressible, zcull,
// depth, multisample and GPU-only storage stays GPU-owned.
static int tex_mirrored(const tex_t *t) {
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem)) return 0;
    if (t->has_layout) return 1;
    if (!(t->pool_flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL |
                      NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL))
        return 0;
    if (t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;   // stencil / depth formats
    return 1;
}

static void log_limited(volatile int32_t *n, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void log_limited(volatile int32_t *n, const char *fmt, ...) {
    if (plat_inc32(n) > 16) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    rt_log("vk: %s", buf);
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}
#define ALL_RW VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT

// ---------------------------------------------------------------- images
static VkFormatFeatureFlags features_of(VkFormat f) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, f, &p);
    return p.optimalTilingFeatures;
}

#ifdef GAME_NS2
VkFormat vk_storage_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_SRGB: return VK_FORMAT_R8_UNORM;
    case VK_FORMAT_R8G8_SRGB: return VK_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return f;
    }
}
#endif

static VkImageAspectFlags aspect_of(const TexVkFormat *f) {
    VkImageAspectFlags a = 0;
    if (f->depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f->stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Binds img into the pool memory behind its GPU address (physical pools, virtual pools through the page table) when
// the memory type, alignment and contiguous size allow; 0 otherwise.
// A depth / stencil image that needs more bytes than its NVN storage (guest_size) is not bound: AMD's RDNA2 driver
// (26.10.1) puts the depth compression metadata after the depth data, and above 1080p that tail lies over the next
// texture of the game's heap; each depth write then scrambled it (garbage band at the top, pink glow, blocks in the
// grass, a blue checkered scene in the menu). A colour image sized above its storage (the RX 9070 does it to some, the
// RDNA 2 integrated GPU to small ones: 262,144 bytes for a 131,072-byte target) kept its place until 1.2.9, as nothing
// had shown the GPU writing the excess. But the excess lies over the game's next texture either way: whatever the
// driver keeps there is overwritten when that texture is written, and the other way round. RDNA 3 and 4 players had
// a material's colours go wrong and the GPU stop inside a small pass right after a full-screen clear, seconds after a
// map arrival; whether this overlap is behind that is not shown yet (no such card here), but the overlap itself is
// wrong, and on AMD's Windows driver nearly every image already takes memory of its own (its 64 KiB alignment). So
// there such a colour image takes memory of its own too: REKARINA_COLOR_TAIL_BIND=1 binds it as before, =0 refuses
// it on every driver. The integrated Radeon: about 54 images in 1,600 frames, the same picture, the same video memory.
static int color_tail_refused(const Tex *e, uint64_t size, uint64_t guest_size) {
    static int on = -1;
    static volatile int32_t count;
    if (on < 0) {
        const char *v = cfg_get("REKARINA_COLOR_TAIL_BIND");
        on = v && *v ? atoi(v) == 0 : vk_caps.amd_prop;
    }
    if (!on) return 0;
    int32_t m = plat_inc32(&count);
    if (!(m & (m - 1)))   // the first, then at each power of two
        rt_log("vk: a colour image the driver sizes above its storage (%ux%u format %#x: %llu bytes for %llu) takes memory "
               "of its own, not the pool's (REKARINA_COLOR_TAIL_BIND); %d so far", e->t.d.width, e->t.d.height,
               (unsigned)e->t.d.format, (unsigned long long)size, (unsigned long long)guest_size, (int)m);
    return 1;
}
static int format_compressed(VkFormat f) {   // vkCmdClearColorImage takes no block-compressed format
    return (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
           (f >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK && f <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK);
}

static int suballoc_on(void) {   // REKARINA_TEX_SUBALLOC=0: every texture lets the driver choose dedicated memory
    static int on = -1;
    if (on < 0) {
        const char *v = cfg_env("REKARINA_TEX_SUBALLOC");
        on = !(v && v[0] == '0');
    }
    return on;
}

// REKARINA_TEX_POOL=1: the sub-allocated textures have VMA pools of their own (one per memory type, blocks of
// REKARINA_VMA_BLOCK_MB), apart from the buffers the rest of the runtime places in the same memory type. In a long
// session the small buffers of the game's map pools (their VRAM mirrors) sat in the textures' blocks: of the blocks
// a tenth to a half full, a third held buffers too (VMA's map, 2026-11-16), so moving the textures out (compaction
// below) could not empty them. Apart, a block of textures empties when its textures move.
static VmaPool tex_vma_pool[VK_MAX_MEMORY_TYPES];
static PlatMutex tex_vma_pool_lock = PLAT_MUTEX_INIT;
static int tex_pool_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_POOL", "0");
    return on;
}
// REKARINA_TEX_GEN_POOLS=1 (off by default; a layout for cleaning at loading screens): the sub-allocated textures go to a
// VMA pool of their memory type and their map generation (map_gen: one more at each loading screen), apart from the
// mirrors and other buffers in VMA's default pools. In a 4-map tour ~95 % of a map's texture bytes went within a minute
// of the next map's arrival, but the few that stayed and the mirrors sat in nearly every 64 MiB block (the dungeon's
// end: 800 MiB free inside 45 blocks; the Market's 101 MiB left in 43 of them, mirrors in 44; 2026-11-17), so no block
// emptied. Apart by generation, a map's blocks empty once its textures go together (a bulk drop at the loading screen);
// a generation's pool goes when it holds nothing (tex_gen_pools_trim). Blocks of REKARINA_VMA_BLOCK_MB.
static struct { VmaPool p; uint16_t gen; uint8_t type; } genp[64];
static int ngenp;
static uint64_t genp_made, genp_gone;
static int tex_gen_pools_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_GEN_POOLS", "0");
    return on;
}
static VmaPool tex_gen_pool(VkCtx *c, uint32_t t) {   // texture lock held (create_tex)
    for (int i = 0; i < ngenp; i++)
        if (genp[i].type == t && genp[i].gen == map_gen) return genp[i].p;
    if (ngenp == 64) return NULL;   // 64 generations still holding textures: VMA's default pools
    const char *bm = cfg_get("REKARINA_VMA_BLOCK_MB");
    int mb = bm && atoi(bm) >= 16 ? atoi(bm) : 16;
    VmaPoolCreateInfo pi = {.memoryTypeIndex = t, .blockSize = (VkDeviceSize)mb << 20};
    VmaPool p;
    if (vmaCreatePool(c->vma, &pi, &p) != VK_SUCCESS) return NULL;
    genp[ngenp++] = (typeof(genp[0])){p, map_gen, (uint8_t)t};
    genp_made++;
    return p;
}
static void tex_gen_pools_trim(void) {   // vk_texture_new_submit, texture lock held: the pools of past generations left empty
    VkCtx *c = vk_ctx();
    for (int i = 0; c && i < ngenp;) {
        VmaStatistics st;
        if (genp[i].gen != map_gen && (vmaGetPoolStatistics(c->vma, genp[i].p, &st), !st.allocationCount)) {
            vmaDestroyPool(c->vma, genp[i].p);
            genp[i] = genp[--ngenp];
            genp_gone++;
        } else {
            i++;
        }
    }
}

// The pool for an image of these requirements (NULL: VMA's default pools)
static VmaPool tex_pool_for(VkCtx *c, uint32_t type_bits) {
    if (!tex_pool_on() && !tex_gen_pools_on()) return NULL;
    VmaAllocationCreateInfo fi = {.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    uint32_t t;
    if (vmaFindMemoryTypeIndex(c->vma, type_bits, &fi, &t) != VK_SUCCESS || t >= VK_MAX_MEMORY_TYPES) return NULL;
    if (tex_gen_pools_on()) return tex_gen_pool(c, t);
    VmaPool p = __atomic_load_n(&tex_vma_pool[t], __ATOMIC_ACQUIRE);
    if (p) return p;
    plat_mutex_lock(&tex_vma_pool_lock);
    if (!(p = tex_vma_pool[t])) {
        const char *bm = cfg_get("REKARINA_VMA_BLOCK_MB");
        int mb = bm && atoi(bm) >= 16 ? atoi(bm) : 16;
        VmaPoolCreateInfo pi = {.memoryTypeIndex = t, .blockSize = (VkDeviceSize)mb << 20};
        if (vmaCreatePool(c->vma, &pi, &p) != VK_SUCCESS) p = NULL;
        else {
            __atomic_store_n(&tex_vma_pool[t], p, __ATOMIC_RELEASE);
            rt_log("vk: sub-allocated textures of memory type %u have a pool of their own (blocks of %d MiB)", t, mb);
        }
    }
    plat_mutex_unlock(&tex_vma_pool_lock);
    return p;
}

// Memory sub-allocated for a texture holds what the texture freed from it last held; dedicated memory comes zeroed. Without
// this the first frames after a load drew doors and cracks of white light: bright leftovers read where the old dedicated
// allocations gave zero. A buffer aliasing the allocation clears it before the upload batch writes the image.
static void zero_suballoc(VkCtx *c, const Tex *e) {
    VmaAllocationInfo info;
    vmaGetAllocationInfo(c->vma, e->alloc, &info);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = info.size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b;
    if (vkCreateBuffer(c->device, &bi, NULL, &b) != VK_SUCCESS) return;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c->device, b, &mr);
    if (!(mr.memoryTypeBits >> info.memoryType & 1) || info.offset % mr.alignment ||
        vmaBindBufferMemory(c->vma, e->alloc, b) != VK_SUCCESS) {
        vkDestroyBuffer(c->device, b, NULL);
        return;
    }
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdFillBuffer(cb, b, 0, VK_WHOLE_SIZE, 0);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_upload_end();
    vk_defer_buffer(b, NULL);
}

// REKARINA_SECTIONS: what asked for the entry being made (texture lock): 0 a render target or vk_texture_get, 1 a draw's
// binding, 2 the texture pool's update. The calls of one creation, also for the TEXSLOW line.
static int sec_mk;
enum { ICS_CREATE, ICS_REQ, ICS_RESOLVE, ICS_BIND, ICS_VMA, ICS_N };
static uint64_t ics[ICS_N];
static inline void ic_end(int split, int kind, uint64_t t0) {
    if (t0) {
        uint64_t d = sec_clock() - t0;
        ics[split] += d;
        sec_add(kind, d);
    }
}
static _Thread_local int pm_thread;   // the thread that makes images ahead (pm_main): no section timers, no log lines
static VkResult timed_create_image(VkCtx *c, const VkImageCreateInfo *ci, VkImage *img) {
    uint64_t t = pm_thread ? 0 : sec_begin();
    VkResult r = vkCreateImage(c->device, ci, NULL, img);
    ic_end(ICS_CREATE, SEC_IC_CREATE_T + sec_mk, t);
    if (r == VK_SUCCESS) vk_live_check_born((uint64_t)(uintptr_t)*img);   // DEBUG-LIVECHECK
    return r;
}

// The texture keys destroyed so far (hash, low bits index, later ones overwrite): how many new images repeat an address and
// shape that was destroyed earlier.
enum { REP_SLOTS = 1 << 18 };
static uint64_t *rep_keys;
static uint64_t rep_hash(const TexKey *k) {
    const uint8_t *p = (const uint8_t *)k;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < sizeof *k; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h | 1;
}

// What this thread's last bind_pool_memory resolved the image's address to, before the twin's shift: an image made
// ahead (pm_take) is used only while its address still resolves to that.
static _Thread_local struct { VmaAllocation alloc; VkDeviceSize at, size; } bind_res;

static uint64_t twin_lo, twin_hi, twin_binds;   // DEBUG-TWINSPAN (twin_lo is the offset + 1; 0: none yet)
static _Thread_local int bind_why;   // bind_pool_memory's last answer: ALIAS_WHY's kinds (0 bound, 1 depth too large,
                                     // 2 not resolved, 3 misaligned, 4 memory type); -1 other
static int bind_pool_memory(VkCtx *c, Tex *e, VkImage img, uint64_t guest_size) {
    VkMemoryRequirements req;
    bind_why = -1;
    uint64_t sec1 = pm_thread ? 0 : sec_begin();
    vkGetImageMemoryRequirements(c->device, img, &req);
    ic_end(ICS_REQ, SEC_IC_REQ, sec1);
    static uint64_t why[5], why_b[5];   // DEBUG-ALIASWHY: bound, depth larger than its guest size, not resolved, misaligned, type
    static uint64_t why_n;
#define ALIAS_WHY(k) do { bind_why = k, why[k]++, why_b[k] += req.size; if (!(++why_n & 1023)) rt_log("ALIASWHY bound %llu (%llu MiB), depth too large %llu (%llu MiB), unresolved %llu, misaligned %llu (%llu MiB), memory type %llu; this one: size %llu guest %llu align %llu", (unsigned long long)why[0], (unsigned long long)(why_b[0] >> 20), (unsigned long long)why[1], (unsigned long long)(why_b[1] >> 20), (unsigned long long)why[2], (unsigned long long)why[3], (unsigned long long)(why_b[3] >> 20), (unsigned long long)why[4], (unsigned long long)req.size, (unsigned long long)guest_size, (unsigned long long)req.alignment); } while (0)
    if ((e->f.depth || e->f.stencil) && req.size > guest_size) { ALIAS_WHY(1); return 0; }
    if (req.size > guest_size && color_tail_refused(e, req.size, guest_size)) { ALIAS_WHY(1); return 0; }
    {   // DEBUG-DEPTHALIAS: REKARINA_DEPTH_ALIAS=0: a depth image never aliases pool memory (memory of its own)
        static int da = -1;
        if (da < 0) da = rt_knob("DEPTH_ALIAS", "1");
        if (!da && (e->f.depth || e->f.stencil)) { ALIAS_WHY(1); return 0; }
    }
    VkMemSpan sp;
    if (!req.size) return 0;   // an image the device could not lay out (image_size)
    sec1 = pm_thread ? 0 : sec_begin();
    int resolved = vk_mem_resolve(e->pub.addr, req.size, &sp);
    ic_end(ICS_RESOLVE, SEC_IC_RESOLVE, sec1);
    if (!resolved || !sp.alloc) { ALIAS_WHY(2); return 0; }
    VkDeviceSize at = sp.alloc_offset;
    bind_res.alloc = sp.alloc, bind_res.at = at, bind_res.size = req.size;
    if (at % req.alignment) {
        // an offset the pool's block cannot take: its twin takes it VK_MEM_TWIN_SHIFT bytes lower (REKARINA_POOL_TWIN=0: memory
        // of its own, as before)
        static int twin = -1;
        // REKARINA_POOL_TWIN: 0 never, 1 always, unset (OoT): while video memory is over 80 % of its budget.
        // Measured in the village: the same frame rate either way (72.3 against 72.7 fps, 63.0 against 62.5 with frame
        // generation), 0.35-0.5 GiB less video memory with it, and an image bound into pool memory costs the replay
        // ~0.3 ms where a recycled image of its own costs nothing, so it is used only when memory is short.
        if (twin < 0) {
            const char *v = cfg_get("REKARINA_POOL_TWIN");
#ifdef GAME_OOT
            twin = v && *v ? atoi(v) : 2;
#else
            twin = v && *v ? atoi(v) : 0;
#endif
        }
        int use = twin == 1;
        if (twin == 2) {
            uint64_t used = 0, budget = 0;
            int exact = 0;
            use = vk_mem_vram(&used, &budget, &exact) && budget && used > budget / 5 * 4;
        }
        VmaAllocation tw = use && at % req.alignment == VK_MEM_TWIN_SHIFT ? vk_mem_phys_twin(sp.alloc) : NULL;
        if (!tw) { ALIAS_WHY(3); return 0; }
        sp.alloc = tw, at -= VK_MEM_TWIN_SHIFT;
        {   // DEBUG-TWINSPAN: the part of the twin images were bound into so far (texture census line)
            if (!twin_lo || at < twin_lo - 1) twin_lo = at + 1;
            if (at + req.size > twin_hi) twin_hi = at + req.size;
            twin_binds++;
        }
    }
    VmaAllocationInfo info;
    vmaGetAllocationInfo(c->vma, sp.alloc, &info);
    if (at + req.size > info.size) { ALIAS_WHY(2); return 0; }
    if (!(req.memoryTypeBits >> info.memoryType & 1)) { ALIAS_WHY(4); return 0; }
    ALIAS_WHY(0);
    sec1 = pm_thread ? 0 : sec_begin();
    uint64_t hb0 = plat_now_ns();   // DEBUG-HRTIME
    VkResult bound = vmaBindImageMemory2(c->vma, sp.alloc, at, img, NULL);
    if (plat_now_ns() - hb0 > 3000000)
        rt_log("HRTIME bind %ux%u fmt %d size %llu KiB onto alloc of %llu MiB: %.2f ms at %.3f s", e->pub.width, e->pub.height,
               (int)e->pub.format, (unsigned long long)(req.size >> 10), (unsigned long long)(info.size >> 20),
               (plat_now_ns() - hb0) / 1e6, hb0 / 1e9);
    ic_end(ICS_BIND, SEC_IC_BIND_ALIAS_T + sec_mk, sec1);
    if (bound != VK_SUCCESS) return 0;
    e->bound_to = sp.alloc;
    return 1;
}

// agl's BC4 encoder draws a BC texture's blocks through an RGBA16UI view, one texel per 8-byte block (refs/nxnative
// docs/agents/04-graphics.md, BotW and TotK sections). Vulkan does it with an uncompressed
// view of a BLOCK_TEXEL_VIEW_COMPATIBLE image; EXTENDED_USAGE lets the BC image carry the COLOR_ATTACHMENT usage that
// only such views support. Whether the device takes that combination for this format and shape:
static int block_views_ok(const VkImageCreateInfo *ci) {
    VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                                           .format = ci->format, .type = ci->imageType, .tiling = ci->tiling,
                                           .usage = ci->usage, .flags = ci->flags};
    VkImageFormatProperties2 p = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    if (vkGetPhysicalDeviceImageFormatProperties2(vk_ctx()->phys, &fi, &p) != VK_SUCCESS) return 0;
    const VkImageFormatProperties *q = &p.imageFormatProperties;
    return ci->extent.width <= q->maxExtent.width && ci->extent.height <= q->maxExtent.height &&
           ci->extent.depth <= q->maxExtent.depth && ci->mipLevels <= q->maxMipLevels &&
           ci->arrayLayers <= q->maxArrayLayers;
}

// The memory the device wants for an image made from ci: 0 when it cannot lay it out. RADV on GFX9 (Vega, Renoir)
// hands back a 0-byte image for a mipmapped block-compressed 3D image at least as deep as it is wide in blocks, with
// COLOR_ATTACHMENT usage and no STORAGE: that usage asks for a thin 3D layout (Mesa radv_image.c), for which the GFX9
// address library has no swizzle mode left (gfx9addrlib.cpp), and the error is dropped; VMA then refuses the 0-byte
// allocation. The format properties call it supported, so only the size shows it.
static VkDeviceSize image_size(const VkImageCreateInfo *ci) {
    VkDeviceImageMemoryRequirements q = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = ci};
    VkMemoryRequirements2 r = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(vk_ctx()->device, &q, &r);
    return r.memoryRequirements.size;
}

// UNDEFINED -> GENERAL; new depth / stencil images start cleared (LAS textures.c: the game copies depth
// without stencil, and the stencil of those images must not be whatever the memory held).
// Not a depth image bound into pool memory (REKARINA_DEPTH_NEW_CLEAR=1: as before): the clear goes into the upload batch,
// which runs ahead of the whole part being recorded, and it writes the pool's bytes, which other render targets of the
// frame share. OoT's transient targets reuse their storage within a frame, and a character's 384x384 shadow map
// gets a new address whenever the frame's set of targets changes (every few frames while the player runs): the new
// image's clear then zeroed the storage of a target the same part still read from, drawn before the map in the
// game's order. Part of the picture and the characters in it went dark for one frame, 5 to 17 times in 2,000 frames
// of running through the Market, where video memory is short and every transient target lies in pool memory
// (REKARINA_POOL_TWIN); none after. The console clears nothing there either: the game clears a target before it draws.
// An image recycle_take handed over while the GPU may still read or write it for its released entry (reused_now) waits
// for all the earlier work on the queue first: the upload batch and every part go to the one graphics queue, so that
// barrier orders the new contents after the old entry's last use.
static void init_image(Tex *e, VkCommandBuffer cb) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = e->reused_now ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_2_NONE,
                                .srcAccessMask = e->reused_now ? VK_ACCESS_2_MEMORY_WRITE_BIT : VK_ACCESS_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image, .subresourceRange = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    static int bound_clear = -1;
    if (bound_clear < 0) bound_clear = cfg_env("REKARINA_DEPTH_NEW_CLEAR") != NULL;   // a test switch, not a setting
    if ((e->f.depth || e->f.stencil) && (!e->bound_to || e->f.stencil || bound_clear)) {
        VkClearDepthStencilValue zero = {0.0f, 0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearDepthStencilImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    } else if (vk_fix_zero() && !e->bound_to && !e->pub.mirrored && !format_compressed(e->f.vk)) {
        // GPU-only storage that got memory of its own starts at zero, as the pool it stands for (vk_memory.c
        // alloc_physical). Images aliasing pool memory are not cleared: that would erase what other textures rendered
        // there; mirrored images are uploaded from their storage on first use.
        VkClearColorValue zero = {0};
        VkImageSubresourceRange rg = {e->pub.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    }
}

// REKARINA_SHADOW_SCALE (1 to 4): the game's shadow maps are made this many times larger on each side. TotK renders its
// sun cascades into a 960x960 DEPTH16 2D array of 4 layers and its static shadows into 1440x1440 and 720x720 arrays of 2,
// whatever the output resolution, and shrinks the cascades under load (872, 848, 800, 480 and so on). Those are made
// larger here: render passes into them scale the game's viewports and scissors (vk_exec.c resolve_rt, pass_scale),
// copies scale their regions (vk_exec_xfer.c sub_resolve), and shaders sample them with normalised coordinates, so the
// game sees nothing of it. A map whose storage the CPU writes (mirrored) or that aliases pool memory keeps its size,
// as its texels must match the guest layout.
//
// auto (OoT's default) picks 4 or 1 for the card. OoT renders its three sun cascades into a 1600x1600 DEPTH16 array of
// 3 layers (on the traced save, the only map shadow_scale_of enlarges), through a light-space perspective projection:
// the casters' matrix divides by the distance along an axis of the light's plane (its w row is that axis plus an
// offset), and the offset and the warp follow the camera. Every camera move slides the texel grid over the world by a
// fraction of a texel, which no snapping can hold still, so leaves a texel or two wide cover other texels each frame
// and their shadows boil while the hero strafes (as much in upstream 1.2.7 as in 1.2.8 and later). A grid four times
// finer keeps the game's projection and gives each texel a sixteenth of the area. The array then grows from 15 to 234
// MiB and its passes draw 16 times the texels, so auto takes 4 only on a discrete card with 12 GiB or more of
// device-local memory.
#ifdef GAME_OOT
static int shadow_scale_auto(void) {
    const VkCtx *c = vk_ctx();
    uint64_t vram = 0;
    for (uint32_t i = 0; i < c->mem_props.memoryHeapCount; i++)
        if ((c->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && c->mem_props.memoryHeaps[i].size > vram)
            vram = c->mem_props.memoryHeaps[i].size;
    if (vk_mem_vram_cap() && vk_mem_vram_cap() < vram) vram = vk_mem_vram_cap();   // DEBUG-VRAMCAP: a smaller card
    int discrete = c->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, s = discrete && vram >= 12ull << 30 ? 4 : 1;
    rt_log("vk: shadow map scale %d (auto: %s gpu, %.1f GiB of video memory)", s, discrete ? "discrete" : "not a discrete",
           (double)vram / 1073741824.0);
    return s;
}
#endif

int vk_shadow_scale(void) {
    static int scale = -1;
    if (scale < 0) {
#ifdef GAME_OOT
        if (!strcmp(cfg_value("REKARINA_SHADOW_SCALE"), "auto")) return scale = shadow_scale_auto();
#endif
        int s = cfg_int("REKARINA_SHADOW_SCALE");
        s = s < 1 ? 1 : s > 4 ? 4 : s;
        rt_log("vk: shadow map scale %d", s);
        scale = s;
    }
    return scale;
}

static int shadow_scale_of(const nvn_texdesc *d, const TexVkFormat *f, uint32_t samples, int mirrored) {
    int scale = vk_shadow_scale();
    if (scale == 1 || mirrored || samples > 1 || d->format != NVN_FORMAT_DEPTH16 || !f->depth) return 1;
    if (d->target != NVN_TEXTURE_TARGET_2D_ARRAY || d->width != d->height || d->width < 256 || d->depth < 2 || d->depth > 4)
        return 1;
    uint32_t most = vk_ctx()->props.limits.maxImageDimension2D;
    return d->width * (uint32_t)scale <= most ? scale : 1;
}

// ---- Images made ahead (REKARINA_TEX_PREMAKE, on for OoT)
// Binding a new image into pool memory costs ~0.3 ms in the driver whatever its size (the kernel reserves and maps GPU
// address space), and OoT gives its transient render targets a new address every frame: 5-15 such images
// a frame in forest village, 1.5-2 ms of the replay thread's 12 ms. The game initializes those textures on its own
// threads well before the replay reaches their first use, so a thread of its own makes the image then (pm_main:
// create_tex up to the bind, nothing else), and create_tex takes the bound image from the table (pm_take) where it
// would create and bind one. What it takes is exactly what it would have made: the same create info, bound where the
// address still resolves to. An entry nothing took within PM_KEEP frames is destroyed; a pool that goes takes its
// entries with it before its memory is freed (pm_range_gone).
// Locks: pm_lock is taken with or without the texture lock held, and nothing is locked while holding it.
#define PM_N 256
#define PM_KEEP 12
enum { PM_FREE, PM_QUEUED, PM_BUSY, PM_READY };
typedef struct {
    volatile int32_t state;
    tex_t t;                    // the texture object when it was initialized
    uint64_t addr, seq, frame;
    VkImageCreateInfo ci;       // PM_READY: what img was created from
    VkImage img;
    VmaAllocation bound_to, res_alloc;
    VkViewsAhead va;            // PM_READY: the views made with img (REKARINA_VIEWS_AHEAD)
    VkDeviceSize res_at, res_size;
    VmaAllocation own;          // PM_READY: img has memory of its own (REKARINA_TEX_PREMAKE_OWN), not pool memory
    uint8_t noalias;            // PM_READY: the image cannot be bound into its pool memory (img, if any, is an own one)
    uint8_t scaled;             // an enlarged shadow map (scale > 1): pm_scaled_n counts img, if any (pm_take_scaled)
} Premade;
static Premade pm[PM_N];
static PlatMutex pm_lock = PLAT_MUTEX_INIT;
static PlatEvent pm_ev;
static volatile int32_t pm_started;
static uint64_t pm_seq;
static _Thread_local Premade *pm_out;   // pm_main: the entry create_tex fills
static _Thread_local VkViewsAhead pm_va_taken;   // pm_take: the views of the image it handed out, for create_tex's tail
static struct { uint64_t asked, made, failed, taken, waited, stolen, stale, full, moved, noalias, own_made, own_taken; } PMS;

// An enlarged shadow map (shadow_scale_of; OoT's sun cascades, 234 MiB at scale 4) has memory of its own, so one image
// serves whichever address the game gives the texture: the game initializes it on its own threads before the replay
// first draws its shadows, possibly at several addresses in turn, and moves it again whenever its set of transient
// targets changes (first on a scene's first 3D frame). The image-ahead thread makes the first one (create_tex), the
// replay takes it at the first use whatever the address (pm_take_scaled), and every later address takes it over from
// the released entry (recycle_take). pm_scaled_n counts the images of that kind alive (ahead, in an entry, released);
// the thread makes one only while there is none, a 234 MiB allocation off the replay thread instead of on it.
static volatile int32_t pm_scaled_n;
// What became of an enlarged shadow map's image (the first 16 times): how it came, the images of the kind alive after.
static void scaled_note(const char *how, const VkImageCreateInfo *ci, uint64_t t0) {
    static volatile int32_t n;
    log_limited(&n, "frame %llu: enlarged shadow map %ux%ux%u %s (%.1f ms; %d alive)",
                (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ci->extent.width,
                ci->extent.height, ci->arrayLayers, how, (plat_now_ns() - t0) / 1e6, (int)__atomic_load_n(&pm_scaled_n, __ATOMIC_RELAXED));
}

// REKARINA_TEX_PREMAKE_OWN (OoT, 1 on, 0 off): six in ten of the game's transient render targets lie at an address the
// driver cannot bind an image to (512-byte steps against its 1,024: vk_texture.c bind_pool_memory, ALIASWHY), so they
// take memory of their own: a released image of the same shape (recycle_take), else a new one from vmaCreateImage.
// The image-ahead thread found that out already, but the replay tried the bind again (an image created and destroyed
// for nothing), and when no released image fits, as after an arrival, it allocated on the frame's thread (3-6 ms a
// frame for ~10 frames). Now the image-ahead thread's verdict "cannot alias" goes with the entry, so the replay goes
// straight to recycling, and while recycling has been failing it also makes the image with memory of its own ahead.
static int pm_own_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_PREMAKE_OWN", "0");
    return on;
}
// How often the replay had to allocate such a target lately: x/8 decays per target, +32 for each allocation.
static volatile int32_t pm_own_score;
static void pm_own_note(int allocated) {
    int32_t s = __atomic_load_n(&pm_own_score, __ATOMIC_RELAXED);
    __atomic_store_n(&pm_own_score, s - s / 8 + (allocated ? 32 : 0), __ATOMIC_RELAXED);
}

static int pm_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_PREMAKE",
#ifdef GAME_OOT
                             "1"
#else
                             "0"
#endif
                             );
    return on;
}

static Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init);
static uint64_t n_made_all;   // create_tex: images made so far (the video memory history before a lost device)

static void pm_main(void *arg) {
    (void)arg;
    pm_thread = 1;
    for (;;) {
        Premade *p = NULL;
        plat_mutex_lock(&pm_lock);
        for (int i = 0; i < PM_N; i++)
            if (pm[i].state == PM_QUEUED && (!p || pm[i].seq < p->seq)) p = &pm[i];
        if (p) __atomic_store_n(&p->state, PM_BUSY, __ATOMIC_RELEASE);
        plat_mutex_unlock(&pm_lock);
        if (!p) {
            plat_event_wait(&pm_ev);
            continue;
        }
        TexKey k;
        make_key(&p->t, &k);
        p->img = VK_NULL_HANDLE, p->own = NULL, p->noalias = 0, p->scaled = 0;
        memset(&p->va, 0, sizeof p->va);
        vk_drv_bg_turn();   // queue work first (gfx_present.c "driver turns")
        pm_out = p;
        create_tex(&p->t, &k, 1);   // on this thread: only the image and its bind, left in *pm_out
        pm_out = NULL;
        plat_mutex_lock(&pm_lock);
        if (p->img) PMS.made++;
        else if (!p->scaled) PMS.failed++;   // (an enlarged shadow map whose image exists already: nothing to make)
        if (p->noalias) PMS.noalias++;
        if (p->own) PMS.own_made++;
        p->frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        __atomic_store_n(&p->state, p->img || p->noalias ? PM_READY : PM_FREE, __ATOMIC_RELEASE);
        plat_mutex_unlock(&pm_lock);
    }
}

// Whether an image is queued or being made ahead (no lock: a glance for the reaper, gfx_present.c vk_drv_reap_turn).
int vk_texture_premake_pending(void) {
    if (!__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE)) return 0;
    for (int i = 0; i < PM_N; i++) {
        int32_t s = __atomic_load_n(&pm[i].state, __ATOMIC_RELAXED);
        if (s == PM_QUEUED || s == PM_BUSY) return 1;
    }
    return 0;
}

// nvnTextureInitialize (any thread): a texture in device-only pool memory will be bound into that memory at its first
// use; queued so that the image is there by then.
static void lead_note(const tex_t *t, int reg);
void vk_texture_premake(const tex_t *t) {
    lead_note(t, 0);   // DEBUG-TEXLEAD
    static int at_init = -1;
    if (at_init < 0) at_init = rt_knob("TEX_STREAM_AT_INIT", "0") != 0;
    if (at_init) ts_ask(t);   // CPU-written storage: the texture stream (asked at registration by default)
    if (!t->mem || !nvn_gpu_only((uintptr_t)t->mem) || t->d.target == NVN_TEXTURE_TARGET_BUFFER || !pm_on() || !vk_ctx()) return;
    uint64_t addr = (uintptr_t)t->mem, now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    plat_mutex_lock(&pm_lock);
    if (!pm_started) {
        plat_event_init(&pm_ev, 0, 0);
        if (!plat_thread_spawn(pm_main, NULL, "vk images ahead", 0)) {
            plat_mutex_unlock(&pm_lock);
            return;
        }
        // REKARINA_TEX_PREMAKE_THREADS (default 1): more threads make them, so that a burst (the ~15 re-sized render
        // targets a frame while the camera turns) is made before the replay needs it instead of by the replay
        const char *pt = cfg_get("REKARINA_TEX_PREMAKE_THREADS");
        int more = (pt && *pt ? atoi(pt) : 1) - 1, extra = 0;
        for (int i = 0; i < more && i < 3; i++) extra += plat_thread_spawn(pm_main, NULL, "vk images ahead", 0) != 0;
        __atomic_store_n(&pm_started, 1, __ATOMIC_RELEASE);
        rt_log("vk: images in pool memory are made ahead of their first use by %d thread(s) of their own (REKARINA_TEX_PREMAKE=1)",
               1 + extra);
    }
    PMS.asked++;
    Premade *slot = NULL;
    for (int i = 0; i < PM_N; i++) {
        Premade *q = &pm[i];
        if (q->state == PM_FREE) {
            if (!slot) slot = q;
        } else if (q->addr == addr && !memcmp(&q->t.d, &t->d, sizeof t->d) && q->t.L.size == t->L.size) {
            q->frame = now;   // the same texture at the same place again: the entry serves it
            slot = NULL;
            goto out;
        }
    }
    if (!slot) {
        PMS.full++;
        goto out;
    }
    slot->t = *t, slot->addr = addr, slot->seq = ++pm_seq, slot->frame = now;
    __atomic_store_n(&slot->state, PM_QUEUED, __ATOMIC_RELEASE);
out:
    plat_mutex_unlock(&pm_lock);
    if (slot) plat_event_set(&pm_ev);
}

static int pm_ci_same(const VkImageCreateInfo *a, const VkImageCreateInfo *b) {
    return a->flags == b->flags && a->imageType == b->imageType && a->format == b->format && a->extent.width == b->extent.width &&
           a->extent.height == b->extent.height && a->extent.depth == b->extent.depth && a->mipLevels == b->mipLevels &&
           a->arrayLayers == b->arrayLayers && a->samples == b->samples && a->usage == b->usage;
}

// The image made ahead for e's address from create info ci, bound (e->bound_to set); VK_NULL_HANDLE when there is none.
// An entry still queued is dropped (the caller makes the image now); one being made is waited for.
static VkImage pm_take(VkCtx *c, Tex *e, const VkImageCreateInfo *ci, int *noalias) {
    if (!__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE)) return VK_NULL_HANDLE;
    uint64_t addr = e->pub.addr;
    for (int spins = 0;; spins++) {
        Premade got = {0};
        int busy = 0;
        plat_mutex_lock(&pm_lock);
        for (int i = 0; i < PM_N && !got.img; i++) {
            Premade *q = &pm[i];
            if (q->state == PM_FREE || q->addr != addr) continue;
            if (q->state == PM_QUEUED) {
                __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
                PMS.stolen++;
            } else if (q->state == PM_BUSY) {
                busy = 1;
            } else if (q->noalias) {   // the image-ahead thread found it cannot be bound there (an own image may wait)
                if (pm_ci_same(&q->ci, ci)) *noalias = 1;
                if (!q->img) __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
            } else if (pm_ci_same(&q->ci, ci)) {
                got = *q;
                __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
            }
        }
        if (busy && !got.img && !spins) PMS.waited++;
        if (*noalias) busy = 0;
        plat_mutex_unlock(&pm_lock);
        if (got.img) {
            VkMemSpan sp;
            if (vk_mem_resolve(addr, got.res_size, &sp) && sp.alloc == got.res_alloc && sp.alloc_offset == got.res_at) {
                e->bound_to = got.bound_to;
                PMS.taken++;
                if (pm_va_taken.image) vk_exec_views_ahead_drop(&pm_va_taken);   // one create_tex did not reach its end with
                pm_va_taken = got.va;   // create_tex's tail puts them into the view cache
                return got.img;
            }
            vk_exec_views_ahead_drop(&got.va);
            vkDestroyImage(c->device, got.img, NULL);   // the address was mapped elsewhere since
            PMS.moved++;
            return VK_NULL_HANDLE;
        }
        if (!busy) return VK_NULL_HANDLE;
        if (spins < 2000) _mm_pause();
        else plat_yield();
    }
}

// The image with memory of its own made ahead for e's address from create info ci (REKARINA_TEX_PREMAKE_OWN), its
// allocation in *out; VK_NULL_HANDLE when there is none.
static VkImage pm_take_own(Tex *e, const VkImageCreateInfo *ci, VmaAllocation *out) {
    if (!__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE) || !pm_own_on()) return VK_NULL_HANDLE;
    Premade got = {0};
    plat_mutex_lock(&pm_lock);
    for (int i = 0; i < PM_N && !got.img; i++) {
        Premade *q = &pm[i];
        if (q->state != PM_READY || q->addr != e->pub.addr || !q->own || !pm_ci_same(&q->ci, ci)) continue;
        got = *q;
        __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
    }
    if (got.img) PMS.own_taken++;
    plat_mutex_unlock(&pm_lock);
    if (!got.img) return VK_NULL_HANDLE;
    *out = got.own;
    if (pm_va_taken.image) vk_exec_views_ahead_drop(&pm_va_taken);
    pm_va_taken = got.va;
    return got.img;
}

// A ready entry goes (pm_lock held): its views and image are destroyed.
static void pm_drop(VkCtx *c, Premade *q) {
    vk_exec_views_ahead_drop(&q->va);
    if (c && q->img) {
        if (q->own) vmaDestroyImage(c->vma, q->img, q->own);
        else vkDestroyImage(c->device, q->img, NULL);
        if (q->scaled) __atomic_fetch_sub(&pm_scaled_n, 1, __ATOMIC_ACQ_REL);
    }
    __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
}

// The enlarged shadow map made ahead with create info ci for e's descriptor, at whatever address (pm_scaled_n), its
// allocation in *out; VK_NULL_HANDLE when there is none. An entry of e's descriptor still queued is dropped (the
// caller makes the image now); one being made is waited for. One of another shape is destroyed: a single image of
// that kind is made ahead, and this texture is the one the scene draws.
static VkImage pm_take_scaled(const Tex *e, const VkImageCreateInfo *ci, VmaAllocation *out) {
    if (!__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE)) return VK_NULL_HANDLE;
    for (int spins = 0;; spins++) {
        Premade got = {0};
        int busy = 0;
        plat_mutex_lock(&pm_lock);
        for (int i = 0; i < PM_N && !got.img; i++) {
            Premade *q = &pm[i];
            int same = q->state != PM_FREE && !memcmp(&q->t.d, &e->t.d, sizeof e->t.d);
            if (q->state == PM_QUEUED && same) {
                __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
                PMS.stolen++;
            } else if (q->state == PM_BUSY && same) {
                busy = 1;
            } else if (q->state == PM_READY && q->scaled && q->own) {
                if (pm_ci_same(&q->ci, ci)) {
                    got = *q;
                    __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
                    PMS.own_taken++;
                } else {
                    pm_drop(vk_ctx(), q);
                    PMS.stale++;
                }
            }
        }
        if (busy && !got.img && !spins) PMS.waited++;
        plat_mutex_unlock(&pm_lock);
        if (got.img) {
            *out = got.own;
            if (pm_va_taken.image) vk_exec_views_ahead_drop(&pm_va_taken);
            pm_va_taken = got.va;
            return got.img;
        }
        if (!busy) return VK_NULL_HANDLE;
        if (spins < 2000) _mm_pause();
        else plat_yield();
    }
}

// The entries of [lo, lo + size) and those bound into alloc go: the pool's memory is about to be freed. One being made
// is waited for. keep_frames: also every ready entry older than that many frames (0: only the range).
// An enlarged shadow map's image made ahead lies in no pool and may wait out a loading screen for the scene's first
// frame (the age rule would destroy the only one): only the shutdown's call (the whole range) takes it.
static void pm_gone(uint64_t lo, uint64_t size, VmaAllocation alloc, uint64_t keep_frames) {
    if (!__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE)) return;
    VkCtx *c = vk_ctx();
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    int all = !lo && size == ~0ull;
    for (;;) {
        int busy = 0;
        plat_mutex_lock(&pm_lock);
        for (int i = 0; i < PM_N; i++) {
            Premade *q = &pm[i];
            int in = q->addr >= lo && q->addr - lo < size;
            if (q->state == PM_QUEUED && in) {
                __atomic_store_n(&q->state, PM_FREE, __ATOMIC_RELEASE);
            } else if (q->state == PM_BUSY) {
                busy |= in || alloc != NULL;   // which block it binds into shows only when it is done
            } else if (q->state == PM_READY && q->scaled && !all) {
                continue;
            } else if (q->state == PM_READY &&
                       (in || (alloc && (q->bound_to == alloc || q->res_alloc == alloc)) || (keep_frames && now - q->frame > keep_frames))) {
                pm_drop(c, q);
                PMS.stale++;
            }
        }
        plat_mutex_unlock(&pm_lock);
        if (!busy) break;
        plat_yield();
    }
}

// The image of e bound into the pool memory behind its address, from create info ci: the one made ahead, else made
// now. VK_NULL_HANDLE when the image cannot be bound there.
static VkImage alias_image(VkCtx *c, Tex *e, const VkImageCreateInfo *ci, uint64_t guest_size) {
    int noalias = 0;
    bind_why = -1;
    VkImage img = pm_thread ? VK_NULL_HANDLE : pm_take(c, e, ci, &noalias);
    if (img) return img;
    if (noalias && pm_own_on()) return VK_NULL_HANDLE;   // the image-ahead thread tried: memory of its own (create_tex)
    if (!pm_thread) vk_drv_urgent(1);   // made where it is needed: the image-ahead thread and the reaper let it go first
    int ok = timed_create_image(c, ci, &img) == VK_SUCCESS;
    if (ok && !bind_pool_memory(c, e, img, guest_size)) {
        vkDestroyImage(c->device, img, NULL);
        img = VK_NULL_HANDLE, ok = 0;
    }
    if (!pm_thread) vk_drv_urgent(-1);
    return ok ? img : VK_NULL_HANDLE;
}

// defer_init: the caller records the image's first transition (init_image or alias_copy).
// DEBUG-R8LOG: REKARINA_R8LOG=1 logs what happens to R8 textures 256 or more wide and 400 or less high (the subtitle lines):
// creation with the storage layout, CPU uploads, copies into them and passes that draw into them, up to 400 lines.
void vk_dbg_r8(const char *what, const tex_t *t, long a, long b) {
    static int on = -1, n;
    if (on < 0) on = cfg_get("REKARINA_R8LOG") != NULL;
    if (!on || !t || t->d.format != 0x1 || t->d.width < 256 || t->d.height > 400 || n >= 400) return;
    n++;
    rt_log("R8LOG frame %llu %s off %#llx mem %p %ux%u flags %#x levels %u stride %u pkg_bh %d has_layout %d lay0 %#x "
           "linear %d tileH %u size %llu | %ld %ld",
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), what, (unsigned long long)t->off,
           (const void *)t->mem, t->d.width, t->d.height, (unsigned)t->d.flags, t->d.levels, t->d.stride,
           (int)t->d.pkg_block_h, t->has_layout, t->layout[0], nvn_tex_is_linear(&t->d), t->L.tileH,
           (unsigned long long)t->L.size, a, b);
}

// The image record's fields from what create_tex decided (all but the image and its serial).
static void pub_fill(VkTexImage *p, const TexVkFormat *f, VkImageType type, VkImageUsageFlags usage, uint32_t w, uint32_t h,
                     uint32_t depth, uint32_t levels, uint32_t layers, uint32_t samples, const nvn_texdesc *d,
                     VkImageCreateFlags flags, const tex_t *t, int bound, int block_views, int fb_usage, int scale) {
    p->format = f->vk;
    p->type = type;
    p->usage = usage;
    p->aspect = aspect_of(f);
    p->width = w, p->height = h, p->depth = depth, p->levels = levels, p->layers = layers, p->samples = samples;
    p->view_type = type == VK_IMAGE_TYPE_1D ? (d->target == NVN_TEXTURE_TARGET_1D_ARRAY ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                 : type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE
                 : d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY && (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                 : layers > 1 || d->target == NVN_TEXTURE_TARGET_2D_ARRAY || d->target == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    p->alpha_one = f->alpha_one;
    memcpy(p->chan, f->chan, sizeof p->chan);
    p->swz[0] = (uint8_t)d->swizzle_r, p->swz[1] = (uint8_t)d->swizzle_g;
    p->swz[2] = (uint8_t)d->swizzle_b, p->swz[3] = (uint8_t)d->swizzle_a;
    if (!(p->swz[0] | p->swz[1] | p->swz[2] | p->swz[3]))   // descriptors built without nvnTextureBuilderSetDefaults
        p->swz[0] = NVN_TEXTURE_SWIZZLE_R, p->swz[1] = NVN_TEXTURE_SWIZZLE_G, p->swz[2] = NVN_TEXTURE_SWIZZLE_B,
        p->swz[3] = NVN_TEXTURE_SWIZZLE_A;
    p->mirrored = (uint8_t)tex_mirrored(t);
    p->bound = bound;
    p->block_views = (uint8_t)block_views;
    p->fb_usage = (uint8_t)fb_usage;
    p->scale = (uint8_t)scale;
}

// What create_tex makes for t: its Vulkan format and the image's create info, with what the rest of create_tex and the
// record need. It reads no table and takes no lock (only the device's format queries and settings read once), so the
// texture stream's workers plan exactly the image the replay would make (REKARINA_TEX_MAKE_AHEAD, below the stream).
// bc1: entry_of chose BC1 for this opaque ASTC texture (bc1_choose). 0: no image can be made for t.
typedef struct {
    TexVkFormat f;
    nvn_texlayout L;
    VkImageType type;
    uint32_t w, h, depth, levels, layers, samples;
    VkImageUsageFlags usage, ns2_usage;
    VkImageCreateFlags flags, ns2_flags;
    int scale, block_views, fb_usage;
    VkImageCreateInfo ci;
} TexPlan;
static int tex_plan(const tex_t *t, int bc1, TexPlan *P) {
    static volatile int32_t n_unmapped, n_unsupported, n_thick;
    VkCtx *c = vk_ctx();
    const nvn_texdesc *d = &t->d;
    if (!c || d->target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    TexVkFormat f;
    if (!tex_vk_format(d->format, &f)) {
        log_limited(&n_unmapped, "texture format %#x has no Vulkan format: %s", d->format, f.reason);
        return 0;
    }
    if (!tex_vk_format_supported(&f)) {
        log_limited(&n_unsupported, "texture format %#x: VkFormat %d lacks features %#x", d->format, (int)f.vk, (unsigned)f.need);
        return 0;
    }
    if (bc1 && f.astc_bc)   // entry_of found the storage's BC3 blocks opaque in the texture cache
        f.vk = f.vk == VK_FORMAT_BC3_SRGB_BLOCK ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    uint32_t w = umax(d->width, 1), h = umax(d->height, 1), depth = 1, layers = 1;
    VkImageType type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: type = VK_IMAGE_TYPE_1D; h = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: type = VK_IMAGE_TYPE_1D; h = 1; layers = L.layers; break;
    case NVN_TEXTURE_TARGET_3D: type = VK_IMAGE_TYPE_3D; depth = umax(d->depth, 1); break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    case NVN_TEXTURE_TARGET_CUBEMAP: case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = L.layers; break;
    default: break;
    }
    int scale = shadow_scale_of(d, &f, d->samples > 1 ? (uint32_t)d->samples : 1, tex_mirrored(t));
    w *= (uint32_t)scale, h *= (uint32_t)scale;
    uint32_t levels = umax(d->levels, 1), most = 1;
    for (uint32_t s = umax(umax(w, h), depth); s > 1; s >>= 1) most++;
    if (levels > most) levels = most;
    uint32_t samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    if (samples > 1) levels = 1;
    VkFormatFeatureFlags ff = features_of(f.vk);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    int ds = f.depth || f.stencil;
    if (ds && (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!ds && (ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1)
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    // contact shadows (vk_contact.c) darken the game's B10G11R11 scene colour in place from a compute shader
    if (f.vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && samples == 1 &&
        (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && w >= 640 && vk_contact_strength())
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateFlags flags = ds ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // format-reinterpreting views
    VkImageUsageFlags ns2_usage = 0;
    VkImageCreateFlags ns2_flags = 0;
#ifdef GAME_NS2
    // NVN2 games write through image slots into textures made without NVN_TEXTURE_FLAGS_IMAGE: OoT's movie converts its
    // YUV frame into an RGBA8 texture of flags 0 and composes its output into an sRGB one of flags COMPRESSIBLE, with
    // nothing to say they are images until the dispatch binds them (MNG nvn_mtl.m image_prepare upgrades a texture
    // then). Without the usage the replay bound a stand-in and the writes were lost: the movie came out black. So every
    // single-sample colour image a storage view can serve gets it here; an sRGB one through its UNORM format
    // (EXTENDED_USAGE; vk_exec_draw.c get_view makes such views, and leaves the usage out of its sRGB views). An image
    // in device-only pool memory that cannot alias that memory with the usage is made without it (below).
    if (!ds && samples == 1 && !(usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        VkFormat sf = vk_storage_format(f.vk);
        if (ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ns2_usage = VK_IMAGE_USAGE_STORAGE_BIT;
        else if (sf != f.vk && (features_of(sf) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
            ns2_usage = VK_IMAGE_USAGE_STORAGE_BIT, ns2_flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        usage |= ns2_usage, flags |= ns2_flags;
    }
#endif
    // Cube compatible only for cube targets: the replay makes cube views of nothing else (vk_exec_draw.c get_view checks
    // the image's own view type). The flag on every square array of 6+ layers changed how AMD's Windows driver lays such
    // an image out, and terrain material arrays sampled through it came out in rainbow blocks that moved with the mip
    // level (Savage Wind Recomp issue #59, BotW on an RX 580 and a 7800 XT).
    if (type == VK_IMAGE_TYPE_2D && layers >= 6 && w == h && samples == 1 &&
        (d->target == NVN_TEXTURE_TARGET_CUBEMAP || d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY))
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // NVN renders into a 3D texture slice by slice (a view's min layer is the depth plane, nvn_mtl.m), e.g. the
    // grading LUT of the menu background blur. Vulkan needs a 2D view of the slice for that.
    if (type == VK_IMAGE_TYPE_3D && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    // The 512x512 float maps of the ground fog's fluid simulation are only read and written by compute shaders. Created as
    // possible render targets too, one of their passes (cs 3a7117b072154c38) took 9 ms where its twin took 0.2 ms, on the frames
    // the simulation runs; without those usages the same frames took 30 ms instead of 39 in two runs of three.
    // REKARINA_SIM_NO_ATTACH=0 keeps the usages.
    if (w == 512 && h == 512 && !tex_mirrored(t) && (f.vk == VK_FORMAT_R16G16_SFLOAT || f.vk == VK_FORMAT_R16G16B16A16_SFLOAT) &&
        (usage & VK_IMAGE_USAGE_STORAGE_BIT) && rt_knob("SIM_NO_ATTACH", "1"))
        usage &= ~(VkImageUsageFlags)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = type, .format = f.vk,
                            .extent = {w, h, depth}, .mipLevels = levels, .arrayLayers = layers,
                            .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    int bw, bh, block_views = 0;
    // REKARINA_BLOCK_VIEWS_3D (default 0): a 3D BC image also gets block views. On RDNA2 cards (driver 26.10.1, issue #5)
    // the 32x32x32 BC4 volume the game creates at the logo, made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE,
    // took a GPU write just past the end of its memory: device lost at the first submit after it when the image ended
    // its memory block. Draws into a 3D BC image go through a stand-in instead (vk_exec.c bc_target).
    static int bv3d = -1;
    if (bv3d < 0) {
        bv3d = rt_knob("BLOCK_VIEWS_3D", "0");
        rt_log("vk: 3D BC textures %s (REKARINA_BLOCK_VIEWS_3D=%d)", bv3d ? "get block views (renderable)" : "have no block views",
               bv3d);
    }
    if (!ds && samples == 1 && f.vk == f.ideal && nvn_fmt_bpb(d->format, &bw, &bh) && bw == 4 && bh == 4 &&
        (type != VK_IMAGE_TYPE_3D || bv3d)) {
        VkImageCreateInfo bc = ci;   // BC formats: renderable through views of their blocks (block_views_ok)
        bc.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        bc.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if (type == VK_IMAGE_TYPE_3D) bc.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        VkImageUsageFlags hidden = 0;   // creation usage the replay never uses (views restrict theirs, vk_exec.c)
        int ok = block_views_ok(&bc);
        if (ok && type == VK_IMAGE_TYPE_3D && !image_size(&bc)) {
            // STORAGE usage keeps RADV on GFX9 from the thin layout (image_size), so agl's BC4 encoder still renders
            // into the volume; else the image is made without block views and its encoder passes are left out
            bc.usage |= VK_IMAGE_USAGE_STORAGE_BIT, hidden = VK_IMAGE_USAGE_STORAGE_BIT;
            ok = block_views_ok(&bc) && image_size(&bc);
            log_limited(&n_thick, "texture fmt %#x %ux%ux%u: the device cannot lay out a 3D image with block views; %s",
                        d->format, w, h, depth, ok ? "made with storage usage (a thick layout)" : "made without them");
        }
        if (ok) ci = bc, usage = bc.usage & ~hidden, flags = bc.flags, block_views = 1;
    }
    // Feedback loops (vk.h VkCtx.feedback): every attachable image may be sampled by a draw of a pass that renders it,
    // so it gets the usage now; adding it when a loop shows up would mean recreating the image (that lost the
    // grass tile maps once).
    int fb_usage = c->feedback == VK_FEEDBACK_LAYOUT && !block_views &&
                   (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
    if (fb_usage) ci.usage = usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    *P = (TexPlan){.f = f, .L = L, .type = type, .w = w, .h = h, .depth = depth, .levels = levels, .layers = layers,
                   .samples = samples, .usage = usage, .ns2_usage = ns2_usage, .flags = flags, .ns2_flags = ns2_flags,
                   .scale = scale, .block_views = block_views, .fb_usage = fb_usage, .ci = ci};
    return 1;
}

// REKARINA_TEX_SUBALLOC_KB (default 2048): the largest CPU-filled texture that takes a piece of a block (create_tex)
#define TEX_SUBALLOC_MAX (32 << 20)
static VkDeviceSize tex_suballoc_max(void) {
    static VkDeviceSize m;
    if (!m) {
        const char *v = cfg_get("REKARINA_TEX_SUBALLOC_KB");
        m = (VkDeviceSize)(v && atoi(v) > 0 ? atoi(v) : 2048) << 10;
    }
    return m;
}
static int ma_on(void);   // REKARINA_TEX_MAKE_AHEAD (below the texture stream)
static _Thread_local uint64_t cs_plan, cs_made, cs_tail;   // DEBUG-TEXLEAD: create_tex's stamps (plan done, image made, tail done)
static int lead_on(void);
static VkImage ma_take(const tex_t *t, const VkImageCreateInfo *ci, VmaAllocation *alloc, int *sub);

static Tex *create_tex(const tex_t *t, const TexKey *k, int defer_init) {
    static volatile int32_t n_failed, n_unbound;
    VkCtx *c = vk_ctx();
    const nvn_texdesc *d = &t->d;
    TexPlan P;
    int planned = tex_plan(t, bc1_next && !pm_thread, &P);
    bc1_next = 0;
    if (!planned) return NULL;
    if (lead_on()) cs_plan = plat_now_ns();   // DEBUG-TEXLEAD
    TexVkFormat f = P.f;
    nvn_texlayout L = P.L;
    VkImageType type = P.type;
    uint32_t w = P.w, h = P.h, depth = P.depth, levels = P.levels, layers = P.layers, samples = P.samples;
    VkImageUsageFlags usage = P.usage;
    VkImageCreateFlags flags = P.flags;
    int scale = P.scale, block_views = P.block_views, fb_usage = P.fb_usage;
    VkImageCreateInfo ci = P.ci;
#ifdef GAME_NS2
    VkImageUsageFlags ns2_usage = P.ns2_usage;
    VkImageCreateFlags ns2_flags = P.ns2_flags;
#endif
    int ds = f.depth || f.stencil;
    vk_dbg_r8("create", t, defer_init, 0);   // DEBUG-R8LOG
    if (d->width >= 4096 && d->height >= 4096 && cfg_get("REKARINA_BIGLOG"))   // DEBUG-BIGLOG
    {
        rt_log("BIGLOG create %p fmt %#x flags %#x pool_flags %#x frame %llu", (const void *)t->mem, d->format, d->flags,
               (unsigned)t->pool_flags, (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        void vk_dump_arm_soon(void);
        void boot_shots_arm(uint64_t n);
        if (d->width == 12288) boot_shots_arm(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        const char *aw = cfg_env("REKARINA_BIGLOG_ARM_W");   // the width whose creation arms the dump (default 4096)
        if (d->width == (uint32_t)(aw && *aw ? atoi(aw) : 4096) && (d->format == 0x34 || d->format == 0x32)) vk_dump_arm_soon();
    }
    {   // DEBUG-TEXFLAGS: REKARINA_TEXFLAGS=1: each distinct (flags, format, target) of a new non-depth image, once, with its size
        static struct { uint32_t fl, fm, tg; } seen[256];
        static int ns;
        if (cfg_env("REKARINA_TEXFLAGS") && !ds && ns < 256) {
            int found = 0;
            for (int i = 0; i < ns; i++) found |= seen[i].fl == (uint32_t)d->flags && seen[i].fm == (uint32_t)d->format && seen[i].tg == (uint32_t)d->target;
            if (!found) {
                seen[ns].fl = (uint32_t)d->flags, seen[ns].fm = (uint32_t)d->format, seen[ns].tg = (uint32_t)d->target, ns++;
                rt_log("TEXFLAGS flags %#x nvn fmt %#x target %d %ux%u levels %u usage %#x mirrored %d", (unsigned)d->flags, (unsigned)d->format, (int)d->target, w, h, levels, (unsigned)usage, tex_mirrored(t));
            }
        }
    }
    Tex *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("vk: out of memory for a texture");
    e->key = *k;
    e->t = *t;
    e->f = f;
    e->pub.addr = k->addr;
    VkImage img = VK_NULL_HANDLE;
    uint64_t spk_t0 = plat_now_ns();   // DEBUG-SPIKE
    int spk_alias = 0;
    if (__builtin_expect(sec_on, 0) && !pm_thread) memset(ics, 0, sizeof ics);
    // REKARINA_TEX_FRESH_OWN=1 (a test, off): a render target at storage no other descriptor names takes memory of its own,
    // so that a released image can go to the next target of its shape (recycle_take). OoT puts its transient
    // targets at ever new addresses, 5-15 a frame in the village, and an image bound into pool memory serves no other
    // address. Measured there it lost: the game then names the same storage through a second descriptor (a depth target
    // and its R32F view, a 2560x1440 target and a 1280x720 one), which pool memory shares for free and separate images
    // share by a copy (alias_copy): GPU time went from 15 to 25.7 ms a frame, 46 to 32 fps.
    static int fresh_own = -1;
    if (fresh_own < 0) fresh_own = rt_knob("TEX_FRESH_OWN", "0");
    int fresh = fresh_own && !tex_mirrored(t) && samples == 1 &&
                (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
                !nvn_hm_get(&by_addr, &k->addr);
    if (scale == 1 && nvn_gpu_only(k->addr) && !fresh) {   // storage in a physical or virtual pool: alias the pool memory (section 3.2)
        spk_alias = 1;
        img = alias_image(c, e, &ci, L.size);
        if (!img) {
#ifdef GAME_NS2
            // REKARINA_STORAGE_ALIAS_DROP=1 (as before): an image that cannot alias its pool memory with the storage usage
            // aliases it without the usage, and a storage binding of it then gets a stand-in. By default it keeps the
            // usage and takes memory of its own (below), so a store into it lands.
            static int drop = -1;
            if (drop < 0) drop = rt_knob("STORAGE_ALIAS_DROP", "0") != 0;
            if (ns2_usage && !block_views && drop) {   // the storage usage changed the image's layout: alias without it
                ci.usage = usage &= ~ns2_usage, ci.flags = flags &= ~ns2_flags;
                img = alias_image(c, e, &ci, L.size);
                if (!img) ci.usage = usage |= ns2_usage, ci.flags = flags |= ns2_flags;   // own memory, with the usage
                else if (!pm_thread)
                    log_limited(&n_unbound, "texture %#llx (%ux%u fmt %#x) made without storage usage to alias its pool "
                                "memory: an image slot bound to it gets a stand-in", (unsigned long long)k->addr, w, h, d->format);
            }
            if (!img && !pm_thread)
#else
            if (!pm_thread)
#endif
            log_limited(&n_unbound, "texture %#llx (%ux%u fmt %#x) cannot alias its pool memory; own memory instead",
                        (unsigned long long)k->addr, w, h, d->format);
        }
    }
    if (pm_thread) {   // made ahead (pm_main): the bound image goes into the table, the rest happens at its first use
        if (img) {
            pm_out->img = img, pm_out->ci = ci, pm_out->bound_to = e->bound_to;
            if (vk_exec_views_ahead_on()) {   // the views its first uses ask for (vk_exec_draw.c vk_exec_views_ahead)
                VkTexImage tmp = {.image = img, .addr = k->addr};
                pub_fill(&tmp, &f, type, usage, w, h, depth, levels, layers, samples, d, flags, t, e->bound_to != NULL,
                         block_views, fb_usage, scale);
                vk_drv_bg_turn();
                vk_exec_views_ahead(&tmp, &pm_out->va);
            }
            pm_out->res_alloc = bind_res.alloc, pm_out->res_at = bind_res.at, pm_out->res_size = bind_res.size;
        } else if (scale > 1 && !tex_mirrored(t)) {
            // an enlarged shadow map: one image with memory of its own for any address, made here only while there is
            // none (pm_scaled_n); else the replay takes the one there is over (pm_take_scaled, recycle_take)
            pm_out->scaled = (uint8_t)scale;
            int32_t none = 0;
            if (__atomic_compare_exchange_n(&pm_scaled_n, &none, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
                VmaAllocation own = NULL;
                vk_drv_bg_turn();
                uint64_t t0 = plat_now_ns();
                if (vmaCreateImage(c->vma, &ci, &ai, &img, &own, NULL) == VK_SUCCESS) {
                    vk_live_check_born((uint64_t)(uintptr_t)img);   // DEBUG-LIVECHECK
                    scaled_note("made ahead", &ci, t0);
                    pm_out->img = img, pm_out->own = own, pm_out->ci = ci, pm_out->bound_to = NULL;
                    if (vk_exec_views_ahead_on()) {
                        VkTexImage tmp = {.image = img, .addr = k->addr};
                        pub_fill(&tmp, &f, type, usage, w, h, depth, levels, layers, samples, d, flags, t, 0, block_views,
                                 fb_usage, scale);
                        vk_drv_bg_turn();
                        vk_exec_views_ahead(&tmp, &pm_out->va);
                    }
                } else {
                    img = VK_NULL_HANDLE;
                    __atomic_fetch_sub(&pm_scaled_n, 1, __ATOMIC_ACQ_REL);
                }
            }
        } else if (spk_alias && pm_own_on() && !tex_mirrored(t) && (bind_why == 1 || bind_why == 3 || bind_why == 4)) {
            // cannot alias for a reason that holds for the replay too (its shape, size or address, not a pool not yet
            // there): the replay need not try again
            pm_out->noalias = 1, pm_out->ci = ci;
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
            VmaAllocation own = NULL;
            // and while recycling a released image has been failing, the image with memory of its own is made now
            if (__atomic_load_n(&pm_own_score, __ATOMIC_RELAXED) >= 8 && (vk_drv_bg_turn(), 1) &&
                vmaCreateImage(c->vma, &ci, &ai, &img, &own, NULL) == VK_SUCCESS) {
                vk_live_check_born((uint64_t)(uintptr_t)img);   // DEBUG-LIVECHECK
                pm_out->img = img, pm_out->own = own, pm_out->bound_to = NULL;
                if (vk_exec_views_ahead_on()) {
                    VkTexImage tmp = {.image = img, .addr = k->addr};
                    pub_fill(&tmp, &f, type, usage, w, h, depth, levels, layers, samples, d, flags, t, 0, block_views,
                             fb_usage, scale);
                    vk_drv_bg_turn();
                    vk_exec_views_ahead(&tmp, &pm_out->va);
                }
            }
        }
        free(e);
        return NULL;
    }
    // A texture whose content the CPU provides (tex_mirrored), up to SUBALLOC_MAX, takes memory from VMA's device blocks:
    // the driver prefers a dedicated allocation for most images and vmaCreateImage follows that, so loading a scene makes
    // one vkAllocateMemory (a kernel call) per new texture, thousands of them in the frames right after a save loads.
    // vmaAllocateMemory with the plain requirements sub-allocates instead, from the blocks the headroom thread keeps ahead
    // (vk_headroom.c). Render targets keep vmaCreateImage and the driver's preference.
    // Above REKARINA_TEX_SUBALLOC_KB (default 2048) a texture has memory of its own: freed, it goes back to the system at
    // once. A freed piece of a block stays the allocator's until the whole block is empty, and after a few map
    // changes the blocks held one to three gigabytes with nothing in it, which the driver counts as the game's all
    // the same (vk_texture.c sweep_pressure). The large textures are few and hold most of the bytes.
    VkDeviceSize suballoc_max = tex_suballoc_max();
    enum { SUBALLOC_MAX = TEX_SUBALLOC_MAX };
    int own_memory = 0;   // the size rule sent it to memory of its own
    if (!img && tex_mirrored(t) && samples == 1 && !pm_thread && ma_on()) {
        // REKARINA_TEX_MAKE_AHEAD: the image a texture stream worker made for this storage and create info, with its
        // memory and views (ma_make, below the stream); a piece of a block is zeroed here, as below
        int sub = 0;
        if ((img = ma_take(t, &ci, &e->alloc, &sub)) && sub) {
            uint64_t sec1 = sec_begin();
            zero_suballoc(c, e);
            sec_end(SEC_IC_ZERO, sec1);
        }
    }
    // the frame makes a CPU-filled image itself: the stream's workers making images ahead let it go first (their
    // vk_drv_bg_turn waits while this is set), as the image-ahead thread does for the replay's own pool images
    int urgent = !img && tex_mirrored(t) && !pm_thread && ma_on();
    if (urgent) vk_drv_urgent(1);
    if (!img && tex_mirrored(t) && samples == 1 && suballoc_on()) {
        if (timed_create_image(c, &ci, &img) == VK_SUCCESS) {
            // an image the driver requires (not only prefers) dedicated memory for keeps it: vmaCreateImage below
            VkMemoryDedicatedRequirements dr = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
            VkMemoryRequirements2 mr2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dr};
            VkImageMemoryRequirementsInfo2 ri = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = img};
            uint64_t sec1 = sec_begin();
            vkGetImageMemoryRequirements2(c->device, &ri, &mr2);
            ic_end(ICS_REQ, SEC_IC_REQ, sec1);
            VkMemoryRequirements mr = mr2.memoryRequirements;
            VmaAllocationCreateInfo ai = {.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          .pool = mr.size <= suballoc_max ? tex_pool_for(c, mr.memoryTypeBits) : NULL};
            sec1 = sec_begin();
            own_memory = mr.size > suballoc_max;
            int ok = !dr.requiresDedicatedAllocation && mr.size <= SUBALLOC_MAX && !own_memory &&
                     vmaAllocateMemory(c->vma, &mr, &ai, &e->alloc, NULL) == VK_SUCCESS;
            sec_end(SEC_IC_ALLOC_SUB, sec1);
            sec1 = sec_begin();
            VkResult bound = ok ? vmaBindImageMemory(c->vma, e->alloc, img) : VK_SUCCESS;
            ic_end(ICS_BIND, SEC_IC_BIND_SUB_T + sec_mk, sec1);
            if (bound != VK_SUCCESS) {
                vmaFreeMemory(c->vma, e->alloc);
                e->alloc = NULL;
                ok = 0;
            }
            sec1 = sec_begin();
            if (ok) zero_suballoc(c, e);
            sec_end(SEC_IC_ZERO, sec1);
            if (!ok) {
                vkDestroyImage(c->device, img, NULL);
                img = VK_NULL_HANDLE, e->alloc = NULL;
            }
        } else {
            img = VK_NULL_HANDLE;
        }
    }
    if (!img && !tex_mirrored(t)) {
        uint64_t t0 = plat_now_ns();
        img = recycle_take(&ci, scale, &e->alloc);
        e->reused_now = img && scale > 1;   // the old entry's last parts may still run (init_image)
        if (img && scale > 1) scaled_note("taken over from its released entry", &ci, t0);
        if (!img && scale > 1 && (img = pm_take_scaled(e, &ci, &e->alloc))) scaled_note("taken from the image-ahead thread", &ci, t0);
    }
    if (spk_alias && !tex_mirrored(t) && pm_own_on()) {   // a pool target with memory of its own (REKARINA_TEX_PREMAKE_OWN)
        pm_own_note(!img);
        if (!img) img = pm_take_own(e, &ci, &e->alloc);
    }
    if (!img) {
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      .flags = own_memory ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0};
        uint64_t sec1 = sec_begin();
        VkResult r = vmaCreateImage(c->vma, &ci, &ai, &img, &e->alloc, NULL);
        ic_end(ICS_VMA, SEC_IC_VMA_T + sec_mk, sec1);
        if (r == VK_SUCCESS) vk_live_check_born((uint64_t)(uintptr_t)img);   // DEBUG-LIVECHECK
        if (r == VK_SUCCESS && scale > 1) {   // none was made ahead
            __atomic_fetch_add(&pm_scaled_n, 1, __ATOMIC_ACQ_REL);
            scaled_note("made by the replay", &ci, spk_t0);
        }
        if (r != VK_SUCCESS) {
            if (urgent) vk_drv_urgent(-1);
            log_limited(&n_failed, "image creation failed (%d): nvn fmt %#x %ux%ux%u target %d levels %u layers %u", (int)r,
                        d->format, w, h, depth, d->target, levels, layers);
            free(e);
            return NULL;
        }
    }
    if (urgent) vk_drv_urgent(-1);
    if (cs_plan) cs_made = plat_now_ns();   // DEBUG-TEXLEAD
    n_made_all++;
    vk_addr_label((uint64_t)img, "tex %#llx fmt %#x %ux%ux%u lv %u%s%s", (unsigned long long)(uintptr_t)t->mem, d->format, w, h, depth, levels, e->alloc ? " own" : " pool",
                  tex_mirrored(t) ? "" : " gpu");
    {   // DEBUG-SPIKE: a slow image creation
        uint64_t dt = plat_now_ns() - spk_t0;
        if ((vk_spike_on > 0 || sec_on) && dt > 3000000) {
            char split[96] = "";   // the calls are timed only under REKARINA_SECTIONS
            if (sec_on) snprintf(split, sizeof split, "; create %.1f requirements %.1f resolve %.1f bind %.1f vma %.1f", sec_ticks_ms(ics[ICS_CREATE]),
                                 sec_ticks_ms(ics[ICS_REQ]), sec_ticks_ms(ics[ICS_RESOLVE]), sec_ticks_ms(ics[ICS_BIND]), sec_ticks_ms(ics[ICS_VMA]));
            rt_log("TEXSLOW %.1f ms %s %ux%ux%u fmt %#x target %d levels %u layers %u%s", dt / 1e6, spk_alias ? (e->alloc ? "own memory after alias failed" : "alias") : "own memory",
                   w, h, depth, d->format, d->target, levels, layers, split);
        }
        if (__builtin_expect(sec_on, 0)) {   // a new image: was this address and shape destroyed before?
            int al = spk_alias && !e->alloc;
            uint64_t h = rep_hash(k);
            sec_count(al ? SEC_REP_ALIAS_NEW : SEC_REP_OWN_NEW, 1);
            if (rep_keys && rep_keys[h & (REP_SLOTS - 1)] == h) sec_count(al ? SEC_REP_ALIAS_REPEAT : SEC_REP_OWN_REPEAT, 1);
        }
    }
    {   // DEBUG-TEXNEW: REKARINA_TEXNEW=<frame>:<frames>: every image made in those frames
        static int on = -1;
        static uint64_t from, cnt;
        if (on < 0) { const char *v = cfg_get("REKARINA_TEXNEW"); on = v && sscanf(v, "%llu:%llu", (unsigned long long *)&from, (unsigned long long *)&cnt) == 2; }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= from && f < from + cnt) {
            uint64_t h = rep_hash(k);
            if (!rep_keys) rep_keys = calloc(REP_SLOTS, sizeof *rep_keys);
            rt_log("TEXNEW %llu addr %#llx %ux%ux%u fmt %#x target %d levels %u layers %u %s%s", (unsigned long long)f,
                   (unsigned long long)k->addr, w, h ? (unsigned)0 + (unsigned)d->height : 0, depth, d->format, d->target, levels, layers,
                   e->alloc ? "own" : "alias", rep_keys && rep_keys[h & (REP_SLOTS - 1)] == h ? " repeat" : "");
        }
    }
    e->ci_flags = ci.flags;
    e->ci_type = (uint8_t)ci.imageType, e->ci_format = ci.format, e->ci_usage = ci.usage;
    if (e->alloc) {
        VmaAllocationInfo own;
        vmaGetAllocationInfo(c->vma, e->alloc, &own);
        e->bytes = own.size;
        vk_vma_name(e->alloc, "tex:%s:%d:%ux%ux%u:lv%u:g%u", f.astc_bc ? "astc" : tex_mirrored(t) ? "cpu" : "gpu", (int)f.vk, w,
                    h, layers * depth, levels, (unsigned)map_gen);   // DEBUG-VMANAME
    }
    e->born_gen = map_gen;
    // A new image's handle can be one a destroyed image had. Views are purged with the image (destroy_tex), so none
    // should be cached under it: one that is was made from a destroyed texture's record after its purge, and would
    // reach this image's memory through the dead one's. Dropped here, and counted in the log.
    {
        static int32_t n_stale;
        int stale = vk_exec_views_gone(img);
        if (stale) vk_tpool_image_gone(img);   // a pool slot written with one of those views resolves again
        if (stale && n_stale++ < 16)
            rt_log("vk: new image %#llx had %d cached views of a destroyed image with its handle; dropped",
                   (unsigned long long)(uintptr_t)img, stale);
    }
    VkTexImage *p = &e->pub;
    p->image = img;
    p->serial = vk_texture_image_serial();
    pub_fill(p, &f, type, usage, w, h, depth, levels, layers, samples, d, flags, t, e->bound_to != NULL, block_views, fb_usage,
             scale);
    if (pm_va_taken.image) {   // views made with the image ahead (REKARINA_VIEWS_AHEAD)
        if (pm_va_taken.image == img) vk_exec_views_adopt(p, &pm_va_taken);
        else vk_exec_views_ahead_drop(&pm_va_taken);
    }

    if (cs_plan) cs_tail = plat_now_ns();   // DEBUG-TEXLEAD
    if (!defer_init) {
        init_image(e, vk_upload_begin());
        vk_upload_end();
    }
    return e;
}

static int owned_lower(uint64_t a) {
    int lo = 0, hi = nowned;
    while (lo < hi) {
        int m = (lo + hi) / 2;
        if (owned[m]->pub.addr < a) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static void own_add(Tex *e) {
    if (e->listed) return;
    plat_mutex_lock(&owned_lock);
    if (nowned == cowned) {
        cowned = cowned ? 2 * cowned : 256;
        if (!(owned = realloc(owned, sizeof *owned * (size_t)cowned))) rt_fatal("vk: out of memory for the owned-texture list");
    }
    int i = owned_lower(e->pub.addr);
    memmove(owned + i + 1, owned + i, sizeof *owned * (size_t)(nowned - i));
    owned[i] = e;
    __atomic_store_n(&nowned, nowned + 1, __ATOMIC_RELEASE);
    if (e->t.L.size > max_owned) max_owned = e->t.L.size;
    e->listed = 1;
    plat_mutex_unlock(&owned_lock);
}

static void own_remove(Tex *e) {
    if (!e->listed) return;
    plat_mutex_lock(&owned_lock);
    for (int i = owned_lower(e->pub.addr); i < nowned && owned[i]->pub.addr == e->pub.addr; i++)
        if (owned[i] == e) {
            memmove(owned + i, owned + i + 1, sizeof *owned * (size_t)(nowned - i - 1));
            __atomic_store_n(&nowned, nowned - 1, __ATOMIC_RELEASE);
            break;
        }
    e->listed = 0;
    plat_mutex_unlock(&owned_lock);
}

// The texture stream's set of content hashes adopt_released can find an image for: released entries and parked images
// (ts_lock). A storage that hashes to one gets its image by adoption, so the stream reads or stages no bytes for it:
// in the field four in ten files the stream read went to textures that were then adopted (2026-11-16).
#define RH_N (1u << 15)
static struct { uint64_t h; int32_t n; } *rh;
static volatile int32_t ts_started;
static PlatMutex ts_lock;
static void rh_add(uint64_t h, int d) {   // ts_lock held
    if (!rh && !(rh = calloc(RH_N, sizeof *rh))) return;
    uint32_t i = (uint32_t)((h * 0x9E3779B97F4A7C15ull) >> 49) & (RH_N - 1), free_at = RH_N;
    for (int k = 0; k < 64; k++, i = (i + 1) & (RH_N - 1)) {
        if (rh[i].n > 0 && rh[i].h == h) {
            rh[i].n += d;
            return;
        }
        if (rh[i].n <= 0 && free_at == RH_N) free_at = i;
    }
    if (d > 0 && free_at != RH_N) rh[free_at].h = h, rh[free_at].n = d;
}
static int rh_has(uint64_t h) {   // ts_lock held
    if (!rh) return 0;
    uint32_t i = (uint32_t)((h * 0x9E3779B97F4A7C15ull) >> 49) & (RH_N - 1);
    for (int k = 0; k < 64; k++, i = (i + 1) & (RH_N - 1))
        if (rh[i].n > 0 && rh[i].h == h) return 1;
    return 0;
}
static void rh_note(uint64_t h, int d) {   // texture lock held
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    plat_mutex_lock(&ts_lock);
    rh_add(h, d);
    plat_mutex_unlock(&ts_lock);
}

static void rel_append(Tex *e) {
    if (e->has_chash) e->rh = 1, e->rh_hash = e->chash, rh_note(e->chash, 1);
    e->rel_next = NULL;
    e->rel_prev = rel_tail;
    if (rel_tail) rel_tail->rel_next = e;
    else rel_head = e;
    rel_tail = e;
}

static void rel_unlink(Tex *e) {
    if (e->rh) e->rh = 0, rh_note(e->rh_hash, -1);
    if (e->rel_prev) e->rel_prev->rel_next = e->rel_next;
    else rel_head = e->rel_next;
    if (e->rel_next) e->rel_next->rel_prev = e->rel_prev;
    else rel_tail = e->rel_prev;
    e->rel_prev = e->rel_next = NULL;
}

// DEBUG-WHONAMES: the last images destroy_tex let go, with who let them go (the caller's return address as an offset
// from destroy_tex, to look up in the link map) and the entry's state, for the device fault report: the fault comes
// seconds after the read and names an image nothing holds any more. A ring under the texture lock.
#define DT_N 32768
static struct { uint64_t image, addr, ns, frame, rel_frame, use_frame, draw_frame; int64_t ra; uint32_t w, h, fmt;
                uint8_t released, bound, gpu; } *dt_ring;
static uint32_t dt_next;

static __attribute__((noinline)) void destroy_tex(Tex *e) {
    if (e->pub.image && (dt_ring || (dt_ring = calloc(DT_N, sizeof *dt_ring)))) {   // DEBUG-WHONAMES
        dt_ring[dt_next++ % DT_N] = (typeof(*dt_ring)){
            (uint64_t)(uintptr_t)e->pub.image, e->key.addr, plat_now_ns(), __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
            e->released ? e->rel_frame : 0, e->use_frame, e->draw_frame,
            (int64_t)((const char *)__builtin_return_address(0) - (const char *)(uintptr_t)destroy_tex), e->key.d.width,
            e->key.d.height, e->key.d.format, (uint8_t)e->released, (uint8_t)(e->bound_to != NULL), (uint8_t)e->pub.gpu_owned};
    }
    if (__builtin_expect(sec_on, 0)) {
        if (!rep_keys) rep_keys = calloc(REP_SLOTS, sizeof *rep_keys);
        if (rep_keys) {
            uint64_t h = rep_hash(&e->key);
            rep_keys[h & (REP_SLOTS - 1)] = h;
        }
    }
    own_remove(e);
    tex_gen_bump_entry(e->key.addr, e->texq_base);
    nvn_hm_del(&by_key, &e->key);
    Tex *head = nvn_hm_get(&by_addr, &e->key.addr);
    if (head == e) {
        if (e->next_at) nvn_hm_put(&by_addr, &e->key.addr, e->next_at);
        else nvn_hm_del(&by_addr, &e->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == e) { p->next_at = e->next_at; break; }
    }
    if (e->released) nreleased--, rel_unlink(e);
    if (e->pub.image) vk_exec_image_gone(e->pub.image);
    // an enlarged shadow map's image goes (recycle_take clears pub.image of the entry it takes one from): the
    // image-ahead thread may make the next one
    if (e->pub.image && e->pub.scale > 1) __atomic_fetch_sub(&pm_scaled_n, 1, __ATOMIC_ACQ_REL);
    // an image bound into pool memory is held a while (vk_memory.c vk_mem_pool_image_hold); others go with the GPU
    int held = 0, parked_now = park_image(e);
    if (!parked_now && !(e->pub.image && !e->alloc && e->bound_to && (held = vk_mem_pool_image_hold(e->pub.image, e->bound_to, e->f.depth || e->f.stencil))))
        iw_defer_check(e->pub.image, e, __LINE__), vk_defer_image(e->pub.image, e->alloc);
    if (e->pub.image)   // DEBUG-PARKWATCH: the fault report's label says how the image's texture ended
        vk_addr_note((uint64_t)(uintptr_t)e->pub.image, " end %llu %s",
                     (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                     parked_now ? "park" : held ? "hold" : "free");
    if (held) {   // DEBUG-HOLDCOUNT: when each held image went in, depth ones apart (the 60-frame perf line counts them)
        uint64_t ns = plat_now_ns();
        hold_ring[hold_n++ % HOLD_RING] = (e->f.depth || e->f.stencil) ? ns | 1 : ns & ~1ull;
    }
    free(e->snap);
    free(e->edge);
    // The record goes once the GPU is past the submission being recorded, as its image: the replay keeps &e->pub as a
    // render target for the rest of a command buffer (vk_exec.c resolve_target) and DLSS for the frame, and this runs on
    // whichever thread released the texture. Freed here, a pass that began again read the freed record, cached a view of
    // the dead image (vk_exec_target_view), and a later image given the same handle drew through it into freed memory.
    e->pub.gone = 1;
    vk_defer_free(e);
}

// Whether released entry r's image was made from create info ci (recycle_take).
static int recycle_fits(const Tex *r, const VkImageCreateInfo *ci) {
    return r->pub.format == ci->format && r->pub.width == ci->extent.width && r->pub.height == ci->extent.height &&
           r->pub.depth == ci->extent.depth && r->pub.levels == ci->mipLevels && r->pub.layers == ci->arrayLayers &&
           r->pub.samples == (uint32_t)ci->samples && r->pub.usage == ci->usage && r->ci_flags == ci->flags;
}

// REKARINA_TEX_RECYCLE=<frames> (default 30, 0: off): a new GPU-only image takes the image and memory of a released entry of the same
// shape that has been released at least that long, instead of allocating. The console renders transient targets into the same
// memory one after the other; here each address got an image of its own, about 3 GB released and waiting in heavy scenes.
// An enlarged shadow map (scale > 1, shadow_scale_of) is taken back at once: OoT's transient target allocator gives its
// cascade array a new address whenever the frame's set of targets changes, and with the age rule the new address made
// a second 234 MiB array on the replay thread (41-74 ms in a shown frame) while the old one waited out its frames, and
// the two took turns from then on. The array has memory of its own and the game renders all of it before it samples
// it, so the new entry may have it as soon as no part being recorded reads or writes the old one: neither written in
// the submission being recorded nor drawn with this frame. Earlier parts on the queue are ordered before its new
// contents by init_image (reused_now). Every map of that shape is enlarged alike, so the scale need not match on its
// own, but it is compared all the same.
static VkImage recycle_take(const VkImageCreateInfo *ci, int scale, VmaAllocation *out) {
    static int age = -1;
    static uint64_t taken;
    if (age < 0) {
        const char *s = cfg_env("REKARINA_TEX_RECYCLE");
        // OoT: 6 frames. Its transient targets never come back to an address (REKARINA_TEXNEW: 3195 distinct
        // address and shape pairs of 3273 new images), so 30 frames only kept ~240 dead targets, ~850 MiB, waiting.
        age = s ? atoi(s) :
#ifdef GAME_OOT
              6;
#else
              30;
#endif
    }
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    Tex *r = NULL;
    if (scale > 1) {
        uint64_t rec = vk_exec_recording_fence();   // (0 off the replay: nothing qualifies)
        for (Tex *q = rel_head; q && !r; q = q->rel_next)   // the newest at the tail: the whole list
            if (q->alloc && !q->bound_to && !q->pub.mirrored && q->pub.scale == scale && q->wr_fence < rec &&
                q->draw_frame < now && recycle_fits(q, ci))
                r = q;
    } else {
        if (!age) return VK_NULL_HANDLE;
        uint64_t done = vk_sync_fence_done();
        for (Tex *q = rel_head; q && !r && now - q->rel_frame >= (uint64_t)age; q = q->rel_next)
            if (q->alloc && !q->bound_to && !q->pub.mirrored && q->wr_fence <= done && !released_in_use(q, now) &&
                recycle_fits(q, ci))
                r = q;
    }
    if (!r) return VK_NULL_HANDLE;
    VkImage img = r->pub.image;
    *out = r->alloc;
    vk_exec_image_gone(img);
    r->pub.image = VK_NULL_HANDLE, r->alloc = NULL;
    destroy_tex(r);
    if (!(++taken & 255)) rt_log("vk: recycled %llu released images", (unsigned long long)taken);
    return img;
}

// ---- REKARINA_TEX_ADOPT (default on; 0: off): a new texture whose storage holds the bytes a released one was filled
// from takes that one's image instead of an upload.
// The game's texture manager compacts its memory all the time in some places (the thread aglTextureDataMgr::
// Compaction; the Market and the field, not the forest): it copies a texture's bytes to another address,
// finalizes the texture object and initializes one of the same shape there, a block of 9 to 14 MiB every few frames
// and back again later. On the console that is a memcpy. Here the new address was a new image, created and filled
// again: about 1.5 images a frame, 0.6 to 0.8 ms of the replay for the staging copies, as much again for the GPU's
// copies, and frames of 25 to 33 ms where a few large ones met (the GPU busy for 20 ms; 2026-11-12).
// The bytes are hashed when an entry with memory of its own is made (64 bits over the whole storage, 4 lanes; a
// collision needs the same shape and size too). A released entry of the same descriptor and hash that the GPU never
// wrote and whose storage was not uploaded a second time holds exactly these texels: the two entries trade images and
// the released one goes, with the new, empty image. Among the newest ADOPT_LOOK released entries only: the move's
// source was released a moment ago.
#define ADOPT_LOOK 512
static uint64_t n_adopted, adopted_bytes, n_adopt_miss;

static inline uint64_t rotl64(uint64_t v, int r) { return (v << r) | (v >> (64 - r)); }
static uint64_t content_hash(const uint8_t *p, uint64_t n) {
    const uint64_t M = 0xff51afd7ed558ccdull;
    uint64_t h[4] = {0x9e3779b97f4a7c15ull ^ n, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull};
    for (; n >= 32; p += 32, n -= 32) {
        uint64_t w[4];
        memcpy(w, p, 32);
        for (int i = 0; i < 4; i++) h[i] = rotl64(h[i] ^ w[i], 29) * M;
    }
    if (n) {
        uint64_t w[4] = {0};
        memcpy(w, p, (size_t)n);
        for (int i = 0; i < 4; i++) h[i] = rotl64(h[i] ^ w[i], 29) * M;
    }
    uint64_t r = h[0] ^ rotl64(h[1], 17) ^ rotl64(h[2], 31) ^ rotl64(h[3], 47);
    r ^= r >> 33, r *= 0xc4ceb9fe1a85ec53ull, r ^= r >> 29;
    return r;
}

static int comes_back_intact(const Tex *e);

// ---- moved storage
// The Market's texture compaction moves a texture's bytes with one memcpy of 9 to 14 MiB and registers the texture at
// the new address; a few moves later it moves them back. The new entry found its image by the hash of its storage
// (adopt_released), and hashing was what such a frame cost: six textures in every sixth frame, 0.4 to 0.5 ms each,
// 2.5 to 3 ms on the replay thread (17 % of the Market's frames, 2026-11-13). The game's large copies pass through
// nvn_api_core.c ov_memcpy, which notes them here. A new entry whose storage the newest such copy filled, from the
// storage address of a released or parked texture with the same descriptor, size and content print, has that
// texture's bytes and takes its hash. The print is the hash of 64 samples of 64 bytes spread over the storage and
// its two ends: it stands between the copy's record and a storage written again since, by other means than a large
// copy. REKARINA_TEX_MOVED=0: every new entry hashes its storage.
#define MOVES 4096
static struct { uint64_t dst, src, n; } moves[MOVES];
static uint32_t nmoves;
static PlatMutex moves_lock = PLAT_MUTEX_INIT;
static uint64_t n_moved_hits;

void gfx_guest_moved(uint64_t dst, uint64_t src, uint64_t n) {   // any guest thread
    plat_mutex_lock(&moves_lock);
    moves[nmoves++ % MOVES] = (typeof(moves[0])){dst, src, n};
    plat_mutex_unlock(&moves_lock);
}

// 1: the newest noted copy that touches [addr, addr + size) filled all of it; *src: where those bytes came from.
static int moved_from(uint64_t addr, uint64_t size, uint64_t *src) {
    int found = 0;
    plat_mutex_lock(&moves_lock);
    for (uint32_t k = 0; k < MOVES && k < nmoves; k++) {
        const typeof(moves[0]) *m = &moves[(nmoves - 1 - k) % MOVES];
        if (addr >= m->dst + m->n || addr + size <= m->dst) continue;   // does not touch it
        if (addr >= m->dst && addr + size <= m->dst + m->n) *src = m->src + (addr - m->dst), found = 1;
        break;   // the newest one that touches it decides
    }
    plat_mutex_unlock(&moves_lock);
    return found;
}

// 1: the newest noted copy that read from [addr, addr + size) took all of it; *dst: where those bytes went.
static int moved_to(uint64_t addr, uint64_t size, uint64_t *dst) {
    int found = 0;
    plat_mutex_lock(&moves_lock);
    for (uint32_t k = 0; k < MOVES && k < nmoves; k++) {
        const typeof(moves[0]) *m = &moves[(nmoves - 1 - k) % MOVES];
        if (addr >= m->src + m->n || addr + size <= m->src) continue;
        if (addr >= m->src && addr + size <= m->src + m->n) *dst = m->dst + (addr - m->src), found = 1;
        break;
    }
    plat_mutex_unlock(&moves_lock);
    return found;
}

static uint64_t content_print(const uint8_t *p, uint64_t n) {
    uint8_t buf[66 * 64];
    if (n <= sizeof buf) return content_hash(p, n) ^ 0x5bd1e995u;
    memcpy(buf, p, 64);
    memcpy(buf + 64, p + n - 64, 64);
    for (uint64_t i = 0; i < 64; i++) memcpy(buf + 128 + i * 64, p + ((n - 64) / 64 * i & ~(uint64_t)63), 64);
    return content_hash(buf, sizeof buf) ^ n;
}

// Images of destroyed entries, kept by what they hold: the move's source is often destroyed before its copy is made
// (another texture is initialized at its address, or it leaves the released list), and the same block comes back to
// an address it had a few moves ago. At most PARK_MAX images and REKARINA_TEX_PARK_MB of video memory (default 384; 0: none
// are kept), each for PARK_FRAMES presents at most; the oldest goes first. The cache is outside the released entries'
// pressure rule (sweep_pressure destroys those above 75 % of the device's budget, which a 12 GiB card at 1440p is
// always above, and their images come here), so it has its own: never more than 1/32 of the budget (128 MiB on a
// 4 GiB card), and above PARK_FULL percent of the budget nothing is kept and what is here goes, a few images a sweep.
// tex_lock held everywhere.
#define PARK_MAX 512
#define PARK_FRAMES 1800
#define PARK_FULL 97
typedef struct {
    uint64_t chash, size, bytes, frame;
    uint64_t cprint, addr;        // the entry's content_print and storage address (adopt_released, "moved storage")
    uint8_t moved;                // kept for its bytes' new address, outside the budget (park_image)
    nvn_texdesc d;
    uint8_t has_layout, layout[8];
    VkImage image;
    VmaAllocation alloc;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageCreateFlags ci_flags;
    uint32_t width, height, depth, levels, layers, samples;
} Parked;
static Parked parked[PARK_MAX];
static int nparked;
static uint64_t parked_bytes, park_budget = ~0ull, park_cap = ~0ull, n_parked, n_park_taken;
static uint64_t moved_bytes, n_moved_parked;   // of parked_bytes: images kept for a moved texture's new address
#define MOVED_FRAMES 240   // such an image waits this long for its texture to be made at the new address
static int park_hold;   // the device is above PARK_FULL percent of its budget (sweep_pressure)

static void park_drop(int i) {
    rh_note(parked[i].chash, -1);
    // DEBUG-PARKWATCH: the texture's end purged the replay's views and the pool's slots of this image (destroy_tex) and
    // nothing has taken it since (adopt_released would have removed it from here), so any view found now was cached
    // while it was parked: a holder the image's destruction would leave reading freed memory. Purged with it all the same.
    static uint64_t n_late;
    int late = vk_exec_image_gone(parked[i].image);
    if (late && n_late++ < 32)
        rt_log("vk: image cache: a parked image (%#llx, %ux%u, parked at frame %llu) still had %d views cached when it was "
               "dropped (%llu such drops)", (unsigned long long)(uintptr_t)parked[i].image, parked[i].width, parked[i].height,
               (unsigned long long)parked[i].frame, late, (unsigned long long)n_late);
    vk_addr_note((uint64_t)(uintptr_t)parked[i].image, " drop %llu",
                 (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    {   // DEBUG-IMGWATCH: the image cache's own record goes now; another record of the same image, or an entry, is a holder
        VkImage im = parked[i].image;
        parked[i].image = VK_NULL_HANDLE;
        iw_defer_check(im, NULL, __LINE__);
        parked[i].image = im;
    }
    vk_defer_image(parked[i].image, parked[i].alloc);
    parked_bytes -= parked[i].bytes;
    if (parked[i].moved) moved_bytes -= parked[i].bytes;
    parked[i] = parked[--nparked];
}

static void park_trim(uint64_t room) {
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int i = nparked - 1; i >= 0; i--)
        if (now - parked[i].frame > (parked[i].moved ? MOVED_FRAMES : PARK_FRAMES)) park_drop(i);
    uint64_t limit = park_budget < park_cap ? park_budget : park_cap;
    // the images kept for moved textures are outside the budget: they go by age, or oldest first when the table is full
    while (nparked && (nparked >= PARK_MAX || parked_bytes - moved_bytes + room > limit)) {
        int old = -1;
        for (int i = 0; i < nparked; i++)
            if ((!parked[i].moved || nparked >= PARK_MAX) && (old < 0 || parked[i].frame < parked[old].frame)) old = i;
        if (old < 0) break;
        park_drop(old);
    }
}

// The storage of e was copied elsewhere by a noted copy and the bytes there are still e's: the game is moving the
// texture, and the entry it makes at the new address looks for this image (adopt_released). Kept for that entry it is
// no extra video memory: the entry would otherwise get an image of its own and an upload. Most moved textures lost
// their image before that entry was made (another texture was moved into their old storage first, 94 % of the moves
// noted while running through the Market), and where video memory is short the cache kept nothing.
static int moved_away(const Tex *e) {
    static int moved_on = -1;
    if (moved_on < 0) { const char *v = cfg_env("REKARINA_TEX_MOVED"); moved_on = !v || *v != '0'; }
    uint64_t dst;
    VkMemSpan m;
    if (!moved_on || !moved_to(e->key.addr, e->t.L.size, &dst) || !vk_mem_resolve(dst, e->t.L.size, &m)) return 0;
    return content_print((const uint8_t *)(uintptr_t)dst, e->t.L.size) == e->cprint;
}

// destroy_tex: e's image goes into the cache instead of being destroyed. 1: taken (the caller must not destroy it).
static int park_image(Tex *e) {
    if (park_budget == ~0ull) {
        const char *v = cfg_env("REKARINA_TEX_PARK_MB"), *on = cfg_env("REKARINA_TEX_ADOPT");
        park_budget = (on && *on == '0') ? 0 : (uint64_t)(v && *v ? atoi(v) : 384) << 20;
    }
    uint64_t limit = park_budget < park_cap ? park_budget : park_cap;
    if (!park_budget || !e->has_chash || !e->uploaded || !e->alloc || e->bound_to || !e->pub.image || !comes_back_intact(e) ||
        !e->bytes)
        return 0;
    int moved = moved_away(e);
    if (!moved && (park_hold || e->bytes > limit / 4)) return 0;
    for (int i = 0; i < nparked; i++)   // one image for a content is enough
        if (parked[i].chash == e->chash && parked[i].size == e->t.L.size && !memcmp(&parked[i].d, &e->key.d, sizeof e->key.d)) return 0;
    park_trim(moved ? 0 : e->bytes);
    if (nparked >= PARK_MAX) return 0;
    if (moved) {
        moved_bytes += e->bytes;
        if (!(++n_moved_parked & 2047))
            rt_log("vk: image cache: %llu images kept for moved textures so far (%.0f MiB waiting now)",
                   (unsigned long long)n_moved_parked, moved_bytes / 1048576.0);
    }
    parked[nparked++] = (Parked){.chash = e->chash, .size = e->t.L.size, .bytes = e->bytes, .cprint = e->cprint,
                                 .addr = e->key.addr, .moved = (uint8_t)moved,
                                 .frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), .d = e->key.d,
                                 .has_layout = e->key.has_layout, .image = e->pub.image, .alloc = e->alloc,
                                 .format = e->pub.format, .usage = e->pub.usage, .ci_flags = e->ci_flags,
                                 .width = e->pub.width, .height = e->pub.height, .depth = e->pub.depth,
                                 .levels = e->pub.levels, .layers = e->pub.layers, .samples = e->pub.samples};
    memcpy(parked[nparked - 1].layout, e->key.layout, sizeof e->key.layout);
    rh_note(e->chash, 1);
    parked_bytes += e->bytes;
    n_parked++;
    vk_live_check_dead((uint64_t)(uintptr_t)e->pub.image);   // DEBUG-PARKWATCH: a recording that still names it is stale
    // e->pub.image stays as it is: the replay may hold &e->pub for the rest of the command buffer it is recording
    // (destroy_tex), and the image it names lives on here, with the same texels. Cleared, a draw recorded after this
    // sampled a null image and the device was lost in the Market within two minutes.
    return 1;
}

static void park_clear(void) {
    while (nparked) park_drop(0);
}

// sweep_pressure, with the device's use and budget: the cache's cap, and its retreat when the device is nearly full.
static void park_pressure(uint64_t used, uint64_t budget) {
    static int full = -1;   // REKARINA_TEX_PARK_FULL=<percent> overrides PARK_FULL
    if (full < 0) { const char *v = cfg_env("REKARINA_TEX_PARK_FULL"); full = v && atoi(v) > 0 ? atoi(v) : PARK_FULL; }
    park_cap = budget / 32;
    park_hold = used > budget / 100 * (uint64_t)full;
    static uint64_t held, sweeps;
    held += park_hold, sweeps++;
    if (!(sweeps & 16383))
        rt_log("vk: image cache: %d images, %.0f MiB of at most %.0f; held back by video memory in %llu of %llu sweeps (%.0f of %.0f MiB in use)",
               nparked, parked_bytes / 1048576.0, park_cap / 1048576.0, (unsigned long long)held, (unsigned long long)sweeps,
               used / 1048576.0, budget / 1048576.0);
    for (int i = 0; park_hold && nparked && i < 8; i++) {
        int old = -1;
        for (int k = 0; k < nparked; k++)
            if (!parked[k].moved && (old < 0 || parked[k].frame < parked[old].frame)) old = k;
        if (old < 0) break;
        park_drop(old);
    }
}

// e was just made for t and nothing has used its image yet (entry_of, tex_lock held). 1: e holds its texels now.
static int adopt_released(Tex *e, const tex_t *t) {
    static int on = -1;
    if (on < 0) { const char *v = cfg_env("REKARINA_TEX_ADOPT"); on = !v || *v != '0'; }
    e->has_chash = 0;
    if (!on || !e->pub.mirrored || e->pub.gpu_owned || !e->alloc || e->bound_to || !t->mem || t->L.size < 4096) return 0;
    uint32_t gen;
    uint64_t pre_cp = 0, pre_ch = 0;
    int pre = up_worker && up_pre.valid && up_pre.addr == (uintptr_t)t->mem && up_pre.size == t->L.size;
    if (pre) pre_cp = up_pre.cprint, pre_ch = up_pre.chash;
    // the hash bc1_choose took for this storage just before (with its page generation)
    if (!pre && bc1_pre.valid && bc1_pre.addr == (uintptr_t)t->mem && bc1_pre.size == t->L.size) {
        pre = 1, pre_cp = bc1_pre.cprint, pre_ch = bc1_pre.chash, gen = bc1_pre.gen;
        bc1_pre.valid = 0;
    } else
    // the texture stream's hash, taken ahead (ts_take_hash checks that the storage still holds those bytes)
    if (!pre && !up_worker && ts_take_hash(t, &pre_cp, &pre_ch, &gen)) pre = 1;
    else vk_mem_changed(e->pub.addr, t->L.size, 0, &gen);   // resets the write bits before the storage is read, as make_fresh
    uint64_t hsec = pre ? 0 : sec_begin();
    e->cprint = pre ? pre_cp : content_print((const uint8_t *)(uintptr_t)t->mem, t->L.size);
    Tex *src_rel = NULL;
    {   // "moved storage": the hash of the texture these bytes were copied from
        static int moved_on = -1;
        if (moved_on < 0) { const char *v = cfg_env("REKARINA_TEX_MOVED"); moved_on = !v || *v != '0'; }
        uint64_t src = 0;
        int have = 0;
        if (moved_on && moved_from((uintptr_t)t->mem, t->L.size, &src)) {
            TexKey sk = e->key;
            sk.addr = src;
            Tex *r = nvn_hm_get(&by_key, &sk);   // the texture these bytes were: the same descriptor at the source
            if (r && r != e && r->released && r->has_chash && r->cprint == e->cprint && r->t.L.size == t->L.size)
                e->chash = r->chash, have = 1, src_rel = r;
            for (int i = 0; i < nparked && !have; i++)
                if (parked[i].addr == src && parked[i].cprint == e->cprint && parked[i].size == t->L.size &&
                    !memcmp(&parked[i].d, &e->key.d, sizeof e->key.d))
                    e->chash = parked[i].chash, have = 1;
        }
        static uint64_t n_calls, n_copy, n_hashed_bytes;
        n_calls++, n_copy += src != 0;
        if (have) n_moved_hits++;
        else if (pre) e->chash = pre_ch;
        else e->chash = content_hash((const uint8_t *)(uintptr_t)t->mem, t->L.size), n_hashed_bytes += t->L.size;
        sec_end(SEC_TS_HASH, hsec);
        if (!(n_calls & 8191))
            rt_log("vk: new textures looked for their image by content: %llu so far, %llu filled by a noted copy, %llu of "
                   "those took the copy's source hash, %.1f GiB hashed", (unsigned long long)n_calls,
                   (unsigned long long)n_copy, (unsigned long long)n_moved_hits, n_hashed_bytes / 1073741824.0);
        e->has_chash = 1;
    }
    uint64_t done = vk_sync_fence_done();
    int looked = 0;
    // the released texture at the copy's source first: the list below is only looked at ADOPT_LOOK entries deep, and
    // the game's transient targets push a moved texture's old entry past that within a frame or two
    for (Tex *r = src_rel ? src_rel : rel_tail; r && looked < ADOPT_LOOK; r = r == src_rel ? rel_tail : r->rel_prev, looked++) {
        if (r == e || !r->has_chash || r->chash != e->chash || r->t.L.size != t->L.size) continue;
        if (!r->uploaded || !r->alloc || r->bound_to || !comes_back_intact(r) || r->wr_fence > done || !r->pub.image)
            continue;
        if (memcmp(&r->key.d, &e->key.d, sizeof e->key.d) || r->key.has_layout != e->key.has_layout ||
            memcmp(r->key.layout, e->key.layout, sizeof e->key.layout))
            continue;
        if (r->pub.format != e->pub.format || r->pub.usage != e->pub.usage || r->ci_flags != e->ci_flags ||
            r->pub.levels != e->pub.levels || r->pub.layers != e->pub.layers || r->pub.width != e->pub.width ||
            r->pub.height != e->pub.height || r->pub.depth != e->pub.depth || r->pub.samples != e->pub.samples)
            continue;
        vk_exec_image_gone(e->pub.image);
        vk_exec_image_gone(r->pub.image);
        VkImage img = e->pub.image;
        VmaAllocation al = e->alloc;
        uint64_t by = e->bytes;
        e->pub.image = r->pub.image, e->alloc = r->alloc, e->bytes = r->bytes;
        r->pub.image = img, r->alloc = al, r->bytes = by;
        e->img_how = "a released entry's", e->img_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);   // DEBUG-IMGWATCH
        e->pub.serial = vk_texture_image_serial(), r->pub.serial = vk_texture_image_serial();   // (view cache keys)
        r->has_chash = 0;   // it holds the empty image now: nothing to keep (park_image)
        vk_addr_note((uint64_t)(uintptr_t)e->pub.image, " adopt %llu",   // DEBUG-PARKWATCH
                     (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        destroy_tex(r);
        e->wgen = gen;
        e->uploaded = 1;
        e->n_up++;
        n_adopted++, adopted_bytes += t->L.size, lf_count(LF_ADOPT, 1);
        ts_forget(t);   // filled without an upload: what the stream read for it goes
        if (!(n_adopted & 1023))
            rt_log("vk: %llu moved textures took their released images (%.1f GiB not uploaded again; %llu new textures found none)",
                   (unsigned long long)n_adopted, adopted_bytes / 1073741824.0, (unsigned long long)n_adopt_miss);
        return 1;
    }
    for (int i = 0; i < nparked; i++) {
        Parked *q = &parked[i];
        if (q->chash != e->chash || q->size != t->L.size || memcmp(&q->d, &e->key.d, sizeof e->key.d) ||
            q->has_layout != e->key.has_layout || memcmp(q->layout, e->key.layout, sizeof e->key.layout) ||
            q->format != e->pub.format || q->usage != e->pub.usage || q->ci_flags != e->ci_flags || q->levels != e->pub.levels ||
            q->layers != e->pub.layers || q->width != e->pub.width || q->height != e->pub.height || q->depth != e->pub.depth ||
            q->samples != e->pub.samples)
            continue;
        vk_exec_image_gone(e->pub.image);
        iw_defer_check(e->pub.image, e, __LINE__), vk_defer_image(e->pub.image, e->alloc);   // the empty image made for e
        e->pub.image = q->image, e->alloc = q->alloc, e->bytes = q->bytes;
        e->img_how = "the image cache's", e->img_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);   // DEBUG-IMGWATCH
        e->pub.serial = vk_texture_image_serial();   // (view cache keys)
        vk_live_check_born((uint64_t)(uintptr_t)q->image);   // DEBUG-PARKWATCH: a texture holds it again
        vk_addr_note((uint64_t)(uintptr_t)q->image, " adopt %llu",
                     (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        parked_bytes -= q->bytes;
        if (q->moved) moved_bytes -= q->bytes;
        rh_note(q->chash, -1);
        *q = parked[--nparked];
        e->wgen = gen;
        e->uploaded = 1;
        e->n_up++;
        n_adopted++, n_park_taken++, adopted_bytes += t->L.size, lf_count(LF_ADOPT, 1);
        ts_forget(t);   // filled without an upload: what the stream read for it goes
        if (!(n_adopted & 1023))
            rt_log("vk: %llu moved textures took their released images (%.1f GiB not uploaded again; %llu new textures found none)",
                   (unsigned long long)n_adopted, adopted_bytes / 1073741824.0, (unsigned long long)n_adopt_miss);
        return 1;
    }
    n_adopt_miss++;
    return 0;
}

// ---- REKARINA_TEX_FOLLOW (1 on, the default; 0 off): a moved texture keeps its entry.
// The game's texture compaction (above) copies a block of textures to another address, finalizes each texture object
// and initializes one of the same descriptor at the new address, 2-3 textures a frame while the hero stands in the
// Market. Each move made a new entry: an empty image created (the driver's object lock), then the released entry's
// image taken by content (adopt_released) or, when another texture had replaced the old one at its address first, an
// upload from the cache file; views made for the new image, the slots that showed the old one marked as gone and
// written again. One frame in five of the Market had a pool update of 1.2-14 ms that way, and it was most of the
// replay's frame-to-frame variance (2026-11-18). Here the entry itself moves with its bytes: when the newest noted copy
// filled the new storage from the old one, the entry at the old address has the same descriptor, holds exactly the
// texels it was filled from (comes_back_intact, no write since its upload: has_chash, no WriteTexels) and the new
// storage still holds them (content_print), the entry is keyed to the new address. Its image, its views and its
// record stay, so a slot registered again with the moved texture resolves to the view it already shows and the pool
// writes nothing. Two places: a lookup of the new address that finds no entry (entry_of), and an entry about to be
// replaced at its old address by another texture moved there (entry_of's replace loop), which is kept, released, at
// the new address for the registration that follows.
// Measured with REKARINA_PRESSURE_GRACE=30 at the Market's square (ABBA, 2026-11-18): shown frames' jumps over 4 ms 8.0
// against 11.9 a second, over 8 ms 1.2 against 2.2, the replay's p90 10.2 against 10.7 ms, +2.3 fps; the pool's
// resolutions of slots whose image went from ~130 ms to ~4 ms per 600 updates.
static uint64_t n_follow, n_follow_kept, follow_bytes;

static int follow_on(void) {
    static int on = -1;
    if (on < 0) {
        on = rt_knob("TEX_FOLLOW", "1") != 0;
        rt_log("vk: a texture the game moves %s (REKARINA_TEX_FOLLOW=%d)",
               on ? "keeps its entry and image at the new address" : "gets a new entry at the new address", on);
    }
    return on;
}

static int follow_ok(const Tex *r, uint64_t size) {
    return r->has_chash && r->uploaded && comes_back_intact(r) && !r->bound_to && r->alloc && r->pub.image && !r->pub.gone &&
           !r->snap && !r->wt_n && !r->texq_base[0] && !r->listed && r->t.L.size == size;
}

// r goes to key k, its bytes now at k->addr; the caller checked them (texture lock held).
static void follow_rekey(Tex *r, const TexKey *k) {
    tex_gen_bump_entry(r->key.addr, r->texq_base);   // the slots' kept resolutions at the old address go
    nvn_hm_del(&by_key, &r->key);
    Tex *head = nvn_hm_get(&by_addr, &r->key.addr);
    if (head == r) {
        if (r->next_at) nvn_hm_put(&by_addr, &r->key.addr, r->next_at);
        else nvn_hm_del(&by_addr, &r->key.addr);
    } else {
        for (Tex *p = head; p; p = p->next_at)
            if (p->next_at == r) { p->next_at = r->next_at; break; }
    }
    r->key = *k;
    r->pub.addr = k->addr;
    r->t.mem = (uint8_t *)(uintptr_t)k->addr;
    edge_keep(r);   // its first and last page lie elsewhere now
    r->next_at = nvn_hm_get(&by_addr, &r->key.addr);
    nvn_hm_put(&by_addr, &r->key.addr, r);
    nvn_hm_put(&by_key, &r->key, r);
    tex_gen_bump(r->key.addr);
    n_follow++, follow_bytes += r->t.L.size;
    if (!(n_follow & 1023))
        rt_log("vk: %llu moved textures kept their entries (%.1f GiB; %llu of them kept before another texture took their "
               "old address)", (unsigned long long)n_follow, follow_bytes / 1073741824.0, (unsigned long long)n_follow_kept);
}

// The new storage [addr, addr + size) of key k: whether r, the entry at the old address `src`, may move there. Reads
// the new storage's write state first (its generation becomes r's), then its content print. may_destroy 0: the caller
// walks a list that released entries at the new address may be on, so any entry there means no.
static int follow_check(Tex *r, const TexKey *k, uint64_t size, int may_destroy) {
    if (!follow_ok(r, size) || nvn_hm_get(&by_key, k)) return 0;
    for (Tex *p = nvn_hm_get(&by_addr, &k->addr); p; p = p->next_at)
        if (!p->released || !may_destroy) return 0;   // a live texture at the new address: an alias, the normal path decides
    uint32_t gen;
    vk_mem_changed(k->addr, size, 0, &gen);
    if (content_print((const uint8_t *)(uintptr_t)k->addr, size) != r->cprint) return 0;
    for (Tex *p = nvn_hm_get(&by_addr, &k->addr), *nx; p; p = nx) {   // released ones of another descriptor go, as in entry_of
        nx = p->next_at;
        destroy_tex(p);
    }
    r->wgen = gen;
    return 1;
}

// entry_of found no entry for t: the entry its bytes were copied from, moved here; NULL if there is none.
static Tex *follow_new(const tex_t *t, const TexKey *k) {
    uint64_t src;
    if (!follow_on() || !t->mem || t->L.size < 4096 || !tex_mirrored(t) || !moved_from(k->addr, t->L.size, &src))
        return NULL;
    TexKey sk = *k;
    sk.addr = src;
    Tex *r = nvn_hm_get(&by_key, &sk);
    if (!r || !follow_check(r, k, t->L.size, 1)) return NULL;
    follow_rekey(r, k);
    ts_forget(t);   // what the texture stream read ahead for the new address goes
    return r;
}

// The released entry p is about to be destroyed (entry_of: another texture takes its address; sweep_pressure: video
// memory is short): 1 when p's bytes were moved and p went to their new address instead, released still and counted as
// looked up now, so the registration that follows there finds it (a moved texture's old entry otherwise went before
// the game registered the texture at its new address, and the new entry found its image in the image cache or not
// at all: 3,000 of ~14,000 moves kept at the square without this, 2026-11-18). Kept for video memory exactly as the
// image cache keeps a moved texture's image (park_image), which it replaces.
static int follow_away(Tex *p, int may_destroy) {
    uint64_t dst, now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!follow_on() || !p->released || !p->pub.mirrored || p->t.L.size < 4096) return 0;
    // Moved once in this frame already: the game's compaction moves a block back and forth, so the newest copy that
    // read from the new address may lead back to the old one, and a sweep that met the entry again at its list's tail
    // moved it there and back without end (a tour hung at the crater with the pressure grace 0, 2026-11-18). A sweep
    // leaves it be (1); entry_of, which needs its address, destroys it as before (0).
    if (p->follow_frame == now + 1) return !may_destroy;
    // A sweep moves an entry once: one it moved before that nothing looked up since goes as it would have (else a
    // block the game moves back and forth would keep its released entries alive and bouncing for ever).
    if (!may_destroy && p->follow_frame && p->use_frame < p->follow_frame) return 0;
    if (!moved_to(p->key.addr, p->t.L.size, &dst)) return 0;
    TexKey k = p->key;
    k.addr = dst;
    if (!follow_check(p, &k, p->t.L.size, may_destroy)) return 0;
    follow_rekey(p, &k);
    p->follow_frame = now + 1;   // (0: never moved)
    rel_unlink(p);   // released anew: the list stays oldest first
    p->rel_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    rel_append(p);
    p->use_frame = p->rel_frame;   // (released_in_use's clock) kept for the registration at the new address
    n_follow_kept++;
    return 1;
}

// Released entries with another descriptor at the same address are replaced; past RELEASED_MAX released entries the
// oldest one goes at each release (it comes back on demand, a mirrored one re-uploaded from its storage). TotK
// finalizes and initializes transient textures every frame, so a released entry is mostly taken back within frames.
// Dropping all of them at once when the count passed 4096 destroyed thousands of images in one frame: 333-422 ms
// frames every 40-50 s on a GTX 1050 Ti, then the textures still in use were created and uploaded again.
#define RELEASED_MAX 2048
// A released entry that lookups still reach is in use, whatever the game did with its texture object. OoT
// finalizes the object of its far shadow map (12288x12288 D16, 288 MiB of its own) and of the 4096x4096 one at a
// map's arrival and goes on reading them through their slots for as long as the map is on screen; the entries wait
// here, released, with the only copy of what the GPU drew. Counted from the release, the age rule (sweep_pressure)
// destroyed them 600 frames after the arrival, the limit of this list earlier still when a burst of transient
// targets filled it, and the lookup that came next made an empty image: at the castle the ground lost its sunlight
// six seconds after the scene appeared, while the map's name was still on screen, and stayed in shade (the step in
// the brightness of every frame's thumbnail, exactly 600 frames after the first lit one, 2026-11-14). So the clock
// of a released entry is its last lookup (touch: use_frame), not its release.
#define RELEASED_IDLE 120   // frames without a lookup before a released entry counts as let go for good
static int released_in_use(const Tex *e, uint64_t now) {
    static int on = -1;   // REKARINA_REL_KEEP=0: the old clock, the release (a test: the castle's ground goes dark again)
    if (on < 0) on = rt_knob("REL_KEEP", "1") != 0;
    return on && e->use_frame > e->rel_frame && now - e->use_frame < RELEASED_IDLE;
}

// DEBUG-RELDEPTH: REKARINA_REL_DEPTH_MAX=<n> (0, the default: no limit of its own) keeps at most n released depth
// entries that no lookup reached lately, the oldest going first: NVIDIA's driver holds ~0.2-0.3 MiB outside the
// allocations for every depth image alive and keeps the high-water mark (scratch drvreplay.c, 2026-11-17), and in the
// forest the outside count rose by +0.17 MiB per new image made while the released list filled.
static uint32_t rel_depth_n;   // released depth entries after the last sweep (the vram line)
static void sweep_released(void) {
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    Tex *e = rel_head;
    for (int looked = 0; nreleased > RELEASED_MAX && e && looked < 256; looked++) {   // the oldest that nothing reads
        Tex *next = e->rel_next;
        if (!released_in_use(e, now) && !follow_away(e, 0)) destroy_tex(e);   // (REKARINA_TEX_FOLLOW: kept if moved)
        e = next;
    }
    static int dmax = -1;
    if (dmax < 0) { const char *v = cfg_get("REKARINA_REL_DEPTH_MAX"); dmax = v && *v ? atoi(v) : 0; }
    if (dmax <= 0 && !(rt_trace & RT_TRACE_PERF)) return;   // the count is for the knob and the vram line only
    uint32_t nd = 0;
    for (Tex *p = rel_head; p; p = p->rel_next) nd += p->f.depth || p->f.stencil;
    for (Tex *p = rel_head, *nx; dmax > 0 && p && nd > (uint32_t)dmax; p = nx) {
        nx = p->rel_next;
        if ((p->f.depth || p->f.stencil) && !released_in_use(p, now)) destroy_tex(p), nd--;
    }
    rel_depth_n = nd;
}

// REKARINA_TEX_VRAM_PRESSURE (percent of the device-local budget, default 85; 0 = off): above it, each submit drops up to
// PRESSURE_STEP of the oldest released entries, so a 4 GiB card does not keep RELEASED_MAX released images (each ASTC
// one decoded to RGBA8) while the heap spills into system memory. A few per submit, for the reason above. First the
// entries that come back intact: mirrored ones the GPU never wrote, re-uploaded from their storage. A render target
// lives only in its image; dropping it brought the texture back empty (black flashes in a Depths boss fight, ~170
// "left uninitialised" aliases a minute: transient targets released and taken back within frames). So a target goes
// only once it has stayed released for REKARINA_TEX_DROP_AGE frames (default 600; 0: never). With 30, the value from
// 2026-11-10 to 2026-11-12, OoT lost what its lighting reads: it lets go of targets for longer than that
// and takes them back with their content, and in the Market the lit scene came out black for one frame 20 to
// 30 times in 95 s (fog, particles and HUD on a dark scene: the "shadow flicker" of the performance video at 1:22;
// measured with the flight recorder's flash watch at 80 % of the mean, 0 to 1 with 600). OoT kept ~2,000
// released entries of 2.9 GiB at 95 % of a 12 GiB card's budget, while pools that wanted a VRAM mirror (which needs
// used + 1/8 of the budget to fit, vk_memory.c) stayed in host memory and the GPU read them over the bus. The default
// is below that 87.5 % line, so the cache gives memory back before mirrors are refused.
#define PRESSURE_STEP 32
// REKARINA_PRESSURE_GRACE (frames, default 30; 0 = none): an entry released less than this long ago is not dropped for
// the pressure. The game's texture compaction releases a texture and registers its moved bytes at
// another address within a frame or two, and the pool's update finds the moved texture's image by the released entry
// (adopt_released); dropped first at a full budget, the moved texture was read from its cache file and uploaded again
// on the frame's thread: 32 textures (20-22 MiB) every ~270 frames in the village at 97 % of the budget (2026-11-18).
static int pressure_grace(void) {
    static int g = -1;
    if (g < 0) { const char *e = cfg_get("REKARINA_PRESSURE_GRACE"); g = e && *e ? atoi(e) : 30; if (g < 0) g = 0; }
    return g;
}
static int vram_pressure = -1, drop_age = -1;
static uint64_t n_pressure_drops, n_age_drops;

static int comes_back_intact(const Tex *e) { return e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty; }

// When a lookup last reached an entry: entry_of (every texture get and slot resolution) and the per-submit check of a
// slot lookup (vk_texture_by_id_locked). The texture pool refresh looks every registered slot up once in each
// submission that draws through the pool, so an entry no lookup reached for IDLE_ROUNDS such submissions and STALE_FRAMES
// presents has no live descriptor naming it: the game keeps the texture object, nothing samples it. The clock is
// look_round, which only a submission whose slot checks reached half of the registered pool slots moves on
// (vk_texture_new_submit): counted in submissions, a pause screen or a fade that draws nothing through the pool would
// make the whole world's textures idle within seconds.
#define IDLE_ROUNDS 240
#define STALE_FRAMES 120
static uint32_t look_round = 1;     // submissions whose lookups walked the texture pool
static uint32_t round_checks;       // slot checks of the submission being recorded (vk_texture_by_id_locked)
static uint32_t pool_slots;         // registered slots
// What draws use. The texture pool's set (vk_texpool.c) looks every registered slot up in turn, so the rule above
// only ever found textures the game had unregistered, and the game keeps the textures of the maps it left registered
// as long as its own heap has room: the pool made all of them resident. The forest, then the field, the Market
// and the castle held 7.3 GiB of textures on a 12 GiB card, video memory went over the budget and the frame rate from
// 95 to 28 with frames of 100 ms (2026-11-14; the title screen afterwards ran at 8). So an entry also keeps when a
// draw's binding last named it: a lookup that is not the pool's (pool_lookup). The clock is draw_round, moved on by
// the submissions that bound DRAW_SUBMIT_MIN slots or more for the first time in them: a pause screen or a movie
// draws a handful of textures and must not make the scene behind it look unused. An entry no draw named for
// COLD_ROUNDS such submissions and COLD_FRAMES presents is cold: evict_cold takes those, least recently drawn first,
// while video memory is over the line, and the pool leaves an evicted texture out (cold_keys) until a draw asks for it.
// Three ages, by how badly room is needed (sweep_pressure). Long cold (COLD_LONG of both clocks, half a minute and
// more of play): the textures of a map the player left. Short cold (COLD_SHORT, some seconds): also what the camera
// turned away from, uploaded again at its next turn. And while no scene is on screen (a loading screen: scene_frame)
// whatever no draw named for COLD_LOAD presents: the map behind the loading screen is gone from the screen, and what
// goes now is not there when the next map's textures arrive, so the two maps never have to fit together.
#define DRAW_SUBMIT_MIN 24
#define COLD_SHORT 600
#define COLD_LONG 3600
#define COLD_LOAD 30
// Over the game's video memory budget (used > budget, REKARINA_TEX_OVER=1), what no draw named for this many presents
// and scene rounds goes too: a second or so, so only what the picture of this moment does not use.
#define COLD_OVER 90
static uint64_t scene_frame;        // gfx_counters.frames when a submission last drew a scene
static int room_short;              // sweep_pressure: the room for new textures is under its reserve
static int cs_trim_due;   // sweep_pressure found video memory short: vk_texture_new_submit trims the unused compute pipelines
static uint64_t n_over_sweeps;      // sweeps that found the driver's use over its budget (mode 3 of evict_cold)
// REKARINA_TEX_OVER=1 (off by default until a run shows its cost): over the budget, evict_cold takes what no draw
// named for COLD_OVER presents.
static int over_budget_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_OVER", "0") != 0;
    return on;
}

// The game's format is ASTC, whatever the image holds (BC3 blocks, decoded texels, or ASTC itself where the card
// samples it): the class the picture test showed no shader to read through the pool (evictable).
static int fmt_astc(const TexVkFormat *f) {
    return f->astc_bc || f->astc_bw || (f->vk >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f->vk <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}
static uint32_t draw_round = 1;     // submissions that drew a scene
uint32_t vk_texture_scene_rounds(void) { return __atomic_load_n(&draw_round, __ATOMIC_RELAXED); }
// perf.c's long-frame report: presents since a submission last drew a scene (a loading screen draws none)
int perf_scene_frames_ago(void) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), s = __atomic_load_n(&scene_frame, __ATOMIC_RELAXED);
    return !s ? 1 << 20 : f > s ? (int)(f - s < (1 << 20) ? f - s : 1 << 20) : 0;
}
static uint32_t draw_checks;        // slots a draw looked up first in the submission being recorded
// vk_arrive.c, at the present of the frame just replayed (gfx_counters.frames still the one before): 1 when a submission
// of it drew a scene (DRAW_SUBMIT_MIN first draw lookups; the one being recorded counts too, its round is not closed).
// A submission's round is closed at the next one's start, which marks the frame number current then: the last
// submission of the frame before marks this one as well, a frame late at the end of a scene and never at its start.
int vk_texture_scene_in_frame(void) {
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), s = __atomic_load_n(&scene_frame, __ATOMIC_RELAXED);
    return (s && s >= f) || draw_checks >= DRAW_SUBMIT_MIN;
}

static int pool_lookup;             // the lookup under way is the texture pool's (pool_slot_locked; tex_lock)
static void drawn(Tex *e) {
    e->draw_round = draw_round;
    e->draw_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
}
static void touch(Tex *e) {
    e->use_round = look_round;
    // The texture pool resolves every registered slot, also the stale ones no draw reads: its lookup does not keep a
    // released entry (released_in_use). Counted, the pool kept ~2,000 released targets "in use" past the limit of the
    // list, and the sweeps dropped and the pool made them again in bursts (frames of 50-68 ms, 2026-11-15).
    if (!(pool_lookup && e->released)) e->use_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!pool_lookup || !e->draw_round) drawn(e);   // a new entry ages from its making
}
static int cold(const Tex *e, uint64_t now, uint32_t age) {
    return draw_round - e->draw_round >= age && now - e->draw_frame >= age;
}

// The keys of evicted textures (evict_cold). The pool's lookups make no entry for one (slot_resolve): its slot shows
// the stand-in, which nothing reads, since what was evicted is of a class no shader reads through the pool
// (evictable). A draw's lookup makes the entry as for a texture seen the first time and takes the key out (entry_of).
// When the game finalizes the texture the key is done: what comes to that storage next is another texture, which the
// pool makes resident as any new one. It stays for COLD_MOVE_FRAMES as a mark of where an evicted texture was, because
// the game's heap compaction moves textures (a copy, a new texture object at the new address, the old one finalized,
// in an order not relied on here): the texture at the new address is the evicted one still and stays out
// (cold_kept_out by the noted copy, vk_texture_register by the slot). The keys of a memory pool go with it
// (vk_texture_range_gone).
// Values: COLD_KEY (a draw's binding brings it back), COLD_TAB (a texture no draw ever bound: a handle table naming
// it brings it back too, ht_take), or COLD_GONE + (the frame the texture was finalized in) * 2 + (1 for COLD_TAB).
#define COLD_KEY ((void *)(uintptr_t)1)
#define COLD_TAB ((void *)(uintptr_t)2)
static int cold_val(const void *v) { return v == COLD_KEY || v == COLD_TAB; }
#define COLD_GONE 16
#define COLD_MOVE_FRAMES 240
static nvn_hmap cold_keys = {.klen = sizeof(TexKey)};
static uint32_t n_cold;        // keys of evicted textures (not the marks of finalized ones)
static uint64_t n_cold_kept;   // pool lookups left without an entry so far
static void cold_put(const TexKey *k, void *kind) {
    if (!cold_val(nvn_hm_put(&cold_keys, k, kind))) n_cold++;
}
static void *cold_del(const TexKey *k) {
    void *v = cold_keys.n ? nvn_hm_del(&cold_keys, k) : NULL;
    if (cold_val(v)) n_cold--;
    return v;
}
// k is evicted, or was when its texture was finalized a moment ago (what moved from there is the same texture): its
// kind, COLD_KEY or COLD_TAB. NULL otherwise.
static void *cold_or_just_gone(const TexKey *k) {
    void *v = cold_keys.n ? nvn_hm_get(&cold_keys, k) : NULL;
    if (!v || cold_val(v)) return v;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), w = (uint64_t)(uintptr_t)v - COLD_GONE;
    if (now - (w >> 1) <= COLD_MOVE_FRAMES) return w & 1 ? COLD_TAB : COLD_KEY;
    nvn_hm_del(&cold_keys, k);
    return NULL;
}
static void cold_released(const TexKey *k) {   // vk_texture_release
    void *v = cold_keys.n ? nvn_hm_get(&cold_keys, k) : NULL;
    if (!cold_val(v)) return;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    nvn_hm_put(&cold_keys, k, (void *)(uintptr_t)(COLD_GONE + (now << 1 | (v == COLD_TAB))));
    n_cold--;
}
static int moved_from(uint64_t addr, uint64_t size, uint64_t *src);
// slot_resolve, before it makes an entry for k (storage of size bytes): 1 when the lookup is the pool's and the
// texture is an evicted one.
static int cold_kept_out(const TexKey *k, uint64_t size) {
    if (!pool_lookup || !cold_keys.n) return 0;
    if (!cold_val(nvn_hm_get(&cold_keys, k))) {
        uint64_t src;
        TexKey sk = *k;
        if (!size || !moved_from(k->addr, size, &src)) return 0;
        sk.addr = src;
        void *kind = cold_or_just_gone(&sk);
        if (!kind) return 0;
        cold_put(k, kind);   // the evicted texture, moved here
    }
    n_cold_kept++;
    return 1;
}

// ---------------------------------------------------------------- handle tables
// What the shaders read through the texture pool's set (vk_texpool.c) no binding shows: a fragment shader loads a
// texture handle from a storage buffer of its draw (the game's material and baked-lighting tables, megabytes of
// records) and indexes the pool with it. In every one of 500 cached shaders with pool arrays that were traced, the
// index came from loads of the draw's storage buffers and from nothing else (2026-11-14). So the storage buffers of
// the draws whose pipelines read the pool are the handle tables (vk_exec_draw.c names them: vk_texture_handle_table),
// and a thread of its own reads them a few times a second and notes every word that has the form of a handle: its
// sampler id (bits 20 and up) is a registered sampler, its texture id (bits 0-19) a slot. The replay then counts the
// entries of those slots as used by a draw (vk_texture_new_submit: ht_take), and an evicted texture a table names
// again comes back through the pool. A word that only looks like a handle keeps a texture that could have gone;
// nothing a shader can read through the pool is missed as long as its table is one of these buffers.
// The tables are read with plat_mem_read, which fails where the game has unmapped the memory meanwhile.
#define HT_N 64
#define HT_IDS (1u << 20)
#define HT_KEEP_FRAMES 1800       // a table no draw named for this long is not read any more
#define HT_TABLE_MAX (64ull << 20)
typedef struct { uint64_t addr, size, frame; } HTab;
static HTab htab[HT_N];
static PlatMutex ht_lock = PLAT_MUTEX_INIT;   // htab, ht_pub, ht_gen
static PlatEvent ht_ev;                       // a new table: read now
static volatile int32_t ht_started;
static uint64_t *ht_pub;                      // the newest finished reading: a bit per texture id
static uint32_t ht_gen, ht_taken;             // readings finished / taken in by the replay (tex_lock)
static uint64_t ht_frame;                     // the frame the newest reading finished in
static uint8_t samp_reg[4096];                // registered sampler ids (racy reads by the thread: a late id is seen next time)
static struct { uint64_t scans, bytes, words, fails, ids, marked, revived; } HS;   // since the last perf line

static void ht_main(void *arg) {
    (void)arg;
    enum { CHUNK = 1 << 20 };
    plat_thread_background();   // nobody waits for a reading: off the cores the frame runs on, where there are slower ones
    uint8_t *buf = malloc(CHUNK);
    uint64_t *bits = malloc(HT_IDS / 8);
    if (!buf || !bits) return;
    for (;;) {
        plat_event_wait_ns(&ht_ev, 250000000ull);
        HTab t[HT_N];
        int n = 0;
        uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        plat_mutex_lock(&ht_lock);
        for (int i = 0; i < HT_N; i++)
            if (htab[i].addr && now - htab[i].frame <= HT_KEEP_FRAMES) t[n++] = htab[i];
        plat_mutex_unlock(&ht_lock);
        memset(bits, 0, HT_IDS / 8);
        uint64_t bytes = 0, words = 0, fails = 0;
        for (int i = 0; i < n; i++) {
            uint64_t size = t[i].size < HT_TABLE_MAX ? t[i].size : HT_TABLE_MAX;
            for (uint64_t off = 0; off < size; off += CHUNK) {
                size_t m = (size_t)(size - off < CHUNK ? size - off : CHUNK) & ~(size_t)3;
                if (!m) break;
                // A read fails as a whole when one page of it is not there (the game commits the large tables piece by
                // piece, or has unmapped the buffer meanwhile): then the chunk again a page at a time, with what is not
                // there read as zeros, so the pages that are there are never skipped.
                if (!plat_mem_read(t[i].addr + off, buf, m)) {
                    for (size_t pg = 0, pn; pg < m; pg += pn) {
                        pn = 4096 - (size_t)((t[i].addr + off + pg) & 4095);   // up to the page's end
                        if (pn > m - pg) pn = m - pg;
                        if (!plat_mem_read(t[i].addr + off + pg, buf + pg, pn)) memset(buf + pg, 0, pn), fails++;
                    }
                }
                bytes += m;
                const uint32_t *w = (const uint32_t *)buf;
                for (size_t k = 0; k < m / 4; k++) {
                    uint32_t v = w[k];
                    if (!v || !samp_reg[v >> 20]) continue;
                    bits[(v & (HT_IDS - 1)) >> 6] |= 1ull << (v & 63);
                    words++;
                }
            }
        }
        plat_mutex_lock(&ht_lock);
        if (!ht_pub) ht_pub = calloc(HT_IDS / 8, 1);
        if (ht_pub) memcpy(ht_pub, bits, HT_IDS / 8), ht_gen++;
        ht_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        HS.scans++, HS.bytes += bytes, HS.words += words, HS.fails += fails;
        plat_mutex_unlock(&ht_lock);
    }
}

// vk_exec_draw.c, any thread that records draws: [addr, addr + size) is a storage buffer of a draw whose pipeline reads
// the texture pool.
static int evict_cls(void);
void vk_texture_handle_table(uint64_t addr, uint64_t size) {
    static _Thread_local struct { uint64_t addr, frame; } last[8];
    static _Thread_local uint32_t turn;
    if (!addr || !size || evict_cls() < 3) return;   // the tables are read only where their word decides (evictable)
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int i = 0; i < 8; i++)
        if (last[i].addr == addr && now - last[i].frame < 30) return;   // told lately by this thread
    last[turn++ & 7] = (typeof(last[0])){addr, now};
    int fresh = 0, at = 0;
    plat_mutex_lock(&ht_lock);
    for (int i = 0; i < HT_N; i++) {
        if (htab[i].addr == addr) { at = i; goto found; }
        if (htab[i].frame < htab[at].frame) at = i;   // the one named longest ago goes
    }
    htab[at] = (HTab){addr, 0, 0}, fresh = 1;
found:
    if (size > htab[at].size) htab[at].size = size;
    htab[at].frame = now ? now : 1;
    plat_mutex_unlock(&ht_lock);
    if (!__atomic_load_n(&ht_started, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&ht_lock);
        if (!ht_started) {
            plat_event_init(&ht_ev, 0, 0);
            if (plat_thread_spawn(ht_main, NULL, "vk handle tables", 0)) __atomic_store_n(&ht_started, 1, __ATOMIC_RELEASE);
            else __atomic_store_n(&ht_started, -1, __ATOMIC_RELEASE);
        }
        plat_mutex_unlock(&ht_lock);
    }
    if (fresh && ht_started > 0) plat_event_set(&ht_ev);
}

// 1 while the tables are being read: the thread runs and its newest reading is not older than a second or two.
// Without that, only the textures no table can name may be evicted (evictable).
static int ht_fresh(void) {
    if (__atomic_load_n(&ht_started, __ATOMIC_ACQUIRE) <= 0) return 0;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    return ht_gen && now - ht_frame < 240;
}

// vk_texture_new_submit (tex_lock): the newest reading's texture ids count as used now.
static void ht_take(void) {
    static uint64_t *mine;
    if (__atomic_load_n(&ht_started, __ATOMIC_ACQUIRE) <= 0 || ht_gen == ht_taken) return;
    if (!mine && !(mine = malloc(HT_IDS / 8))) return;
    plat_mutex_lock(&ht_lock);
    memcpy(mine, ht_pub, HT_IDS / 8);
    ht_taken = ht_gen;
    plat_mutex_unlock(&ht_lock);
    int top = ntslots < (int)HT_IDS ? ntslots : (int)HT_IDS;
    for (int wi = 0; wi < (top + 63) / 64; wi++) {
        for (uint64_t b = mine[wi]; b; b &= b - 1) {
            int id = wi * 64 + __builtin_ctzll(b);
            if (id >= top || !tslots[id].used || !tslots[id].t.mem) continue;
            TexSlot *s = &tslots[id];
            HS.ids++;
            TexKey k[2];
            int nk = 1;
            make_key(&s->t, &k[0]);
            tex_t half;
            if (texq_mode > 0 && texq_shape(&s->t, &half)) make_key(&half, &k[nk++]);   // "texture quality": the smaller one
            for (int j = 0; j < nk; j++) {
                Tex *e = nvn_hm_get(&by_key, &k[j]);
                if (e && !e->released) {
                    // only for what nothing else shows the use of (evictable): the game's material tables name every
                    // texture it has loaded, the maps it left too, and counted for all of them nothing would ever go
                    if (!e->ever_drawn && !fmt_astc(&e->f)) drawn(e), HS.marked++;
                } else if (cold_keys.n && nvn_hm_get(&cold_keys, &k[j]) == COLD_TAB) {
                    cold_del(&k[j]);   // evicted, and a table names it again: the pool makes it
                    HS.revived++;
                    vk_tpool_slot_dirty(id);
                } else if (!e && room_short && !j) {
                    // held back while room is short (pool_holds_back) and named now: the pool looks at it at once
                    TexVkFormat f;
                    if (tex_vk_format(s->t.d.format, &f) && !(fmt_astc(&f) && s->t.d.depth <= 1)) vk_tpool_slot_dirty(id);
                }
            }
        }
    }
}

// Whether e may be evicted while the game keeps it registered: only what a draw binds by handle when it uses it.
// Baked lighting and the terrain's layers are read through the pool's set by handles in a storage buffer, with no
// binding to see; those stay as long as they are registered. Seen on screen with the pool showing the stand-in for
// one class at a time (REKARINA_TPOOL_HIDE, vk_texpool.c): without the textures decoded from ASTC the forest and the field
// look as they do with them (two runs of the same setting differ as much), without the other single-layer textures
// the lightmapped surfaces go dark and the stone path loses its stones.
// A texture some draw has bound (ever_drawn) is taken to be used through bindings: it goes by when a draw last
// named it, like the textures decoded from ASTC. The material tables name those too, for the maps the game left as
// well, so a table's word for them means nothing; with the pool showing the stand-in for every texture a draw had
// bound (REKARINA_TPOOL_HIDE=8) the picture did not change. A texture no draw ever bound is used through the pool if at
// all (the baked lighting's arrays among them): it goes by when a table last named it (ht_take), and only while the
// tables are being read (ht_fresh).
// The tables are read a few times a second, so a texture a table names for the first time is seen up to a quarter
// of a second late. The one that carries the handles (the game's table of baked lighting and material records, one
// buffer for everything loaded) is written when a map loads, behind the loading screen, and read from then on, so
// the textures of an arriving map are named before its first frame. Without this class the textures no draw ever
// bound stayed for every map visited (the masks and lighting of the parts of a map the player did not look at): 3.7
// GiB after six maps, and the card was over its budget again in the fifth minute of the tour.
// REKARINA_TEX_EVICT_CLASS: 0 none, 1 single-layer textures decoded from ASTC only, 2 those and every texture a draw
// has bound, 3 also what only the handle tables name, arrays and volumes included (default).
static int evict_class = -1;
static int evict_cls(void) {
    if (evict_class < 0) { const char *v = cfg_get("REKARINA_TEX_EVICT_CLASS"); evict_class = v && *v ? atoi(v) : 3; }
    return evict_class;
}
static int evictable(const Tex *e) {
    int cls = evict_cls(), many = e->pub.layers > 1 || e->pub.depth > 1;
    if (!cls) return 0;
    if (fmt_astc(&e->f) && !many) return 1;
    if (cls >= 2 && e->ever_drawn) return 1;
    return cls >= 3 && ht_fresh();
}

// slot_resolve, before it makes an entry for slot texture t on the pool's behalf (id: its slot). While room is short
// the pool makes no image for a texture nothing is known to use: an ASTC one (a draw makes it when it binds it), or
// one no handle table names. Made ahead as before, such textures were only uploaded to be evicted half a minute
// later: in a tour of six maps the evictions came to 11.8 GiB while 4 GiB of what stayed had never been bound by a
// draw. Nothing is noted for the slot: the pool asks again at its next turn and makes the image once there is room.
static int pool_holds_back(const tex_t *t, int id) {
    if (!pool_lookup || !room_short) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f)) return 0;
    if (fmt_astc(&f) && t->d.depth <= 1) return 1;
    if (evict_cls() < 3 || !ht_fresh()) return 0;   // what only a table can name is made ahead as ever (evictable)
    int named = 0;   // by the newest reading (ht_pub is replaced whole under ht_lock)
    plat_mutex_lock(&ht_lock);
    if (ht_pub && (unsigned)id < HT_IDS) named = (int)(ht_pub[id >> 6] >> (id & 63) & 1);
    plat_mutex_unlock(&ht_lock);
    return !named;
}

// REKARINA_TEX_EVICT (percent of the device-local budget, default 88, 0 = off): above it, once the released entries that
// come back intact are gone, live entries go too, least recently looked up first: those that come back intact (the CPU
// provides their bytes) and that are idle. A later lookup creates the image again and uploads it from its storage, as
// for a texture seen the first time. Up to EVICT_STEP entries per check; the next check waits until the GPU is past the
// submissions that may still name the images, as their memory goes back to the driver only then (vk_defer_image) and
// the budget would still show the old use. Going over the budget makes Windows move video memory to system memory, and
// the GPU stops for 100-250 ms at a time (2026-11-09: 9005 of 8789 MiB on the forest porch).
#define EVICT_STEP 128
static int tex_evict = -1;
static uint64_t evict_wait;                       // vk_sync_fence_done value the next eviction waits for
static uint64_t n_evict, n_evict_bytes;           // since the last perf line
static uint64_t n_evict_all, n_evict_all_bytes;   // so far
static uint64_t n_cold_back;                      // evicted textures a draw brought back so far
static int evict_logs;

// mode: 0 long cold, 1 short cold, 2 anything no draw named for COLD_LOAD presents (no scene is on screen), 3 over the
// budget: anything no draw named for COLD_OVER presents (over_budget_on).
static uint64_t evict_until_ns;   // no eviction batch before then (REKARINA_EVICT_HOLD_MS, at the end of evict_cold)
// Returns the bytes of the textures it evicted (0: none, or not yet: the wait above).
static uint64_t evict_cold(uint64_t want, uint64_t used, uint64_t budget, int mode) {   // tex_lock held
    if (vk_sync_fence_done() < evict_wait || plat_now_ns() < evict_until_ns) return 0;
    uint32_t min_age = mode == 3 ? COLD_OVER : mode ? COLD_SHORT : COLD_LONG;
    Tex *v[EVICT_STEP];
    int nv = 0;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < by_key.cap; i++) {   // the EVICT_STEP candidates no draw named for the longest, those first
        Tex *e = by_key.vals[i];
        if (!e || e->released || !e->bytes || !comes_back_intact(e) || !evictable(e)) continue;
        if (mode == 2 ? now - e->draw_frame < COLD_LOAD : !cold(e, now, min_age)) continue;
        uint64_t age = now - e->draw_frame;   // presents: the round clock stands still behind a loading screen
        int at = nv;
        while (at > 0 && now - v[at - 1]->draw_frame < age) at--;
        if (at == EVICT_STEP) continue;
        if (nv < EVICT_STEP) nv++;
        memmove(v + at + 1, v + at, sizeof *v * (size_t)(nv - 1 - at));
        v[at] = e;
    }
    uint64_t freed = 0;
    int n = 0;
    if (cold_keys.n > 65536) nvn_hm_clear(&cold_keys, NULL), n_cold = 0;   // mostly marks of finalized textures: start over
    for (; n < nv && freed < want; n++) {
        freed += v[n]->bytes;
        cold_put(&v[n]->key, v[n]->ever_drawn || fmt_astc(&v[n]->f) ? COLD_KEY : COLD_TAB);
        destroy_tex(v[n]);
    }
    if (!n) return 0;
    n_evict += (uint64_t)n, n_evict_bytes += freed;
    n_evict_all += (uint64_t)n, n_evict_all_bytes += freed;
    lf_count(LF_EVICT, n), lf_count(LF_EVICT_BYTES, (int64_t)freed);
    evict_wait = vk_sync_issued() + 2;
    {   // The next batch waits (REKARINA_EVICT_HOLD_MS, 500 ms by default): at most EVICT_STEP textures go per wait.
        // With 8000, the value that came with the eviction hold of vk_memory.c and stayed when that hold went, a
        // session through 22 maps could not free fast enough and ended over the budget at 10-39 fps; with 0 a check
        // every other submission freed over 1 GiB within a second on arriving in a large map, and both such
        // sessions lost the device seconds later (2026-11-16).
        static int64_t hold_ms = -1;
        if (hold_ms < 0) { const char *e = cfg_get("REKARINA_EVICT_HOLD_MS"); hold_ms = e && *e ? atoll(e) : 500; }
        if (hold_ms > 0) evict_until_ns = plat_now_ns() + (uint64_t)hold_ms * 1000000ull;
    }
    if (evict_logs < 4 || !(++evict_logs & ((rt_trace & RT_TRACE_PERF) ? 31 : 255))) {
        if (evict_logs < 4) evict_logs++;
        rt_log("vk: video memory %.0f of %.0f MiB in use: %llu textures no draw used lately evicted so far (%.2f GiB), "
               "the longest unused first%s; one comes back when a draw uses it (%llu did, %u wait)", used / 1048576.0,
               budget / 1048576.0, (unsigned long long)n_evict_all, n_evict_all_bytes / 1073741824.0,
               mode == 2 ? " (behind a loading screen)" : mode == 3 ? " (over the budget)" : mode ? " (room is short)" : "",
               (unsigned long long)n_cold_back, n_cold);
    }
    return freed;
}

// vk_memory.c mirror_create: a buffer pool's VRAM mirror missed the card's room by `bytes` (0: no longer wanted). A
// pool without its mirror makes every draw read its buffers over PCIe, so the textures make room for it: the next
// sweep (sweep_pressure_) evicts at least that much of what no draw used for COLD_SHORT presents, the pool is asked
// again a few dozen frames later. Any thread; the newest request stands, for ASK_FRAMES at most.
#define ASK_FRAMES 600
static volatile uint64_t room_ask, room_ask_frame;
void vk_texture_room_for(uint64_t bytes) {
    __atomic_store_n(&room_ask_frame, __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), __ATOMIC_RELAXED);
    __atomic_store_n(&room_ask, bytes, __ATOMIC_RELEASE);
}

// Live textures with their own memory by what brings them back, the idle part apart, then the formats holding the most
// and the largest entries (REKARINA_TRACE=perf, every 600 frames).
static void texture_census(uint64_t frame) {   // tex_lock held
    enum { K_BC, K_CPU, K_GPU, K_N };
    static const char *const kn[K_N] = {"ASTC as BC", "other CPU-written", "GPU-written"};
    uint64_t n[K_N] = {0}, b[K_N] = {0}, in[K_N] = {0}, ib[K_N] = {0}, bound = 0, nd[K_N] = {0}, ndb[K_N] = {0};
    struct { int fmt; uint64_t n, b; } fm[48];
    int nfm = 0;
    struct { uint64_t b; uint32_t w, h, d; int fmt, gpu; } big[6] = {{0}};
    const double M = 1048576.0;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || e->released) continue;
        if (!e->bytes) { bound += e->bound_to != NULL; continue; }
        int k = !comes_back_intact(e) ? K_GPU : e->f.astc_bc ? K_BC : K_CPU;
        n[k]++, b[k] += e->bytes;
        if (!e->ever_drawn) nd[k]++, ndb[k] += e->bytes;
        if (evictable(e) && comes_back_intact(e) && cold(e, now, COLD_LONG)) in[k]++, ib[k] += e->bytes;
        int j = 0;
        while (j < nfm && fm[j].fmt != (int)e->f.vk) j++;
        if (j == nfm && nfm < 48) fm[nfm++] = (typeof(fm[0])){(int)e->f.vk, 0, 0};
        if (j < nfm) fm[j].n++, fm[j].b += e->bytes;
        for (int q = 0; q < 6; q++)
            if (e->bytes > big[q].b) {
                memmove(big + q + 1, big + q, sizeof *big * (size_t)(5 - q));
                big[q] = (typeof(big[0])){e->bytes, e->pub.width, e->pub.height, e->pub.layers * e->pub.depth, (int)e->f.vk, k == K_GPU};
                break;
            }
    }
    char line[2048];
    int o = snprintf(line, sizeof line, "mem: textures frame %llu: live with own memory by kind, the long cold part that may be evicted in brackets:", (unsigned long long)frame);
    for (int k = 0; k < K_N; k++)
        o += snprintf(line + o, sizeof line - (size_t)o, " %s %llu %.0f MiB (%llu %.0f MiB), no draw ever bound %llu %.0f MiB;",
                      kn[k], (unsigned long long)n[k], b[k] / M, (unsigned long long)in[k], ib[k] / M,
                      (unsigned long long)nd[k], ndb[k] / M);
    snprintf(line + o, sizeof line - (size_t)o, " bound into pool memory %llu; evicted and not back %u, pool lookups left "
             "empty %llu", (unsigned long long)bound, n_cold, (unsigned long long)n_cold_kept);
    rt_log("%s", line);
    for (int x = 0; x < nfm; x++)
        for (int y = x + 1; y < nfm; y++)
            if (fm[y].b > fm[x].b) { typeof(fm[0]) t = fm[x]; fm[x] = fm[y], fm[y] = t; }
    o = snprintf(line, sizeof line, "mem: textures frame %llu: formats:", (unsigned long long)frame);
    for (int x = 0; x < nfm && x < 8; x++)
        o += snprintf(line + o, sizeof line - (size_t)o, " %d: %llu %.0f MiB;", fm[x].fmt, (unsigned long long)fm[x].n, fm[x].b / M);
    o += snprintf(line + o, sizeof line - (size_t)o, " largest:");
    for (int q = 0; q < 6 && big[q].b; q++)
        o += snprintf(line + o, sizeof line - (size_t)o, " %ux%ux%u fmt %d %s %.0f MiB;", big[q].w, big[q].h, big[q].d, big[q].fmt,
                      big[q].gpu ? "gpu" : "cpu", big[q].b / M);
    rt_log("%s", line);
    // What holds memory without being read: live entries by the age of their last lookup (under 60 frames, 60 to 600,
    // older) for each kind, and the released ones by the age of their release, with how many a lookup reached since.
    uint64_t an[K_N][3] = {{0}}, ab[K_N][3] = {{0}}, rn[2][3] = {{0}}, rb[2][3] = {{0}};
    // Depth images, every entry that has one (own memory, pool-bound, released): NVIDIA's driver reserves memory outside
    // the allocations for each depth image it creates (about 1.5 bits a texel: 687 KiB for 2560x1440, any depth
    // format, sampled or not; scratch drvdepth.c, 2026-11-17) and keeps it for later depth images when one goes.
    uint64_t dn[2] = {0}, dt[2] = {0};
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (e && e->pub.image && (e->f.depth || e->f.stencil)) {
            int r = e->released != 0;
            dn[r]++, dt[r] += (uint64_t)e->pub.width * e->pub.height * (e->pub.layers ? e->pub.layers : 1);
        }
        if (!e || !e->bytes) continue;
        int k = !comes_back_intact(e) ? K_GPU : e->f.astc_bc ? K_BC : K_CPU;
        if (e->released) {
            uint64_t age = now > e->rel_frame ? now - e->rel_frame : 0;
            int a = age < 60 ? 0 : age < 600 ? 1 : 2, g = k == K_GPU;
            rn[g][a]++, rb[g][a] += e->bytes;
            continue;
        }
        uint64_t age = now > e->use_frame ? now - e->use_frame : 0;
        int a = age < 60 ? 0 : age < 600 ? 1 : 2;
        an[k][a]++, ab[k][a] += e->bytes;
    }
    o = snprintf(line, sizeof line, "mem: textures frame %llu: by the last lookup (under 60 frames / 60-600 / older):", (unsigned long long)frame);
    for (int k = 0; k < K_N; k++)
        o += snprintf(line + o, sizeof line - (size_t)o, " %s %llu %.0f / %llu %.0f / %llu %.0f MiB;", kn[k],
                      (unsigned long long)an[k][0], ab[k][0] / M, (unsigned long long)an[k][1], ab[k][1] / M,
                      (unsigned long long)an[k][2], ab[k][2] / M);
    o += snprintf(line + o, sizeof line - (size_t)o, " released by age (same bands): CPU-provided %llu %.0f / %llu %.0f / %llu %.0f MiB,"
                  " GPU-written %llu %.0f / %llu %.0f / %llu %.0f MiB",
                  (unsigned long long)rn[0][0], rb[0][0] / M, (unsigned long long)rn[0][1], rb[0][1] / M,
                  (unsigned long long)rn[0][2], rb[0][2] / M, (unsigned long long)rn[1][0], rb[1][0] / M,
                  (unsigned long long)rn[1][1], rb[1][1] / M, (unsigned long long)rn[1][2], rb[1][2] / M);
    o += snprintf(line + o, sizeof line - (size_t)o, "; depth images live %llu (%.0f Mtexels, ~%.0f MiB of the driver's), released %llu "
                  "(%.0f Mtexels, ~%.0f MiB)", (unsigned long long)dn[0], dt[0] / 1e6, dt[0] * 0.19 / M,
                  (unsigned long long)dn[1], dt[1] / 1e6, dt[1] * 0.19 / M);
    o += snprintf(line + o, sizeof line - (size_t)o, "; twin bound %llu times, span %.1f..%.1f MiB",
                  (unsigned long long)twin_binds, twin_lo ? (twin_lo - 1) / M : 0.0, twin_hi / M);
    {   // live CPU-provided texture memory by the loading screen it was made after (now: map_gen)
        uint64_t gb[8] = {0};
        for (uint32_t i = 0; i < by_key.cap; i++) {
            const Tex *e = by_key.vals[i];
            if (!e || e->released || !e->bytes || !comes_back_intact(e)) continue;
            uint32_t age = map_gen - e->born_gen;
            gb[age < 7 ? age : 7] += e->bytes;
        }
        o += snprintf(line + o, sizeof line - (size_t)o, "; CPU textures by map generation (now %u; this map, 1, 2, .. 7+ back):",
                      (unsigned)map_gen);
        for (int g = 0; g < 8; g++) o += snprintf(line + o, sizeof line - (size_t)o, " %.0f", gb[g] / M);
        if (tex_gen_pools_on())
            o += snprintf(line + o, sizeof line - (size_t)o, "; generation pools %d (made %llu, gone %llu)", ngenp,
                          (unsigned long long)genp_made, (unsigned long long)genp_gone);
    }
    o += snprintf(line + o, sizeof line - (size_t)o, "; BC1 images filled %llu (%.0f MiB, as BC3 %.0f), made again %llu, "
                  "keys failed %u, .bc1 files written %lld, filled again from another file %lld", (unsigned long long)bc1_made,
                  bc1_made_bytes / M, 2 * bc1_made_bytes / M, (unsigned long long)bc1_redone, bc1_nfailed, (long long)bc1_written,
                  (long long)bc1_refilled);
    rt_log("%s", line);
}

// ---- video memory in the minute before a lost device
// Two players' devices were lost on small cards while the game was over its video memory budget (2026-11-19); their
// logs said so only in an occasional line. The pressure sweep notes, once a second, the budget, the use, the share of
// the use no allocation fills (holes the allocator keeps) and what the eviction, the pressure drops and the image
// cache did since the second before; vk_device_fault_report prints the last minute (vk_texture_vram_history_report).
#define VH_N 64
static struct { uint64_t ns; uint32_t used_mb, budget_mb, holes_mb, parked_mb, evicted, evicted_mb, dropped, made; } vh[VH_N];
static volatile uint32_t vh_n;
static uint64_t vh_last_ns, vh_ev0, vh_evb0, vh_dr0, vh_made0;

static void vh_note(uint64_t used, uint64_t budget) {   // tex_lock held
    uint64_t now = plat_now_ns();
    if (now - vh_last_ns < 1000000000ull) return;
    vh_last_ns = now;
    uint64_t dr = n_pressure_drops + n_age_drops, holes = vk_mem_vma_free();
    uint32_t i = vh_n % VH_N;
    vh[i].ns = now, vh[i].used_mb = (uint32_t)(used >> 20), vh[i].budget_mb = (uint32_t)(budget >> 20);
    vh[i].holes_mb = (uint32_t)(holes >> 20), vh[i].parked_mb = (uint32_t)(parked_bytes >> 20);
    vh[i].evicted = (uint32_t)(n_evict_all - vh_ev0), vh[i].evicted_mb = (uint32_t)((n_evict_all_bytes - vh_evb0) >> 20);
    vh[i].dropped = (uint32_t)(dr - vh_dr0), vh[i].made = (uint32_t)(n_made_all - vh_made0);
    vh_ev0 = n_evict_all, vh_evb0 = n_evict_all_bytes, vh_dr0 = dr, vh_made0 = n_made_all;
    __atomic_store_n(&vh_n, vh_n + 1, __ATOMIC_RELEASE);
}

// The last minute of vh_note's seconds, oldest first (any thread, at a lost device: racy reads, a report).
void vk_texture_vram_history_report(void) {
    uint32_t all = __atomic_load_n(&vh_n, __ATOMIC_ACQUIRE), n = all < 60 ? all : 60;
    if (!n) {
        rt_log("vk: video memory before the loss: no budget reading was taken");
        return;
    }
    uint64_t now = plat_now_ns();
    rt_log("vk: video memory in the last %u seconds before the loss (seconds before: in use / budget MiB, holes in the "
           "allocator's blocks, image cache; textures made, evicted (MiB), released ones dropped):", n);
    char line[1100];
    int o = 0;
    for (uint32_t k = 0; k < n; k++) {
        const typeof(vh[0]) *s = &vh[(all - n + k) % VH_N];
        o += snprintf(line + o, sizeof line - (size_t)o, " -%.0f %u/%u h%u c%u m%u e%u(%u) d%u;",
                      (double)(now - s->ns) / 1e9, s->used_mb, s->budget_mb, s->holes_mb, s->parked_mb, s->made,
                      s->evicted, s->evicted_mb, s->dropped);
        if (o > 900 || k + 1 == n) {
            rt_log("vk:%s", line);
            o = 0;
        }
    }
}

static void sweep_pressure_(int no_scene);
static void sweep_pressure(void) {   // tex_lock held
    iw_drain();   // DEBUG-IMGWATCH
    if (vram_pressure < 0) {
        const char *s = cfg_get("REKARINA_TEX_VRAM_PRESSURE");
        vram_pressure = s && *s ? atoi(s) : 75;
        if (vram_pressure < 0 || vram_pressure > 100) vram_pressure = 75;
        const char *a = cfg_get("REKARINA_TEX_DROP_AGE");
        drop_age = a && *a ? atoi(a) : 600;
        if (drop_age < 0) drop_age = 0;
        rt_log("vk: released textures under VRAM pressure: %s%d%%", vram_pressure ? "dropped above " : "off", vram_pressure);
    }
    if (tex_evict < 0) {
        const char *s = cfg_get("REKARINA_TEX_EVICT");
        tex_evict = s && *s ? atoi(s) : 88;
        if (tex_evict < 0 || tex_evict > 100) tex_evict = 88;
        if (tex_evict) rt_log("vk: textures no draw used lately are evicted while the room under %d%% of the video memory budget is short", tex_evict);
        else rt_log("vk: textures are never evicted (REKARINA_TEX_EVICT=0)");
    }
    // the budget query every 8 submits; every second one behind a loading screen, where textures arrive by the hundred
    uint64_t now_f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    int no_scene = now_f - scene_frame >= COLD_LOAD;
    if ((!vram_pressure && !tex_evict) || submit_epoch & (no_scene ? 1 : 7)) return;
    // What goes from here is destroyed only once the GPU is past every timeline value handed out so far, not only the
    // submission being recorded: a live texture evicted while it is cold may still be named by a submission the
    // replay recorded and holds (the queue interleave), or by one it records later from the game's earlier commands;
    // the device was lost once in the field after evictions (2026-11-15, a read inside an evicted texture's old
    // place). Eviction runs only while video memory is short, so the wait costs a few submissions' worth of nothing.
    // REKARINA_EVICT_QUIET=0: the stamp of any other texture.
    static int quiet = -1;
    if (quiet < 0) quiet = rt_knob("EVICT_QUIET", "1") != 0;
    if (quiet) vk_defer_past_reserved(1);
    sweep_pressure_(no_scene);
    if (quiet) vk_defer_past_reserved(0);
}

static void sweep_pressure_(int no_scene) {   // tex_lock held
    uint64_t used, budget;
    int exact, dropped = 0;
    if (!vk_mem_vram(&used, &budget, &exact)) return;
    vh_note(used, budget);
    park_pressure(used, budget);
    uint64_t freed = 0;
    if (vram_pressure && rel_head && used > budget / 100 * (uint64_t)vram_pressure) {
        uint64_t now0 = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        static int keep_used = -1;   // REKARINA_PRESSURE_KEEP_USED=0: drop them too, as before (a test switch)
        if (keep_used < 0) keep_used = rt_knob("PRESSURE_KEEP_USED", "1") != 0;
        uint64_t grace = (uint64_t)pressure_grace();
        for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP; e = next) {
            next = e->rel_next;
            if (grace && now0 - e->rel_frame < grace) break;   // the list is in release order: the rest are younger
            if (!comes_back_intact(e)) continue;
            // Not one that is read still (a slot or a binding follows it: released_in_use), as the drops by age and
            // sweep_released leave those alone; the pressure drops took them too, and three times in the night the
            // device was lost seconds later with a read where only such textures' images had lain.
            if (keep_used && released_in_use(e, now0)) continue;
            if (follow_away(e, 0)) continue;   // REKARINA_TEX_FOLLOW: its bytes moved; kept at their new address
            freed += e->bytes;
            destroy_tex(e);
            dropped++, n_pressure_drops++, lf_count(LF_DROP, 1);
        }
        if (drop_age && dropped < PRESSURE_STEP) {
            // Then anything released long enough ago, oldest first (the list is in release order).
            uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
            for (Tex *e = rel_head, *next; e && dropped < PRESSURE_STEP && now - e->rel_frame >= (uint64_t)drop_age; e = next) {
                next = e->rel_next;
                if (released_in_use(e, now)) continue;   // read through its slot still (above sweep_released)
                if (follow_away(e, 0)) continue;   // REKARINA_TEX_FOLLOW
                freed += e->bytes;
                destroy_tex(e);
                dropped++, n_age_drops++, lf_count(LF_DROP, 1);
            }
        }
    }
    // Room for what comes next: what is left under the line (REKARINA_TEX_EVICT percent of the budget) plus what the
    // allocator's blocks hold free. Freed textures leave their bytes in the blocks (a block goes back to the system
    // only when nothing is left in it), so the use the driver reports hardly falls after an eviction, and judged by it
    // alone every texture went the moment it was long cold, for no gain: the tour of six maps evicted 4.3 GiB that
    // way while 1.9 GiB of the blocks stood free. While the room is at least the reserve, nothing goes.
    uint64_t line = budget / 100 * (uint64_t)tex_evict;
    if (tex_evict) {
        uint64_t holes = vk_mem_vma_free(), filled = used > holes ? used - holes : 0;   // what really holds something
        uint64_t room = line > filled ? line - filled : 0;
        uint64_t reserve = budget / 12 > (512ull << 20) ? budget / 12 : 512ull << 20;
        room_short = room < reserve;   // the pool makes nothing ahead that nothing is known to use (pool_holds_back)
        // A buffer pool's mirror waits for room (vk_texture_room_for): at least that much goes, short cold first.
        uint64_t ask0 = __atomic_load_n(&room_ask, __ATOMIC_ACQUIRE), ask = ask0;
        if (ask && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) -
                       __atomic_load_n(&room_ask_frame, __ATOMIC_RELAXED) > ASK_FRAMES) {
            ask = 0;   // stale: the pool went, or was not used again
            __atomic_compare_exchange_n(&room_ask, &ask0, 0, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
        }
        ask = ask > freed ? ask - freed : 0;
        if (room + freed < reserve || ask) {
            int over = exact && used > budget && over_budget_on();
            int mode = no_scene ? 2 : over ? 3 : ask || room < reserve / 3 || used > budget ? 1 : 0;
            if (over && !n_over_sweeps++)
                rt_log("vk: video memory over the budget (%.0f of %.0f MiB in use): textures no draw named for %d presents "
                       "are evicted (REKARINA_TEX_OVER)", used / 1048576.0, budget / 1048576.0, COLD_OVER);
            uint64_t want = room + freed < reserve ? reserve - room - freed + (64ull << 20) : 0;
            uint64_t got = evict_cold(want > ask ? want : ask, used, budget, mode);
            if (ask && got) {   // what is still missing (unless the pool asked anew meanwhile); its next try says it anew
                uint64_t left = ask > got ? ask - got : 0;
                __atomic_compare_exchange_n(&room_ask, &ask0, left, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
                static int said;
                if (said < 16 && ++said)
                    rt_log("vk: textures evicted for a buffer pool's mirror: %.1f MiB (%.0f MiB were asked for, %.0f of "
                           "%.0f MiB in use)", got / 1048576.0, (ask + freed) / 1048576.0, used / 1048576.0,
                           budget / 1048576.0);
            }
            cs_trim_due = 1;   // the compute pipelines nothing dispatched go too (vk_texture_new_submit, outside the lock)
        }
    }
}

// REKARINA_TRACE=perf: what the released entries hold (tex_lock held). Own memory only: an entry bound into pool memory
// costs nothing of its own.
static void released_census(uint64_t frame) {
    struct { uint64_t n, bytes; } cls[3] = {{0}};   // 0 droppable (clean mirror), 1 GPU content, 2 other
    struct { uint32_t w, h, fmt, n; uint64_t bytes; } top[8] = {{0}};
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), old = 0, old_bytes = 0;
    for (Tex *e = rel_head; e; e = e->rel_next) {
        uint64_t b = 0;
        if (e->alloc) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(vk_ctx()->vma, e->alloc, &ai);
            b = ai.size;
        }
        int k = e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty ? 0 : e->pub.gpu_owned || e->gpu_dirty ? 1 : 2;
        cls[k].n++, cls[k].bytes += b;
        if (now - e->rel_frame >= 600) old++, old_bytes += b;
        int j = 0;
        while (j < 8 && top[j].n && !(top[j].w == e->key.d.width && top[j].h == e->key.d.height && top[j].fmt == (uint32_t)e->key.d.format)) j++;
        if (j == 8) {   // replace the smallest
            j = 0;
            for (int i = 1; i < 8; i++) if (top[i].bytes < top[j].bytes) j = i;
            if (top[j].bytes > b) continue;
            top[j].n = 0, top[j].bytes = 0;
        }
        top[j].w = e->key.d.width, top[j].h = e->key.d.height, top[j].fmt = (uint32_t)e->key.d.format;
        top[j].n++, top[j].bytes += b;
    }
    rt_log("perf: frame %llu: released textures: clean mirrors %llu (%.0f MiB), GPU content %llu (%.0f MiB), other %llu "
           "(%.0f MiB); released 600+ frames ago %llu (%.0f MiB); dropped by age %llu", (unsigned long long)frame,
           (unsigned long long)cls[0].n, cls[0].bytes / 1048576.0, (unsigned long long)cls[1].n, cls[1].bytes / 1048576.0,
           (unsigned long long)cls[2].n, cls[2].bytes / 1048576.0, (unsigned long long)old, old_bytes / 1048576.0,
           (unsigned long long)n_age_drops);
    char line[512];
    size_t o = 0;
    for (int i = 0; i < 8 && o < sizeof line - 64; i++)
        if (top[i].n)
            o += (size_t)snprintf(line + o, sizeof line - o, " %ux%u/%#x*%u=%.0fMiB", top[i].w, top[i].h, top[i].fmt, top[i].n,
                                  top[i].bytes / 1048576.0);
    rt_log("perf: frame %llu: released textures by size:%s", (unsigned long long)frame, o ? line : " none");
    n_age_drops = 0;
}

// ---------------------------------------------------------------- uploads
static uint32_t texel_alpha_one(int format, uint32_t *elem) {
    switch (format) {
    case 0x19: *elem = 1; return 0xFF;
    case 0x1A: *elem = 1; return 0x7F;
    case 0x1B: case 0x1C: *elem = 1; return 1;
    case 0x1D: *elem = 2; return 0x3C00;
    case 0x1E: *elem = 2; return 0xFFFF;
    case 0x1F: *elem = 2; return 0x7FFF;
    case 0x20: case 0x21: *elem = 2; return 1;
    case 0x22: *elem = 4; return 0x3F800000;
    default: *elem = 4; return 1;   // 0x23, 0x24
    }
}

// 3-component texels to 4 (n texels).
static void expand_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e, one = texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 3 * e, dst += 4 * e) {
        memcpy(dst, src, 3 * e);
        memcpy(dst + 3 * e, &one, e);
    }
}

static void shrink_rgb(int format, const uint8_t *src, uint8_t *dst, size_t n) {
    uint32_t e;
    texel_alpha_one(format, &e);
    for (size_t i = 0; i < n; i++, src += 4 * e, dst += 3 * e) memcpy(dst, src, 3 * e);
}

// REKARINA_ASTC=check: the compute decoder's texels of n regions (gpu: the regions one after the other) against
// astc_decode_ldr of the same raw blocks (raw: the staging bytes the regions point into).
static void astc_check(const Tex *e, const uint8_t *raw, const VkAstcRegion *ar, int n, const uint8_t *gpu) {
    static volatile int32_t n_bad;
    static uint64_t n_tex, n_texels, n_diff_tex;
    uint32_t bw = e->f.astc_bw, bh = e->f.astc_bh;
    uint64_t diff = 0, first = UINT64_MAX;
    uint32_t want = 0, got = 0;
    int first_k = 0;
    for (int k = 0; k < n; k++) {
        const VkAstcRegion *r = &ar[k];
        size_t sb = (size_t)((r->w + bw - 1) / bw) * ((r->h + bh - 1) / bh) * 16, st = (size_t)r->w * r->h * 4;
        uint8_t *cpu = malloc(st);
        for (uint32_t z = 0; z < r->slices; z++, gpu += st) {
            astc_decode_ldr(raw + r->src + z * sb, r->w, r->h, bw, bh, e->f.astc_srgb, cpu);
            if (!memcmp(cpu, gpu, st)) continue;
            for (size_t i = 0; i < st / 4; i++)
                if (memcmp(cpu + 4 * i, gpu + 4 * i, 4)) {
                    if (first == UINT64_MAX) {
                        first = i, first_k = k;
                        memcpy(&want, cpu + 4 * i, 4), memcpy(&got, gpu + 4 * i, 4);
                    }
                    diff++;
                }
        }
        free(cpu);
        n_texels += (uint64_t)r->w * r->h * r->slices;
    }
    n_tex++;
    if (diff) {
        n_diff_tex++;
        log_limited(&n_bad, "ASTC check: texture %#llx fmt %#x %ux%u: %llu texels differ, first region %d texel %llu "
                    "cpu %08x gpu %08x", (unsigned long long)e->pub.addr, e->t.d.format, bw, bh,
                    (unsigned long long)diff, first_k, (unsigned long long)first, want, got);
    }
    if (n_tex == 1 || !(n_tex & 63) || diff)
        rt_log("vk: ASTC check: %llu textures, %llu texels compared, %llu textures differ", (unsigned long long)n_tex,
               (unsigned long long)n_texels, (unsigned long long)n_diff_tex);
}

// Staging bytes of one part of a texture upload decoded or copied on the CPU (see upload). With REKARINA_ASTC=cpu a
// large ASTC array decodes to RGBA8 in one go: a staging buffer of 676682336 bytes failed (-2) on an Intel UHD with
// 8 GB of shared memory and the game closed before the title screen.
#define UPLOAD_PART (32ull << 20)   // fits the staging pool (vk_staging takes up to half of its 64 MiB block)

// Copies the whole storage into the image (every level and layer the image has).
// ---- the texture cache (tex_cache, REKARINA_TEX_CACHE; on by default since 2026-11-13): the BC3 form of ASTC textures on disk.
// With REKARINA_ASTC_BC the GPU decodes an ASTC texture and encodes it to BC3 at its first use: 16 to 35 ms of GPU time
// for one of the field's large textures, in the frame that first draws it, so running through the field had 58
// frames over 25 ms a minute where standing and turning had none (2026-11-12). The blocks that encode makes are the
// same every time, so they are kept: tex-cache/<hash of the storage>-<format>-<size>-<levels>x<layers>.bc3 beside the
// game (REKARINA_TEX_CACHE_DIR names another folder), every level and layer in upload's order. A texture whose storage
// has that hash is filled from the file with one buffer-to-image copy. A miss decodes as before and also copies the
// blocks to a host buffer, which the reaper thread writes out once the GPU is past the batch (vk_defer_call); the
// storage is hashed again after it was read, and a texture the game wrote meanwhile is not kept.
// The folder holds at most tex_cache_mb MiB (REKARINA_TEX_CACHE_MB, default 14336): a thread counts it at start and deletes
// files, in the order the listing gives them, until it fits; while the game runs nothing new is written once the
// count and what this run wrote reach the limit. The setup fills it ahead of play (tools/oot_prepare_textures.py,
// runtime/tex_prepare.c), with the keys and bytes play writes.
// On by default since 2026-11-13. What it gains depends on what decides the frame: docs/oot-notes.md, "The morning's
// control runs" (nothing in the Market, the field's long frames where the GPU is the limit). The device
// losses in the Market that were first laid at its door came from the texture pool (vk_texpool.c update).
typedef struct {
    char path[600];
    uint8_t *map;
    uint64_t size;
    VkBuffer buf;
    VmaAllocation alloc;
    int keep;
} TexCacheJob;
static volatile int64_t tc_loaded, tc_saved, tc_load_bytes, tc_failed;
static volatile int64_t tc_disk = -1;   // bytes in the folder: -1 until the start's count is done, then kept by the writes
static int64_t tc_limit;

typedef struct { const char *dir; int64_t total, limit, deleted; int files; } TexCacheScan;
static int tex_cache_count(const char *name, void *ctx) {
    TexCacheScan *s = ctx;
    char path[700];
    snprintf(path, sizeof path, "%s/%s", s->dir, name);
    int64_t n = plat_file_size(path);
    if (n < 0) return 1;
    if (s->total + n > s->limit) {   // over the limit from here on: this file goes
        plat_file_delete(path);
        s->deleted += n;
        return 1;
    }
    s->total += n, s->files++;
    return 1;
}
// The folder's files into the system's file cache, once at start. After a restart of the machine they are on the disk
// only, and the replay thread read each at disk speed at its texture's first use: the first run through the Market
// after a restart had 38 frames over 25 ms a minute where the next had 15 and a run without the cache 12, and
// the 180 frames after an arrival in the field took 22 ms each instead of 18 (2026-11-13). The count's thread reads
// them at the lowest priority, in the listing's order: REKARINA_TEX_CACHE_WARM_MB at most (default: the whole folder
// where the machine's memory holds it beside the game, less where it does not, see tex_cache_scan; 0: none). With the
// setup's prepared cache (12.9 GiB) a read stopped at 8 GiB left the field's files on the disk, and a first visit of
// the field had 128 frames over 25 ms a minute, its pool updates reading them inside the frame (2026-11-16). It
// gives up when the disk gives less than 150 MiB a second over the first 256 MiB: a hard disk, where these reads
// would hold up the game's own loading.
typedef struct { const char *dir; uint8_t *buf; int64_t done, limit; uint64_t t0; int files, slow; } TexCacheWarm;
static int tex_cache_warm_one(const char *name, void *ctx) {
    TexCacheWarm *w = ctx;
    char path[700];
    snprintf(path, sizeof path, "%s/%s", w->dir, name);
    PlatFile *f = plat_file_open(path);
    if (!f) return 1;
    for (uint64_t off = 0;;) {
        int64_t n = plat_file_pread(f, w->buf, 1u << 20, off);
        if (n <= 0) break;
        off += (uint64_t)n, w->done += n;
    }
    plat_file_close(f);
    w->files++;
    if (w->done >= w->limit) return 0;
    if (!w->slow && w->done >= (256ll << 20)) {   // judged once
        double sec = (double)(plat_now_ns() - w->t0) / 1e9;
        w->slow = (double)w->done / 1048576.0 < 150.0 * sec ? 1 : -1;
        if (w->slow > 0) return 0;
    }
    return 1;
}

// ---- BC1 for opaque ASTC textures (REKARINA_TEX_BC1, default on)
// An ASTC texture becomes BC3 here (vk_astc.c): 8 bits a texel. Of the prepared cache's 13.3 GiB of BC3 blocks, 3.9 GiB
// (29 %) are textures whose every block has the alpha half the encoder writes for alpha 255 (a0 = a1 = 255, all
// indices 0; bc3_encode.comp), and the encoder's colour half is a BC1 block in its four-colour order (c0 > c1, or
// c0 == c1 with every index 0): such a texture is the same picture, texel for texel, as BC1 at 4 bits a texel. The
// colour halves are what the cache keeps for it, in "<key>.bc1" instead of "<key>.bc3" (tex_cache_write, and the setup's
// tex_prepare.c), and a texture whose storage hashes to a key with a .bc1 file gets a BC1 image (entry_of, bc1_choose):
// the cache's names are read once when the cache is first asked, so the choice costs a table lookup and the storage
// hash adopt_released takes anyway. A BC1 image is only ever filled from its file: the decoder writes BC3, so a BC1
// entry whose file cannot be read, whose storage the game writes again or into which it writes texels is made again
// (bc1_redo: destroyed at the next submission's start, the lookup after that chooses again).
// REKARINA_TEX_BC1=0: no BC1 image and no .bc1 file written (the files there are left alone).
static int bc1_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_BC1", "1");
    return on;
}
static uint64_t *bc1_tab;   // open addressing, 0 = empty; filled before the first texture is made, read-only after
static uint32_t bc1_mask, bc1_n;

static uint64_t bc1_key(uint64_t hash, unsigned format, unsigned w, unsigned h, unsigned d, unsigned levels, unsigned layers) {
    uint64_t k = hash ^ (uint64_t)format * 0x9E3779B97F4A7C15ull ^
                 ((uint64_t)w << 42 | (uint64_t)h << 21 | (uint64_t)d) * 0xC2B2AE3D27D4EB4Full ^
                 ((uint64_t)levels << 32 | layers) * 0x165667B19E3779F9ull;
    return k ? k : 1;
}
static void bc1_insert(uint64_t k) {
    if (bc1_n * 2 >= bc1_mask) {   // grow at half full
        uint32_t cap = bc1_mask ? (bc1_mask + 1) * 2 : 8192;
        uint64_t *t = calloc(cap, sizeof *t), *old = bc1_tab;
        if (!t) return;
        uint32_t om = bc1_mask;
        bc1_tab = t, bc1_mask = cap - 1, bc1_n = 0;
        for (uint32_t i = 0; old && i <= om; i++)
            if (old[i]) bc1_insert(old[i]);
        free(old);
    }
    uint32_t i = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 32) & bc1_mask;
    while (bc1_tab[i] && bc1_tab[i] != k) i = (i + 1) & bc1_mask;
    if (!bc1_tab[i]) bc1_tab[i] = k, bc1_n++;
}
static int bc1_has(uint64_t k) {
    if (!bc1_n) return 0;
    uint32_t i = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 32) & bc1_mask;
    for (; bc1_tab[i]; i = (i + 1) & bc1_mask)
        if (bc1_tab[i] == k) return 1;
    return 0;
}
static int bc1_list_one(const char *name, void *ctx) {
    (void)ctx;
    unsigned long long h;
    unsigned f, w, hh, d, lv, ly;
    if (sscanf(name, "%16llx-%x-%ux%ux%u-%ux%u.bc1", &h, &f, &w, &hh, &d, &lv, &ly) == 7) bc1_insert(bc1_key(h, f, w, hh, d, lv, ly));
    return 1;
}

// 1 when every BC3 block of blocks (n bytes, 16 a block) has the opaque alpha half bc3_encode.comp writes; then the
// colour halves, BC1 blocks, are written to out (n / 2 bytes; out may be blocks itself).
int vk_texture_bc3_to_bc1(const uint8_t *blocks, uint64_t n, uint8_t *out) {
    if (!n || n % 16) return 0;
    for (uint64_t i = 0; i < n; i += 16) {
        uint64_t a;
        memcpy(&a, blocks + i, 8);
        if (a != 0xFFFFull) return 0;
    }
    for (uint64_t i = 0; i < n; i += 16) memmove(out + i / 2, blocks + i + 8, 8);
    return 1;
}

static int is_bc1(const Tex *e) {
    return e->f.astc_bc && (e->f.vk == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || e->f.vk == VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
}

static void tex_cache_scan(void *arg) {
    TexCacheScan s = {.dir = arg, .limit = tc_limit};
    plat_file_list(s.dir, ".bc3", tex_cache_count, &s);
    plat_file_list(s.dir, ".bc1", tex_cache_count, &s);
    __atomic_store_n(&tc_disk, s.total, __ATOMIC_RELEASE);
    rt_log("vk: texture cache: %d textures, %.2f GiB of at most %.2f%s", s.files, s.total / 1073741824.0,
           tc_limit / 1073741824.0, s.deleted ? "; files over the limit deleted" : "");
    // The default leaves the file cache what the machine has beyond the game, up to the whole folder: the game's
    // working set is 15-17.5 GiB in play (2026-11-16), so with 16 GiB of memory files read ahead could only push the
    // game's own pages out, and 32 GiB hold the whole prepared cache beside it.
    const char *wm = cfg_env("REKARINA_TEX_CACHE_WARM_MB");
    float ram_used, ram_total, ram_proc;
    plat_mem_info(&ram_used, &ram_total, &ram_proc);
    const int reserve_mb = 18432;
    int folder_mb = (int)(s.total >> 20) + 1;
    int warm_mb = ram_total < 0 ? 8192 : ram_total < reserve_mb + 512 ? 0 : (int)ram_total - reserve_mb;
    if (warm_mb > folder_mb) warm_mb = folder_mb;
    TexCacheWarm w = {.dir = s.dir, .limit = (int64_t)(wm && *wm ? atoi(wm) : warm_mb) << 20, .t0 = plat_now_ns()};
    if (w.limit <= 0 || !s.files || !(w.buf = malloc(1u << 20))) return;
    plat_thread_priority(PLAT_PRIO_LOWEST);
    // The read waits for the first scene to be shown (at most 45 s) and runs as background work (low I/O priority on
    // Windows): at the start it opened its 14,646 files beside the replay's and the texture stream's own reads of the
    // title's files, 52-65 s of it on a cold file cache (the virus scanner looks at each file once), and every file of
    // the title waited behind it (2026-11-17). REKARINA_UPFRONT=0: at once, as before.
    int vk_upfront(void), vk_arrive_first_shown(void);
    if (vk_upfront()) {
        uint64_t t0 = plat_now_ns();
        while (!vk_arrive_first_shown() && plat_now_ns() - t0 < 45000000000ull) plat_sleep_ms(100);
#ifdef _WIN32
        plat_thread_priority(0x00010000);   // THREAD_MODE_BACKGROUND_BEGIN (plat_thread_priority passes it through)
#endif
        w.t0 = plat_now_ns();
        w.slow = -1;   // not judged: at background priority it holds up none of the game's reads, a hard disk's either
    }
    plat_file_list(s.dir, ".bc3", tex_cache_warm_one, &w);
    if (w.done < w.limit && w.slow <= 0) plat_file_list(s.dir, ".bc1", tex_cache_warm_one, &w);
    free(w.buf);
    rt_log("vk: texture cache: %d files, %.2f GiB, read into the system's file cache in %.1f s%s", w.files,
           (double)w.done / 1073741824.0, (double)(plat_now_ns() - w.t0) / 1e9,
           w.slow > 0 ? " and left there: the disk gave under 150 MiB a second" : "");
}

static const char *tex_cache_dir(void) {
    static char dir[512];
    static int state;   // 0 not asked, 1 on, -1 off
    if (!state) {
        const char *d = cfg_env("REKARINA_TEX_CACHE_DIR"), *mb = cfg_get("REKARINA_TEX_CACHE_MB");
        state = cfg_bool("REKARINA_TEX_CACHE") ? 1 : -1;
        // cfg_get gives no table default: unset, the limit was 6144 MiB whatever the table said, and a folder that had
        // reached it took no new texture (2026-11-13)
        tc_limit = (int64_t)(mb && atoi(mb) >= 256 ? atoi(mb) : 14336) << 20;
        if (state > 0) {
            if (d && *d) snprintf(dir, sizeof dir, "%s", d);
            else {
                char exe[400] = "";
                plat_exe_path(exe, sizeof exe);
                char *cut = strrchr(exe, '/'), *cut2 = strrchr(exe, 92);
                if (cut2 > cut) cut = cut2;
                if (cut) *cut = 0;
                snprintf(dir, sizeof dir, "%s/tex-cache", exe);
            }
            plat_file_mkdir_p(dir);
            uint64_t t0 = plat_now_ns();
            if (bc1_on()) plat_file_list(dir, ".bc1", bc1_list_one, NULL);   // before any texture is made: read-only after
            rt_log("vk: texture cache on: BC3 blocks of ASTC textures in %s (tex_cache); %u opaque ones kept as BC1 (%s, "
                   "listed in %.1f ms)", dir, bc1_n, bc1_on() ? "REKARINA_TEX_BC1=1" : "off", (plat_now_ns() - t0) / 1e6);
            if (!plat_thread_spawn(tex_cache_scan, dir, "texture cache", 0)) __atomic_store_n(&tc_disk, 0, __ATOMIC_RELEASE);
        }
    }
    return state > 0 ? dir : NULL;
}

static void tex_cache_path(char *out, size_t n, const Tex *e, uint64_t hash, uint32_t levels, uint32_t layers) {
    const nvn_texdesc *d = &e->t.d;
    snprintf(out, n, "%s/%016llx-%x-%ux%ux%u-%ux%u.%s", tex_cache_dir(), (unsigned long long)hash, (unsigned)d->format,
             d->width, d->height, d->depth, levels, layers, is_bc1(e) ? "bc1" : "bc3");
}

// The reaper thread, after the GPU wrote the blocks: to a file of its own name first, then renamed, so a reader never
// sees half of one. Blocks that are all opaque go to the .bc1 name as their colour halves (the BC1 section above).
static void tex_cache_write(void *arg) {
    TexCacheJob *j = arg;
    VkCtx *c = vk_ctx();
    if (j->keep && c) {
        vmaInvalidateAllocation(c->vma, j->alloc, 0, VK_WHOLE_SIZE);
        uint64_t size = j->size;
        size_t pl = strlen(j->path);
        if (bc1_on() && pl > 4 && !strcmp(j->path + pl - 4, ".bc3") && vk_texture_bc3_to_bc1(j->map, size, j->map)) {
            size /= 2;
            j->path[pl - 1] = '1';
            plat_inc64(&bc1_written);
        }
        char tmp[640];
        snprintf(tmp, sizeof tmp, "%s.%u.tmp", j->path, (unsigned)plat_thread_id());
        FILE *f = fopen(tmp, "wb");
        int ok = f && fwrite(j->map, 1, (size_t)size, f) == size;
        if (f) ok = fclose(f) == 0 && ok;
        if (ok && rename(tmp, j->path) == 0) plat_inc64(&tc_saved), plat_add64(&tc_disk, (int64_t)size);
        else remove(tmp), plat_inc64(&tc_failed);   // (a texture with the same bytes got there first: rename does not replace)
    }
    if (c) vmaDestroyBuffer(c->vma, j->buf, j->alloc);
    free(j);
}

// 1: the image was filled from the cache's file (one copy in the upload batch).
// Back under the texture lock after a fill. 0 when the entry went or was uploaded by another thread meanwhile (the
// caller drops what it filled). A destroyed entry's record is freed once the GPU is past the submission that was being
// recorded (destroy_tex), which a file read can outlast, so e is not read before the table says its key still names
// it; the image is compared too, for an entry made again at the same address in the same record.
static int fill_leave(const TexKey *k, const Tex *e, VkImage image, int was_uploaded) {
    lock();
    plat_mutex_unlock(&fill_lock);
    return nvn_hm_get(&by_key, k) == e && !e->pub.gone && e->pub.image == image && e->uploaded == was_uploaded;
}

// *skip (the worker only): the entry went or was uploaded by the replay while the file was read; nothing was recorded.
// tex_prepare.c (--prepare-textures) keys and places its files as play does
uint64_t vk_texture_content_hash(const void *p, uint64_t n) { return content_hash((const uint8_t *)p, n); }
const char *vk_texture_cache_dir(void) { return tex_cache_dir(); }

static int tex_cache_load(Tex *e, const char *path, const VkBufferImageCopy *rg, uint32_t n, uint64_t total, int *skip) {
    VkBuffer sb = VK_NULL_HANDLE;
    VkCommandBuffer cb;
    int ok = 0;
    if (up_worker) {
        int was = e->uploaded;
        TexKey key = e->key;
        VkImage image = e->pub.image;
        VmaAllocation sa = NULL;
        fill_enter();
        FILE *f = fopen(path, "rb");
        if (f) {
            uint8_t *stg = vk_staging(total, &sb, &sa);
            ok = fread(stg, 1, (size_t)total, f) == total && fgetc(f) == EOF;
            fclose(f);
        }
        int alive = fill_leave(&key, e, image, was);
        if (f && !alive) vmaDestroyBuffer(vk_ctx()->vma, sb, sa);
        if (!alive) *skip = 1;
        if (!f || !alive) return 0;
        cb = vk_upload_begin();
        owned_staging_later(sb, sa);   // after the batch that copies from it is open: the deferral is stamped with it
    } else {
        FILE *f = fopen(path, "rb");
        if (!f) return 0;
        cb = vk_upload_begin();
        uint8_t *stg = vk_staging(total, &sb, NULL);
        ok = fread(stg, 1, (size_t)total, f) == total && fgetc(f) == EOF;
        fclose(f);
    }
    if (ok) {
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        plat_inc64(&tc_loaded);
        plat_add64(&tc_load_bytes, (int64_t)total);
    } else {
        remove(path);   // another size than this texture's blocks: not ours, or cut short
        plat_inc64(&tc_failed);
    }
    vk_upload_end();
    return ok;
}

// ---- the texture stream (REKARINA_TEX_STREAM; on by default since 2026-11-17, off with REKARINA_UPFRONT=0)
// The user's start (their save, the prepared cache) went from 151 / 75 / 35 frames over 25 / 50 / 100 ms in its first
// minute to 10-12 / 2-3 / 1-2 with the stream, its jobs for freshly loaded textures and the arrival hold (vk_arrive.c);
// on a cold file cache from 66 / 22 / 9 to 18 / 2 / 1 (docs/oot-notes.md, "The start and the arrivals, 2026-11-17").
// A texture's first use (the pool set's update or a draw's binding, on the replay thread) hashed its storage (up to
// 220 MiB; adopt_released and the texture cache key by the hash) and read its cache file from the disk, inside the
// frame that needed it: with the cache prepared at install the field had frames of 300 ms of "texture upload" and one
// of 2.35 s (2026-11-16), worse than decoding on a fast GPU. So when the game initializes a texture in CPU-written
// memory (nvnTextureInitialize, any thread: vk_texture_premake) it is queued here, and worker threads do that part
// ahead: the page generation first (vk_mem_changed), then the hash, then the cache file's bytes into staging memory of
// the job's own. The replay takes the results where it would compute them: adopt_released the hash, upload the file's
// bytes (one copy recorded). A page of the storage written after the generation was taken voids the hash, and a file
// whose name or size differs from the one upload computes is not taken, so a job can only save work, never change a
// texel. Not ready at the first use: a job a worker is running is waited for (TS_WAIT, at most
// REKARINA_TEX_STREAM_WAIT_MS, default 250, then the replay does the work as before); one still queued is taken back
// and done as before (TS_MISS, "not ready"). The pool update leaves a slot that shows the stand-in and whose job is in
// flight for a later update, at most TS_DEFER_FRAMES frames after the texture was initialized (vk_texpool.c
// resolve_slot; such slots wait for their turn under the update's budget today as well). The workers never touch an
// entry, an image or a command buffer. Their reads of the game's memory are made under ts_read (shared) after
// vk_mem_resolve said the storage lies in a live pool; vk_texture_range_gone takes it exclusively once the pool has
// left the index, so no read outlasts the memory. Jobs are keyed by the storage (address and size): the hash is the
// storage's whatever descriptor names it, and the file's name carries the descriptor.
#define TS_N 16384             // jobs kept (a power of two)
#define TS_PROBE 64
#define TS_DEFER_FRAMES 30
#define TS_KEEP_FRAMES 300     // a result nobody took is dropped after this many frames
#define TS_REASK_FRAMES 120    // a texture initialized again at the same storage within this many frames is not queued again
enum { TS_FREE, TS_QUEUED, TS_BUSY, TS_READY, TS_DONE };
typedef struct {
    uint64_t addr, size;       // the storage
    tex_t t;                   // the texture as initialized
    int32_t state;
    uint8_t hashed;            // TS_READY: cprint and chash are the storage's as of page generation gen
    uint8_t staged;            // TS_READY: buf holds the bytes upload stages (no cache file was read), nreg regions
    uint32_t nreg;
    uint32_t gen;
    uint8_t *edge;             // TS_READY: the storage's bytes in its first and last page as hashed (all of it when no
                               // page lies wholly inside), 0x2000 bytes: ts_valid
    uint64_t seq, ask_ns, ask_frame, cprint, chash;
    uint64_t prev_chash;       // asked again after it was done: the hash it had then (had_prev), to count repeats
    uint8_t had_prev;
    uint8_t nofile;            // TS_READY: the cache holds no file of the name in path (upload need not look)
    uint8_t unwanted;          // its texture was found made already while a worker ran it: keep no bytes
    uint8_t used;              // a result was taken (or the image adopted): not queued again while the storage lies in
    VmaAllocation pool;        // the same pool (pool: vk_mem_resolve's alloc when it was hashed)
    VkBuffer buf;              // TS_READY: the cache file's bytes (VK_NULL_HANDLE: none read)
    VmaAllocation alloc;
    uint64_t bytes;
    struct MadeAhead *ma;      // TS_READY: the image made for this storage ahead of its first use (REKARINA_TEX_MAKE_AHEAD)
    char path[600];            // the file buf holds
} TsJob;
static TsJob *ts;
static uint32_t *tsq;          // job indices in the order asked, with the seq they were asked under (ts_lock)
static uint64_t *tsq_seq;
static uint32_t tsq_head, tsq_tail;
static PlatMutex ts_lock = PLAT_MUTEX_INIT;
static PlatMutex ts_read = PLAT_MUTEX_INIT;
static PlatCond ts_cv;         // a job finished
static PlatEvent ts_ev;        // a job was queued
static volatile int32_t ts_started = 0;   // 0 not yet, 1 running, -1 could not start
static uint64_t ts_seq, ts_held, ts_cap, ts_wait_ns;
static int ts_live;            // jobs queued, running or ready
static struct {
    uint64_t asked, dup, full, hashed, gone, files, nofile, capped, taken_hash, taken_file, stale, waited, wait_ns,
             wait_max_ns, timeout, stolen, missing, deferred, expired, mismatch, adopted, staged, taken_staged, reask, same, resident, bytes_expired, adoptable, nofile_skip, moved, lead_ns,
             lead_n;
} TSS;

static int ts_on(void) {
    static int on = -1;
    if (on < 0) {
        int vk_upfront(void);
        on = rt_knob("TEX_STREAM", vk_upfront() ? "1" : "0") != 0;
    }
    return on;
}

static uint32_t ts_home(uint64_t addr, uint64_t size) {
    return (uint32_t)(((addr ^ size * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull) >> 50) & (TS_N - 1);
}

static TsJob *ts_find(uint64_t addr, uint64_t size) {   // ts_lock held
    for (uint32_t i = 0, h = ts_home(addr, size); i < TS_PROBE; i++, h = (h + 1) & (TS_N - 1)) {
        TsJob *j = &ts[h];
        if (j->state == TS_FREE) return NULL;
        if (j->addr == addr && j->size == size) return j;
    }
    return NULL;
}

// ---- images made ahead by the texture stream (REKARINA_TEX_MAKE_AHEAD; on unless 0: at the square, walking in, the
// frames over 25 ms from the arrival were 128 / 54 / 41 / 118 in an off / on / on / off series, 2026-11-18)
// While the hero moves the game streams in textures, up to ~40 in a frame, and the frame's threads made each one's
// image under the texture lock (the pool's update on the replay thread, or a draw's binding): image, memory, bind and
// views ~0.25 ms of a texture's ~0.35 (REKARINA_SECTIONS MADE_P, IC_*, VKALLOC_*; 12 ms in the field's bad frames,
// 2026-11-18), and the replay waited for the texture lock behind whoever made them. The texture stream already takes
// every texture the game registers in CPU-filled memory ahead of its first use, on the game's own registering call,
// 2 ms to seconds before the replay makes it (DEBUG-TEXLEAD; under a millisecond for ~5 %). So the stream's worker,
// once it has hashed the storage, also makes the image create_tex would make (tex_plan; BC1 chosen from that hash as
// entry_of would), its memory by create_tex's rule (a piece of a block, else memory of its own), the bind and the
// views its first uses ask for (vk_exec_views_ahead), with NO texture lock held and no table touched. The record hangs
// on the stream's job (ts_lock). create_tex takes it where it would make the image (ma_take: the same storage and the
// same create info), zeroes a piece of a block itself (zero_suballoc records into the upload batch and stamps its
// buffer with the submission being recorded: it belongs to the taker) and goes on as before; one whose create info
// differs, or nobody took, is destroyed. A texture whose record is not there makes its image as before: the worker
// only ever takes work away from the frame.
// What a worker reads without the texture lock, and why it cannot move or go under it: the job's own copy of the
// texture object (tex_t, copied under ts_lock when the job is taken), the device and its format properties (fixed),
// settings read once, bc1_tab (filled before the first texture is made, read-only after), bc1_failed / bc1_nfailed
// (a fixed array of 64 words, read racily: a wrong guess only makes an image nobody takes), texq_mode and room_short
// (single words, read racily: a stale value only makes or skips one image). It writes nothing the texture tables hold.
// Freeing: every record leaves through ma_trash (a list under ts_lock) and is destroyed by ma_drain OUTSIDE ts_lock
// (vk_defer_image takes cmd_lock, which the upload batch holds while it calls into the stream), through the same
// deferred path as any image; its views were never handed out and go at once (vk_exec_views_ahead_drop).
typedef struct MadeAhead {
    VkImage img;
    VmaAllocation alloc;
    uint8_t sub;               // alloc is a piece of a block (zeroed by the taker); else memory of its own
    VkImageCreateInfo ci;
    VkViewsAhead va;
    uint64_t bytes, frame;     // the memory's size (counted in ma_held), the frame it was made in
    uint64_t addr;             // DEBUG: the storage it was made for
    struct MadeAhead *next;    // in ma_trash
} MadeAhead;
static MadeAhead *ma_trash;    // ts_lock
static uint64_t ma_held, ma_cap;   // ts_lock: bytes of records made and not taken or trashed; the cap
static int ma_live;            // ts_lock: records hanging on jobs
static struct {
    uint64_t made, sub, own, failed, capped, taken, mismatch, none, waited, wait_ns, dropped, expired, make_ns;
    uint64_t no_job, queued, done, no_rec;   // "none" by the job's state at first use
} MAS;
static volatile uint64_t ma_short;   // jobs whose image was not made while video memory was short (any thread)
static PlatMutex ma_drv_lock = PLAT_MUTEX_INIT;   // ma_make's driver calls, one worker at a time (taken with no other lock)                         // ts_lock (made / failed / make_ns: added under it after the worker's making)
static void ma_trash_put(MadeAhead *m, int counted) {   // ts_lock held
    if (!m) return;
    if (counted) ma_held -= m->bytes, ma_live--;
    m->next = ma_trash, ma_trash = m;
}

static void ts_drop(TsJob *j) {   // ts_lock held: the job is done, its staging memory goes
    if (j->ma) ma_trash_put(j->ma, 1), j->ma = NULL, MAS.dropped++;   // an image made ahead that nobody took
    if (j->buf) vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc), ts_held -= j->bytes;
    j->buf = VK_NULL_HANDLE, j->alloc = NULL, j->bytes = 0, j->path[0] = 0, j->staged = 0, j->nofile = 0;
    free(j->edge);
    j->edge = NULL;
    if (j->state == TS_QUEUED || j->state == TS_BUSY || j->state == TS_READY) ts_live--;
    j->state = TS_DONE;
}

// The storage's bytes in its first and last page, which neighbours share (all of it when no page lies wholly inside
// it): taken before the hash, so a write there after it shows as a difference (ts_valid). NULL when out of memory.
static uint8_t *ts_edge_take(uint64_t addr, uint64_t size) {
    uint64_t lo = (addr + 0xfff) & ~0xfffull, hi = (addr + size) & ~0xfffull;
    uint8_t *e = malloc(0x2000);
    if (!e) return NULL;
    if (hi <= lo) memcpy(e, (const void *)(uintptr_t)addr, size < 0x2000 ? size : 0x2000);
    else memcpy(e, (const void *)(uintptr_t)addr, lo - addr), memcpy(e + 0x1000, (const void *)(uintptr_t)hi, addr + size - hi);
    return e;
}

// ts_lock held, the storage live (its texture is being made): 1 when j's results still describe the storage: no page
// wholly inside it was written since j->gen and the bytes it shares with neighbours in its first and last page are as
// they were hashed. Page writes alone threw away a third of the field's results (2026-11-16): the game loads the next
// texture beside one, into the same page. *now: the generation of a vk_mem_changed over all of the storage (its write
// bits reset, as before a read).
static int ts_valid(const TsJob *j, uint32_t *now) {
    uint64_t addr = j->addr, size = j->size, lo = (addr + 0xfff) & ~0xfffull, hi = (addr + size) & ~0xfffull;
    vk_mem_changed(addr, size, j->gen, now);
    if (!j->edge) return 0;
    if (hi <= lo) return size <= 0x2000 && !memcmp(j->edge, (const void *)(uintptr_t)addr, size);
    uint32_t x;
    if (vk_mem_changed(lo, hi - lo, j->gen, &x) != 0) return 0;
    return !memcmp(j->edge, (const void *)(uintptr_t)addr, lo - addr) &&
           !memcmp(j->edge + 0x1000, (const void *)(uintptr_t)hi, addr + size - hi);
}

// ts_lock held, a worker about to hold `need` more bytes of staging: when that passes the cap, the bytes of the oldest
// results nobody took yet go first (their hash stays). A result asked at a registration is used within a frame or a
// few if at all, so the newest are worth more than the oldest. 1: there is room now.
static int ts_room(uint64_t need) {
    while (ts_held + need > ts_cap) {
        TsJob *old = NULL;
        for (uint32_t i = 0; i < TS_N; i++)
            if (ts[i].state == TS_READY && ts[i].buf && (!old || ts[i].seq < old->seq)) old = &ts[i];
        if (!old) return 0;
        vmaDestroyBuffer(vk_ctx()->vma, old->buf, old->alloc), ts_held -= old->bytes;
        old->buf = VK_NULL_HANDLE, old->alloc = NULL, old->bytes = 0, old->path[0] = 0, old->staged = 0, old->nofile = 0;
        TSS.bytes_expired++;
    }
    return 1;
}

// The levels and layers of the image create_tex makes for t (upload's regions run over them).
static void ts_shape(const tex_t *t, uint32_t *levels_out, uint32_t *layers_out) {
    const nvn_texdesc *d = &t->d;
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    int is3d = d->target == NVN_TEXTURE_TARGET_3D;
    int one_d = d->target == NVN_TEXTURE_TARGET_1D || d->target == NVN_TEXTURE_TARGET_1D_ARRAY;
    uint32_t w = umax(d->width, 1), h = one_d ? 1 : umax(d->height, 1), depth = is3d ? umax(d->depth, 1) : 1, layers = 1;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D_ARRAY: case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY:
    case NVN_TEXTURE_TARGET_CUBEMAP: case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = L.layers; break;
    default: break;
    }
    uint32_t levels = umax(d->levels, 1), most = 1;
    for (uint32_t s = umax(umax(w, h), depth); s > 1; s >>= 1) most++;
    if (levels > most) levels = most;
    if (nvn_tex_is_linear(d)) levels = 1, layers = 1;
    *levels_out = levels, *layers_out = layers;
}

// The cache file upload would read for t's first upload when its storage hashes to hash, and its size; 0 when t would
// not go through the cache (create_tex's levels and layers, upload's regions).
static int ts_cache_file(const tex_t *t, uint64_t hash, char *path, size_t n, uint64_t *bytes) {
    const nvn_texdesc *d = &t->d;
    TexVkFormat f;
    const char *dir = tex_cache_dir();
    if (!dir || texq_mode > 0 || d->samples > 1 || !tex_vk_format(d->format, &f) || !f.astc_bw || !f.astc_bc ||
        !vk_astc_gpu() || vk_astc_mode() == VK_ASTC_CHECK)
        return 0;
    uint32_t levels, layers;
    ts_shape(t, &levels, &layers);
    int is3d = d->target == NVN_TEXTURE_TARGET_3D;
    int one_d = d->target == NVN_TEXTURE_TARGET_1D || d->target == NVN_TEXTURE_TARGET_1D_ARRAY;
    uint64_t total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        total += (uint64_t)layers * vk_astc_bc_bytes(mip(d->width, lv), one_d ? 1 : mip(d->height, lv), is3d ? mip(d->depth, lv) : 1);
    int bc1 = bc1_has(bc1_key(hash, (unsigned)d->format, d->width, d->height, d->depth, levels, layers));   // bc1_choose's
    snprintf(path, n, "%s/%016llx-%x-%ux%ux%u-%ux%u.%s", dir, (unsigned long long)hash, (unsigned)d->format, d->width,
             d->height, d->depth, levels, layers, bc1 ? "bc1" : "bc3");
    *bytes = bc1 ? total / 2 : total;
    return 1;
}

// entry_of, before create_tex: 1 when t's image is to be BC1, its storage hashing to a key the cache holds as .bc1. The
// hash is taken as adopt_released takes it (the stream's, else the page generation reset and the storage hashed) and
// left in bc1_pre for it, so a texture is not hashed twice.
static int bc1_choose(const tex_t *t) {
    bc1_pre.valid = 0;
    if (!bc1_n || !t->mem || t->L.size < 4096 || t->d.samples > 1 || !tex_mirrored(t) || !vk_astc_gpu() ||
        vk_astc_mode() == VK_ASTC_CHECK || !tex_cache_dir())
        return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || !f.astc_bc || !f.astc_bw) return 0;
    uint64_t cp = 0, ch = 0, src;
    uint32_t gen = 0;
    // "moved storage" (adopt_released): bytes the game copied here from a texture whose entry or image is still kept
    // take that one's hash and image, with no hash of their own; the same format keeps the image adoptable
    if (moved_from((uintptr_t)t->mem, t->L.size, &src)) {
        TexKey sk;
        make_key(t, &sk);
        sk.addr = src;
        Tex *r = nvn_hm_get(&by_key, &sk);
        uint64_t mp = content_print((const uint8_t *)(uintptr_t)t->mem, t->L.size);
        if (r && r->released && r->has_chash && r->cprint == mp && r->t.L.size == t->L.size) return is_bc1(r);
        for (int i = 0; i < nparked; i++)
            if (parked[i].addr == src && parked[i].cprint == mp && parked[i].size == t->L.size &&
                !memcmp(&parked[i].d, &sk.d, sizeof sk.d))
                return parked[i].format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || parked[i].format == VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    }
    if (up_worker && up_pre.valid && up_pre.addr == (uintptr_t)t->mem && up_pre.size == t->L.size) {
        return 0;   // the upload worker's own hash goes to adopt_released as before; its textures stay BC3 this time
    } else if (!up_worker && ts_take_hash(t, &cp, &ch, &gen)) {
    } else {
        vk_mem_changed((uintptr_t)t->mem, t->L.size, 0, &gen);   // the write bits reset before the storage is read
        uint64_t hsec = sec_begin();
        cp = content_print((const uint8_t *)(uintptr_t)t->mem, t->L.size);
        ch = content_hash((const uint8_t *)(uintptr_t)t->mem, t->L.size);
        sec_end(SEC_TS_HASH, hsec);
    }
    bc1_pre.addr = (uintptr_t)t->mem, bc1_pre.size = t->L.size, bc1_pre.cprint = cp, bc1_pre.chash = ch, bc1_pre.gen = gen;
    bc1_pre.valid = 1;
    uint32_t levels, layers;
    ts_shape(t, &levels, &layers);
    uint64_t k = bc1_key(ch, (unsigned)t->d.format, t->d.width, t->d.height, t->d.depth, levels, layers);
    if (!bc1_has(k)) return 0;
    for (uint32_t i = 0; i < bc1_nfailed && i < 64; i++)
        if (bc1_failed[i] == k) return 0;
    return 1;
}

// A BC1 entry whose content cannot come from its cache file: destroyed at the next submission's start (the lookups after
// that make it again, BC3 when the key failed). why: for the log. Texture lock held.
static void bc1_redo(Tex *e, const char *why) {
    static int32_t n_said;
    if (!e->bc1_redo) e->bc1_redo = 1, bc1_pending++;
    if (n_said++ < 16)
        rt_log("vk: BC1 texture %#llx (%ux%u fmt %#x) made again: %s", (unsigned long long)e->pub.addr, e->pub.width,
               e->pub.height, e->t.d.format, why);
}
static void bc1_fail_key(const Tex *e) {   // its file did not fill it: the key is not taken as BC1 again
    if (!e->has_chash) return;
    uint32_t levels = nvn_tex_is_linear(&e->t.d) ? 1 : e->pub.levels, layers = nvn_tex_is_linear(&e->t.d) ? 1 : e->pub.layers;
    bc1_failed[bc1_nfailed++ % 64] = bc1_key(e->chash, (unsigned)e->t.d.format, e->t.d.width, e->t.d.height, e->t.d.depth,
                                             levels, layers);
}
static void bc1_redo_run(void) {   // vk_texture_new_submit, texture lock held
    if (!bc1_pending) return;
    bc1_pending = 0;
    Tex *v[64];
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {   // collected first: destroy_tex changes the table
        Tex *e = by_key.vals[i];
        if (!e || !e->bc1_redo) continue;
        if (n == 64) { bc1_pending = 1; break; }   // the rest at the next submission
        v[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_tex(v[i]), bc1_redone++;
}

// The bytes upload stages for t's first upload when no cache file fills it: every region deswizzled at upload's offsets
// (an ASTC texture's raw blocks, which the GPU decodes; RGB expanded to RGBA). 1 with *buf, *alloc, *bytes, *nreg set; 0
// where upload would not stage them so (the CPU's ASTC decode, a region without a CPU layout, the staging cap reached,
// the storage gone). Reads the storage under ts_read, after the caller took the page generation.
static int ts_stage(const tex_t *t, VkBuffer *buf, VmaAllocation *alloc, uint64_t *bytes, uint32_t *nreg) {
    const nvn_texdesc *d = &t->d;
    TexVkFormat f;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || d->samples > 1 || d->target == NVN_TEXTURE_TARGET_BUFFER || !tex_vk_format(d->format, &f)) return 0;
    int astc = f.astc_bw != 0, rgb = f.expand_rgb, agpu = astc && vk_astc_gpu();
    if ((astc && !agpu) || (agpu && vk_astc_mode() == VK_ASTC_CHECK)) return 0;
    uint32_t levels, layers;
    ts_shape(t, &levels, &layers);
    int is3d = d->target == NVN_TEXTURE_TARGET_3D;
    int one_d = d->target == NVN_TEXTURE_TARGET_1D || d->target == NVN_TEXTURE_TARGET_1D_ARRAY;
    uint64_t total = 0;
    for (uint32_t lv = 0; lv < levels; lv++) {
        uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
        uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)(w + bw - 1) / bw * bpb * ((h + bh - 1) / bh) * sl;
        uint64_t out = agpu ? raw : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
        total += layers * ((out + 15) & ~15ull);
    }
    plat_mutex_lock(&ts_lock);
    int room = ts_room(total);
    if (room) ts_held += total;   // reserved before the copy
    else TSS.capped++;
    plat_mutex_unlock(&ts_lock);
    if (!room) return 0;
    uint8_t *stg = vk_staging(total, buf, alloc);
    VkMemSpan m;
    int ok = 1;
    uint64_t off = 0;
    plat_mutex_lock_shared(&ts_read);
    ok = vk_mem_resolve((uintptr_t)t->mem, t->L.size, &m);
    for (uint32_t lv = 0; lv < levels && ok; lv++)
        for (uint32_t ly = 0; ly < layers && ok; ly++) {
            uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            size_t raw = (size_t)rb * rows * sl;
            uint64_t texels = (uint64_t)w * h * sl, out = agpu ? raw : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            uint8_t *tmp = rgb ? malloc(raw) : NULL;
            if ((rgb && !tmp) || nvn_deswizzle(d, lv, ly, t->mem, tmp ? tmp : stg + off, rb)) ok = 0;
            else if (rgb) expand_rgb(d->format, tmp, stg + off, (size_t)w * h * sl);
            free(tmp);
            off += (out + 15) & ~15ull;
        }
    plat_mutex_unlock_shared(&ts_read);
    if (!ok) {
        vmaDestroyBuffer(vk_ctx()->vma, *buf, *alloc);
        *buf = VK_NULL_HANDLE, *alloc = NULL;
        plat_mutex_lock(&ts_lock);
        ts_held -= total;
        plat_mutex_unlock(&ts_lock);
        return 0;
    }
    *bytes = total, *nreg = levels * layers;
    return 1;
}

#define MA_KEEP_FRAMES 30      // a record nobody took within this many frames of its making is destroyed (ma_expire)
static int ma_on(void) {
    static int on = -1;
    if (on < 0) {
        on = rt_knob("TEX_MAKE_AHEAD", "1") != 0;
        const char *mb = cfg_get("REKARINA_TEX_MAKE_AHEAD_MB");   // the records' memory held at most (default 512 MiB)
        ma_cap = (uint64_t)(mb && atoi(mb) > 0 ? atoi(mb) : 512) << 20;
        if (on) rt_log("vk: texture stream: its workers also make the images of new textures ahead of their first use "
                       "(REKARINA_TEX_MAKE_AHEAD; held at most %llu MiB, %d frames)", (unsigned long long)(ma_cap >> 20),
                       MA_KEEP_FRAMES);
    }
    return on;
}

// bc1_choose's answer for t when its storage hashes to chash (no lock: bc1_tab is read-only once filled; a guess that
// differs from entry_of's makes a record whose create info differs, which is not taken).
static int ma_bc1(const tex_t *t, uint64_t chash) {
    if (!bc1_n || t->L.size < 4096 || t->d.samples > 1 || !vk_astc_gpu() || vk_astc_mode() == VK_ASTC_CHECK || !tex_cache_dir())
        return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || !f.astc_bc || !f.astc_bw) return 0;
    uint32_t levels, layers;
    ts_shape(t, &levels, &layers);
    uint64_t k = bc1_key(chash, (unsigned)t->d.format, t->d.width, t->d.height, t->d.depth, levels, layers);
    if (!bc1_has(k)) return 0;
    uint32_t nf = __atomic_load_n(&bc1_nfailed, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < nf && i < 64; i++)
        if (__atomic_load_n(&bc1_failed[i], __ATOMIC_RELAXED) == k) return 0;
    return 1;
}

// A texture stream worker, no lock held: the image create_tex would make for t (whose storage hashed to chash), with
// its memory, bind and views. NULL: none (not such a texture, over the cap, or the device refused).
static MadeAhead *ma_make(const tex_t *t, uint64_t chash) {
    VkCtx *c = vk_ctx();
    if (!ma_on() || !c || !t->mem || !tex_mirrored(t) || t->d.samples > 1 || !suballoc_on() || tex_gen_pools_on() ||
        __atomic_load_n(&texq_mode, __ATOMIC_RELAXED) != 0)
        return NULL;
    // while video memory is short the pool's lookups make no image of most textures (pool_holds_back) and eviction
    // runs: an image made ahead then would only add to it
    if (__atomic_load_n(&room_short, __ATOMIC_RELAXED)) {
        __atomic_fetch_add(&ma_short, 1, __ATOMIC_RELAXED);
        return NULL;
    }
    uint64_t t0 = plat_now_ns();
    TexPlan P;
    if (!tex_plan(t, ma_bc1(t, chash), &P) || P.samples != 1) return NULL;
    VkMemoryDedicatedRequirements dr = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 mr2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dr};
    VkDeviceImageMemoryRequirements q = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &P.ci};
    vkGetDeviceImageMemoryRequirements(c->device, &q, &mr2);
    uint64_t size = mr2.memoryRequirements.size;
    if (!size) return NULL;
    plat_mutex_lock(&ts_lock);
    int room = ma_held + size <= ma_cap;
    if (room) ma_held += size, ma_live++;   // reserved; given back by ma_trash_put or ma_take
    else MAS.capped++;
    plat_mutex_unlock(&ts_lock);
    if (!room) return NULL;
    // create_tex's rule: a piece of a block unless the driver requires memory of its own or the size rule asks for it
    VkDeviceSize sa_max = tex_suballoc_max();
    int own_memory = size > sa_max;
    int sub = !dr.requiresDedicatedAllocation && size <= TEX_SUBALLOC_MAX && !own_memory;
    VkImage img = VK_NULL_HANDLE;
    VmaAllocation al = NULL;
    // One worker in the driver at a time (REKARINA_TEX_MAKE_AHEAD_ONE=0: all of them). NVIDIA's driver makes its kernel
    // calls for an image's memory under one object lock: with the stream's five workers making images side by side,
    // the replay's own binds of the textures nobody made ahead took 0.069 ms each against 0.013 without them, and one
    // took 57 ms (the field, 2026-11-18).
    static int one = -1;
    if (one < 0) one = rt_knob("TEX_MAKE_AHEAD_ONE", "1") != 0;
    if (one) plat_mutex_lock(&ma_drv_lock);
    vk_drv_bg_turn();   // queue work, and the frame's own image making, first (gfx_present.c "driver turns")
    if (sub) {
        if (vkCreateImage(c->device, &P.ci, NULL, &img) == VK_SUCCESS) {
            vk_live_check_born((uint64_t)(uintptr_t)img);   // DEBUG-LIVECHECK
            VkMemoryRequirements mr;
            vkGetImageMemoryRequirements(c->device, img, &mr);
            VmaAllocationCreateInfo ai = {.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                          .pool = mr.size <= sa_max ? tex_pool_for(c, mr.memoryTypeBits) : NULL};
            vk_drv_bg_turn();
            if (vmaAllocateMemory(c->vma, &mr, &ai, &al, NULL) != VK_SUCCESS) {
                al = NULL;
            } else {
                vk_drv_bg_turn();
                if (vmaBindImageMemory(c->vma, al, img) != VK_SUCCESS) vmaFreeMemory(c->vma, al), al = NULL;
            }
            if (!al) vkDestroyImage(c->device, img, NULL), img = VK_NULL_HANDLE;   // never seen by anyone
        }
    } else {
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      .flags = own_memory ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0};
        if (vmaCreateImage(c->vma, &P.ci, &ai, &img, &al, NULL) == VK_SUCCESS) vk_live_check_born((uint64_t)(uintptr_t)img);
        else img = VK_NULL_HANDLE, al = NULL;
    }
    MadeAhead *m = img ? calloc(1, sizeof *m) : NULL;
    if (img && !m) vmaDestroyImage(c->vma, img, al), img = VK_NULL_HANDLE;
    if (m) {
        m->img = img, m->alloc = al, m->sub = (uint8_t)sub, m->ci = P.ci, m->bytes = size, m->addr = (uintptr_t)t->mem;
        m->frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (vk_exec_views_ahead_on()) {   // the views its first uses ask for (the pool's two, the sampled one)
            VkTexImage tmp = {.image = img, .addr = (uintptr_t)t->mem};
            pub_fill(&tmp, &P.f, P.type, P.usage, P.w, P.h, P.depth, P.levels, P.layers, P.samples, &t->d, P.flags, t, 0,
                     P.block_views, P.fb_usage, P.scale);
            vk_drv_bg_turn();
            vk_exec_views_ahead(&tmp, &m->va);
        }
    }
    if (one) plat_mutex_unlock(&ma_drv_lock);
    plat_mutex_lock(&ts_lock);
    if (m) MAS.made++, MAS.sub += sub, MAS.own += !sub, MAS.make_ns += plat_now_ns() - t0;
    else MAS.failed++, ma_held -= size, ma_live--;
    plat_mutex_unlock(&ts_lock);
    return m;
}

// The records that left (ma_trash) are destroyed: no lock held, or the texture lock (as destroy_tex), never ts_lock.
static void ma_drain(void) {
    if (!__atomic_load_n(&ma_trash, __ATOMIC_RELAXED)) return;   // (a racy look; the list is taken under ts_lock)
    plat_mutex_lock(&ts_lock);
    MadeAhead *m = ma_trash;
    ma_trash = NULL;
    plat_mutex_unlock(&ts_lock);
    while (m) {
        MadeAhead *nx = m->next;
        vk_exec_views_ahead_drop(&m->va);
        vk_defer_image(m->img, m->alloc);   // as any image: once the GPU is past the submission being recorded
        free(m);
        m = nx;
    }
}

// vk_texture_new_submit, every 8 frames (texture lock held): records older than MA_KEEP_FRAMES go.
static void ma_expire(uint64_t frame) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    plat_mutex_lock(&ts_lock);
    for (uint32_t i = 0; ma_live && i < TS_N; i++)
        if (ts[i].ma && ts[i].ma->frame + MA_KEEP_FRAMES < frame) ma_trash_put(ts[i].ma, 1), ts[i].ma = NULL, MAS.expired++;
    plat_mutex_unlock(&ts_lock);
    ma_drain();
}

// create_tex, texture lock held: the image made ahead for t's storage from create info ci, its memory in *alloc, *sub
// when that is a piece of a block (the caller zeroes it); the views made with it go to pm_va_taken for create_tex's
// tail. Nothing is waited for: the worker hangs the record on its job as soon as it is made, before it reads the cache
// file, so a job still running has it already or the frame makes the image itself (a wait here was a frame's wait for
// the file read: 57 waits, 99 ms in 600 frames of the square's arrival, 2026-11-18). VK_NULL_HANDLE: none.
static VkImage ma_take(const tex_t *t, const VkImageCreateInfo *ci, VmaAllocation *alloc, int *sub) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || !t->mem || t->L.size < 4096) return VK_NULL_HANDLE;
    MadeAhead *m = NULL;
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (j && (j->state == TS_READY || j->state == TS_BUSY) && j->ma) {
        if (pm_ci_same(&j->ma->ci, ci) && j->ma->ci.tiling == ci->tiling) {
            m = j->ma, j->ma = NULL;
            ma_held -= m->bytes, ma_live--;
            MAS.taken++;
        } else {   // another create info (a BC1 guess that missed, another descriptor of the storage): not this one
            ma_trash_put(j->ma, 1), j->ma = NULL, MAS.mismatch++;
        }
    } else {
        MAS.none++;
        if (!j) MAS.no_job++;
        else if (j->state == TS_QUEUED) MAS.queued++;
        else if (j->state == TS_BUSY) MAS.waited++;   // running, its image not made yet
        else if (j->state == TS_READY) MAS.no_rec++;
        else MAS.done++;
    }
    plat_mutex_unlock(&ts_lock);
    if (!m) return VK_NULL_HANDLE;
    VkImage img = m->img;
    *alloc = m->alloc, *sub = m->sub;
    if (pm_va_taken.image) vk_exec_views_ahead_drop(&pm_va_taken);   // (one create_tex did not reach its end with them)
    pm_va_taken = m->va;   // create_tex's tail puts them into the view cache (vk_exec_views_adopt)
    free(m);
    return img;
}

static void ma_log(uint64_t frame) {   // every 600 frames while it is on (texture lock held)
    if (!ma_on() || __atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    plat_mutex_lock(&ts_lock);
    rt_log("vk: frame %llu: images made ahead %llu (pieces of blocks %llu, own memory %llu; %.3f ms each on the stream's "
           "threads), failed %llu, over the cap %llu, not made while video memory was short %llu; at first use taken %llu, "
           "other create info %llu, none %llu (no job %llu, job queued %llu, job done %llu, job without an image %llu; "
           "job running, its image not made yet %llu; %.1f ms); dropped with their job %llu, expired %llu; %d held, %.1f MiB",
           (unsigned long long)frame, (unsigned long long)MAS.made, (unsigned long long)MAS.sub, (unsigned long long)MAS.own,
           MAS.made ? MAS.make_ns / 1e6 / MAS.made : 0.0, (unsigned long long)MAS.failed, (unsigned long long)MAS.capped,
           (unsigned long long)__atomic_exchange_n(&ma_short, 0, __ATOMIC_RELAXED),
           (unsigned long long)MAS.taken, (unsigned long long)MAS.mismatch, (unsigned long long)MAS.none,
           (unsigned long long)MAS.no_job, (unsigned long long)MAS.queued, (unsigned long long)MAS.done,
           (unsigned long long)MAS.no_rec, (unsigned long long)MAS.waited, MAS.wait_ns / 1e6, (unsigned long long)MAS.dropped,
           (unsigned long long)MAS.expired, ma_live, ma_held / 1048576.0);
    memset(&MAS, 0, sizeof MAS);
    plat_mutex_unlock(&ts_lock);
}

static void ts_main(void *arg) {
    (void)arg;
    sec_thread_off = 1;
    for (;;) {
        plat_mutex_lock(&ts_lock);
        TsJob *j = NULL;
        uint64_t seq = 0;
        while (tsq_head != tsq_tail && !j) {
            uint32_t i = tsq[tsq_head % TS_N];
            seq = tsq_seq[tsq_head % TS_N];
            tsq_head++;
            if (ts[i].state == TS_QUEUED && ts[i].seq == seq) j = &ts[i];
        }
        if (!j) {
            plat_mutex_unlock(&ts_lock);
            plat_event_wait(&ts_ev);
            continue;
        }
        j->state = TS_BUSY;
        tex_t t = j->t;
        int more = tsq_head != tsq_tail;
        plat_mutex_unlock(&ts_lock);
        if (more) plat_event_set(&ts_ev);   // another worker takes the next one
        uint64_t addr = (uintptr_t)t.mem, size = t.L.size, cprint = 0, chash = 0, bytes = 0;
        uint32_t gen = 0;
        uint8_t *edge = NULL;
        int hashed = 0, file = 0, staged = 0;   // file: 1 read, -1 none of that name and size, -2 over the staging cap
        uint32_t nreg = 0;
        VkMemSpan m;
        plat_mutex_lock_shared(&ts_read);
        if (vk_mem_resolve(addr, size, &m)) {
            vk_mem_changed(addr, size, 0, &gen);   // before the read: a write after it carries a newer generation
            edge = ts_edge_take(addr, size);          // and the shared pages' bytes, for a write there after this
            cprint = content_print((const uint8_t *)(uintptr_t)addr, size);
            chash = content_hash((const uint8_t *)(uintptr_t)addr, size);
            hashed = 1;
        }
        plat_mutex_unlock_shared(&ts_read);
        VmaAllocation pool = hashed ? m.alloc : NULL;
        char path[600];
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = NULL;
        int adoptable = 0;
        if (hashed) {
            plat_mutex_lock(&ts_lock);
            adoptable = rh_has(chash);
            TSS.adoptable += adoptable;
            plat_mutex_unlock(&ts_lock);
        }
        // REKARINA_TEX_MAKE_AHEAD: its image, before the cache file's read (an adoptable one takes a released image)
        MadeAhead *made = hashed && !adoptable ? ma_make(&t, chash) : NULL;
        if (made) {   // hung on the job now: create_tex takes it from a job still reading its file (ma_take)
            plat_mutex_lock(&ts_lock);
            if (j->state == TS_BUSY && j->seq == seq && !j->unwanted) {
                if (j->ma) ma_trash_put(j->ma, 1);
                j->ma = made, made = NULL;
            }
            plat_mutex_unlock(&ts_lock);
        }
        if (hashed && !adoptable && ts_cache_file(&t, chash, path, sizeof path, &bytes)) {
            file = plat_file_size(path) == (int64_t)bytes ? 1 : -1;
            if (file > 0) {
                plat_mutex_lock(&ts_lock);
                if (!ts_room(bytes)) file = -2;
                else ts_held += bytes;   // reserved before the read
                plat_mutex_unlock(&ts_lock);
            }
            if (file > 0) {
                uint8_t *stg = vk_staging(bytes, &buf, &alloc);
                PlatFile *f = plat_file_open(path);
                uint64_t off = 0;
                while (f && off < bytes) {
                    uint64_t want = bytes - off < (8u << 20) ? bytes - off : (8u << 20);
                    int64_t got = plat_file_pread(f, stg + off, (uint32_t)want, off);
                    if (got <= 0) break;
                    off += (uint64_t)got;
                }
                if (f) plat_file_close(f);
                if (off != bytes) {
                    vmaDestroyBuffer(vk_ctx()->vma, buf, alloc);
                    buf = VK_NULL_HANDLE, alloc = NULL, file = -1;
                    plat_mutex_lock(&ts_lock);
                    ts_held -= bytes;
                    plat_mutex_unlock(&ts_lock);
                }
            }
        }
        if (hashed && !adoptable && file <= 0) staged = ts_stage(&t, &buf, &alloc, &bytes, &nreg);   // what upload would stage instead
        plat_mutex_lock(&ts_lock);
        if (j->state == TS_BUSY && j->seq == seq) {
            j->gen = gen, j->cprint = cprint, j->chash = chash, j->hashed = (uint8_t)hashed, j->pool = pool;
            free(j->edge);
            j->edge = edge, edge = NULL;
            if (j->unwanted && buf) {   // made meanwhile from a registration of the same storage: no bytes are needed
                vmaDestroyBuffer(vk_ctx()->vma, buf, alloc);
                ts_held -= bytes;
                buf = VK_NULL_HANDLE, alloc = NULL, staged = 0;
            }
            j->buf = buf, j->alloc = alloc, j->bytes = buf ? bytes : 0, j->staged = (uint8_t)staged, j->nreg = nreg;
            if (file) snprintf(j->path, sizeof j->path, "%s", path);   // the name looked for (read when buf && !staged)
            j->nofile = file == -1;
            if (j->unwanted && j->ma) ma_trash_put(j->ma, 1), j->ma = NULL, MAS.dropped++;   // its texture was made meanwhile
            j->state = TS_READY;
            if (!hashed) ts_drop(j), TSS.gone++;
            else TSS.hashed++, TSS.same += j->had_prev && j->prev_chash == chash;
            TSS.files += file > 0, TSS.nofile += file == -1, TSS.capped += file == -2, TSS.staged += staged;
        } else if (buf) {   // dropped meanwhile (its storage went)
            vmaDestroyBuffer(vk_ctx()->vma, buf, alloc);
            ts_held -= bytes;
        }
        if (made) ma_trash_put(made, 1), MAS.dropped++;   // the job went meanwhile, or its texture was made already
        plat_cond_broadcast(&ts_cv);
        plat_mutex_unlock(&ts_lock);
        free(edge);   // (not taken: the job was dropped meanwhile)
        ma_drain();
    }
}

static void ts_start(void) {
    plat_mutex_lock(&ts_lock);
    if (!ts_started) {
        ts = calloc(TS_N, sizeof *ts), tsq = calloc(TS_N, sizeof *tsq), tsq_seq = calloc(TS_N, sizeof *tsq_seq);
        plat_cond_init(&ts_cv);
        plat_event_init(&ts_ev, 0, 0);
        const char *e = cfg_get("REKARINA_TEX_STREAM_THREADS"), *mb = cfg_get("REKARINA_TEX_STREAM_MB"),
                   *wm = cfg_get("REKARINA_TEX_STREAM_WAIT_MS");
        // a third of the logical processors the process may use (an affinity mask counts, as for the other pools),
        // 2 to 6: a cache file's first open costs the scanner's look at it (2-5 ms on a cold file), which the workers
        // wait out side by side
        int auto_n = (int)plat_cpu_usable() / 3;
        auto_n = auto_n < 2 ? 2 : auto_n > 6 ? 6 : auto_n;
        int nt = e && atoi(e) > 0 ? atoi(e) : auto_n, started = 0;
        if (nt > 8) nt = 8;
        ts_cap = (uint64_t)(mb && atoi(mb) > 0 ? atoi(mb) : 512) << 20;
        ts_wait_ns = (uint64_t)(wm && atoi(wm) > 0 ? atoi(wm) : 250) * 1000000ull;
        tex_cache_dir();   // decided here, before a worker asks
        for (int i = 0; ts && tsq && tsq_seq && i < nt; i++) started += plat_thread_spawn(ts_main, NULL, "vk texture stream", 0) != 0;
        __atomic_store_n(&ts_started, started ? 1 : -1, __ATOMIC_RELEASE);
        rt_log("vk: texture stream: %d threads hash new textures' storage and read their cache files ahead of their first "
               "use (REKARINA_TEX_STREAM; staging held at most %llu MiB, a wait at most %llu ms)", started,
               (unsigned long long)(ts_cap >> 20), (unsigned long long)(ts_wait_ns / 1000000));
    }
    plat_mutex_unlock(&ts_lock);
}

// The game registers t in its texture pool (vk_backend.c, the game's thread, before the registration is queued for the
// replay): t's storage is hashed and its bytes read or staged ahead of the replay's first use. Asking at
// nvnTextureInitialize instead (REKARINA_TEX_STREAM_AT_INIT=1) gave far more lead but was mostly wasted: in the field
// and the Market three in four textures initialized were not used within 300 frames (25,604 asks, 5,587 hashes taken,
// 2026-11-16), and their bytes filled the staging cap.
// The game copied t's bytes from src (moved_from): 1 when adopt_released will find them there, a released entry of the
// same descriptor and size at src with its hash, or an image kept for src (park_image). Takes the texture lock (the game's
// registering thread, off its virtual core while it waits, as vk_texture_register).
static int moved_adoptable(const tex_t *t, uint64_t src) {
    TexKey sk;
    make_key(t, &sk);
    sk.addr = src;
    lock_off_core();
    Tex *r = nvn_hm_get(&by_key, &sk);
    int found = r && r->released && r->has_chash && r->t.L.size == t->L.size;
    for (int i = 0; i < nparked && !found; i++)
        found = parked[i].addr == src && parked[i].size == t->L.size && !memcmp(&parked[i].d, &sk.d, sizeof sk.d);
    unlock();
    return found;
}

static void ts_ask(const tex_t *t) {
    if (!t->mem || t->L.size < 4096 || !ts_on() || !tex_mirrored(t) || !vk_ctx()) return;
    if (!__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE)) ts_start();
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    uint64_t addr = (uintptr_t)t->mem, size = t->L.size, now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    // Storage the game filled by moving a texture there (its compaction: one large copy, then the registration at the
    // new address) needs no job: adopt_released takes the hash of the texture the bytes came from ("moved storage")
    // and its image with it. Those were six in ten of the field's registrations (17,021 in 30 s, 2026-11-16), each
    // hashed by a worker for nothing.
    // Only when that texture is kept: the game also loads a texture by copying its bytes from the file's buffer into
    // place, a large copy from no texture at all, and the start's and every arrival's textures came that way: none got
    // a job, and the replay hashed each and read its cache file itself (TS_READ with no miss counted, 2026-11-17).
    // REKARINA_UPFRONT=0: no job for any moved storage, as before.
    uint64_t src;
    int vk_upfront(void);
    if (moved_from(addr, size, &src) && (!vk_upfront() || moved_adoptable(t, src))) {
        plat_mutex_lock(&ts_lock);
        TSS.asked++, TSS.moved++;
        plat_mutex_unlock(&ts_lock);
        return;
    }
    int wake = 0;
    plat_mutex_lock(&ts_lock);
    TSS.asked++;
    TsJob *j = NULL, *reuse = NULL;
    for (uint32_t i = 0, h = ts_home(addr, size); i < TS_PROBE; i++, h = (h + 1) & (TS_N - 1)) {
        TsJob *q = &ts[h];
        if (q->state == TS_FREE) {
            if (!reuse) reuse = q;
            break;
        }
        if (q->addr == addr && q->size == size) {
            j = q;
            break;
        }
        if (!reuse && q->state == TS_DONE) reuse = q;
    }
    VkMemSpan m;
    if (j && j->state == TS_DONE && j->used && !(vk_mem_resolve(addr, size, &m) && m.alloc == j->pool)) j->used = 0;   // another pool now
    if (j && (j->state != TS_DONE || j->used || now - j->ask_frame < TS_REASK_FRAMES)) {
        TSS.dup++;
        if (j->state != TS_DONE) j->t = *t;   // (the same storage: the newest descriptor names the cache file)
        goto out;
    }
    if (j) {   // the same storage initialized again after its job was done: are its bytes the same? (counted)
        j->had_prev = j->hashed, j->prev_chash = j->chash;
        TSS.reask++;
    } else if (reuse) {
        reuse->had_prev = 0;
    }
    if (!j) j = reuse;
    if (!j || tsq_tail - tsq_head >= TS_N) {
        TSS.full++;
        goto out;
    }
    j->addr = addr, j->size = size, j->t = *t, j->hashed = 0, j->used = 0, j->unwanted = 0;
    j->seq = ++ts_seq, j->ask_ns = plat_now_ns(), j->ask_frame = now, j->state = TS_QUEUED;
    ts_live++;
    tsq[tsq_tail % TS_N] = (uint32_t)(j - ts), tsq_seq[tsq_tail % TS_N] = j->seq, tsq_tail++;
    wake = 1;
out:
    plat_mutex_unlock(&ts_lock);
    if (wake) plat_event_set(&ts_ev);
}

// Texture lock held (adopt_released, upload): the hash a worker took of t's storage, when it still describes the
// storage (ts_valid; *now: the generation to read the storage as of, its write bits reset). Waits for a job a worker
// is running; a job still queued is taken back. 0: no hash (the caller hashes, as before). The job stays for another
// descriptor of the same storage and for upload's bytes.
static int ts_take_hash(const tex_t *t, uint64_t *cprint, uint64_t *chash, uint32_t *now) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || !t->mem) return 0;
    int r = 0;
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (!j || j->state == TS_DONE) {
        TSS.missing++;
        goto out;
    }
    if (j->state == TS_QUEUED) {
        ts_drop(j);
        TSS.stolen++;
        if (sec_on) sec_count(SEC_TS_MISS, 1);
        goto out;
    }
    if (j->state == TS_BUSY) {
        uint64_t seq = j->seq, t0 = plat_now_ns(), sec0 = sec_begin(), el = 0;
        while (j->state == TS_BUSY && j->seq == seq && (el = plat_now_ns() - t0) < ts_wait_ns)
            plat_cond_wait_ns(&ts_cv, &ts_lock, ts_wait_ns - el);
        el = plat_now_ns() - t0;
        sec_end(SEC_TS_WAIT, sec0);
        TSS.waited++, TSS.wait_ns += el;
        if (el > TSS.wait_max_ns) TSS.wait_max_ns = el;
        if (j->seq != seq || j->state != TS_READY) {
            if (j->seq == seq && j->state == TS_BUSY) {
                TSS.timeout++;
                if (sec_on) sec_count(SEC_TS_MISS, 1);
            }
            goto out;
        }
    }
    if (j->state == TS_READY && j->hashed) {
        if (!ts_valid(j, now)) {
            TSS.stale++;
            ts_drop(j);   // its bytes and file name were the old storage's too
            goto out;
        }
        *cprint = j->cprint, *chash = j->chash, j->used = 1;
        TSS.taken_hash++, TSS.lead_ns += plat_now_ns() - j->ask_ns, TSS.lead_n++;
        r = 1;
    }
out:
    plat_mutex_unlock(&ts_lock);
    return r;
}

// Texture lock held (upload, the first upload of an entry): the bytes of cache file path a worker read, when they are
// that file's and bytes long. The caller owns *buf / *alloc then (owned_staging_later). -1: the worker found no such file.
static int ts_take_file(const tex_t *t, const char *path, uint64_t bytes, VkBuffer *buf, VmaAllocation *alloc) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || !t->mem) return 0;
    int r = 0;
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (j && j->state == TS_READY && j->nofile && !strcmp(j->path, path)) {
        r = -1;   // no such file: upload decodes without looking (an open of a missing file costs the replay 0.1-0.5 ms)
        TSS.nofile_skip++;
    } else if (j && j->state == TS_READY && !j->staged) {
        if (j->buf && j->bytes == bytes && !strcmp(j->path, path)) {
            *buf = j->buf, *alloc = j->alloc;
            ts_held -= j->bytes;
            j->buf = VK_NULL_HANDLE, j->alloc = NULL, j->bytes = 0;
            TSS.taken_file++, j->used = 1;
            r = 1;
        } else if (j->buf) {
            TSS.mismatch++;
        }
        ts_drop(j);
    }
    plat_mutex_unlock(&ts_lock);
    return r;
}

// Texture lock held (upload, the first upload of an entry): the bytes a worker staged for t, when they were made for t's
// descriptor in n regions of total bytes and the storage still holds what they were made from (ts_valid); waits for a
// job a worker runs. The caller owns *buf / *alloc then (owned_staging_later); *hash is the storage's hash they go with.
static int ts_take_staged(const tex_t *t, uint64_t total, uint32_t n, VkBuffer *buf, VmaAllocation *alloc, uint64_t *hash) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || !t->mem) return 0;
    int r = 0;
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (j && j->state == TS_QUEUED) {
        ts_drop(j);
        TSS.stolen++;
        if (sec_on) sec_count(SEC_TS_MISS, 1);
    } else if (j && j->state == TS_BUSY) {
        uint64_t seq = j->seq, t0 = plat_now_ns(), sec0 = sec_begin(), el = 0;
        while (j->state == TS_BUSY && j->seq == seq && (el = plat_now_ns() - t0) < ts_wait_ns)
            plat_cond_wait_ns(&ts_cv, &ts_lock, ts_wait_ns - el);
        el = plat_now_ns() - t0;
        sec_end(SEC_TS_WAIT, sec0);
        TSS.waited++, TSS.wait_ns += el;
        if (el > TSS.wait_max_ns) TSS.wait_max_ns = el;
        if (j->seq != seq) j = NULL;
        else if (j->state == TS_BUSY) {
            TSS.timeout++;
            if (sec_on) sec_count(SEC_TS_MISS, 1);
            j = NULL;
        }
    }
    if (j && j->state == TS_READY && j->staged) {
        uint32_t now;
        if (j->bytes != total || j->nreg != n || memcmp(&j->t.d, &t->d, sizeof t->d)) {
            TSS.mismatch++;
        } else if (!ts_valid(j, &now)) {
            TSS.stale++;
        } else {
            *buf = j->buf, *alloc = j->alloc, *hash = j->chash;
            ts_held -= j->bytes;
            j->buf = VK_NULL_HANDLE, j->alloc = NULL, j->bytes = 0;
            TSS.taken_staged++, j->used = 1;
            r = 1;
        }
        ts_drop(j);
    }
    plat_mutex_unlock(&ts_lock);
    return r;
}

// vk_texture_register, texture lock held: t's storage is registered again while its texture is made and uploaded
// already (the game names its loaded textures in new slots all the time): what the stream prepares for it is not
// needed. In the field such jobs held the staging cap full (3,211 jobs without room in 50 s, 2026-11-16).
static void ts_resident(const tex_t *t) {
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (j && (j->state == TS_QUEUED || j->state == TS_READY)) ts_drop(j), TSS.resident++;
    else if (j && j->state == TS_BUSY) j->unwanted = 1, TSS.resident++;
    if (j) j->used = 1;
    plat_mutex_unlock(&ts_lock);
}

// Texture lock held: t's entry was filled without an upload (adopt_released took an image): its job goes.
static void ts_forget(const tex_t *t) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || !t->mem) return;
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    if (j && j->state == TS_READY) ts_drop(j), TSS.adopted++;
    if (j) j->used = 1;
    plat_mutex_unlock(&ts_lock);
}

static void lead_note(const tex_t *t, int reg);   // DEBUG-TEXLEAD
void vk_texture_stream_ask(const tex_t *t) {   // the game's registration of t in a pool slot, on the game's thread
    lead_note(t, 1);
    ts_ask(t);
}

// vk_arrive.c: the stream's jobs still in its queue (a racy read: a count for a decision; results ready and not taken
// do not count, many are of textures nothing uses soon).
int vk_texture_stream_live(void) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return 0;
    uint32_t h = __atomic_load_n(&tsq_head, __ATOMIC_RELAXED), t = __atomic_load_n(&tsq_tail, __ATOMIC_RELAXED);
    return t - h < (1u << 30) ? (int)(t - h) : 0;
}

// vk_texture_range_gone, after the pool left the index: no worker reads storage in the range once this returns, and
// the jobs there go.
static void ts_range_gone(uint64_t lo, uint64_t size) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    plat_mutex_lock(&ts_read);
    plat_mutex_unlock(&ts_read);
    if (!size) return;
    plat_mutex_lock(&ts_lock);
    for (uint32_t i = 0; ts_live && i < TS_N; i++) {
        TsJob *j = &ts[i];
        if ((j->state == TS_QUEUED || j->state == TS_BUSY || j->state == TS_READY) && j->addr >= lo && j->addr - lo < size) ts_drop(j);
    }
    plat_mutex_unlock(&ts_lock);
}

// vk_texpool.c resolve_slot, texture lock held: 1 when slot id's texture has a job in flight that was asked less than
// TS_DEFER_FRAMES frames ago, so the update leaves the slot (it shows the stand-in) for a later one. While no scene is
// shown (a loading screen, the start's logos and opening, or a scene's first frames held unshown: vk_arrive.c) the
// slot waits for its job however long it takes: nothing on screen reads it, and taking the job back made the replay
// thread open and read the texture's cache file itself, 5 ms a file on a cold file cache (the start's 5.7 s of reads
// beside three busy workers, 2026-11-17). REKARINA_UPFRONT=0: the frame limit in every case, as before.
int vk_arrive_unshown(void);
int vk_texture_stream_defer(int id) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0 || id < 0 || id >= ntslots || !tslots[id].used) return 0;
    const tex_t *t = &tslots[id].t;
    if (!t->mem) return 0;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    int unshown = vk_arrive_unshown();
    plat_mutex_lock(&ts_lock);
    TsJob *j = ts_find((uintptr_t)t->mem, t->L.size);
    int r = j && (j->state == TS_QUEUED || j->state == TS_BUSY) && (unshown || now - j->ask_frame < TS_DEFER_FRAMES);
    if (r) TSS.deferred++;
    plat_mutex_unlock(&ts_lock);
    if (r && sec_on) sec_count(SEC_TS_DEFER, 1);
    return r;
}

// vk_texture_new_submit, once per 60 frames: results nobody took go, and the line of what the stream did.
static void ts_tick(uint64_t frame, int log) {
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) <= 0) return;
    plat_mutex_lock(&ts_lock);
    for (uint32_t i = 0; ts_live && i < TS_N; i++)
        if (ts[i].state == TS_READY && frame - ts[i].ask_frame > TS_KEEP_FRAMES) ts_drop(&ts[i]), TSS.expired++;
        else if (ts[i].state == TS_READY && ts[i].buf && frame - ts[i].ask_frame > TS_KEEP_FRAMES / 3) {   // bytes go sooner
            vmaDestroyBuffer(vk_ctx()->vma, ts[i].buf, ts[i].alloc), ts_held -= ts[i].bytes;
            ts[i].buf = VK_NULL_HANDLE, ts[i].alloc = NULL, ts[i].bytes = 0, ts[i].path[0] = 0, ts[i].staged = 0;
            TSS.bytes_expired++;
        }
    if (log)
        rt_log("perf: frame %llu: texture stream: asked %llu (moved storage, no job %llu; again %llu, table full %llu; asked again after done %llu, the same bytes %llu), hashed %llu (storage gone %llu), "
               "cache files read %llu (none %llu, over the staging cap %llu), bytes staged %llu; at first use: hash taken %llu "
               "(%.1f ms after it was asked), file taken %llu, staged bytes taken %llu, written since %llu, waited %llu (%.2f ms, longest %.2f, ran out %llu), not "
               "ready %llu, no job %llu; pool slots left for later %llu, results dropped unused %llu, other file %llu, image adopted %llu; "
               "made already at registration %llu, bytes dropped unused %llu, no bytes for an adoptable hash %llu, no file to look for %llu; %d jobs live, staging held %.1f MiB",
               (unsigned long long)frame, (unsigned long long)TSS.asked, (unsigned long long)TSS.moved, (unsigned long long)TSS.dup,
               (unsigned long long)TSS.full, (unsigned long long)TSS.reask, (unsigned long long)TSS.same, (unsigned long long)TSS.hashed, (unsigned long long)TSS.gone,
               (unsigned long long)TSS.files, (unsigned long long)TSS.nofile, (unsigned long long)TSS.capped,
               (unsigned long long)TSS.staged, (unsigned long long)TSS.taken_hash,
               TSS.lead_n ? TSS.lead_ns / 1e6 / TSS.lead_n : 0.0, (unsigned long long)TSS.taken_file,
               (unsigned long long)TSS.taken_staged, (unsigned long long)TSS.stale, (unsigned long long)TSS.waited,
               TSS.wait_ns / 1e6, TSS.wait_max_ns / 1e6, (unsigned long long)TSS.timeout, (unsigned long long)TSS.stolen,
               (unsigned long long)TSS.missing, (unsigned long long)TSS.deferred, (unsigned long long)TSS.expired,
               (unsigned long long)TSS.mismatch, (unsigned long long)TSS.adopted, (unsigned long long)TSS.resident,
               (unsigned long long)TSS.bytes_expired, (unsigned long long)TSS.adoptable, (unsigned long long)TSS.nofile_skip, ts_live, ts_held / 1048576.0);
    if (log) memset(&TSS, 0, sizeof TSS);
    plat_mutex_unlock(&ts_lock);
}

static void upload(Tex *e) {
    static volatile int32_t n_bad, n_undecoded, n_split;
    // the worker reads the storage with the texture lock released, when e may go: it reads through copies
    tex_t t_own;
    const tex_t *t = &e->t;
    if (up_worker) t_own = e->t, t = &t_own;
    const nvn_texdesc *d = &t->d;
    const TexKey key0 = e->key;
    const VkImage image0 = e->pub.image;
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || !t->mem) return;
    int bc1 = is_bc1(e);
    if (bc1 && e->bc1_redo) return;   // made again at the next submission's start
    if (bc1 && e->uploaded) {   // only a cache file fills a BC1 image (the decoder writes BC3)
        if (e->last_ch == -1 || up_worker) return;   // storage that is not write-tracked keeps its content; the replay's
        uint64_t h = content_hash((const uint8_t *)(uintptr_t)t->mem, t->L.size);
        if (e->has_chash && h == e->chash) return;   // the same bytes written again
        uint32_t lv = nvn_tex_is_linear(d) ? 1 : e->pub.levels, ly = nvn_tex_is_linear(d) ? 1 : e->pub.layers;
        uint64_t k = bc1_key(h, (unsigned)d->format, d->width, d->height, d->depth, lv, ly);
        int failed = 0;
        for (uint32_t i = 0; i < bc1_nfailed && i < 64; i++) failed |= bc1_failed[i] == k;
        if (!bc1_has(k) || failed) {   // other bytes, not kept as BC1: a BC3 image
            bc1_redo(e, "its storage holds other bytes now");
            return;
        }
        e->chash = h, e->has_chash = 1;   // other bytes kept as BC1 too: their file fills the same image below
        plat_inc64(&bc1_refilled);
    }
    vk_dbg_r8("upload", t, (long)e->pub.levels, 0);   // DEBUG-R8LOG
    uint64_t p0 = perf_begin(), u0 = kern_now_ns(), sec0 = sec_begin();
    edge_keep(e);
    int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D, one_d = e->pub.type == VK_IMAGE_TYPE_1D;
    uint32_t levels = linear ? 1 : e->pub.levels, layers = linear ? 1 : e->pub.layers;
    int astc = e->f.astc_bw != 0, rgb = e->f.expand_rgb;
    int agpu = astc && vk_astc_gpu(), acheck = agpu && vk_astc_mode() == VK_ASTC_CHECK;   // raw blocks to the GPU
    uint32_t n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    VkAstcRegion *ar = agpu ? calloc(n, sizeof *ar) : NULL;
    VkDeviceSize total = 0, decoded = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) {
            uint32_t w = mip(d->width, lv), h = one_d ? 1 : mip(d->height, lv), sl = is3d ? mip(d->depth, lv) : 1;
            uint64_t texels = (uint64_t)w * h * sl, raw = (uint64_t)(w + bw - 1) / bw * bpb * ((h + bh - 1) / bh) * sl;
            uint64_t out = agpu ? raw : astc ? texels * 4 : rgb ? texels * (uint64_t)bpb / 3 * 4 : raw;
            rg[k++] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {e->pub.aspect, lv, ly, 1},
                                          .imageExtent = {w, h, sl}};
            total += (out + 15) & ~15ull;
            decoded += texels * 4;
        }
    VkBuffer sb, rbuf = VK_NULL_HANDLE;
    VmaAllocation ralloc = NULL;
    uint8_t *rmap = NULL, *raw_copy = NULL;
    int checked = 0, parts = 1;
    // the texture cache: BC3 images the GPU would decode and encode, by the hash of their storage
    TexCacheJob *tcj = NULL;
    uint64_t tc_hash = 0;
    int tc_done = 0, w = up_worker, was = e->uploaded, skip = 0;
    if (agpu && e->f.astc_bc && !acheck && (!e->uploaded || bc1) && t->L.size >= 4096 &&
        tex_cache_dir()) {
        // the entry's own hash where it was just taken (adopt_released): the same function over the same bytes
        uint64_t scp, sch;
        uint32_t snow;
        if (e->has_chash) tc_hash = e->chash;
        else if (!w && ts_take_hash(t, &scp, &sch, &snow)) tc_hash = sch;   // the texture stream's
        else {
            uint64_t hsec = sec_begin();
            tc_hash = content_hash((const uint8_t *)(uintptr_t)t->mem, t->L.size);
            sec_end(SEC_TS_HASH, hsec);
        }
        VkBufferImageCopy *br = calloc(n, sizeof *br);
        uint64_t bc_total = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t w = rg[i].imageExtent.width, h = rg[i].imageExtent.height, sl = rg[i].imageExtent.depth;
            br[i] = rg[i];
            br[i].bufferOffset = bc_total, br[i].bufferRowLength = (w + 3) / 4 * 4, br[i].bufferImageHeight = (h + 3) / 4 * 4;
            bc_total += bc1 ? vk_astc_bc_bytes(w, h, sl) / 2 : vk_astc_bc_bytes(w, h, sl);   // BC1: 8 bytes a block
        }
        char path[600];
        tex_cache_path(path, sizeof path, e, tc_hash, levels, layers);
        VkBuffer fb = VK_NULL_HANDLE;
        VmaAllocation fa = NULL;
        int tf = w ? 0 : ts_take_file(t, path, bc_total, &fb, &fa);
        if (tf < 0) {
            tc_done = 0;   // the texture stream found no such file
        } else if (tf > 0) {   // the texture stream read the file: one copy
            VkCommandBuffer cb = vk_upload_begin();
            owned_staging_later(fb, fa);   // after the batch that copies from it is open: the deferral is stamped with it
            barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            vkCmdCopyBufferToImage(cb, fb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, br);
            barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
            plat_inc64(&tc_loaded);
            plat_add64(&tc_load_bytes, (int64_t)bc_total);
            vk_upload_end();
            tc_done = 1;
        } else {
            uint64_t rsec = sec_begin();
            tc_done = tex_cache_load(e, path, br, n, bc_total, &skip);
            sec_end(SEC_TS_READ, rsec);
        }
        free(br);
        int64_t on_disk = __atomic_load_n(&tc_disk, __ATOMIC_ACQUIRE);
        if (!tc_done && !w && !bc1 && on_disk >= 0 && on_disk + (int64_t)bc_total <= tc_limit && (tcj = calloc(1, sizeof *tcj))) {
            snprintf(tcj->path, sizeof tcj->path, "%s", path);
            tcj->size = bc_total;
            tcj->map = vk_staging(bc_total, &tcj->buf, &tcj->alloc);
        }
    }
    if (bc1 && !tc_done) {   // no decode into a BC1 image: made again (bc1_redo), BC3 now that its key failed
        if (!skip && !w) {
            bc1_fail_key(e);
            bc1_redo(e, "its cache file could not be read");
        }
        free(ar);
        free(rg);
        return;
    }
    if (bc1 && !e->uploaded) bc1_made++, bc1_made_bytes += e->bytes;
    // A CPU decode or copy stages at most UPLOAD_PART bytes per staging buffer (a region larger than that alone) and
    // records one copy per buffer; every part stays in the same upload batch, submitted as before. The GPU decoder
    // takes the raw blocks of every region from one buffer.
    // The worker leaves every ASTC decode to the replay thread (the GPU decoder's recording is the replay's, and the
    // CPU decoder reads the entry while it runs), and anything it lost a race for.
    if (w && !tc_done && (astc || acheck)) skip = 1;
    // the texture stream staged these bytes ahead (every region at the offsets above, in one buffer): one copy, or the
    // GPU's decode of the raw blocks
    VkBuffer pb = VK_NULL_HANDLE;
    VmaAllocation pa = NULL;
    uint64_t phash = 0;
    int staged = !w && !tc_done && !skip && !acheck && (agpu || !astc) && !e->uploaded &&
                 ts_take_staged(t, total, n, &pb, &pa, &phash);
    if (staged) {
        VkCommandBuffer cb = vk_upload_begin();
        owned_staging_later(pb, pa);   // after the batch that copies from it is open: the deferral is stamped with it
        for (k = 0; k < n; k++) {
            uint32_t lv = rg[k].imageSubresource.mipLevel, ly = rg[k].imageSubresource.baseArrayLayer;
            uint32_t w = rg[k].imageExtent.width, h = rg[k].imageExtent.height, sl = rg[k].imageExtent.depth;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            if (agpu) ar[k] = (VkAstcRegion){.src = rg[k].bufferOffset, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly};
            else if (rgb) rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            else rg[k].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[k].bufferImageHeight = rows * (uint32_t)bh;
        }
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (agpu) {
            uint64_t a0 = perf_begin();
            if (tcj) vk_astc_bc_readback(tcj->buf);
            checked = vk_astc_record(cb, pb, total, ar, (int)n, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc,
                                     e->pub.image, e->pub.aspect, VK_NULL_HANDLE);
            perf_end(PERF_ASTC, a0);
            if (!checked)
                log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: left as it was",
                            (unsigned long long)e->pub.addr, d->format, d->width, d->height);
        } else {
            vkCmdCopyBufferToImage(cb, pb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
        }
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        vk_upload_end();
    }
    for (uint32_t k0 = 0, k1; k0 < n && !tc_done && !skip && !staged; k0 = k1) {
        VkDeviceSize base = rg[k0].bufferOffset, size;
        for (k1 = k0 + 1; k1 < n && (agpu || (k1 + 1 < n ? rg[k1 + 1].bufferOffset : total) - base <= UPLOAD_PART);) k1++;
        size = (k1 < n ? rg[k1].bufferOffset : total) - base;
        if (k0) parts++;
        VmaAllocation sa = NULL;
        VkCommandBuffer cb = w ? VK_NULL_HANDLE : vk_upload_begin();
        uint8_t *stg = vk_staging(size, &sb, w ? &sa : NULL);
        if (acheck) rmap = vk_staging(decoded, &rbuf, &ralloc);
        if (w) fill_enter();   // the copy out of the game's memory, with the texture lock released
        for (k = k0; k < k1; k++) {
            uint32_t lv = rg[k].imageSubresource.mipLevel, ly = rg[k].imageSubresource.baseArrayLayer;
            uint32_t w = rg[k].imageExtent.width, h = rg[k].imageExtent.height, sl = rg[k].imageExtent.depth;
            uint32_t rb = (w + bw - 1) / bw * (uint32_t)bpb, rows = (h + bh - 1) / bh;
            size_t raw = (size_t)rb * rows * sl;
            rg[k].bufferOffset -= base;
            uint8_t *dst = stg + rg[k].bufferOffset, *tmp = (astc && !agpu) || rgb ? malloc(raw) : NULL;
            uint64_t sec1 = sec_begin();
            int swz_bad = nvn_deswizzle(d, lv, ly, t->mem, tmp ? tmp : dst, rb);
            sec_end(raw < (64u << 10) ? SEC_DESW_S : raw < (1u << 20) ? SEC_DESW_M : raw < (4u << 20) ? SEC_DESW_L : SEC_DESW_XL, sec1);
            if (swz_bad) {
                log_limited(&n_bad, "texture %#llx fmt %#x target %d: level %u layer %u has no CPU layout; uploaded as zeros",
                            (unsigned long long)(uintptr_t)t->mem, d->format, d->target, lv, ly);
                memset(tmp ? tmp : dst, 0, raw);
            }
            if (agpu) {
                ar[k] = (VkAstcRegion){.src = rg[k].bufferOffset, .w = w, .h = h, .slices = sl, .level = lv, .layer = ly};
            } else if (astc) {
                uint64_t a0 = perf_begin();
                for (uint32_t z = 0; z < sl; z++)
                    astc_decode_ldr_mt(tmp + (size_t)z * rb * rows, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb,
                                       dst + (size_t)z * w * h * 4);
                perf_end(PERF_ASTC, a0);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else if (rgb) {
                expand_rgb(d->format, tmp, dst, (size_t)w * h * sl);
                rg[k].bufferRowLength = w, rg[k].bufferImageHeight = h;
            } else {
                rg[k].bufferRowLength = rb / (uint32_t)bpb * (uint32_t)bw, rg[k].bufferImageHeight = rows * (uint32_t)bh;
            }
            free(tmp);
        }
        if (acheck) {
            raw_copy = malloc(size);
            memcpy(raw_copy, stg, size);
        }
        if (w) {
            if (!fill_leave(&key0, e, image0, was)) {
                vmaDestroyBuffer(vk_ctx()->vma, sb, sa);
                skip = 1;
                break;
            }
            cb = vk_upload_begin();
            owned_staging_later(sb, sa);   // after the batch that copies from it is open: the deferral is stamped with it
        }
        barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (agpu) {
            uint64_t a0 = perf_begin();
            if (tcj) vk_astc_bc_readback(tcj->buf);
            checked = vk_astc_record(cb, sb, size, ar, (int)n, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc,
                                     e->pub.image, e->pub.aspect, rbuf);
            perf_end(PERF_ASTC, a0);
            if (!checked)
                log_limited(&n_undecoded, "ASTC texture %#llx (fmt %#x %ux%u) not decoded on the GPU: left as it was",
                            (unsigned long long)e->pub.addr, d->format, d->width, d->height);
        } else {
            vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, k1 - k0, rg + k0);
        }
        barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
        vk_upload_end();
    }
    if (tcj) {   // written once the GPU is past this batch; not kept when the decode failed or the game wrote meanwhile
        // staged by the stream: the bytes decoded are those its hash names (checked when they were taken)
        tcj->keep = checked && (staged ? phash == tc_hash : content_hash((const uint8_t *)(uintptr_t)t->mem, t->L.size) == tc_hash);
        vk_defer_call(tex_cache_write, tcj);
    }
    if (tc_hash && !((tc_loaded + tc_saved) & 255) && tc_loaded + tc_saved)
        rt_log("vk: texture cache: %lld textures filled from it (%.2f GiB), %lld written to it, %lld failed",
               (long long)tc_loaded, tc_load_bytes / 1073741824.0, (long long)tc_saved, (long long)tc_failed);
    if (parts > 1)
        log_limited(&n_split, "texture %#llx fmt %#x %ux%u, %u levels, %u layers: %.1f MiB staged in %d parts",
                    (unsigned long long)e->pub.addr, d->format, d->width, d->height, levels, layers, total / 1048576.0,
                    parts);
    if (acheck) {   // waits for the GPU
        vk_upload_flush(1);
        if (checked) {
            vmaInvalidateAllocation(vk_ctx()->vma, ralloc, 0, VK_WHOLE_SIZE);
            astc_check(e, raw_copy, ar, (int)n, rmap);
        }
        vmaDestroyBuffer(vk_ctx()->vma, rbuf, ralloc);
    }
    free(raw_copy);
    free(ar);
    free(rg);
    if (skip) return;   // the worker: left to the replay thread, the entry is as it was
    uint64_t dt = kern_now_ns() - u0;
    if (e->uploaded) {
        if (!bc1) e->has_chash = 0;   // the image no longer holds the bytes that were hashed (adopt_released); a BC1 one
                                      // was filled from the file of the hash it has now
        n_reuploads++, perf_count(PERF_CNT_TEX_REUP);
        ru_n++, ru_bytes += t->L.size, ru_ns += dt;
        if (e->wt_n) {
            ru_wt_n++, ru_wt_bytes += t->L.size, ru_wt_ns += dt;
            if (rt_trace & RT_TRACE_PERF)
                rt_log("perf: frame %llu: re-upload after %u WriteTexels: texture %#llx fmt %#x %ux%u, %.1f KiB, %.3f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), e->wt_n,
                       (unsigned long long)e->pub.addr, d->format, d->width, d->height, t->L.size / 1024.0, dt / 1e6);
        }
    } else {
        n_uploads++, perf_count(PERF_CNT_TEX_UP);
    }
    lf_count(LF_UP_BYTES, (int64_t)t->L.size);
    e->wt_n = 0;
    e->uploaded = 1;
    e->n_up++;
    perf_end(PERF_TEX, p0);
    sec_end(SEC_UP_TOTAL, sec0);
}

// Whether e's storage fills no 4 KiB page of its own, so the write tracking can only see writes to pages it shares.
static int fills_no_page(const Tex *e) {
    return ((e->pub.addr + e->t.L.size) & ~0xfffull) <= ((e->pub.addr + 0xfff) & ~0xfffull);
}

// A texture that fills no page keeps a copy of its storage bytes from when a GPU write to it was recorded
// (vk_texture_gpu_written), up to SNAP_MAX bytes; wb_exact (REKARINA_WB_EXACT, default on) enables it.
enum { SNAP_MAX = 4096 };
static int wb_exact = -1;
static volatile int32_t n_kept;

// vk_mem_changed over the pages wholly inside e's storage (all of its pages when it fills none): a neighbour that shares
// the first or last page does not count as a write to e. A texture that fills no page shares all of its pages, so
// there a page write counts only when e's own bytes differ from the copy taken at the GPU write. TotK's tablet
// camera renders the ID, screen rect and depth of what is under the reticle into 4x4 targets of 512 bytes, two
// slots used in turn, and the CPU reads them a frame later. One slot's rect target shares its page with data the CPU
// writes every frame: without the copy its renders were never written back, the game read a zero rect every other
// frame and the target's name label blinked. REKARINA_WB_EXACT=0 restores the page-only decision.
static int storage_rewritten(const Tex *e, uint32_t since, uint32_t *now) {
    uint64_t lo = (e->pub.addr + 0xfff) & ~0xfffull, hi = (e->pub.addr + e->t.L.size) & ~0xfffull;
    if (hi <= lo) lo = e->pub.addr, hi = e->pub.addr + e->t.L.size;
    int ch = vk_mem_changed(lo, hi - lo, since, now);
    if (ch != 1 || !since || !e->snap || memcmp(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size)) return ch == 1;
    log_limited(&n_kept, "write-back of %#llx kept: fmt %#x %ux%u, %llu bytes; a neighbour wrote its page",
                (unsigned long long)e->pub.addr, e->t.d.format, e->t.d.width, e->t.d.height,
                (unsigned long long)e->t.L.size);
    return 0;
}

// The page check cannot tell a texture's bytes from its neighbours' in the first and last page of its storage. OoT's
// texture compaction moves blocks of 10 to 13 MiB every few frames in the Market, and whenever
// one landed beside the 8-layer R16 array its lighting reads (1775x1175, 32 MiB) that array was uploaded again from
// its storage, every 18 frames or so, although nobody had written it: 3 ms of copying each time, and one frame in
// three or four of those came out a third darker (the upload's frame; the flight recorder's flash watch counted 21 in
// 95 s, 2026-11-12). So a texture of more than two pages keeps its own bytes of those two pages from its last upload
// (edge_keep), and a write that touched only them counts when those bytes differ. REKARINA_TEX_EDGE=0: every write counts.
static int tex_edge = -1;
static uint64_t n_edge_spared;
static void edge_span(const Tex *e, uint64_t *lo, uint64_t *hi) {
    *lo = (e->pub.addr + 0xfff) & ~0xfffull, *hi = (e->pub.addr + e->t.L.size) & ~0xfffull;
}
static void edge_keep(Tex *e) {   // before the upload reads the storage
    uint64_t lo, hi;
    edge_span(e, &lo, &hi);
    if (tex_edge < 0) tex_edge = rt_knob("TEX_EDGE", "1");
    if (!tex_edge || hi <= lo + 0x2000) return;
    uint64_t head = lo - e->pub.addr, tail = e->pub.addr + e->t.L.size - hi;
    if (!e->edge && !(e->edge = malloc(0x2000))) return;
    memcpy(e->edge, (const void *)(uintptr_t)e->pub.addr, head);
    memcpy(e->edge + 0x1000, (const void *)(uintptr_t)hi, tail);
}
// 1: the write the page check reported (since the generation `since`) left e's own bytes as they were uploaded.
static int edge_only(const Tex *e, uint32_t since, int snap) {
    uint64_t lo, hi;
    edge_span(e, &lo, &hi);
    if (!tex_edge || !e->edge || hi <= lo + 0x2000) return 0;
    uint32_t now;
    int in = snap ? vk_mem_changed_snap(lo, hi - lo, since, submit_epoch, &now) : vk_mem_changed(lo, hi - lo, since, &now);
    if (in) return 0;   // written inside, or not tracked
    uint64_t head = lo - e->pub.addr, tail = e->pub.addr + e->t.L.size - hi;
    if (memcmp(e->edge, (const void *)(uintptr_t)e->pub.addr, head) || memcmp(e->edge + 0x1000, (const void *)(uintptr_t)hi, tail)) return 0;
    n_edge_spared++;
    return 1;
}

// A mirrored texture is uploaded on first use and again when its storage pages carry a newer write generation.
static void make_fresh(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint64_t size = e->t.L.size;
    if (!e->uploaded) {
        vk_mem_changed(e->pub.addr, size, 0, &e->wgen);   // resets the write bits before the storage is read
        upload(e);
        return;
    }
    uint32_t now, since = e->wgen;
    int ch = vk_mem_changed(e->pub.addr, size, e->wgen, &now);
    e->wgen = now;
    if (ch == 1 && edge_only(e, since, 0)) ch = 0;
    e->last_ch = ch;
    if (ch) upload(e);   // 1 written, -1 not tracked: both upload again
}

// make_fresh for draw bindings: the page check answers from this submission's write-watch snapshot (vk_memory.c
// vk_mem_changed_snap) instead of a system call per texture. A write the snapshot of the texture's region missed (made
// after it, during this replay) keeps its write-watch bit and is caught by the next submission's snapshot.
static void make_fresh_snap(Tex *e) {
    if (!e->pub.mirrored || e->pub.gpu_owned) return;
    uint32_t now, since = e->wgen;
    n_checks++;
    int ch = vk_mem_changed_snap(e->pub.addr, e->t.L.size, e->uploaded ? e->wgen : 0, submit_epoch, &now);
    e->wgen = now;
    if (!e->uploaded) { upload(e); return; }
    if (ch == 1 && edge_only(e, since, 1)) ch = 0;
    e->last_ch = ch;
    if (ch) upload(e);
}

// ---------------------------------------------------------------- aliases (REKARINA_TEX_ALIAS)
// Bytes per block and block size of the image's VkFormat (ASTC decodes to RGBA8, RGB expands to RGBA); 0 when the
// VkFormat substitutes another one of unknown size.
static int img_block(const Tex *e, int *bw, int *bh) {
    if (e->f.astc_bc) { *bw = *bh = 4; return is_bc1(e) ? 8 : 16; }
    if (e->f.astc_bw) { *bw = *bh = 1; return 4; }
    int b = nvn_fmt_bpb(e->t.d.format, bw, bh);
    if (e->f.expand_rgb) { *bw = *bh = 1; return b / 3 * 4; }
    return e->f.vk == e->f.ideal ? b : 0;
}

// The new entry e can take src's content texel for texel: colour, one sample, the same shape and storage layout, and
// image formats of one block size.
static int alias_compatible(const Tex *e, const Tex *src) {
    const VkTexImage *p = &e->pub, *q = &src->pub;
    if (e->f.depth || e->f.stencil || src->f.depth || src->f.stencil || p->samples > 1 || q->samples > 1) return 0;
    if (p->type != q->type || p->width != q->width || p->height != q->height || p->depth != q->depth) return 0;
    if (e->t.L.size != src->t.L.size || nvn_tex_is_linear(&e->t.d) != nvn_tex_is_linear(&src->t.d)) return 0;
    int ew, eh, sw, sh, eb = img_block(e, &ew, &eh), sb = img_block(src, &sw, &sh);
    return eb && eb == sb && ew == sw && eh == sh;
}

// src holds content the storage bytes do not: the GPU wrote it and, for a mirrored texture, it was not written back
// and the CPU has not rewritten the storage since (the new entry's upload then reads the newer bytes).
static int alias_has_gpu_content(const Tex *src) {
    if (!src->pub.gpu_owned) return 0;
    if (!src->pub.mirrored) return 1;
    uint32_t now;
    return src->gpu_dirty && !storage_rewritten(src, src->gpu_gen, &now);
}

// Maxwell reads the same bytes through both descriptors: src's content goes into the new image e at the replay's
// current position, before any later command of the submission uses e. Through a buffer, since bound images share
// their memory and image copies may not overlap; e leaves UNDEFINED between the two halves, so its first transition
// cannot clobber src's bytes before they are read. 0 when no replay is recording (e is then not initialised yet).
static int alias_copy(Tex *e, const Tex *src) {
    int reopen;
    VkCommandBuffer cb = vk_exec_inline_begin(&reopen);
    if (!cb) return 0;
    int bw, bh, bpb = img_block(e, &bw, &bh);
    uint32_t levels = e->pub.levels < src->pub.levels ? e->pub.levels : src->pub.levels;
    uint32_t layers = e->pub.layers < src->pub.layers ? e->pub.layers : src->pub.layers, n = levels * layers, k = 0;
    VkBufferImageCopy *rg = calloc(n, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for an alias copy");
    VkDeviceSize total = 0;
    for (uint32_t lv = 0; lv < levels; lv++)
        for (uint32_t ly = 0; ly < layers; ly++, k++) {
            uint32_t w = mip(e->pub.width, lv), h = mip(e->pub.height, lv), z = mip(e->pub.depth, lv);
            rg[k] = (VkBufferImageCopy){.bufferOffset = total, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1},
                                        .imageExtent = {w, h, z}};
            total += ((VkDeviceSize)(w + bw - 1) / bw * ((h + bh - 1) / bh) * z * bpb + 15) & ~15ull;
        }
    VkBuffer buf = vk_exec_inline_buffer(total);
    if (!buf) {
        free(rg);
        vk_exec_inline_end(reopen);
        return 0;
    }
    vk_crumb(cb, VK_CRUMB_ALIAS, src->pub.addr >> 12);   // DEBUG-COPYCRUMB
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, src->pub.image, VK_IMAGE_LAYOUT_GENERAL, buf, n, rg);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = e->pub.image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
    vkCmdCopyBufferToImage(cb, buf, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_crumb(cb, VK_CRUMB_ALIAS_END, src->pub.addr >> 12);   // DEBUG-COPYCRUMB
    vk_exec_inline_end(reopen);
    free(rg);
    return 1;
}

// DEBUG-TEXWATCH: REKARINA_TEXWATCH=<hex addr>,... logs from frame REKARINA_TEXWATCH_FROM (default 4400) which entry and
// VkImage each lookup of a texture at those addresses resolves to (600 lines at most)
static void tex_watch(const char *where, const TexKey *k, const Tex *e) {
    static int n = -1, nl;
    static uint64_t w[8], from;
    if (n < 0) {
        n = 0;
        const char *s = cfg_get("REKARINA_TEXWATCH"), *f = cfg_get("REKARINA_TEXWATCH_FROM");
        from = f && *f ? strtoull(f, NULL, 0) : 4400;
        for (char *x; s && *s && n < 8; s = *x == ',' ? x + 1 : x) {
            w[n++] = strtoull(s, &x, 16);
            if (x == s) break;
        }
    }
    if (!n || nl >= 600) return;
    const char *wk = cfg_get("REKARINA_TEXWATCH_KEYS");   // =stale: only lookups that ended on an entry older than a live one of its format
    if (e && wk && !strcmp(wk, "stale")) {
        int stale = 0;
        for (const Tex *p = nvn_hm_get(&by_addr, &k->addr); p && !stale; p = p->next_at)
            stale = p != e && !p->released && p->key.d.format == e->key.d.format && wr_at(p) > wr_at(e);
        if (!stale) return;
    }
    int hit = 0;
    for (int i = 0; i < n; i++) hit |= w[i] == k->addr;
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!hit || fr < from) return;
    int live = 0, all = 0;
    for (const Tex *p = nvn_hm_get(&by_addr, &k->addr); p; p = p->next_at) all++, live += !p->released;
    nl++;
    rt_log("TEXWATCH frame %llu %s addr %#llx key fmt %#x flags %#x %ux%u -> entry %p image %p fmt %#x flags %#x%s%s own %d "
           "gpu_owned %d mirrored %d dirty %d run %u pool_flags %#x; entries at addr %d live %d", (unsigned long long)fr, where,
           (unsigned long long)k->addr,
           k->d.format, k->d.flags, k->d.width, k->d.height, (const void *)e, e ? (void *)e->pub.image : NULL,
           e ? e->key.d.format : 0, e ? e->key.d.flags : 0, e && e->released ? " RELEASED" : "",
           e && memcmp(&e->key, k, sizeof *k) ? " (other key)" : "", e ? e->alloc != NULL : -1,
           e ? e->pub.gpu_owned : -1, e ? e->pub.mirrored : -1, e ? e->gpu_dirty : -1, e ? (unsigned)e->run : 0u,
           e ? (unsigned)e->t.pool_flags : 0u, all, live);
    if (cfg_get("REKARINA_TEXWATCH_KEYS"))   // every entry at the address with its whole key and its write order
        for (const Tex *p = nvn_hm_get(&by_addr, &k->addr); p; p = p->next_at)
            rt_log("TEXWATCH   %s%p tgt %d fmt %#x flags %#x %ux%ux%u lv %u smp %d stride %u bh %d layout %d %02x%02x%02x%02x "
                   "wr %llu (as of %llu) gpu_owned %d mirrored %d%s", p == e ? "> " : "  ", (const void *)p, p->key.d.target,
                   p->key.d.format, p->key.d.flags, p->key.d.width, p->key.d.height, p->key.d.depth, p->key.d.levels,
                   p->key.d.samples, p->key.d.stride, p->key.d.pkg_block_h, p->key.has_layout, p->key.layout[0],
                   p->key.layout[1], p->key.layout[2], p->key.layout[3], (unsigned long long)p->wr_serial,
                   (unsigned long long)wr_at(p), p->pub.gpu_owned, p->pub.mirrored, p->released ? " RELEASED" : "");
}

// DEBUG-ALIASLOG: REKARINA_ALIASLOG=<from frame>: every new entry at an address that already has entries (800 lines at
// most): the new key, each entry there and what became of it (released ones are replaced, a live one may be the
// content source), and whether the new image got content. For the whole-scene black frames (flight.c flash watch).
static int alias_log_on(void) {
    static int on = -1;
    static uint64_t from;
    if (on < 0) { const char *e = cfg_get("REKARINA_ALIASLOG"); on = e && *e; from = on ? strtoull(e, NULL, 0) : 0; }
    return on && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) >= from;
}
static void alias_log(const char *what, const TexKey *k, const Tex *p) {
    static int n, only = -1;
    // REKARINA_ALIASLOG_EMPTY=1: only the new images that got no content or a copy, without the square target whose size
    // changes every frame while the camera turns; 6000 lines
    if (only < 0) only = cfg_get("REKARINA_ALIASLOG_EMPTY") ? atoi(cfg_get("REKARINA_ALIASLOG_EMPTY")) : 0;
    // =2: also the released entries a new one replaces and the new ones that take a released one's image
    if (only && (!(strstr(what, "EMPTY") || strstr(what, "copied") || strstr(what, "storage") ||
                   (only == 2 && (strstr(what, "replaces") || strstr(what, "released one")))) ||
                 (k->d.width == k->d.height && k->d.width > 512 && k->d.format == 0x34)))
        return;
    if (n >= (only ? 6000 : 800)) return;
    n++;
    rt_log("ALIASLOG frame %llu %s addr %#llx new %ux%u fmt %#x flags %#x | entry %ux%u fmt %#x flags %#x%s gpu_owned %d "
           "dirty %d mirrored %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), what,
           (unsigned long long)k->addr, k->d.width, k->d.height, k->d.format, k->d.flags, p ? p->key.d.width : 0,
           p ? p->key.d.height : 0, p ? p->key.d.format : 0, p ? p->key.d.flags : 0, p && p->released ? " RELEASED" : "",
           p ? p->pub.gpu_owned : -1, p ? p->gpu_dirty : -1, p ? p->pub.mirrored : -1);
}

// DEBUG-TEXLEAD (REKARINA_TEX_LEAD=1): how long before the replay makes a CPU-filled texture's image the texture first
// became known: the game's initialization (vk_texture_premake) and the game's registration of it in a pool slot (the
// game's thread, vk_backend.c vk_be_texture_register; the replay applies the registration later, in queue order). The
// time a worker would have to make the image ahead. A direct-mapped table of keys under its own lock (taken inside the
// texture lock or with no lock; it takes none).
#define LEAD_N (1u << 16)
static struct { uint64_t h, init_ns, reg_ns, init_frame, reg_frame; } *lead_tab;
static PlatMutex lead_lock = PLAT_MUTEX_INIT;
enum { LEAD_B = 10 };
static const double lead_edge[LEAD_B - 1] = {0.5, 1, 2, 4, 8, 16, 33, 100, 1000};
static struct { uint64_t n, made_ns, bc1, pool, by_reg[LEAD_B + 1], by_init[LEAD_B + 1], reg_frames, reg_n; } LS;
static int lead_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_env("REKARINA_TEX_LEAD") != NULL;
    return on;
}
static void lead_note(const tex_t *t, int reg) {   // any thread
    if (!lead_on() || !t->mem || !tex_mirrored(t)) return;
    TexKey k;
    make_key(t, &k);
    uint64_t h = rep_hash(&k), now = plat_now_ns(), f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    plat_mutex_lock(&lead_lock);
    if (!lead_tab) lead_tab = calloc(LEAD_N, sizeof *lead_tab);
    if (lead_tab) {
        typeof(*lead_tab) *r = &lead_tab[h & (LEAD_N - 1)];
        if (r->h != h) memset(r, 0, sizeof *r), r->h = h;
        if (reg && !r->reg_ns) r->reg_ns = now, r->reg_frame = f;
        if (!reg) r->init_ns = now, r->init_frame = f, r->reg_ns = 0;   // initialized (again): its registrations come after
    }
    plat_mutex_unlock(&lead_lock);
}
static int lead_bucket(uint64_t from, uint64_t now) {
    if (!from) return LEAD_B;   // never seen
    double ms = (now - from) / 1e6;
    int b = 0;
    while (b < LEAD_B - 1 && ms >= lead_edge[b]) b++;
    return b;
}
static void lead_made(const Tex *e, uint64_t t0) {   // texture lock held: entry_of made e's image
    if (!lead_on() || !e->pub.mirrored || up_worker) return;
    uint64_t h = rep_hash(&e->key), now = plat_now_ns(), f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    uint64_t rn = 0, in = 0, rf = 0;
    plat_mutex_lock(&lead_lock);
    if (lead_tab && lead_tab[h & (LEAD_N - 1)].h == h) {
        rn = lead_tab[h & (LEAD_N - 1)].reg_ns, in = lead_tab[h & (LEAD_N - 1)].init_ns;
        rf = lead_tab[h & (LEAD_N - 1)].reg_frame;
        lead_tab[h & (LEAD_N - 1)].h = 0;   // counted once
    }
    LS.n++, LS.made_ns += now - t0, LS.bc1 += is_bc1(e), LS.pool += pool_lookup;
    LS.by_reg[lead_bucket(rn, now)]++, LS.by_init[lead_bucket(in, now)]++;
    if (rn) LS.reg_frames += f - rf, LS.reg_n++;
    plat_mutex_unlock(&lead_lock);
}
// entry_of's making of an entry, split (ns sums, texture lock held): [0] bc1 choice, [1] create_tex's plan, [2] its image
// and memory (or the one made ahead), [3] its tail (labels, record, views), [4] its init / rest, [5] the replace loop
// (released entries at the address destroyed), [6] the fill (alias copy / adoption / init), [7] the rest; by CPU-filled
// (0) or not (1); hows: alias copies, adoptions, inits.
static uint64_t lsp[2][8], lsp_n[2], lsp_how[4];
static void lead_split(uint64_t t0, uint64_t bc1, uint64_t plan, uint64_t made, uint64_t tail, uint64_t repl, uint64_t end,
                       int how, int mirrored) {
    int m = !mirrored;
    if (!plan) plan = made = tail = bc1;   // (create_tex returned early: none of its parts)
    uint64_t *a = lsp[m];
    a[0] += bc1 - t0, a[1] += plan - bc1, a[2] += made - plan, a[3] += tail - made, a[5] += 0;
    // create_tex's init runs after tail; the replace loop starts when create_tex returned (unknown): counted from tail
    a[4] += 0, a[5] += repl - tail, a[6] += end - repl;
    lsp_n[m]++, lsp_how[how & 3]++;
}
static void lead_log(uint64_t frame) {   // every 600 frames
    if (!lead_on() || !LS.n) return;
    char a[256], b[256];
    int ka = 0, kb = 0;
    for (int i = 0; i <= LEAD_B; i++) {
        ka += snprintf(a + ka, sizeof a - (size_t)ka, " %llu", (unsigned long long)LS.by_reg[i]);
        kb += snprintf(b + kb, sizeof b - (size_t)kb, " %llu", (unsigned long long)LS.by_init[i]);
    }
    for (int m = 0; m < 2; m++)
        if (lsp_n[m])
            rt_log("TEXLEAD frame %llu: %s entries made %llu, ms: bc1 choice %.2f, plan %.2f, image+memory %.2f, record+views %.2f, "
                   "init+replace loop %.2f, fill %.2f; fills: alias copies %llu, adoptions %llu, inits %llu", (unsigned long long)frame,
                   m ? "GPU-owned" : "CPU-filled", (unsigned long long)lsp_n[m], lsp[m][0] / 1e6, lsp[m][1] / 1e6, lsp[m][2] / 1e6,
                   lsp[m][3] / 1e6, lsp[m][5] / 1e6, lsp[m][6] / 1e6, (unsigned long long)lsp_how[1], (unsigned long long)lsp_how[2],
                   (unsigned long long)lsp_how[3]);
    memset(lsp, 0, sizeof lsp), memset(lsp_n, 0, sizeof lsp_n), memset(lsp_how, 0, sizeof lsp_how);
    rt_log("TEXLEAD frame %llu: CPU-filled images made %llu (%.3f ms each in entry_of, %llu by the pool update, %llu BC1); "
           "ms since the game registered them [<0.5 <1 <2 <4 <8 <16 <33 <100 <1000 more never]:%s (mean %.2f frames); "
           "since initialized:%s", (unsigned long long)frame, (unsigned long long)LS.n, LS.made_ns / 1e6 / LS.n,
           (unsigned long long)LS.pool, (unsigned long long)LS.bc1, a, LS.reg_n ? (double)LS.reg_frames / LS.reg_n : 0.0, b);
    memset(&LS, 0, sizeof LS);
}

static Tex *entry_of(const tex_t *t) {
    TexKey k;
    make_key(t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    if (!e) e = follow_new(t, &k);   // REKARINA_TEX_FOLLOW: the entry its bytes were moved from
    if (!e) {
        uint64_t lead_t0 = lead_on() ? plat_now_ns() : 0;   // DEBUG-TEXLEAD
        if (tex_alias < 0) {
            tex_alias = cfg_bool("REKARINA_TEX_ALIAS");
            rt_log("vk: texture aliases: %s", tex_alias ? "a new image takes the live image's content" : "a new image starts empty");
        }
        uint64_t sec0 = sec_begin();
        bc1_next = bc1_choose(t);   // hashes the storage of an ASTC texture the cache knows (bc1_pre)
        uint64_t lt_bc1 = lead_t0 ? plat_now_ns() : 0;
        cs_plan = cs_made = cs_tail = 0;
        if (!(e = create_tex(t, &k, tex_alias))) {
            bc1_pre.valid = 0;
            return NULL;
        }
        perf_count(PERF_CNT_TEX_NEW);
        __atomic_fetch_add(&n_made_total, 1, __ATOMIC_RELAXED);
        if (cold_val(cold_del(&k))) n_cold_back++;
        // released entries of another descriptor at this address are replaced by the new one; a live one holding
        // GPU-written content is the alias source, preferring the one that differs only in swizzle (same_but_swizzle)
        int alias = 0;
        Tex *src = NULL;
        int alog = alias_log_on();
        for (Tex *p = nvn_hm_get(&by_addr, &k.addr), *nx; p; p = nx) {
            nx = p->next_at;
            if (alog) alias_log(p->released ? "replaces released" : "beside live", &k, p);
            if (p->released) {
                if (!follow_away(p, 1)) destroy_tex(p);   // REKARINA_TEX_FOLLOW: moved away, kept at its bytes' new address
                continue;
            }
            alias = 1;
            // (under a limit, "lookups as of a place": an image only later passes write holds nothing yet at this place)
            if (tex_alias && alias_compatible(e, p) && alias_has_gpu_content(p) && !(asof && p->wr_serial && !wr_at(p)) &&
                (!src || wr_at(p) > wr_at(src) ||
                 (wr_at(p) == wr_at(src) && same_but_swizzle(&p->key, &k) && !same_but_swizzle(&src->key, &k))))
                src = p;   // the newest GPU writer of the storage (as slot_resolve follows it)
        }
        if (alias) perf_count(PERF_CNT_TEX_ALIAS);
        uint64_t lt_replace = lead_t0 ? plat_now_ns() : 0;
        int how = 0;   // DEBUG-TEXLEAD: 1 alias copy, 2 adopted, 3 init
        if (tex_alias) {
            if (src && !up_worker && alias_copy(e, src) && (how = 1)) {
                n_alias_copied++;
                if (alog) alias_log("new image copied from", &k, src);
                if (e->pub.mirrored) {   // the content is src's GPU render, newer than the storage bytes
                    vk_mem_changed(e->pub.addr, e->t.L.size, 0, &e->wgen);
                    e->uploaded = 1;
                    e->pub.gpu_owned = 1;
                }
            } else if (adopt_released(e, t) && (how = 2)) {
                if (alog) alias_log("new image is a released one with the same bytes", &k, NULL);
            } else {
                how = 3;
                init_image(e, vk_upload_begin());
                vk_upload_end();
                if (alias) {
                    if (e->pub.mirrored) n_alias_upload++;
                    else n_alias_uninit++;
                    if (alog) alias_log(e->pub.mirrored ? "new image filled from storage" : "new image LEFT EMPTY", &k, src);
                    if (vk_dump_armed && !e->pub.mirrored && k.d.width == k.d.height && k.d.format == 0x32) {   // DEBUG-DUMPEMPTY
                        void vk_dump_arm_now(const char *why);
                        vk_dump_arm_now("the shadow layers' image at a new address");
                    }
                }
            }
        }
        bc1_pre.valid = 0;   // adopt_released took it, or no one did
        e->next_at = nvn_hm_get(&by_addr, &k.addr);
        nvn_hm_put(&by_addr, &k.addr, e);
        nvn_hm_put(&by_key, &k, e);
        tex_gen_bump(k.addr);
        sec_end(SEC_MADE_T + sec_mk, sec0);
        if (lead_t0) {   // DEBUG-TEXLEAD
            uint64_t now = plat_now_ns();
            lead_split(lead_t0, lt_bc1, cs_plan, cs_made, cs_tail, lt_replace, now, how, e->pub.mirrored);
            lead_made(e, lead_t0);
        }
    }
    if (e->released) e->released = 0, nreleased--, rel_unlink(e), tex_gen_bump_entry(k.addr, e->texq_base);
    e->t = *t;
    touch(e);
    return e;
}

// vk_dump: upload bookkeeping of an image vk_texture_get / vk_texture_by_id returned. pages_written: whether its
// storage pages were written since the last check (1, 0, -1 not tracked); the query leaves that check's state alone.
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written) {
    lock();
    const Tex *e = (const Tex *)img;   // pub is the first member
    uint32_t now;
    *uploads = e->n_up;
    *last_change = e->last_ch;
    *pages_written = e->pub.mirrored && e->uploaded ? vk_mem_changed(e->pub.addr, e->t.L.size, e->wgen, &now) : 0;
    unlock();
}

const VkTexImage *vk_texture_get(const tex_t *t) {
    if (!t || !t->mem) return NULL;
    lock();
    Tex *e = entry_of(t);
    if (e) make_fresh(e);
    {   // DEBUG-TEXWATCH
        TexKey k;
        make_key(t, &k);
        tex_watch("get", &k, e);
    }
    unlock();
    return e ? &e->pub : NULL;
}

// Render targets the CPU reads without NVN's invalidate (vk_texture_write_back_small); small_pending: one may be dirty.
// wb_limit: targets up to this size (bytes) always go back when the CPU waits for a fence, REKARINA_WB_FENCE_LIMIT, read
// once. A larger one goes back unless the GPU wrote it in each of the last WB_RUN frames (wb_wanted). wb_trace:
// REKARINA_TRACE=gfx, read together with it.
enum { WB_SMALL_DEFAULT = 64 << 10, WB_RUN = 16 };
static int small_pending;
static uint64_t wb_limit = WB_SMALL_DEFAULT;
static int wb_trace;
static int wb_cfg_read;

static void wb_cfg_init(void) {
    if (wb_cfg_read) return;
    int v = cfg_int("REKARINA_WB_FENCE_LIMIT");
    wb_limit = v < 0 ? 0 : (uint64_t)v;
    wb_trace = (rt_trace & RT_TRACE_GFX) != 0;
    wb_cfg_read = 1;
    if (wb_limit != WB_SMALL_DEFAULT) rt_log("vk: fence write-back limit %llu bytes", (unsigned long long)wb_limit);
}

// Whether the fence write-back takes a dirty target. A target the GPU draws every frame is scratch the CPU does not
// read: in the field three 256x256 RGBA8 targets and a 256x256 RG8 one, which cost 2.7 ms a frame to write back.
// TotK's 260x260 RG8 grass maps, which its fire spread reads after a fence wait (#132), are drawn once per change, or
// up to 9 frames in a row while a tile loads.
static int wb_wanted(const Tex *e) { return e->t.L.size <= wb_limit || e->run < WB_RUN; }

void vk_texture_gpu_written(const tex_t *t) {
    if (!t || !t->mem) return;
    {   // DEBUG-BIGLOG: REKARINA_BIGLOG=1: who writes 4096x4096 textures (the far shadow maps the fog reads), and when
        static int on = -1, k;
        if (on < 0) on = cfg_get("REKARINA_BIGLOG") != NULL;
        if (on && t->d.width >= 4096 && t->d.height >= 4096 && (k++ < 20 || !(k % 200)))
            rt_log("BIGLOG written#%d %p fmt %#x by %p (vk_texture_gpu_written at %p) frame %llu", k, (const void *)t->mem,
                   t->d.format, __builtin_return_address(0), (void *)vk_texture_gpu_written,
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    }
    {   // DEBUG-LUMLOG: who writes the 1x1 RGBA32F luminance targets (caller address, look it up in rekarina.map)
        static int on = -1, k;
        if (on < 0) on = cfg_get("REKARINA_LUMLOG") != NULL;
        if (on && t->d.format == 0x2e && t->d.width == 1 && t->d.height == 1 && (k++ < 6 || !(k % 300)))
            rt_log("LUMLOG written#%d %p by %p (vk_texture_gpu_written at %p) frame %llu pool_flags %#x mirrored %d", k,
                   (const void *)t->mem, __builtin_return_address(0), (void *)vk_texture_gpu_written,
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), (unsigned)t->pool_flags,
                   tex_mirrored(t));
    }
    vk_exec_frame_written(t);
    lock();
    Tex *e = entry_of(t);
    if (e) {
        make_fresh(e);   // a partial render keeps what the storage held before
        e->pub.gpu_owned = 1;
        uint64_t f = vk_exec_recording_fence();
        if (f > e->wr_fence) e->wr_fence = f;
        note_writer(e);
        if (e->pub.mirrored) {
            e->gpu_dirty = 1;
            storage_rewritten(e, 0, &e->gpu_gen);   // CPU writes up to here are older than this render
            if (wb_exact < 0) wb_exact = cfg_bool("REKARINA_WB_EXACT");
            if (wb_exact && !e->snap && e->t.L.size <= SNAP_MAX && fills_no_page(e) && !(e->snap = malloc(SNAP_MAX)))
                rt_fatal("vk: out of memory for a texture's storage copy");
            if (e->snap) memcpy(e->snap, (const void *)(uintptr_t)e->pub.addr, e->t.L.size);
            e->gpu_seq = ++gpu_writes;
            e->gpu_fence = vk_exec_recording_fence();
            if (mem_result_watch_on() && cfg_env("REKARINA_RESULT_WATCH_TEX") && e->t.L.size <= 65536)   // DEBUG-RESULTWATCH: who reads it before the write-back
                mem_result_watch(e->pub.addr, e->t.L.size, e->gpu_fence);
            uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
            if (fr != e->run_frame || !e->run) {
                e->run = fr == e->run_frame + 1 && e->run ? (uint8_t)(e->run < WB_RUN ? e->run + 1 : WB_RUN) : 1;
                e->run_frame = fr;
            }
            wb_cfg_init();
            if (wb_wanted(e)) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
            own_add(e);
        }
    }
    unlock();
}

static void release_key(const TexKey *k) {   // tex lock held
    Tex *e = nvn_hm_get(&by_key, k);
    if (e && !e->released) {
        e->released = 1;
        e->rel_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        nreleased++;
        rel_append(e);
        tex_gen_bump_entry(k->addr, e->texq_base);
        sweep_released();
    }
}

void vk_texture_release(const tex_t *t) {
    if (!t || !t->mem) return;
    TexKey k;
    make_key(t, &k);
    lock_off_core();
    release_key(&k);
    cold_released(&k);
    tex_t half;
    if (texq_mode > 0 && texq_shape(t, &half)) {   // "texture quality": the entry its slots resolved to
        make_key(&half, &k);
        release_key(&k);
        cold_released(&k);
    }
    unlock();
}

void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc) {
    pm_gone(lo, size, alloc, 0);
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && ((e->key.addr >= lo && e->key.addr - lo < size) || (alloc && e->bound_to == alloc))) list[n++] = e;
    }
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    if (cold_keys.n) {   // evicted textures whose storage goes with the pool
        TexKey *gone = malloc(sizeof *gone * cold_keys.n);
        uint32_t ng = 0;
        for (uint32_t i = 0; i < cold_keys.cap && gone; i++) {
            if (!cold_keys.vals[i]) continue;
            TexKey k;
            memcpy(&k, cold_keys.keys + (size_t)i * cold_keys.klen, sizeof k);
            if (k.addr >= lo && k.addr - lo < size) gone[ng++] = k;
        }
        for (uint32_t i = 0; i < ng; i++) cold_del(&gone[i]);
        free(gone);
    }
    unlock();
    up_range_gone();   // a fill in flight reads storage in the range no longer
    ts_range_gone(lo, size);   // nor a stream worker
}

// ---------------------------------------------------------------- descriptor slots
static uint64_t n_same_regs;   // vk_texture_register calls that changed nothing
uint64_t vk_texture_same_regs(void) { return __atomic_load_n(&n_same_regs, __ATOMIC_RELAXED); }
static int same_reg_on(void) {   // REKARINA_TEX_SAME_REG=0: every registration goes the full way
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_SAME_REG", "1");
    return on;
}

void vk_texture_register(int id, const tex_t *t, const view_t *v) {
    if (id < 0 || id >= (1 << 20)) rt_fatal("vk: texture slot %d out of range", id);
    uint64_t sec0 = sec_begin();
    lock_off_core();
    sec_end(SEC_TEXREG_LOCK, sec0);
    if (sec0) sec_count(SEC_TEXREG, 1);
    if (id >= ntslots) {
        int n = ntslots ? ntslots : 1024;
        while (n <= id) n *= 2;
        // A new table, the old one left as it is: a lock-free reader (vk_texture_by_id_fast) may still be in it. realloc
        // freed it when it moved the block, and a draw chunk's lookup read the freed table: an access violation in
        // vk_texture_by_id_fast at the village's first play frames (2026-11-18, the frame the table grew). A few MiB
        // in all; the table doubles a handful of times in a session. The table is published before its size, and the
        // reader takes the size first, so a size it sees always fits the table it reads.
        TexSlot *s = malloc(sizeof *s * (size_t)n);
        if (!s) rt_fatal("vk: out of memory for texture slots");
        if (ntslots) memcpy(s, tslots, sizeof *s * (size_t)ntslots);
        memset(s + ntslots, 0, sizeof *s * (size_t)(n - ntslots));
        __atomic_store_n(&tslots, s, __ATOMIC_RELEASE), __atomic_store_n(&ntslots, n, __ATOMIC_RELEASE);
    }
    if (!sinfo) {
        uint64_t *p = calloc(1 << 20, sizeof *p);
        if (!p) rt_fatal("vk: out of memory for texture slot tokens");
        __atomic_store_n(&sinfo, p, __ATOMIC_RELEASE);
    }
    TexSlot *s = &tslots[id];
    // The same texture and view registered again (the game rewrites much of its descriptor pool every frame): the slot
    // keeps its serial and resolved entry and slots_gen stands, so nothing that was resolved from it goes stale: the
    // pool refresh, the descriptor build skip and the shader memo all key on slots_gen. Only while that entry is live:
    // after a release, a registration is what makes a slot current again (slot_resolve's stale rule).
    // s->ce may point at a record destroy_tex let go of: it bumps the generation of the address and frees the record a
    // few submissions later, and the pointer stays in the slot. So the record is read only when it is known to be
    // there: the generation is the one it was resolved under, or (the generations share 65,536 counters, so another
    // address moves them too) it still is the entry of the slot's own key. Reading it unchecked was an access violation
    // in play once the heap had given the freed page back.
    int same = same_reg_on() && s->used && s->ce && !memcmp(&s->t, t, sizeof *t) && s->has_view == (v != NULL) &&
               (!v || !memcmp(&s->v, v, sizeof *v));
    if (same && s->cgen != agen[sinfo[id] & 0xffff]) {
        TexKey k;
        make_key(&s->t, &k);
        same = nvn_hm_get(&by_key, &k) == s->ce;
        tex_t half;
        if (!same && texq_mode > 0 && texq_shape(&s->t, &half)) {   // "texture quality": the slot resolved to the smaller one
            make_key(&half, &k);
            same = nvn_hm_get(&by_key, &k) == s->ce;
        }
    }
    if (__atomic_load_n(&ts_started, __ATOMIC_ACQUIRE) > 0 && t->mem) {   // the texture stream: one made already needs nothing
        TexKey k;
        make_key(t, &k);
        Tex *x = nvn_hm_get(&by_key, &k);
        if (x && x->uploaded && !x->released && !x->pub.gone) ts_resident(t);
    }
    if (same && !s->ce->released) {
        n_same_regs++;
        unlock();
        return;
    }
    if (cold_keys.n && s->used && s->t.mem != t->mem && t->mem && !memcmp(&s->t.d, &t->d, sizeof t->d)) {
        // the same descriptor at another address: the game moved the texture (its heap's compaction). An evicted one
        // stays evicted there, or the pool would make it resident again at every move.
        TexKey ko, kn;
        tex_t ho, hn;
        make_key(&s->t, &ko), make_key(t, &kn);
        void *kind = cold_or_just_gone(&ko);
        if (kind && !nvn_hm_get(&by_key, &kn)) cold_put(&kn, kind);
        if (texq_mode > 0 && texq_shape(&s->t, &ho) && texq_shape(t, &hn)) {   // "texture quality": the smaller one's key
            make_key(&ho, &ko), make_key(&hn, &kn);
            kind = cold_or_just_gone(&ko);
            if (kind && !nvn_hm_get(&by_key, &kn)) cold_put(&kn, kind);
        }
    }
    slot_wbegin(s);
    s->t = *t;
    if (!s->used) pool_slots++;
    s->used = 1;
    s->ce = NULL;
    slots_gen++;
    if (!++reg_serial) reg_serial = 1;
    __atomic_store_n(&sinfo[id], (uint64_t)reg_serial << 16 | agen_idx((uintptr_t)t->mem), __ATOMIC_RELEASE);
    vk_tpool_slot_registered(id);
    if (t->mem) up_queue(id, reg_serial);
    s->has_view = v != NULL;
    if (v) s->v = *v;
    else memset(&s->v, 0, sizeof s->v);
    slot_wend(s);
    unlock();
}

// The entry slot s resolves to, and the view and has_view a caller gets with it (the slot's, with the descriptor's
// swizzle when the entry is another descriptor at the same storage). NULL when the slot names no texture.
// The live entry at k's storage that the GPU wrote last; a same-but-swizzle entry wins a tie (both never written).
// cpu_content: the slot's bytes are the CPU's, laid out for its own descriptor: only GPU-owned entries qualify.
// fallback when none: own (may be NULL or released).
static Tex *newest_writer(const TexKey *k, Tex *own, int cpu_content);

// A GPU write into e was recorded (tex_lock held). It becomes the storage's newest writer (slot_resolve's "content
// follows the memory"); when another entry at the address held that place, the slots there resolve again.
static void note_writer(Tex *e) {
    int newest = 1;
    for (const Tex *p = nvn_hm_get(&by_addr, &e->key.addr); p && newest; p = p->next_at)
        newest = p == e || p->wr_serial < e->wr_serial;
    // a stretch handed out after the write being replaced may still look e up: that write stays known
    if (e->wr_serial && e->wr_serial <= __atomic_load_n(&asof_floor, __ATOMIC_RELAXED)) {
        memmove(e->wr_old + 1, e->wr_old, sizeof e->wr_old - sizeof e->wr_old[0]);
        e->wr_old[0] = e->wr_serial;
    }
    e->wr_serial = wr_serials + 1;
    __atomic_store_n(&wr_serials, e->wr_serial, __ATOMIC_RELEASE);
    __atomic_store_n(&awr[agen_idx(e->key.addr)], e->wr_serial, __ATOMIC_RELEASE);
    if (!newest) tex_gen_bump(e->key.addr);
}

static int asof_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_CHUNK_ASOF");
        on = !(e && *e == '0');
        if (!on) rt_log("vk: chunks of a draw stretch look textures up as the replay thread sees them (REKARINA_CHUNK_ASOF=0)");
    }
    return on;
}

// vk_exec.c par_try, before a stretch's chunks are handed out: the limit their lookups take. 0: none.
uint64_t vk_texture_asof_mark(void) {
    if (!asof_on()) return 0;
    uint64_t m = __atomic_load_n(&wr_serials, __ATOMIC_ACQUIRE);
    __atomic_store_n(&asof_floor, m, __ATOMIC_RELEASE);
    return m;
}

// This thread's lookups from now on: GPU writes recorded after limit do not count (0: all count). Returns the limit
// in force before.
uint64_t vk_texture_asof(uint64_t limit) {
    uint64_t was = asof;
    asof = limit;
    return was;
}
uint64_t vk_texture_asof_limit(void) { return asof; }

// 1 when slot id resolves for this thread as it does for the replay thread: no limit, or no GPU write after it at the
// slot's storage (by address bucket). Otherwise what a lookup returns holds for this thread's limit only and must not
// be kept. Read without the lock, as vk_texture_slot_token.
int vk_texture_asof_clean(int id) {
    if (__builtin_expect(!asof, 1) || __atomic_load_n(&wr_serials, __ATOMIC_ACQUIRE) <= asof) return 1;
    const uint64_t *si = __atomic_load_n(&sinfo, __ATOMIC_ACQUIRE);
    if (!si || id < 0 || id >= (1 << 20)) return 1;
    uint64_t v = __atomic_load_n(&si[id], __ATOMIC_ACQUIRE);
    return !v || __atomic_load_n(&awr[v & 0xffff], __ATOMIC_ACQUIRE) <= asof;
}

static Tex *newest_writer(const TexKey *k, Tex *own, int cpu_content) {
    Tex *best = NULL;
    for (Tex *p = nvn_hm_get(&by_addr, &k->addr); p; p = p->next_at) {
        if (p->released || (cpu_content && !p->pub.gpu_owned && !same_but_swizzle(&p->key, k))) continue;
        // Only an entry of the slot's own target: OoT's volumetric fog registers a 3D volume over the storage of a 2D
        // bloom target. Followed, no 3D view can be made of that image and the fog passes read and write stand-ins.
        if (p->key.d.target != k->d.target) continue;
        if (!best || wr_at(p) > wr_at(best) ||
            (wr_at(p) == wr_at(best) && same_but_swizzle(&p->key, k) && !same_but_swizzle(&best->key, k)))
            best = p;
    }
    return best ? best : own;
}

// ---- texture quality (REKARINA_TEX_QUALITY: auto, full, half)
// At 2560x1440 OoT holds about 6.5 GiB of video memory, 2.8 GiB of it the ASTC textures transcoded to BC3
// (twice to four times their size on the console). On a card with 8 GiB, or one another program holds part of, the
// system's budget for the game is less than that, Windows moves images to system memory and the frame rate falls to a
// fraction (13 fps for 104 with After Effects holding 7 GiB of a 12 GiB card, 2026-11-11).
// half: a texture slot that names a plain 2D ASTC texture of 512 texels or more with a mip chain resolves to the same
// texture from its second level on: a quarter of the memory, the picture of the next smaller level. The slot's
// descriptor is replaced by one of half the size and one level less whose storage starts where the second level lies;
// it is taken only when that descriptor's layout is byte for byte the tail of the full one (level offsets and block
// heights compared), so the upload reads the same bytes it would for those levels. A texture something else uses at
// its full size (a copy, a clear, a render target: it has an entry under its own descriptor) stays whole. Sampling
// with normalised coordinates sees no difference but the missing top level; a shader that asks for the size in texels
// gets half.
// quarter: the same from the third level on for textures of 1,024 texels and more (those of 512 lose one level), and
// the game's own block-compressed textures (BC1 to BC7) of 1,024 and more lose their first level too.
// auto (the default) decides by the card's own size, its device-local memory: 9.5 GiB or more (cards of 10 GiB and up):
// full; 7 GiB or more (8 GiB cards): half (about 1.1 GiB less than full at 1440p); less: quarter (about 1.5 GiB
// less), as the setup's texture preparation assumes. It used to decide by what other programs left free at the first
// texture, so a browser or an editor holding 5 GiB of a 12 GiB card at that moment lowered the picture for the whole
// session; other programs are normal on a desktop, and a real shortage is the video memory notice's business. A card
// whose size is not known (no device-local heap) keeps the old rule: the free memory, else the budget (half under
// 6.75 GiB). REKARINA_TEX_QUALITY_BY=free: the old rule on any card.
static int texq_mode = -1;   // 0 full, 1 half, 2 quarter
static uint64_t texq_n, texq_bytes;

static int texq_on(void) {
    if (texq_mode < 0) {
        const char *e = cfg_get("REKARINA_TEX_QUALITY");
        int m = e && !strcmp(e, "quarter") ? 2 : e && !strcmp(e, "half") ? 1 : e && !strcmp(e, "full") ? 0 : -1;
        double free_gib = -1;
        if (m < 0) {
            VkCtx *c = vk_ctx();
            if (!c) return 0;   // asked before the device exists: not decided yet
            uint64_t used = 0, budget = 0, total = 0;
            int exact = 0;
            vk_mem_vram(&used, &budget, &exact);
            for (uint32_t h = 0; h < c->mem_props.memoryHeapCount; h++)
                if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) total += c->mem_props.memoryHeaps[h].size;
            VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
            vkGetPhysicalDeviceProperties2(c->phys, &p2);
            float load = -1, sys_mb = -1;
            PlatLoad *l = plat_load_open(id.deviceLUIDValid ? id.deviceLUID : NULL, -1);
            if (l) {
                plat_gpu_load(l, &load, &sys_mb);
                plat_load_close(l);
            }
            const char *by = cfg_get("REKARINA_TEX_QUALITY_BY");
            int by_free = by && !strcmp(by, "free");
            if (vk_mem_vram_cap()) {   // DEBUG-VRAMCAP: the capped budget stands for the card
                free_gib = (double)vk_mem_vram_cap() / 1073741824.0;
                m = free_gib >= 7.25 ? 0 : free_gib >= 6.2 ? 1 : 2;
            } else if (total && !by_free) {
                double card = (double)total / 1073741824.0;
                m = card >= 9.5 ? 0 : card >= 7.0 ? 1 : 2;
                texq_mode = m;
                rt_log("vk: texture quality %s by the card's %.1f GiB of video memory (REKARINA_TEX_QUALITY: auto, full, half, "
                       "quarter)", m == 0 ? "full" : m == 1 ? "half" : "quarter", card);
                return texq_mode;
            } else if (sys_mb >= 0 && total) {
                double others = sys_mb * 1048576.0 - (double)used;
                free_gib = ((double)total - (others > 0 ? others : 0)) / 1073741824.0;
                m = free_gib >= 7.25 ? 0 : free_gib >= 6.2 ? 1 : 2;
            } else {
                m = exact && budget && budget < (27ull << 28);
                free_gib = budget / 1073741824.0;
            }
        }
        texq_mode = m;
        static const char *const name[3] = {"full", "half: ASTC textures of 512 texels and more load from their second level",
                                            "quarter: ASTC textures of 1,024 texels and more load from their third level, of 512 and "
                                            "BC textures of 1,024 from their second"};
        if (free_gib >= 0)
            rt_log("vk: texture quality %s (%.1f GiB of video memory are free for the game; REKARINA_TEX_QUALITY: auto, full, half, "
                   "quarter)", name[m], free_gib);
        else
            rt_log("vk: texture quality %s (REKARINA_TEX_QUALITY)", name[m]);
    }
    return texq_mode;
}

// The descriptor of t without its first level (texq_shape: without as many as the quality drops).
static int texq_shape1(const tex_t *t, tex_t *out) {
    const nvn_texdesc *d = &t->d;
    if (d->target != NVN_TEXTURE_TARGET_2D || d->levels < 2 || d->samples > 1 || (d->width < 512 && d->height < 512)) return 0;
    if (!t->mem || nvn_gpu_only((uintptr_t)t->mem) || nvn_tex_is_linear(d)) return 0;
    nvn_texdesc d2 = *d;
    d2.width = umax(d->width >> 1, 1), d2.height = umax(d->height >> 1, 1), d2.levels = d->levels - 1;
    uint64_t off1 = nvn_tex_level_offset(d, 1);
    if (!off1) return 0;
    for (unsigned l = 0; l < d2.levels; l++) {
        unsigned th, td, th2, td2;
        nvn_tex_level_tiles(d, l + 1, &th, &td);
        nvn_tex_level_tiles(&d2, l, &th2, &td2);
        if (th != th2 || td != td2 || nvn_tex_level_offset(&d2, l) != nvn_tex_level_offset(d, l + 1) - off1) return 0;
    }
    *out = *t;
    out->d = d2;
    nvn_tex_layout(&out->d, &out->L);
    if (off1 + out->L.size > t->L.size) return 0;
    out->mem = t->mem + off1, out->off = t->off + (int64_t)off1;
    return 1;
}

// The descriptor the quality in force gives t; the levels dropped, 0 when t stays whole.
static int texq_shape(const tex_t *t, tex_t *out) {
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f)) return 0;
    int astc = f.astc_bw != 0;
    int bc = !astc && f.vk >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f.vk <= VK_FORMAT_BC7_SRGB_BLOCK && texq_mode == 2 &&
             (t->d.width >= 1024 || t->d.height >= 1024);
    if ((!astc && !bc) || !texq_shape1(t, out)) return 0;
    tex_t two;
    if (astc && texq_mode == 2 && texq_shape1(out, &two)) {
        *out = two;
        return 2;
    }
    return 1;
}

// Two descriptors whose images hold the same picture: target, format, size, levels and samples (flags, layout, stride
// and swizzle may differ: the images are ours, not the storage).
static int same_shape(const TexKey *a, const TexKey *b) {
    const nvn_texdesc *x = &a->d, *y = &b->d;
    return a->addr == b->addr && x->target == y->target && x->format == y->format && x->width == y->width &&
           x->height == y->height && x->depth == y->depth && x->levels == y->levels &&
           (x->samples > 1) == (y->samples > 1) && (x->samples <= 1 || x->samples == y->samples);
}

// REKARINA_FOLLOW_SHAPE=0: a live descriptor once written keeps its own image (slot_resolve), as before 2026-11-15.
static int follow_same_shape(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_FOLLOW_SHAPE");
        on = !(e && *e == '0');
        if (!on) rt_log("vk: a written descriptor keeps its own image when one of the same shape is written later (REKARINA_FOLLOW_SHAPE=0)");
    }
    return on;
}

// An entry whose image can stand for a slot's descriptor: the same size, the same depth or layer count, one sample or
// several alike, and a format of the same texel size and block (an R11G11B10F target read through an RGB9E5
// descriptor, a depth image through an R32F one). Another size at the storage is the game reusing the memory: a
// lookup of that image with the slot's coordinates reads the wrong texels. REKARINA_FOLLOW_ANY=1: any entry, as before
// 2026-11-15.
static int fits_slot(const TexKey *p, const TexKey *k) {
    static int any = -1;
    if (any < 0) { const char *e = cfg_get("REKARINA_FOLLOW_ANY"); any = e && *e == '1'; }
    if (any) return 1;
    const nvn_texdesc *a = &p->d, *b = &k->d;
    if (a->width != b->width || a->height != b->height || a->depth != b->depth || (a->samples > 1) != (b->samples > 1))
        return 0;
    int aw, ah, bw, bh, ab = nvn_fmt_bpb(a->format, &aw, &ah), bb = nvn_fmt_bpb(b->format, &bw, &bh);
    if (!ab || ab != bb || aw != bw || ah != bh) return 0;
    // A depth image stands for a colour descriptor only when that descriptor has one channel (an R32F one reads the
    // depth values). OoT's own edge smoothing at the end of the frame clears a 640x360 R16G16_UINT edge mask at the
    // storage of a 640x360 D32F target drawn earlier in the frame; one compute pass stores the mask (16 bits per 4x4
    // tile and direction), the next turns it into blend weights, a third blends the scene with them. Followed to the
    // depth image (four bytes a texel both), the store had no storage view of it and went into a stand-in in every
    // frame, the weight pass read the depth image instead, and the weights came out zero: the smoothing did nothing.
    // REKARINA_FOLLOW_DEPTH_ANY=1: as before.
    TexVkFormat fa, fb;
    if (!tex_vk_format(a->format, &fa) || !tex_vk_format(b->format, &fb)) return 1;
    int da = fa.depth || fa.stencil, db = fb.depth || fb.stencil;
    if (da == db) return 1;
    static int depth_any = -1;
    if (depth_any < 0) depth_any = rt_knob("FOLLOW_DEPTH_ANY", "0") != 0;
    VkFormat col = da ? fb.vk : fa.vk;
    return depth_any || (col >= VK_FORMAT_R8_UNORM && col <= VK_FORMAT_R8_SRGB) ||
           (col >= VK_FORMAT_R16_UNORM && col <= VK_FORMAT_R16_SFLOAT) || (col >= VK_FORMAT_R32_UINT && col <= VK_FORMAT_R32_SFLOAT);
}

// newest_writer among the entries that fit the slot's descriptor k
static Tex *newest_writer_fit(const TexKey *k, Tex *own, int cpu_content) {
    Tex *best = NULL;
    for (Tex *p = nvn_hm_get(&by_addr, &k->addr); p; p = p->next_at) {
        if (p->released || (cpu_content && !p->pub.gpu_owned && !same_but_swizzle(&p->key, k))) continue;
        if (p->key.d.target != k->d.target || !fits_slot(&p->key, k)) continue;
        if (!best || wr_at(p) > wr_at(best) ||
            (wr_at(p) == wr_at(best) && same_but_swizzle(&p->key, k) && !same_but_swizzle(&best->key, k)))
            best = p;
    }
    return best ? best : own;
}

static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view) {
    int no_make = 0;   // the pool's lookup leaves the slot unresolved rather than make an image (below)
    *view = s->v;
    *has_view = s->has_view;
    TexKey k;
    make_key(&s->t, &k);
    Tex *e = nvn_hm_get(&by_key, &k);
    tex_t half;
    int drops = (!e || e->released) && texq_on() ? texq_shape(&s->t, &half) : 0;
    if (drops) {   // "texture quality": the slot's texture, smaller
        TexKey hk;
        make_key(&half, &hk);
        Tex *he = nvn_hm_get(&by_key, &hk);
        if ((!he || he->released) && (cold_kept_out(&hk, half.L.size) || pool_holds_back(&half, (int)(s - tslots)))) return NULL;
        if (!he || he->released) {
            he = entry_of(&half);
            if (he) {
                texq_n++, texq_bytes += s->t.L.size - half.L.size;
                if (!(texq_n & 1023))
                    rt_log("vk: texture quality: %llu textures made smaller so far, %.0f MiB of ASTC left out",
                           (unsigned long long)texq_n, texq_bytes / 1048576.0);
            }
        }
        if (he) {
            {   // tex_gen_bump_entry: this slot hears when he goes. A texture and its own second level registered as a
                // texture of its own reach the same smaller entry from two addresses.
                uint64_t base = (uintptr_t)s->t.mem;
                int at = -1;
                for (int i = 0; i < 3 && at != -2; i++)
                    if (he->texq_base[i] == base) at = -2;
                    else if (!he->texq_base[i] && at < 0) at = i;
                if (at == -1) at = 0, tex_gen_bump(he->texq_base[0]);   // full: the one replaced resolves again
                if (at >= 0) he->texq_base[at] = base;
            }
            if (*has_view && view->set_levels)
                view->base_level = (uint8_t)(view->base_level > drops ? view->base_level - drops : 0);
            tex_watch("slot", &hk, he);   // DEBUG-TEXWATCH
            return he;
        }
    }
    if (!e || e->released) {   // MNG hle_nvn.c: a finalized slot follows the live texture at its storage
        // Prefer the live entry that differs from the slot descriptor only in its swizzle: transient render targets
        // share storage, and the newest one at the address is usually another size. The map screen samples its
        // 2250x1080 map target through an (R,G,B,1) descriptor and followed a 960x60 target at the same address.
        // A texture whose content the CPU writes (tex_mirrored) follows another descriptor only when the GPU rendered
        // that one's content: otherwise the bytes are in the storage, laid out for the slot's own descriptor. OoT's
        // subtitles rasterize each line on the CPU into a linear R8 strip at the same storage; a line of another size
        // followed the previous line's 256x45 entry, read its 1300-texel rows 256 at a time and came out as stripes.
        // Content follows the memory, as on the console: of the live entries there, the one the GPU wrote last (a
        // same-but-swizzle one wins a tie). The first live entry of the list was often a target of another size
        // that this frame had not rendered yet, and the whole 3D scene came out black for a frame (the composite
        // read it: one black frame every 600-1,300, flight.c flash watch).
        // Only an entry whose image fits the slot's descriptor (fits_slot): when the game moves its transient targets,
        // an R8 1280x720 mask of a compute chain had its own entry released and followed an RGBA8 2560x1440 target
        // written later at that storage; the chain wrote and read its 1280x720 grid in the corner of the larger
        // image, the sun-shadow pass took a quarter-size copy of the scene for its mask, and the lit objects lost
        // their sunlight for several frames (2026-11-15).
        int cpu_content = tex_mirrored(&s->t);
        Tex *own = e;
        e = newest_writer_fit(&k, e, cpu_content);
        // The pool's lookup of a slot whose own entry is gone, at storage only entries of another shape hold: no new
        // image (the stale slots the pool resolves would be made again after every sweep); a draw that binds the slot
        // makes it. (pool_slot_locked leaves the slot unresolved meanwhile.)
        if (!e && pool_lookup && newest_writer(&k, NULL, cpu_content)) no_make = 1;
        {   // DEBUG-FITLOG: REKARINA_FITLOG=1: lookups the fit rule sent to the slot's own entry (or a new one) where any
            // entry at the storage would have been followed before
            static int on = -1, nl;
            if (on < 0) on = cfg_get("REKARINA_FITLOG") != NULL;
            Tex *any = on && e == own && nl < 3000 ? newest_writer(&k, NULL, cpu_content) : NULL;
            if (any && any != e) {
                nl++;
                rt_log("FITLOG frame %llu addr %#llx slot %ux%u fmt %#x tgt %d -> %s; other %ux%u fmt %#x wr %llu",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                       (unsigned long long)k.addr, k.d.width, k.d.height, k.d.format, k.d.target,
                       !own ? "a new entry" : own->released ? "its released entry" : "its entry", any->key.d.width,
                       any->key.d.height, any->key.d.format, (unsigned long long)wr_at(any));
            }
        }
    } else if (!wr_at(e) && !e->pub.mirrored) {
        // A live descriptor the GPU never wrote, at storage another descriptor rendered: it reads that render (a new
        // descriptor over a transient target the game rendered through another one), when its image fits.
        Tex *w = newest_writer_fit(&k, e, 0);
        if (w && wr_at(w)) e = w;
    } else if (!e->pub.mirrored && __atomic_load_n(&awr[agen_idx(e->key.addr)], __ATOMIC_ACQUIRE) > wr_at(e) &&
               follow_same_shape()) {   // (the bucket's newest write first: most lookups have nothing newer to walk to)
        // A live descriptor the GPU wrote some frames ago, at storage that another descriptor of the same shape (format,
        // size, levels) was written through since: it reads that newer write. When the game moves its transient
        // targets to other addresses (a frame that switches an effect on or off), the sun-shadow pass reads an R8
        // 1280x720 mask that a compute pass of the same frame wrote through a second descriptor at the same storage;
        // the reader kept its own image, filled frames earlier at another layout, and the objects lost their sunlight
        // for several frames (the dark frames while the camera turns, 2026-11-15). Only the same shape: a target of
        // another size written later at the address is the game's reuse of the memory, not this content.
        Tex *w = newest_writer(&k, e, 0);
        if (w && w != e && wr_at(w) > wr_at(e) && same_shape(&w->key, &e->key)) e = w;
    }
    {   // DEBUG-SPIKE: entries made while a slot resolves
        uint64_t spk0 = plat_now_ns();
        int made = !e && s->t.mem;
        int mk0 = sec_mk;   // not the pool: a draw's binding
        if (sec_on && !mk0) sec_mk = 1;
        if (!e && !no_make)
            e = s->t.mem && !cold_kept_out(&k, s->t.L.size) && !pool_holds_back(&s->t, (int)(s - tslots)) ? entry_of(&s->t) : NULL;
        if (sec_on) sec_mk = mk0;
        if (made) vk_spike.tex += plat_now_ns() - spk0, vk_spike.ntex++;
    }
    tex_watch("slot", &k, e);   // DEBUG-TEXWATCH
    // A followed texture is another descriptor at the same storage: its image holds the content, but sampling takes the
    // slot descriptor's swizzle. The UI renders text into RGBA8 and samples it through a (A,A,A,A) descriptor; with the
    // render target's RGBA swizzle the text blur saw R = 1 everywhere and drew solid boxes behind tips and labels.
    if (e && !(s->has_view && s->v.set_swizzle)) {
        const nvn_texdesc *d = &s->t.d;
        uint8_t sw[4] = {(uint8_t)d->swizzle_r, (uint8_t)d->swizzle_g, (uint8_t)d->swizzle_b, (uint8_t)d->swizzle_a};
        if (!(sw[0] | sw[1] | sw[2] | sw[3]))   // descriptors built without nvnTextureBuilderSetDefaults
            sw[0] = NVN_TEXTURE_SWIZZLE_R, sw[1] = NVN_TEXTURE_SWIZZLE_G, sw[2] = NVN_TEXTURE_SWIZZLE_B,
            sw[3] = NVN_TEXTURE_SWIZZLE_A;
        if (memcmp(sw, e->pub.swz, 4)) {
            if (!s->has_view) memset(view, 0, sizeof *view);
            memcpy(view->swz, sw, 4);
            view->set_swizzle = 1;
            *has_view = 1;
        }
    }
#ifdef GAME_NS2
    // The same for the format: OoT samples the RGB9E5 words its 3D frame packed into an R11G11B10F image through an
    // RGB9E5 descriptor at that storage, which follows the R11G11B10F entry. The slot's format goes in as a view format
    // (vk_exec_draw.c reinterpret_format makes the view when both have one texel size).
    if (e && e->t.d.format != s->t.d.format && !(*has_view && view->set_format)) {
        if (!*has_view) memset(view, 0, sizeof *view);
        view->format = s->t.d.format;
        view->set_format = 1;
        *has_view = 1;
    }
#endif
    return e;
}

// ---- uploads ahead of first use (REKARINA_TEX_PREP=1; off by default)
// A texture reaches the GPU at its first use: the pool set's update, or a draw's binding, creates its image and fills
// it, a file read or a copy out of the game's memory of up to 14 MiB, on the replay thread. Arriving on a map the game
// registers textures for about three seconds (15 new ones a frame, 1 to 5 ms of copies and as much again in the pool
// update: 55 to 65 fps where the place then runs at 95), and running through the Market it brings in 45 MiB a
// second the same way (the frames of 20 ms and more there). The game names a texture in a pool slot some time before a
// frame draws with it, so a thread of its own takes each newly registered slot, resolves it as the pool update would
// and uploads it, with the texture lock released while the bytes are read (upload: fill_enter). The replay then finds
// the texture there. What the thread does not do stays as it was: the GPU's ASTC decode of a texture the cache does not
// hold, render targets, and whatever the replay reaches first.
// Off by default (2026-11-13): the 180 frames after an arrival in the field went from 18.4 to 17.4 ms and running
// was the same, and of six two-minute runs of the field with it one lost the device (a read of an image destroyed ten
// seconds before; eight runs without it were clean). docs/oot-notes.md, "Uploads ahead of first use".
#define UPQ_N 8192
static struct { uint32_t id, serial; } upq[UPQ_N];
static uint32_t upq_head, upq_tail;   // upq_lock
static PlatMutex upq_lock = PLAT_MUTEX_INIT;
static PlatEvent upq_ev;
static volatile int32_t up_started;
static uint64_t up_done, up_dropped;

static int up_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = cfg_env("REKARINA_TEX_PREP");
        on = v && *v == '1';
    }
    return on;
}

static void up_range_gone(void) {   // no fill reads a pool's memory once this returns (the entries there are gone)
    if (!__atomic_load_n(&up_started, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&fill_lock);
    plat_mutex_unlock(&fill_lock);
}

static Tex *slot_resolve(TexSlot *s, view_t *view, int *has_view);

static void up_main(void *arg) {
    (void)arg;
    up_worker = 1;
    sec_thread_off = 1;
    for (;;) {
        uint32_t id = 0, serial = 0;
        int have = 0;
        plat_mutex_lock(&upq_lock);
        if (upq_head != upq_tail) id = upq[upq_head % UPQ_N].id, serial = upq[upq_head % UPQ_N].serial, upq_head++, have = 1;
        plat_mutex_unlock(&upq_lock);
        if (!have) {
            plat_event_wait(&upq_ev);
            continue;
        }
        lock();
        if ((int)id < ntslots && tslots[id].used && sinfo && (uint32_t)(sinfo[id] >> 16) == serial) {
            TexSlot *s = &tslots[id];
            VkMemSpan m;
            if (tex_mirrored(&s->t) && vk_mem_resolve((uintptr_t)s->t.mem, s->t.L.size ? s->t.L.size : 1, &m)) {
                // the storage's hash, which a new entry needs, with the lock released (as upload's fill)
                TexKey k;
                make_key(&s->t, &k);
                up_pre.valid = 0;
                if (!nvn_hm_get(&by_key, &k) && s->t.L.size >= 4096) {
                    uint64_t addr = (uintptr_t)s->t.mem, size = s->t.L.size;
                    fill_enter();
                    up_pre.addr = addr, up_pre.size = size;
                    up_pre.cprint = content_print((const uint8_t *)(uintptr_t)addr, size);
                    up_pre.chash = content_hash((const uint8_t *)(uintptr_t)addr, size);
                    lock();
                    plat_mutex_unlock(&fill_lock);
                    up_pre.valid = 1;
                }
                // (the slot table may have moved and the slot been registered again meanwhile)
                if ((int)id < ntslots && tslots[id].used && (uint32_t)(sinfo[id] >> 16) == serial &&
                    vk_mem_resolve((uintptr_t)tslots[id].t.mem, tslots[id].t.L.size ? tslots[id].t.L.size : 1, &m)) {
                    view_t v;
                    int hv;
                    Tex *e = slot_resolve(&tslots[id], &v, &hv);
                    if (e && !e->pub.gone && !e->uploaded) make_fresh(e);
                    up_done++;
                }
                up_pre.valid = 0;
            }
        }
        unlock();
    }
}

// vk_texture_register, lock held: slot id names another texture from now on.
static void up_queue(int id, uint32_t serial) {
    if (!up_on() || !vk_ctx()) return;
    if (!__atomic_load_n(&up_started, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&upq_lock);
        if (!up_started) {
            plat_event_init(&upq_ev, 0, 0);
            if (plat_thread_spawn(up_main, NULL, "vk uploads ahead", 0)) {
                __atomic_store_n(&up_started, 1, __ATOMIC_RELEASE);
                rt_log("vk: textures are uploaded ahead of their first use by a thread of their own (REKARINA_TEX_PREP=1)");
            } else {
                __atomic_store_n(&up_started, -1, __ATOMIC_RELEASE);
            }
        }
        plat_mutex_unlock(&upq_lock);
    }
    if (up_started <= 0) return;
    plat_mutex_lock(&upq_lock);
    if (upq_tail - upq_head < UPQ_N) upq[upq_tail % UPQ_N].id = (uint32_t)id, upq[upq_tail % UPQ_N].serial = serial, upq_tail++;
    else up_dropped++;
    plat_mutex_unlock(&upq_lock);
    plat_event_set(&upq_ev);
}

#undef vk_texture_lock
void vk_texture_lock(void) { lock_in("vk_texture_lock"); }
void vk_texture_lock_from(const char *fn) { lock_in(fn); }
int vk_texture_lock_held(void) { return __atomic_load_n(&tex_lock.owner, __ATOMIC_RELAXED) == plat_thread_id(); }   // DEBUG-VIEWLOCK
void vk_texture_unlock(void) { unlock(); }
uint32_t vk_texture_slots_gen(void) { return slots_gen; }
int vk_texture_slot_count(void) { return __atomic_load_n(&ntslots, __ATOMIC_RELAXED); }

// vk_texpool.c: vk_texture_by_id_locked for a slot nothing binds, only while its storage lies in a live memory pool.
// The game leaves slots registered after it finalized their pool; a draw never names those, but resolving one would
// upload from freed memory (an ASTC decode read unmapped pages in the forest).
static const VkTexImage *pool_slot_locked(int id, view_t *view, int *has_view);
const VkTexImage *vk_texture_pool_slot_locked(int id, view_t *view, int *has_view) {
    if (__builtin_expect(sec_on, 0)) {
        sec_mk = 2;
        const VkTexImage *r = pool_slot_locked(id, view, has_view);
        sec_mk = 0;
        return r;
    }
    return pool_slot_locked(id, view, has_view);
}

static const VkTexImage *pool_slot_locked(int id, view_t *view, int *has_view) {
    if (id < 0 || id >= ntslots || !tslots[id].used) return NULL;
    const tex_t *t = &tslots[id].t;
    if (!t->mem) return NULL;
    VkMemSpan m;
    if (!nvn_gpu_only((uintptr_t)t->mem) && !vk_mem_resolve((uintptr_t)t->mem, t->L.size ? t->L.size : 1, &m)) return NULL;
    // REKARINA_TPOOL_NO_GPU=0 turns this off. A slot of a GPU-only target with no live image gets a stand-in. The pool otherwise makes
    // a new image for every registered slot, so targets dropped or recycled came back as permanent live images and memory never fell.
    static int no_gpu = -1;
    if (no_gpu < 0) { const char *s = cfg_env("REKARINA_TPOOL_NO_GPU"); no_gpu = !(s && s[0] == '0'); }
    if (no_gpu && nvn_gpu_only((uintptr_t)t->mem)) {
        TexKey k;
        make_key(t, &k);
        Tex *e = nvn_hm_get(&by_key, &k);
        if ((!e || e->released) && !newest_writer(&k, NULL, 0)) return NULL;
    }
    pool_lookup = 1;   // not a use: cold_kept_out, drawn
    const VkTexImage *r = vk_texture_by_id_locked(id, view, has_view);
    pool_lookup = 0;
    return r;
}

int vk_texture_pool_class(const VkTexImage *img) {
    const Tex *e = (const Tex *)img;   // pub is the first member
    if (!comes_back_intact(e)) return 0;
    if (e->pub.layers > 1 || e->pub.depth > 1) return 4;
    return (e->f.astc_bc ? 1 : 2) | (e->ever_drawn ? 8 : 0);
}

uint64_t vk_texture_slot_token(int id) {
    const uint64_t *si = __atomic_load_n(&sinfo, __ATOMIC_ACQUIRE);
    if (!si || id < 0 || id >= (1 << 20)) return 0;
    uint64_t v = __atomic_load_n(&si[id], __ATOMIC_ACQUIRE);
    return v ? (v >> 16) << 32 | __atomic_load_n(&agen[v & 0xffff], __ATOMIC_ACQUIRE) : 0;
}
uint32_t vk_sampler_gen(void) { return __atomic_load_n(&samp_gen, __ATOMIC_RELAXED); }

// vk_exec_draw.c, when a draw's handle is bound: slot id's cached resolution and registration word are asked into the
// cache for the lookups of the draw that follows (vk_texture_by_id_locked, vk_texture_slot_token). The tables are
// megabytes and a frame's handles are scattered over them. Read without the lock: a table moved meanwhile only wastes
// the prefetch, which reads nothing.
void vk_texture_slot_prefetch(int id) {
    const TexSlot *s = __atomic_load_n(&tslots, __ATOMIC_RELAXED);
    const uint64_t *si = __atomic_load_n(&sinfo, __ATOMIC_RELAXED);
    if ((unsigned)id >= (unsigned)__atomic_load_n(&ntslots, __ATOMIC_RELAXED)) return;
    if (s) __builtin_prefetch(&s[id]);
    if (si) __builtin_prefetch(&si[id]);
}

// vk_texpool.c: the slots' registration words, NULL before the first registration. Bits 16 and up of word id are the
// slot's registration serial, the upper half of vk_texture_slot_token(id); the pool's update compares only that half,
// for every slot of the pool at every submission, and the token's other half costs a second, scattered load.
const uint64_t *vk_texture_slot_regs(void) { return __atomic_load_n(&sinfo, __ATOMIC_ACQUIRE); }

const VkTexImage *vk_texture_by_id(int id, view_t *view, int *has_view) {
    lock();
    const VkTexImage *r = vk_texture_by_id_locked(id, view, has_view);
    unlock();
    return r;
}

const VkTexImage *vk_texture_by_id_locked(int id, view_t *view, int *has_view) {
    if (id < 0 || id >= ntslots || !tslots[id].used) return NULL;
    TexSlot *s = &tslots[id];
    if (__builtin_expect(asof != 0, 0) && awr[sinfo[id] & 0xffff] > asof) {
        // "lookups as of a place": the GPU wrote at the slot's storage (or at an address of its bucket) after this
        // thread's limit. Resolved for the limit and not kept: the slot's kept resolution is the replay thread's.
        view_t v;
        int hv;
        Tex *e = slot_resolve(s, &v, &hv);
        if (e) {
            if (!e->uploaded || e->checked != submit_epoch) {
                make_fresh_snap(e);
                e->checked = submit_epoch;
                touch(e);
            }
            if (!pool_lookup) drawn(e), e->ever_drawn = 1;
        }
        if (!(++n_asof & 0xfffff))
            rt_log("vk: %llu texture lookups of draw stretches so far resolved as of their stretch's place (REKARINA_CHUNK_ASOF)",
                   (unsigned long long)n_asof);
        if (view) *view = has_view ? v : s->v;
        if (has_view) *has_view = view ? hv : s->has_view;
        return e ? &e->pub : NULL;
    }
    uint32_t g = agen[sinfo[id] & 0xffff];
    if (!s->ce || s->cgen != g) {   // resolved again: an entry at its address was created, dropped or released, or the
        view_t v;                   // slot registered (s->ce cleared); an empty result is not kept: image creation may
        int hv;                     // succeed later
        uint32_t creg = (uint32_t)(sinfo[id] >> 16);
        Tex *ne = slot_resolve(s, &v, &hv);   // entry_of may create an entry and bump g: g read before it
        slot_wbegin(s);
        s->ce = ne;
        s->cview = v, s->chas_view = (uint8_t)hv;
        s->cgen = g, s->creg = creg;
        s->cepoch = 0;
        slot_wend(s);
    }
    if (s->ce && s->ce->pub.gone) {   // must not happen (the rule above tex_gen_bump_entry): resolved again, and said
        static volatile int32_t n_gone;
        log_limited(&n_gone, "vk: slot %d kept a destroyed entry (texture %#llx %ux%u, entry at %#llx %ux%u, %u levels): resolved again",
                    id, (unsigned long long)(uintptr_t)s->t.mem, s->t.d.width, s->t.d.height, (unsigned long long)s->ce->key.addr,
                    s->ce->key.d.width, s->ce->key.d.height, s->ce->key.d.levels);
        view_t v;
        int hv;
        Tex *ne = slot_resolve(s, &v, &hv);
        slot_wbegin(s);
        s->ce = ne;
        s->cview = v, s->chas_view = (uint8_t)hv;
        s->cgen = agen[sinfo[id] & 0xffff], s->creg = (uint32_t)(sinfo[id] >> 16);
        s->cepoch = 0;
        slot_wend(s);
    }
    Tex *e = s->ce;
    // Bindings of one submit share one check, answered from the submit's write-watch snapshot (guest writes during a
    // replay race the GPU read on hardware too). After one check an entry is either uploaded or never uploads (not
    // mirrored, GPU-owned or no CPU layout), so a slot that checked its entry in this submit has nothing left to do.
    if (e && s->cepoch != submit_epoch) {
        round_checks++;
        if (!e->uploaded || e->checked != submit_epoch) {
            make_fresh_snap(e);
            e->checked = submit_epoch;
            touch(e);
        }
        slot_wbegin(s), s->cepoch = submit_epoch, slot_wend(s);
    }
    if (e && !pool_lookup && s->depoch != submit_epoch) {   // a draw's first lookup through the slot in this submission
        slot_wbegin(s), s->depoch = submit_epoch, slot_wend(s);
        draw_checks++;
        drawn(e);
        e->ever_drawn = 1;
    }
    if (view) *view = has_view ? s->cview : s->v;
    if (has_view) *has_view = view ? s->chas_view : s->has_view;
    return e ? &e->pub : NULL;
}

// The lookup above without the texture lock (REKARINA_TEX_FAST, on by default; 0: never): *hit 1 when slot id's kept
// resolution is current and was already checked and noted as drawn in this submission, which is all a draw binding's
// lookup would do with it: then the same image and view the locked lookup would return. *hit 0 (the caller takes the
// lock and asks vk_texture_by_id_locked) when anything is due: a new registration (sinfo's serial), an entry made,
// dropped or released at its address (agen), a check or a "drawn" mark not yet done in this submission, a chunk's
// "lookups as of a place". The slot is read under its sequence word. An entry destroy_tex lets go keeps its record and
// image until the GPU is past the submission being recorded (vk_defer_free), as the locked lookup's result would.
// At the Market's square the replay's draw chunks spent most of their texture-lock time waiting on each other's
// lookups (feedback_scan, answers_for; REKARINA_LOCKMAP, 2026-11-18).
const VkTexImage *vk_texture_by_id_fast(int id, view_t *view, int *has_view, int *hit) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_FAST", "1") != 0;
    *hit = 0;
    if (!on) return NULL;
    int n = __atomic_load_n(&ntslots, __ATOMIC_ACQUIRE);   // the size first: a table published with it is at least as large
    TexSlot *arr = __atomic_load_n(&tslots, __ATOMIC_ACQUIRE);
    const uint64_t *si = __atomic_load_n(&sinfo, __ATOMIC_ACQUIRE);
    if (id < 0 || !arr || !si || id >= n) return NULL;
    TexSlot *s = &arr[id];
    uint32_t q = __atomic_load_n(&s->seq, __ATOMIC_ACQUIRE);
    if (q & 1) return NULL;
    Tex *e = s->ce;
    uint32_t cgen = s->cgen, cep = s->cepoch, dep = s->depoch, creg = s->creg;
    uint8_t used = s->used, chv = s->chas_view, shv = s->has_view;
    view_t cv = s->cview, sv = s->v;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (__atomic_load_n(&s->seq, __ATOMIC_RELAXED) != q) return NULL;
    uint64_t w = __atomic_load_n(&si[id], __ATOMIC_ACQUIRE);
    uint32_t ep = __atomic_load_n(&submit_epoch, __ATOMIC_RELAXED);
    if (__builtin_expect(asof != 0, 0) && __atomic_load_n(&awr[w & 0xffff], __ATOMIC_ACQUIRE) > asof) return NULL;   // as of a place
    if (!used || !e || (uint32_t)(w >> 16) != creg || cgen != __atomic_load_n(&agen[w & 0xffff], __ATOMIC_ACQUIRE) ||
        cep != ep || dep != ep || __atomic_load_n(&e->pub.gone, __ATOMIC_ACQUIRE))
        return NULL;
    *hit = 1;
    if (view) *view = has_view ? cv : sv;
    if (has_view) *has_view = view ? chv : shv;
    return &e->pub;
}

// vk_dump: the swizzle selectors of the descriptor registered in slot id; 0 when the slot is empty.
int vk_texture_slot_swz(int id, uint8_t swz[4]) {
    lock();
    int ok = id >= 0 && id < ntslots && tslots[id].used;
    if (ok) {
        const nvn_texdesc *d = &tslots[id].t.d;
        swz[0] = (uint8_t)d->swizzle_r, swz[1] = (uint8_t)d->swizzle_g, swz[2] = (uint8_t)d->swizzle_b;
        swz[3] = (uint8_t)d->swizzle_a;
    }
    unlock();
    return ok;
}

uint32_t vk_texture_epoch(void) { return __atomic_load_n(&submit_epoch, __ATOMIC_RELAXED); }

// ---- compaction of the textures' pieces of VMA's blocks (REKARINA_VRAM_COMPACT=1 turns it on)
// A texture the CPU fills, up to REKARINA_TEX_SUBALLOC_KB, gets a piece of a VMA block (create_tex), and a block goes
// back to the system only when nothing is left in it. Over a long session the textures of the maps left go (eviction,
// the game's releases) and a few stay in each block: in a ten-map tour (2026-11-16) VMA's blocks grew from 7.2 to
// 9.1 GiB while what they held stayed at 7.1 GiB, the game's video memory went past the budget Windows gives it, and
// the last map's GPU time tripled while Windows moved images out of video memory (the castle grounds at 15-21 ms a
// frame against 5-7 ms from a fresh start). Here the textures of the emptiest blocks move: each gets a new image at a
// free place of a fuller block (VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT: never a new block; a place in another of
// the blocks being emptied is given back), its texels are copied on the GPU in the upload batch, and the old image
// with its memory goes through the deferral as a destroyed one does; VMA frees a block once it is empty. Only
// textures the GPU never wrote (no render target, no storage image) and that are not released move, and only out of
// blocks no other texture entry holds an unmovable piece of. At most CP_MOVES textures and CP_BYTES a frame, and a
// look at the blocks every CP_SCAN_FRAMES while their free space exceeds CP_SLACK.
#define CP_SCAN_FRAMES 120
#define CP_SLACK (384ull << 20)
#define CP_MOVES 8
#define CP_BYTES (16ull << 20)
#define CP_FILL_PCT 35          // a block whose texture pieces fill less than this is emptied
#define CP_HASH 4096
typedef struct { VkDeviceMemory m; VkDeviceSize size, movable, fixed; int cand; } CpBlock;
typedef struct { TexKey k; VkDeviceSize rank; } CpKey;
static struct {
    uint64_t frame, next_scan;
    CpBlock *b;                 // CP_HASH slots by block (scan); the candidates sorted to the front, emptiest first
    int nb;                     // candidates
    CpKey *keys;                // the movable textures in them, block by block, emptiest block first
    int nk, ck, cur;            // keys, room for them, the next one
    uint64_t moved, moved_bytes, scans, no_room, failed;
} CP;

static int compact_on(void) {
    static int on = -1;
    if (on < 0) {
        on = rt_knob("VRAM_COMPACT", "0");
        rt_log("vk: textures %s out of nearly empty memory blocks (REKARINA_VRAM_COMPACT)", on ? "move" : "do not move");
    }
    return on;
}

// the texture pool an allocation is in (NULL: a default pool): its new place is found in the same one
static VmaPool oi_pool(VkCtx *c, VmaAllocation a) {
    VmaAllocationInfo i;
    vmaGetAllocationInfo(c->vma, a, &i);
    for (int t = 0; t < VK_MAX_MEMORY_TYPES; t++)
        if (tex_vma_pool[t] && t == (int)i.memoryType) return tex_vma_pool[t];
    return NULL;
}

static int cp_movable(const Tex *e) {
    return e->alloc && !e->bound_to && e->pub.mirrored && !e->pub.gpu_owned && !e->wr_serial && !e->released &&
           !e->pub.gone && e->uploaded && e->pub.image && e->pub.samples <= 1 && e->pub.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
}

static CpBlock *cp_slot(VkDeviceMemory m) {
    uint32_t h = (uint32_t)(((uint64_t)(uintptr_t)m * 0x9E3779B97F4A7C15ull) >> 52) & (CP_HASH - 1);
    for (int i = 0; i < CP_HASH; i++, h = (h + 1) & (CP_HASH - 1))
        if (!CP.b[h].m || CP.b[h].m == m) return &CP.b[h];
    return NULL;
}

static int cp_cmp_key(const void *a, const void *b) {
    const CpKey *x = a, *y = b;
    return x->rank < y->rank ? -1 : x->rank > y->rank;
}
static int cp_cmp(const void *a, const void *b) {
    const CpBlock *x = a, *y = b;
    return x->movable < y->movable ? -1 : x->movable > y->movable;
}

// Which blocks to empty: every texture entry's piece by block (dedicated memory left out), then the blocks that only
// movable textures hold and that they fill less than CP_FILL_PCT, emptiest first.
static void cp_scan(VkCtx *c) {
    if (!CP.b && !(CP.b = malloc(sizeof *CP.b * CP_HASH))) return;
    memset(CP.b, 0, sizeof *CP.b * CP_HASH);
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || !e->alloc || e->bound_to) continue;
        VmaAllocationInfo2 ai;
        vmaGetAllocationInfo2(c->vma, e->alloc, &ai);
        if (ai.dedicatedMemory || !ai.allocationInfo.deviceMemory) continue;
        CpBlock *s = cp_slot(ai.allocationInfo.deviceMemory);
        if (!s) continue;
        s->m = ai.allocationInfo.deviceMemory, s->size = ai.blockSize;
        if (cp_movable(e)) s->movable += ai.allocationInfo.size;
        else s->fixed += ai.allocationInfo.size;
    }
    for (int i = 0; i < CP_HASH; i++)
        CP.b[i].cand = CP.b[i].m && !CP.b[i].fixed && CP.b[i].movable && CP.b[i].movable * 100 < CP.b[i].size * CP_FILL_PCT;
    CP.nk = CP.cur = 0, CP.scans++;
    for (uint32_t i = 0; i < by_key.cap; i++) {   // the movable textures in those blocks
        const Tex *e = by_key.vals[i];
        if (!e || !cp_movable(e)) continue;
        VmaAllocationInfo ai;
        vmaGetAllocationInfo(c->vma, e->alloc, &ai);
        const CpBlock *s = cp_slot(ai.deviceMemory);
        if (!s || !s->cand) continue;
        if (CP.nk == CP.ck) {
            int nc = CP.ck ? 2 * CP.ck : 512;
            CpKey *nk = realloc(CP.keys, sizeof *nk * (size_t)nc);
            if (!nk) break;
            CP.keys = nk, CP.ck = nc;
        }
        CP.keys[CP.nk].k = e->key, CP.keys[CP.nk++].rank = s->movable;
    }
    qsort(CP.keys, (size_t)CP.nk, sizeof *CP.keys, cp_cmp_key);
    int n = 0;
    for (int i = 0; i < CP_HASH; i++)
        if (CP.b[i].cand) CP.b[n++] = CP.b[i];
    qsort(CP.b, (size_t)n, sizeof *CP.b, cp_cmp);
    CP.nb = n;
}

static int cp_candidate(VkDeviceMemory m) {
    for (int i = 0; i < CP.nb; i++)
        if (CP.b[i].m == m) return 1;
    return 0;
}

// e gets a new image at a free place outside the blocks being emptied, with its texels copied in cb. 1 moved, 0 not
// (an error), -1 no free place outside those blocks.
static int cp_stress;   // DEBUG-CPSTRESS (compact_step): a new place anywhere, from any block
static int cp_move(VkCtx *c, Tex *e, VkCommandBuffer cb) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = e->ci_flags, .imageType = (VkImageType)e->ci_type,
                            .format = e->ci_format, .extent = {e->pub.width, e->pub.height, e->pub.depth ? e->pub.depth : 1},
                            .mipLevels = e->pub.levels ? e->pub.levels : 1, .arrayLayers = e->pub.layers ? e->pub.layers : 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = e->ci_usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img;
    if (vkCreateImage(c->device, &ci, NULL, &img) != VK_SUCCESS) return 0;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(c->device, img, &mr);
    VmaAllocationInfo oi;
    vmaGetAllocationInfo(c->vma, e->alloc, &oi);
    if (mr.size != oi.size || !(mr.memoryTypeBits & (1u << oi.memoryType))) {   // not the image it replaces
        vkDestroyImage(c->device, img, NULL);
        return 0;
    }
    mr.memoryTypeBits = 1u << oi.memoryType;   // the same memory type: the free places of the other blocks of its kind
    VmaAllocationCreateInfo aci = {.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                   .flags = cp_stress ? 0 : VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT, .pool = oi_pool(c, e->alloc)};
    VmaAllocation al;
    if (vmaAllocateMemory(c->vma, &mr, &aci, &al, NULL) != VK_SUCCESS) {
        vkDestroyImage(c->device, img, NULL);
        return -1;
    }
    VmaAllocationInfo2 ni;
    vmaGetAllocationInfo2(c->vma, al, &ni);
    if (ni.dedicatedMemory || (!cp_stress && cp_candidate(ni.allocationInfo.deviceMemory))) {
        vmaFreeMemory(c->vma, al);
        vkDestroyImage(c->device, img, NULL);
        return -1;
    }
    if (vmaBindImageMemory(c->vma, al, img) != VK_SUCCESS) {
        vmaFreeMemory(c->vma, al);
        vkDestroyImage(c->device, img, NULL);
        return 0;
    }
    uint32_t levels = ci.mipLevels, layers = ci.arrayLayers;
    // the old image's writes so far (uploads earlier in this batch included) before the copy; the new one from undefined
    VkImageMemoryBarrier2 ib[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT, .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = e->pub.image,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
         .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers}}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &d0);
    VkImageCopy rg[16];
    uint32_t nr = 0;
    for (uint32_t lv = 0; lv < levels; lv++) {
        if (nr == 16) {
            vkCmdCopyImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_GENERAL, nr, rg);
            nr = 0;
        }
        uint32_t w = ci.extent.width >> lv, h = ci.extent.height >> lv, dp = ci.extent.depth >> lv;
        rg[nr++] = (VkImageCopy){.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, layers},
                                 .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, layers},
                                 .extent = {w ? w : 1, h ? h : 1, dp ? dp : 1}};
    }
    if (nr) vkCmdCopyImage(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_GENERAL, nr, rg);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(cb, &d1);
    // the swap, as adopt_released makes it: the replay's views and the pool's slots of the old image go, and the slots
    // and draw memos that reach the entry resolve again
    VkImage old = e->pub.image;
    VmaAllocation oa = e->alloc;
    {   // DEBUG-CPLOG: REKARINA_COMPACT_LOG=1: every move, to find a moved texture by a fault address in its old block
        static int on = -1;
        if (on < 0) on = cfg_get("REKARINA_COMPACT_LOG") != NULL;
        if (on)
            rt_log("CPLOG frame %llu tex %#llx fmt %#x %ux%u lv %u: image %#llx memory %#llx offset %#llx size %#llx -> image %#llx memory %#llx offset %#llx",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), (unsigned long long)e->key.addr,
                   e->key.d.format, e->pub.width, e->pub.height, e->pub.levels, (unsigned long long)(uintptr_t)old,
                   (unsigned long long)(uintptr_t)oi.deviceMemory, (unsigned long long)oi.offset, (unsigned long long)oi.size,
                   (unsigned long long)(uintptr_t)img, (unsigned long long)(uintptr_t)ni.allocationInfo.deviceMemory,
                   (unsigned long long)ni.allocationInfo.offset);
    }
    vk_exec_image_gone(old);
    e->pub.image = img, e->alloc = al;
    e->img_how = "a compaction move's", e->img_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);   // DEBUG-IMGWATCH
    e->pub.serial = vk_texture_image_serial();
    vk_live_check_born((uint64_t)(uintptr_t)img);   // DEBUG-LIVECHECK
    vk_addr_label((uint64_t)img, "tex %#llx fmt %#x %ux%u (moved)", (unsigned long long)e->key.addr, e->key.d.format,
                  e->pub.width, e->pub.height);
    tex_gen_bump_entry(e->key.addr, e->texq_base);
    iw_defer_check(old, e, __LINE__), vk_defer_image(old, oa);
    return 1;
}

static void compact_step(void) {   // tex_lock held, at a submission's start
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f == CP.frame || !compact_on()) return;
    CP.frame = f;
    VkCtx *c = vk_ctx();
    if (!c || !c->vma) return;
    {   // DEBUG-CPSTRESS: REKARINA_COMPACT_STRESS=<n>: n textures a frame move, any movable ones in turn, whatever the
        // blocks hold: a reproducer for "an image read after it went" (2026-11-16: the compaction lost the device once
        // in two ten-map tours with the live check silent)
        static int n = -1;
        static uint32_t at;
        if (n < 0) { const char *s = cfg_get("REKARINA_COMPACT_STRESS"); n = s && *s ? atoi(s) : 0; }
        if (n > 0) {
            cp_stress = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            int moved = 0;
            for (uint32_t k = 0; k < by_key.cap && moved < n; k++) {
                Tex *e = by_key.vals[(at + k) % by_key.cap];
                if (!e || !cp_movable(e)) continue;
                if (!cb) cb = vk_upload_begin();
                if (cp_move(c, e, cb) > 0) moved++, CP.moved++;
                if (moved == n) at = (at + k + 1) % by_key.cap;
            }
            if (moved < n) at = 0;
            if (cb) vk_upload_end();
            if ((rt_trace & RT_TRACE_PERF) && CP.moved % 4096 < (uint64_t)moved)
                rt_log("vk: compaction stress: %llu textures moved so far", (unsigned long long)CP.moved);
            return;
        }
    }
    if (CP.cur >= CP.nk) {   // nothing being emptied: a look at the blocks now and then, while their free space is large
        if (f < CP.next_scan) return;
        CP.next_scan = f + (f - scene_frame >= COLD_LOAD ? CP_SCAN_FRAMES / 8 : CP_SCAN_FRAMES);
        VmaBudget vb[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(c->vma, vb);
        uint64_t slack = 0;
        for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                slack += vb[h].statistics.blockBytes - vb[h].statistics.allocationBytes;
        if (slack < CP_SLACK) return;
        cp_scan(c);
        if (CP.nb && (rt_trace & RT_TRACE_PERF))
            rt_log("vk: compaction: %.0f MiB free inside blocks; %d blocks to empty with %d textures, the emptiest holding %.1f MiB",
                   slack / 1048576.0, CP.nb, CP.nk, CP.b[0].movable / 1048576.0);
        if (!CP.nk) return;
    }
    // the textures of the blocks being emptied, a few a frame; one that went or changed since the look is passed over.
    // While no scene is drawn (a loading screen: a map change, when most blocks empty) eight times as many.
    uint64_t bytes = 0;
    int moves = 0, mul = f - scene_frame >= COLD_LOAD ? 8 : 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    while (CP.cur < CP.nk && moves < CP_MOVES * mul && bytes < CP_BYTES * mul) {
        Tex *e = nvn_hm_get(&by_key, &CP.keys[CP.cur++].k);
        if (!e || !cp_movable(e)) continue;
        VmaAllocationInfo ai;
        vmaGetAllocationInfo(c->vma, e->alloc, &ai);
        if (!cp_candidate(ai.deviceMemory)) continue;
        if (!cb) cb = vk_upload_begin();
        int r = cp_move(c, e, cb);
        if (r > 0) moves++, bytes += ai.size, CP.moved++, CP.moved_bytes += ai.size;
        else if (r < 0) {   // no free place outside the blocks being emptied: look again later
            CP.no_room++, CP.cur = CP.nk;
            break;
        } else CP.failed++;
    }
    if (cb) vk_upload_end();
    if ((rt_trace & RT_TRACE_PERF) && moves && CP.moved % 256 < (uint64_t)moves)   // at every 256th
        rt_log("vk: compaction: %llu textures moved so far (%.0f MiB), %llu times no room, %llu failed, %llu looks",
               (unsigned long long)CP.moved, CP.moved_bytes / 1048576.0, (unsigned long long)CP.no_room,
               (unsigned long long)CP.failed, (unsigned long long)CP.scans);
}

void vk_texture_new_submit(void) {
    lock();
    if (round_checks && (uint64_t)round_checks * 2 >= pool_slots) look_round++;   // the last one walked the pool
    round_checks = 0;
    if (draw_checks >= DRAW_SUBMIT_MIN)   // the last one drew a scene
        draw_round++, scene_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    draw_checks = 0;
    ht_take();
    if (!++submit_epoch) submit_epoch = 1;
    {   // a loading screen begins a new generation of textures (the census; REKARINA_TEX_GEN_POOLS)
        static int loading;
        int no_scene = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) - scene_frame >= COLD_LOAD;
        if (no_scene && !loading) map_gen++;
        loading = no_scene;
    }
    sweep_pressure();
    {   // a moved texture's old image and views: past every value handed out, as the evicted ones (sweep_pressure)
        static int quiet = -1;
        if (quiet < 0) quiet = rt_knob("EVICT_QUIET", "1") != 0;
        if (quiet) vk_defer_past_reserved(1);
        compact_step();
        bc1_redo_run();   // BC1 images to be made again, the same way
        if (quiet) vk_defer_past_reserved(0);
        if (ngenp > 1) tex_gen_pools_trim();
    }
    pm_gone(0, 0, NULL, PM_KEEP);
    static uint64_t log_at;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) / 60;
    static uint64_t ts_at;
    if (f != ts_at) {
        ts_at = f, ts_tick(f * 60, (rt_trace & RT_TRACE_PERF) != 0);
        if (!(f % 10)) lead_log(f * 60), ma_log(f * 60);   // DEBUG-TEXLEAD; REKARINA_TEX_MAKE_AHEAD
    }
    {   // REKARINA_TEX_MAKE_AHEAD: records nobody took go, every 8 frames; those that left are destroyed
        static uint64_t ma_at;
        uint64_t f8 = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) / 8;
        if (f8 != ma_at) ma_at = f8, ma_expire(f8 * 8);
        else ma_drain();
    }
    if ((rt_trace & RT_TRACE_PERF) && f != log_at && tex_alias > 0) {
        log_at = f;
        rt_log("perf: frame %llu: texture aliases: content copied %llu, filled from storage %llu, left uninitialised %llu",
               (unsigned long long)f * 60, (unsigned long long)n_alias_copied, (unsigned long long)n_alias_upload,
               (unsigned long long)n_alias_uninit);
        n_alias_copied = n_alias_upload = n_alias_uninit = 0;
        rt_log("perf: frame %llu: texture re-uploads %llu (%.1f MiB, %.2f ms), after WriteTexels %llu (%.1f MiB, %.2f ms)",
               (unsigned long long)f * 60, (unsigned long long)ru_n, ru_bytes / 1048576.0, ru_ns / 1e6,
               (unsigned long long)ru_wt_n, ru_wt_bytes / 1048576.0, ru_wt_ns / 1e6);
        ru_n = ru_bytes = ru_ns = ru_wt_n = ru_wt_bytes = ru_wt_ns = 0;
        uint64_t ck, calls, untr;
        vk_mem_snap_stats(&ck, &calls, &untr);
        rt_log("perf: frame %llu: binding storage checks %llu (texture %llu), write-watch snapshot queries %llu, untracked %llu",
               (unsigned long long)f * 60, (unsigned long long)ck, (unsigned long long)n_checks, (unsigned long long)calls,
               (unsigned long long)untr);
        n_checks = 0;
        if (!(f % 10)) texture_census(f * 60);
        if (n_evict) rt_log("perf: frame %llu: cold textures evicted %llu (%.1f MiB)", (unsigned long long)f * 60, (unsigned long long)n_evict, n_evict_bytes / 1048576.0);
        if (ht_started > 0 && !(f % 10)) {
            plat_mutex_lock(&ht_lock);
            int nt = 0;
            for (int i = 0; i < HT_N; i++) nt += htab[i].addr && (int64_t)(f * 60 - htab[i].frame) <= HT_KEEP_FRAMES;
            rt_log("perf: frame %llu: handle tables: %d read, %llu readings of %.1f MiB each, %llu handle-like words a reading, "
                   "%llu reads failed; slots named %llu a reading, entries kept in use %llu, evicted ones brought back %llu",
                   (unsigned long long)f * 60, nt, (unsigned long long)HS.scans, HS.scans ? HS.bytes / 1048576.0 / HS.scans : 0.0,
                   (unsigned long long)(HS.scans ? HS.words / HS.scans : 0), (unsigned long long)HS.fails,
                   (unsigned long long)(HS.scans ? HS.ids / HS.scans : 0), (unsigned long long)(HS.scans ? HS.marked / HS.scans : 0),
                   (unsigned long long)HS.revived);
            memset(&HS, 0, sizeof HS);
            plat_mutex_unlock(&ht_lock);
        }
        n_evict = n_evict_bytes = 0;
        rt_log("perf: frame %llu: released textures kept %d (depth %u), dropped under VRAM pressure %llu", (unsigned long long)f * 60,
               nreleased, rel_depth_n, (unsigned long long)n_pressure_drops);
        {   // depth images held by entries, live and released, and the most seen at these 60-frame points (the driver
            // keeps ~0.25 MiB for each depth image alive, at its high-water mark: sweep_released)
            static uint32_t peak_all, peak_live, peak_rel;
            uint32_t dl = 0, dr = 0;
            for (uint32_t i = 0; i < by_key.cap; i++) {
                const Tex *e = by_key.vals[i];
                if (e && e->pub.image && (e->f.depth || e->f.stencil)) e->released ? dr++ : dl++;
            }
            if (dl > peak_live) peak_live = dl;
            if (dr > peak_rel) peak_rel = dr;
            if (dl + dr > peak_all) peak_all = dl + dr;
            static uint64_t hold_ns;
            static uint64_t hold_dns;
            if (!hold_ns) {
                const char *hv = cfg_get("REKARINA_POOL_IMAGE_HOLD_MS"), *hd = cfg_get("REKARINA_POOL_IMAGE_HOLD_DEPTH_MS");
                long long ms = hv && *hv ? atoll(hv) : 30000;
                hold_ns = (uint64_t)ms * 1000000ull + 1;
                hold_dns = (uint64_t)(hd && *hd ? atoll(hd) : 500) * 1000000ull + 1;
            }
            uint64_t now_ns = plat_now_ns(), hd = 0, hc = 0;
            for (uint64_t i = hold_n > HOLD_RING ? hold_n - HOLD_RING : 0; i < hold_n; i++) {
                uint64_t v = hold_ring[i % HOLD_RING];
                if (now_ns - (v & ~1ull) < (v & 1 ? hold_dns : hold_ns)) v & 1 ? hd++ : hc++;
            }
            static uint64_t peak_hd;
            if (hd > peak_hd) peak_hd = hd;
            rt_log("perf: frame %llu: depth images live %u, released %u (most so far: %u together, %u live, %u released); "
                   "held after their texture went: %llu depth, %llu colour (most depth so far %llu)",
                   (unsigned long long)f * 60, dl, dr, peak_all, peak_live, peak_rel, (unsigned long long)hd,
                   (unsigned long long)hc, (unsigned long long)peak_hd);
        }
        if (__atomic_load_n(&pm_started, __ATOMIC_ACQUIRE)) {
            plat_mutex_lock(&pm_lock);
            rt_log("perf: frame %llu: images made ahead: asked %llu, made %llu, not bindable %llu, taken %llu (waited for %llu), "
                   "made at first use instead %llu, never used %llu, address moved %llu, table full %llu; cannot alias %llu, "
                   "with memory of their own made %llu, taken %llu", (unsigned long long)f * 60,
                   (unsigned long long)PMS.asked, (unsigned long long)PMS.made, (unsigned long long)PMS.failed,
                   (unsigned long long)PMS.taken, (unsigned long long)PMS.waited, (unsigned long long)PMS.stolen,
                   (unsigned long long)PMS.stale, (unsigned long long)PMS.moved, (unsigned long long)PMS.full,
                   (unsigned long long)PMS.noalias, (unsigned long long)PMS.own_made, (unsigned long long)PMS.own_taken);
            memset(&PMS, 0, sizeof PMS);
            plat_mutex_unlock(&pm_lock);
        }
        n_pressure_drops = 0;
        released_census(f * 60);
    }
    unlock();
    {   // every 1800 frames: where the image views came from (REKARINA_VIEWS_AHEAD; vk_exec_draw.c)
        static uint64_t views_at;
        if (f / 30 != views_at) {
            views_at = f / 30;
            int64_t vr, va, vd;
            vk_exec_views_stats(&vr, &va, &vd);
            if (vr || va || vd) {
                plat_mutex_lock(&pm_lock);   // images made ahead (the counters restart with each REKARINA_TRACE=perf line)
                uint64_t taken = PMS.taken, stolen = PMS.stolen, waited = PMS.waited;
                plat_mutex_unlock(&pm_lock);
                rt_log("vk: frame %llu: image views the replay made %lld, made ahead with their image and taken %lld, made ahead "
                       "and not needed %lld (REKARINA_VIEWS_AHEAD=%d); images made ahead and taken %llu, still queued when "
                       "needed (made by the replay) %llu, waited for %llu", (unsigned long long)f * 60, (long long)vr,
                       (long long)va, (long long)vd, vk_exec_views_ahead_on(), (unsigned long long)taken,
                       (unsigned long long)stolen, (unsigned long long)waited);
            }
        }
    }
    // Video memory was short at this submission: the compute pipelines built ahead that nothing has dispatched go as
    // well (vk_pipeline.c vk_pipeline_compute_trim), at most once in ten seconds. NVIDIA's driver does not hand their
    // memory back to the system then; it uses it for what it allocates next, which is still less than keeping them.
    static uint64_t trim_at;
    if (cs_trim_due && f >= trim_at) {
        cs_trim_due = 0, trim_at = f + 10;
        vk_pipeline_compute_trim();
    }
}

// ---------------------------------------------------------------- texel reads and writes
typedef struct { unsigned level, layer0; int nz, is3d, bw, bh, bpb; size_t wb, hb, row, img, x0, y0; } Region;

static int region_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img, Region *g) {
    g->bpb = nvn_fmt_bpb(t->d.format, &g->bw, &g->bh);
    if (!g->bpb || r->width <= 0 || r->height <= 0 || r->depth < 0 || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 ||
        r->xoffset % g->bw || r->yoffset % g->bh)
        return 0;
    g->level = v && v->set_levels ? v->base_level : 0;
    g->layer0 = v && v->set_layers ? v->min_layer : 0;
    g->nz = r->depth ? r->depth : 1;
    g->is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
    g->wb = (size_t)((r->width + g->bw - 1) / g->bw) * (size_t)g->bpb;
    g->hb = (size_t)(r->height + g->bh - 1) / (size_t)g->bh;
    g->row = row ? row : g->wb;
    g->img = img ? img : g->row * g->hb;
    g->x0 = (size_t)(r->xoffset / g->bw) * (size_t)g->bpb;
    g->y0 = (size_t)(r->yoffset / g->bh);
    return 1;
}

// Pool bytes <-> caller rows of the region alone (nvn_texels_box). MacNativeGames' nvn_mtl.m goes through a
// linear copy of the whole level; for a small region of a large texture that cost tens of milliseconds while the guest
// thread held its virtual core and the caller the texture lock (monsters2 replay, frames 765-790).
static int storage_texels(const tex_t *t, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    for (int z = 0; z < g->nz; z++) {
        unsigned layer = g->is3d ? 0 : g->layer0 + (unsigned)r->zoffset + (unsigned)z;
        uint32_t slice = g->is3d ? (uint32_t)(r->zoffset + z) : 0;
        if (nvn_texels_box(&t->d, g->level, layer, (uint32_t)g->x0, (uint32_t)g->y0, slice, (uint32_t)g->wb,
                           (uint32_t)g->hb, t->mem, mem + (size_t)z * g->img, g->row, !write))
            return 0;
    }
    return 1;
}

static int read_aspect(const Tex *e, VkImageAspectFlags *a) {
    if (!e->f.depth && !e->f.stencil) { *a = VK_IMAGE_ASPECT_COLOR_BIT; return 1; }
    switch (e->t.d.format) {   // packed depth-stencil storage has no single-aspect copy
    case 0x31: *a = VK_IMAGE_ASPECT_STENCIL_BIT; return 1;
    case 0x32: case 0x34: *a = VK_IMAGE_ASPECT_DEPTH_BIT; return 1;
    default: return 0;
    }
}

// An image region read through a staging buffer. texels_read_record records the copy into the open upload batch;
// texels_read_take copies the rows out once that batch is done (vk_upload_flush(1)) and frees the buffer. Several
// records before one flush cost one GPU round trip instead of one each.
typedef struct {
    VkBuffer sb;
    VmaAllocation sa;
    const uint8_t *stg;
    size_t slice, ib;   // ib: bytes per texel of an expanded RGB image, 0 when the image holds the storage layout
    uint32_t w;
} TexRead;

static int texels_read_record(Tex *e, const Region *g, const NVNcopyRegion *r, TexRead *t) {
    static volatile int32_t n_astc, n_ds;
    VkImageAspectFlags aspect;
    if (e->f.astc_bw) {
        log_limited(&n_astc, "ReadTexels of ASTC texture %#llx: the image holds decoded RGBA8; zeros returned",
                    (unsigned long long)e->pub.addr);
        return 0;
    }
    if (!read_aspect(e, &aspect)) {
        log_limited(&n_ds, "texels of packed depth-stencil format %#x are not copied", e->t.d.format);
        return 0;
    }
    if (g->level >= e->pub.levels) return 0;
    uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height;
    t->w = w;
    t->ib = e->f.expand_rgb ? (size_t)g->bpb / 3 * 4 : 0;
    t->slice = t->ib ? (size_t)w * h * t->ib : g->wb * g->hb;
    t->stg = vk_staging(t->slice * (size_t)g->nz, &t->sb, &t->sa);
    VkBufferImageCopy *rg = calloc((size_t)g->nz, sizeof *rg);
    if (!rg) rt_fatal("vk: out of memory for a texture read");
    for (int z = 0; z < g->nz; z++)
        rg[z] = (VkBufferImageCopy){.bufferOffset = t->slice * (size_t)z,
                                    .imageSubresource = {aspect, g->level, g->is3d ? 0 : g->layer0 + (uint32_t)r->zoffset + (uint32_t)z, 1},
                                    .imageOffset = {r->xoffset, r->yoffset, g->is3d ? r->zoffset + z : 0},
                                    .imageExtent = {w, h, 1}};
    VkCommandBuffer cb = vk_upload_begin();
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, t->sb, (uint32_t)g->nz, rg);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT);
    vk_upload_end();
    free(rg);
    return 1;
}

static void texels_read_take(const Tex *e, const Region *g, const TexRead *t, uint8_t *mem) {
    VkCtx *c = vk_ctx();
    vmaInvalidateAllocation(c->vma, t->sa, 0, VK_WHOLE_SIZE);
    for (int z = 0; z < g->nz; z++) {
        const uint8_t *s = t->stg + t->slice * (size_t)z;
        uint8_t *m = mem + (size_t)z * g->img;
        for (size_t y = 0; y < g->hb; y++) {
            if (t->ib) shrink_rgb(e->t.d.format, s + y * t->w * t->ib, m + y * g->row, t->w);
            else memcpy(m + y * g->row, s + y * g->wb, g->wb);
        }
    }
    vmaDestroyBuffer(c->vma, t->sb, t->sa);
}

// Image <-> caller rows through a staging buffer; the GPU copy is waited for.
static int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    static volatile int32_t n_astc, n_ds;
    VkCtx *c = vk_ctx();
    VkImageAspectFlags aspect;
    if (!write) {
        TexRead t;
        if (!texels_read_record(e, g, r, &t)) return 0;
        vk_upload_flush(1);
        texels_read_take(e, g, &t, mem);
        return 1;
    }
    if (!read_aspect(e, &aspect)) {
        log_limited(&n_ds, "texels of packed depth-stencil format %#x are not copied", e->t.d.format);
        return 0;
    }
    if (g->level >= e->pub.levels) return 0;
    if (is_bc1(e)) {   // the decoder writes BC3: the image is made again from the storage (bc1_redo)
        bc1_redo(e, "the game wrote texels into it");
        return 1;
    }
    uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height;
    int rgb = e->f.expand_rgb, astc = e->f.astc_bw != 0, agpu = astc && vk_astc_gpu();
    size_t texels = (size_t)w * h, ib = agpu ? 0 : astc ? 4 : rgb ? (size_t)g->bpb / 3 * 4 : 0;
    size_t slice = ib ? texels * ib : g->wb * g->hb;   // agpu: the raw blocks, decoded by vk_astc_record
    VkBuffer sb;
    VmaAllocation sa;
    uint8_t *stg = vk_staging(slice * (size_t)g->nz, &sb, &sa);
    VkBufferImageCopy *rg = calloc((size_t)g->nz, sizeof *rg);
    for (int z = 0; z < g->nz; z++) {
        uint8_t *s = stg + slice * (size_t)z;
        const uint8_t *m = mem + (size_t)z * g->img;
        if (agpu) {
            for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
        } else if (astc) {
            uint8_t *tight = malloc(g->wb * g->hb);
            for (size_t y = 0; y < g->hb; y++) memcpy(tight + y * g->wb, m + y * g->row, g->wb);
            uint64_t a0 = perf_begin();
            astc_decode_ldr_mt(tight, w, h, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, s);
            perf_end(PERF_ASTC, a0);
            free(tight);
        } else if (rgb) {
            for (size_t y = 0; y < g->hb; y++) expand_rgb(e->t.d.format, m + y * g->row, s + y * w * ib, w);
        } else {
            for (size_t y = 0; y < g->hb; y++) memcpy(s + y * g->wb, m + y * g->row, g->wb);
        }
        rg[z] = (VkBufferImageCopy){.bufferOffset = slice * (size_t)z,
                                    .imageSubresource = {aspect, g->level, g->is3d ? 0 : g->layer0 + (uint32_t)r->zoffset + (uint32_t)z, 1},
                                    .imageOffset = {r->xoffset, r->yoffset, g->is3d ? r->zoffset + z : 0},
                                    .imageExtent = {w, h, 1}};
    }
    VkCommandBuffer cb = vk_upload_begin();
    vmaFlushAllocation(c->vma, sa, 0, VK_WHOLE_SIZE);
    barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    int done = 0;
    // A BC3 image takes whole 4x4 blocks, or a part that ends on the mip's edge; a region of a 5x5 or 6x6 footprint
    // may end inside a block (vk_astc_record checks the start corner).
    uint32_t mw = e->t.d.width >> g->level, mh = e->t.d.height >> g->level;
    int bc_cut = e->f.astc_bc && (((w & 3) && (uint32_t)r->xoffset + w != (mw ? mw : 1)) ||
                                  ((h & 3) && (uint32_t)r->yoffset + h != (mh ? mh : 1)));
    if (agpu && !bc_cut) {   // bc_cut: nothing recorded, logged below as not decoded
        VkAstcRegion *ar = calloc((size_t)g->nz, sizeof *ar);
        for (int z = 0; z < g->nz; z++)
            ar[z] = (VkAstcRegion){.src = slice * (size_t)z, .w = w, .h = h, .slices = 1, .level = rg[z].imageSubresource.mipLevel,
                                   .layer = rg[z].imageSubresource.baseArrayLayer, .offset = rg[z].imageOffset};
        uint64_t a0 = perf_begin();
        done = vk_astc_record(cb, sb, slice * (size_t)g->nz, ar, g->nz, e->f.astc_bw, e->f.astc_bh, e->f.astc_srgb, e->f.astc_bc,
                              e->pub.image, aspect, VK_NULL_HANDLE);
        perf_end(PERF_ASTC, a0);
        free(ar);
    }
    if (!agpu) vkCmdCopyBufferToImage(cb, sb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)g->nz, rg);
    else if (!done) log_limited(&n_astc, "WriteTexels of ASTC texture %#llx not decoded on the GPU: left as it was",
                                (unsigned long long)e->pub.addr);
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, ALL_RW);
    vk_upload_end();
    vk_upload_flush(1);
    free(rg);
    vmaDestroyBuffer(c->vma, sb, sa);
    return 1;
}

static void zero_rows(const Region *g, uint8_t *dst) {
    for (int z = 0; z < g->nz; z++)
        for (size_t y = 0; y < g->hb; y++) memset(dst + (size_t)z * g->img + y * g->row, 0, g->wb);
}

int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    Region g;
    if (!t || !region_of(t, v, r, row, img, &g)) return 0;
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
    return ok;
}

// ReadTexels: the pool bytes are the content of a mirrored texture the GPU has not rendered into; every other texture
// is read back from its image.
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g)) {
        log_limited(&n_bad, "ReadTexels: invalid region %d,%d,%d %dx%dx%d (format %#x)", r->xoffset, r->yoffset,
                    r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    int ok = 0;
    if (e && e->pub.mirrored && !e->pub.gpu_owned) ok = storage_texels(t, &g, r, dst, 0);
    else if (e) {
        make_fresh(e);
        ok = image_texels(e, &g, r, dst, 0);
    }
    unlock();
    if (!ok) zero_rows(&g, dst);
}

// WriteTexels: a mirrored texture gets the texels in its pool bytes (and in its image once the GPU owns it); every
// other texture in its image.
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img) {
    static volatile int32_t n_bad;
    Region g;
    if (!region_of(t, v, r, row, img, &g) || (v && v->set_format && v->format != t->d.format)) {
        log_limited(&n_bad, "WriteTexels: invalid region %d,%d,%d %dx%dx%d or view format (format %#x)", r->xoffset,
                    r->yoffset, r->zoffset, r->width, r->height, r->depth, t->d.format);
        return;
    }
    lock();
    Tex *e = entry_of(t);
    if (e && e->pub.mirrored) {
        if (!storage_texels(t, &g, r, (uint8_t *)src, 1))
            log_limited(&n_bad, "WriteTexels: region outside level %u of texture %#llx", g.level, (unsigned long long)e->pub.addr);
        vk_guest_written((uintptr_t)t->mem, t->L.size);
        if (e->pub.gpu_owned) image_texels(e, &g, r, (uint8_t *)src, 1);
        else if (e->uploaded) e->wt_n++;
    } else if (e) {
        image_texels(e, &g, r, (uint8_t *)src, 1);
    }
    unlock();
}

// ---------------------------------------------------------------- CPU reads of GPU-written storage
// A mirrored texture the GPU rendered into holds its texels in its image only; its storage keeps what the CPU last put
// there. The game may still read that storage with the CPU: TotK moves its generated grass maps between pools with
// memcpy once the GPU is done, and the copy then carried zeros. Such a read writes the image back first.
//
// Every level and layer of every texture in v is copied to staging in one upload batch with one wait: one GPU round
// trip per level and layer took 0.1 ms each, and a 260x260 grass map has 9 levels. The storage is then written in the
// order of v, so where two textures share storage the later one wins.
typedef struct {
    Tex *e;
    Region g;
    NVNcopyRegion r;
    TexRead t;
} WbPart;

static void write_back_n(Tex **v, int n) {
    int np = 0, cap = 0;
    WbPart *p = NULL;
    for (int i = 0; i < n && mem_result_watch_on() && cfg_env("REKARINA_RESULT_WATCH_TEX"); i++)   // DEBUG-RESULTWATCH: the pages open for the write-back
        if (v[i]->t.L.size <= 65536) mem_result_release(v[i]->pub.addr, v[i]->t.L.size);
    for (int i = 0; i < n; i++) {
        Tex *e = v[i];
        const nvn_texdesc *d = &e->t.d;
        int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
        uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
        for (uint32_t lv = 0; lv < levels; lv++)
            for (uint32_t ly = 0; ly < layers; ly++) {
                if (np == cap) {
                    cap = cap ? cap * 2 : 16;
                    if (!(p = realloc(p, sizeof *p * (size_t)cap))) rt_fatal("vk: out of memory for a texture write-back");
                }
                WbPart *q = &p[np];
                view_t vw = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)lv, .min_layer = (uint16_t)ly};
                q->e = e;
                q->r = (NVNcopyRegion){.width = (int)mip(d->width, lv),
                                       .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, lv),
                                       .depth = is3d ? (int)mip(d->depth, lv) : 1};
                if (region_of(&e->t, &vw, &q->r, 0, 0, &q->g) && texels_read_record(e, &q->g, &q->r, &q->t)) np++;
            }
    }
    if (np) vk_upload_flush(1);
    for (int i = 0; i < np; i++) {
        WbPart *q = &p[i];
        uint8_t *buf = malloc(q->g.img * (size_t)q->g.nz);
        if (!buf) rt_fatal("vk: out of memory for a texture write-back");
        texels_read_take(q->e, &q->g, &q->t, buf);
        storage_texels(&q->e->t, &q->g, &q->r, buf, 1);
        free(buf);
    }
    free(p);
    for (int i = 0; i < n; i++) v[i]->gpu_dirty = 0;
    {   // DEBUG-LUMLOG: REKARINA_LUMLOG=1: the 1x1 RGBA32F targets the CPU reads back (OoT's auto-exposure luminance), every
        // 10th write-back of them
        static int on = -1, k;
        if (on < 0) on = cfg_get("REKARINA_LUMLOG") != NULL;
        for (int i = 0; on && i < n; i++)
            if (v[i]->t.d.format == 0x2e && v[i]->t.d.width == 1 && v[i]->t.d.height == 1 && !(k++ % 300)) {
                const float *x = (const float *)(uintptr_t)v[i]->pub.addr;
                rt_log("LUMLOG writeback#%d frame %llu %#llx: %g %g %g %g", k,
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                       (unsigned long long)v[i]->pub.addr, x[0], x[1], x[2], x[3]);
            }
    }
}

// Mirrored textures overlapping [addr, addr+size) that the GPU wrote since their last write-back.
static int owned_dirty(uint64_t addr, uint64_t size, Tex **hit) {
    int n = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        Tex *e = owned[i];
        if (__atomic_load_n(&e->gpu_dirty, __ATOMIC_RELAXED) && e->pub.addr + e->t.L.size > addr) {
            if (!hit) return 1;
            hit[n++] = e;
        }
    }
    return n;
}

// A storage image may be written by the shaders of the submission being recorded (texture lock held, vk_exec_draw.c
// tex_info). Its storage is not marked GPU-owned: only the write fence for ReadTexels.
// The colour of one of the game's fill-in textures: a CPU-filled RGBA8 texture of at most 8x8 texels (OoT keeps 4x4
// ones in black, white and clear, and binds them to whatever a material leaves empty, a 3D or cube sampler included).
// 1 with the first texel's bytes in rgba; 0 for any other texture. Texture lock held.
int vk_texture_fill_colour_locked(const VkTexImage *img, uint8_t rgba[4]) {
    const Tex *e = (const Tex *)img;   // pub is the entry's first member
    const nvn_texdesc *d = &e->t.d;
    int bw, bh;
    if (img->gpu_owned || img->gone || !e->t.mem || img->samples > 1 || img->width > 8 || img->height > 8 || img->depth > 1 ||
        (img->format != VK_FORMAT_R8G8B8A8_UNORM && img->format != VK_FORMAT_R8G8B8A8_SRGB) ||
        nvn_fmt_bpb(d->format, &bw, &bh) != 4 || bw != 1 || bh != 1)
        return 0;
    memcpy(rgba, (const uint8_t *)(uintptr_t)e->t.mem, 4);
    return 1;
}

void vk_texture_storage_bound_locked(const VkTexImage *img) {
    Tex *e = (Tex *)img;   // pub is the first member
    uint64_t f = vk_exec_recording_fence();
    if (f > e->wr_fence) e->wr_fence = f;
    note_writer(e);
}

uint64_t vk_texture_last_write(const tex_t *t) {
    lock();
    Tex *e = entry_of(t);
    uint64_t f = e ? e->wr_fence : 0;
    unlock();
    return f;
}

size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img) {
    Region g;
    if (!region_of(t, v, r, row, img, &g)) return 0;
    return (size_t)(g.nz - 1) * g.img + (g.hb - 1) * g.row + g.wb;
}

int vk_texture_cpu_stale(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    plat_mutex_lock_shared(&owned_lock);
    int dirty = owned_dirty(addr, size, NULL);
    plat_mutex_unlock_shared(&owned_lock);
    return dirty;
}

// The newest submission recorded to write a GPU-written mirrored texture overlapping [addr, addr+size); 0: none.
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size) {
    if (!size || !__atomic_load_n(&nowned, __ATOMIC_ACQUIRE)) return 0;
    lock();
    uint64_t f = 0;
    for (int i = owned_lower(addr > max_owned ? addr - max_owned : 0); i < nowned && owned[i]->pub.addr < addr + size; i++) {
        const Tex *e = owned[i];
        if (!e->gpu_dirty || e->pub.addr + e->t.L.size <= addr) continue;
        if (e->gpu_fence > f) f = e->gpu_fence;
        if (e->wr_fence > f) f = e->wr_fence;
    }
    unlock();
    return f;
}

// The image is the newest content only while nothing wrote the storage since the GPU render: TotK finalizes grass maps,
// reuses their storage for new data and memcpys that data on, and writing the old image back first handed the copy
// a stale map or stale resource bytes (wrong textures in the distance). Every hit is checked before any is written
// back, since a write-back is itself a CPU write to storage another hit may share; the hits then go back oldest render
// first, so where two descriptors share storage the newer render wins.
// Readbacks (vk_texture_readback_hold). OoT tests the sun's visibility once after a load (a point draw
// into a 1x1 target, copied into CPU memory) and reads the copy in the next frame without a sync; on the console the
// GPU, a frame behind, has not run the copy by then, so the game keeps the copy's initial value (the sun counts as
// occluded under the forest canopy) and lights the scene with it for good. Delivered at once (the present waiting
// for the GPU, or a write-back on that read), the sun counted as visible and the sky, fog and aerial perspective came
// out 2.4 times too bright: the long-standing haze (docs/oot-notes.md). REKARINA_READBACK_LAG=0 delivers at once.
static int rb_lag = -1;
void vk_texture_readback_hold(const tex_t *t) {
    if (rb_lag < 0) rb_lag = rt_knob("READBACK_LAG", "0");   // DEBUG: on (1) it changed nothing visible
    if (!rb_lag) return;
    lock();
    Tex *e = entry_of(t);
    if (e) e->rb_hold = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 2;   // recorded in frames + 1
    unlock();
}
static int rb_held(const Tex *e) { return e->lagged || e->rb_hold > __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }

void vk_texture_write_back(uint64_t addr, uint64_t size) {
    lock();
    Tex **hit = malloc(sizeof *hit * (size_t)(nowned + 1));
    if (!hit) rt_fatal("vk: out of memory for a texture write-back");
    int n = owned_dirty(addr, size, hit), k = 0;   // owned[] cannot change while tex_lock is held
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (rb_held(hit[i])) {   // a readback the console's GPU has not delivered yet: the CPU reads the old bytes
            __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
            continue;
        }
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    write_back_n(hit, k);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// TotK's grass system renders a 128x128 RGBA8 map of cut and trampled grass into a CPU-cached pool every frame and its
// CPU code reads the pool bytes without nvnMemoryPoolInvalidateMappedRange (cut reactions: blades, insects, the cut
// itself). On the Switch the bytes are there once the frame's fence has passed. So when the
// CPU waits for a fence, mirrored targets rendered by work up to that fence go back to their storage (wb_wanted:
// targets over wb_limit only when not drawn every frame), unless the CPU rewrote it since. Targets drawn every frame
// still go back on an invalidate, memcpy or ReadTexels.
int vk_texture_small_dirty(void) { return __atomic_load_n(&small_pending, __ATOMIC_ACQUIRE); }

// REKARINA_TRACE=gfx: a dirty target the fence write-back leaves alone because it is over the limit and drawn every frame
// (what a CPU read after the wait would miss), or sends back although it is over the default limit. One line per
// target address and outcome; the table is small and a full one stops logging.
static void wb_note(const Tex *e, int skipped) {
    static uint64_t seen[2][64];
    static int n_seen[2];
    for (int i = 0; i < n_seen[skipped]; i++)
        if (seen[skipped][i] == e->pub.addr) return;
    if (n_seen[skipped] == 64) return;
    seen[skipped][n_seen[skipped]++] = e->pub.addr;
    const pool_t *p = e->t.pool;
    unsigned fl = p ? p->flags : 0;
    rt_log("vk: fence write-back %s %#llx %ux%u fmt %#x size %llu pool %s (limit %llu)",
           skipped ? "skipped-drawn-every-frame" : "over-default", (unsigned long long)e->pub.addr, e->t.d.width, e->t.d.height,
           e->t.d.format, (unsigned long long)e->t.L.size,
           !p ? "none" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_CACHED) ? "cpu-cached" : (fl & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) ? "cpu-uncached" : "other",
           (unsigned long long)wb_limit);
}

void vk_texture_write_back_small(uint64_t fence) {
    lock();
    wb_cfg_init();
    __atomic_store_n(&small_pending, 0, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&owned_lock);
    int n = 0, later = 0;
    for (int i = 0; i < nowned; i++) n += owned[i]->gpu_dirty && wb_wanted(owned[i]);
    Tex **hit = n ? malloc(sizeof *hit * (size_t)n) : NULL;
    if (n && !hit) rt_fatal("vk: out of memory for a texture write-back");
    n = 0;
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        if (!e->gpu_dirty) continue;
        if (!wb_wanted(e)) {
            if (wb_trace && e->gpu_fence <= fence) {
                wb_note(e, 1);
                perf_count(PERF_CNT_WB_SKIP);
            }
            continue;
        }
        if (e->gpu_fence > fence || rb_held(e)) later = 1;   // held readbacks go back at a later sync
        else {
            if (wb_trace && e->t.L.size > WB_SMALL_DEFAULT) wb_note(e, 0);
            hit[n++] = e;
        }
    }
    plat_mutex_unlock_shared(&owned_lock);
    if (later) __atomic_store_n(&small_pending, 1, __ATOMIC_RELEASE);
    int k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t now;
        if (storage_rewritten(hit[i], hit[i]->gpu_gen, &now) == 1) hit[i]->gpu_dirty = 0;
        else hit[k++] = hit[i];
    }
    for (int i = 1; i < k; i++)
        for (int j = i; j > 0 && hit[j - 1]->gpu_seq > hit[j]->gpu_seq; j--) {
            Tex *x = hit[j];
            hit[j] = hit[j - 1], hit[j - 1] = x;
        }
    write_back_n(hit, k);
    for (int i = 0; i < k; i++) perf_count(PERF_CNT_WB);
    for (int i = k; i < n; i++) perf_count(PERF_CNT_WB_SKIP);
    free(hit);
    unlock();
}

// ---------------------------------------------------------------- fence readback
// OoT reads some GPU results without an NVN sync, a frame or so after rendering them: the 1x1 luminance
// its auto exposure takes the next frame, among others. Here a render target's bytes reached its storage only when the
// CPU waited for a fence (vk_texture_write_back_small) or read the storage through NVN (vk_texture_write_back), and
// such a write-back is a readback on the graphics queue that waits for everything queued before it. Whenever the GPU
// ran late the game read old luminance, and while the camera moved every other frame came out white (with every
// present waiting for the GPU, REKARINA_PRESENT_FINISH=1: none in 699 shots). So the small targets (wb_wanted, at most
// FB_MAX bytes, plain formats) that a submission renders are copied into host buffers at the end of that
// submission's own command buffer, and the completion callback of its fence puts the bytes into their storage before
// the fence counts as done (vk_sync.c): what the console's GPU writes into memory as it renders, with no CPU wait.
// The target stays gpu_dirty; a later sync wait or NVN read finds its storage written since the render
// (storage_rewritten) and leaves it. Off by default (REKARINA_FENCE_READBACK=1): the white frames went with the blocking
// acquire (nvn_api_obj.c nvnWindowAcquireTexture) and stayed without it, with this on or off.
#define FB_MAX (64u << 10)
typedef struct { tex_t t; Region g; NVNcopyRegion r; VkBuffer sb; VmaAllocation sa; const uint8_t *stg; } FbPart;
typedef struct FbJob { struct FbJob *next; uint64_t frame; volatile int done; int n; FbPart p[]; } FbJob;
static int fb_on = -1;

// Frame-latency readbacks (REKARINA_READBACK_FRAMES=<k>, default 1 for OoT, 0 elsewhere). The console's GPU
// runs about a frame behind its CPU, and OoT's auto exposure is tuned to that: the CPU reads the 1x1
// luminance of its small targets (after nvnMemoryPoolInvalidateMappedRange) and takes whatever the GPU has written by
// then, the result of the frame before. Here such a read used to write back the newest render at once (the GPU done
// with it, or waited for), the controller settled 2.4 times brighter (sun x exposure ~23 instead of ~10) and the forest
// came out hazy and washed out against the publisher's own screenshots; with a frame in flight it came out right but not
// reliably. Now the small targets a submission renders are staged at its end (the fence readback below), and their
// bytes go into storage only once k more frames have been presented after the one that rendered them, in render
// order; the NVN read and fence-wait write-backs leave such a target alone (rb_held). The same latency in every present
// mode, whatever the GPU's own timing.
static int rb_frames = -1;
static int readback_frames(void) {
    if (rb_frames < 0) {
        const char *e = cfg_get("REKARINA_READBACK_FRAMES");
#ifdef GAME_OOT
        rb_frames = e && *e ? atoi(e) : 1;
#else
        rb_frames = e && *e ? atoi(e) : 0;
#endif
        if (rb_frames < 0) rb_frames = 0;
        if (rb_frames > 4) rb_frames = 4;
        if (rb_frames) rt_log("vk: small render targets reach their storage %d frame(s) after their render (REKARINA_READBACK_FRAMES)", rb_frames);
    }
    return rb_frames;
}
static FbJob *dq_head, *dq_tail;   // dq_lock: staged jobs in render order
static PlatMutex dq_lock = PLAT_MUTEX_INIT;

static void fb_write(FbJob *j) {
    VmaAllocator vma = vk_ctx()->vma;
    for (int i = 0; i < j->n; i++) {
        FbPart *q = &j->p[i];
        vmaInvalidateAllocation(vma, q->sa, 0, VK_WHOLE_SIZE);
        // the copy's rows are tight (wb bytes), as region_of makes row and img without a caller pitch
        mem_thaw((uint64_t)(uintptr_t)q->t.mem, q->t.L.size);
        storage_texels(&q->t, &q->g, &q->r, (uint8_t *)(uintptr_t)q->stg, 1);
        vmaDestroyBuffer(vma, q->sb, q->sa);
    }
    free(j);
}

// The staged jobs that are done and due at frame count `now`, in order (a job not done yet holds the later ones back).
static void fb_deliver(uint64_t now) {
    plat_mutex_lock(&dq_lock);
    while (dq_head && __atomic_load_n(&dq_head->done, __ATOMIC_ACQUIRE) &&
           dq_head->frame + 1 + (uint64_t)rb_frames <= now) {
        FbJob *j = dq_head;
        dq_head = j->next;
        if (!dq_head) dq_tail = NULL;
        fb_write(j);
    }
    plat_mutex_unlock(&dq_lock);
}

void vk_texture_readback_tick(uint64_t frames) {   // a present: frames is the new gfx_counters.frames
    if (rb_frames > 0) fb_deliver(frames);
}

static void fb_done(void *arg) {   // completion thread, before the fence counts as done
    FbJob *j = arg;
    if (readback_frames() > 0) {
        __atomic_store_n(&j->done, 1, __ATOMIC_RELEASE);
        fb_deliver(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        return;
    }
    fb_write(j);
}

int vk_texture_fence_readback(VkCommandBuffer cb, uint64_t fence) {
    if (fb_on < 0) fb_on = rt_knob("FENCE_READBACK", readback_frames() ? "1" : "0");
    if (!fb_on || !__atomic_load_n(&small_pending, __ATOMIC_ACQUIRE)) return 0;
    lock();
    wb_cfg_init();
    FbJob *j = NULL;
    int n = 0, cap = 0, rec = 0;
    plat_mutex_lock_shared(&owned_lock);
    for (int i = 0; i < nowned; i++) {
        Tex *e = owned[i];
        VkImageAspectFlags aspect;
        if (!e->gpu_dirty || e->gpu_fence != fence || e->t.L.size > FB_MAX || !wb_wanted(e) || e->f.astc_bw ||
            e->f.expand_rgb || !read_aspect(e, &aspect))
            continue;
        const nvn_texdesc *d = &e->t.d;
        int linear = nvn_tex_is_linear(d), is3d = d->target == NVN_TEXTURE_TARGET_3D;
        uint32_t levels = linear ? 1 : e->pub.levels, layers = linear || is3d ? 1 : e->pub.layers;
        for (uint32_t lv = 0; lv < levels; lv++)
            for (uint32_t ly = 0; ly < layers; ly++) {
                FbPart q = {.t = e->t};
                view_t vw = {.set_levels = 1, .set_layers = 1, .base_level = (uint8_t)lv, .min_layer = (uint16_t)ly};
                q.r = (NVNcopyRegion){.width = (int)mip(d->width, lv),
                                      .height = d->target == 0 || d->target == 3 ? 1 : (int)mip(d->height, lv),
                                      .depth = is3d ? (int)mip(d->depth, lv) : 1};
                if (!region_of(&e->t, &vw, &q.r, 0, 0, &q.g) || q.g.nz > 8) continue;
                size_t slice = q.g.wb * q.g.hb, bytes = slice * (size_t)q.g.nz;
                VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                              .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                       VMA_ALLOCATION_CREATE_MAPPED_BIT};
                VmaAllocationInfo info;
                if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &q.sb, &q.sa, &info) != VK_SUCCESS) continue;
                if (!info.pMappedData) {
                    vmaDestroyBuffer(vk_ctx()->vma, q.sb, q.sa);
                    continue;
                }
                q.stg = info.pMappedData;
                VkBufferImageCopy rg[8];
                for (int z = 0; z < q.g.nz; z++)
                    rg[z] = (VkBufferImageCopy){.bufferOffset = slice * (size_t)z,
                                                .imageSubresource = {aspect, q.g.level, q.g.is3d ? 0 : q.g.layer0 + (uint32_t)z, 1},
                                                .imageOffset = {0, 0, q.g.is3d ? z : 0},
                                                .imageExtent = {(uint32_t)q.r.width, (uint32_t)q.r.height, 1}};
                if (!rec) barrier(cb, ALL_RW, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT), rec++;
                vkCmdCopyImageToBuffer(cb, e->pub.image, VK_IMAGE_LAYOUT_GENERAL, q.sb, (uint32_t)q.g.nz, rg);
                rec++;
                if (n == cap) {
                    cap = cap ? cap * 2 : 8;
                    FbJob *nj = realloc(j, sizeof *j + sizeof(FbPart) * (size_t)cap);
                    if (!nj) rt_fatal("vk: out of memory for a fence readback");
                    j = nj;
                }
                j->p[n++] = q;
                if (rb_frames > 0) e->lagged = 1;   // its bytes go to storage through the staged jobs only
            }
    }
    plat_mutex_unlock_shared(&owned_lock);
    unlock();
    if (!n) return rec;
    barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT);
    rec++;
    j->n = n;
    j->next = NULL, j->done = 0, j->frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (rb_frames > 0) {
        plat_mutex_lock(&dq_lock);
        if (dq_tail) dq_tail->next = j;
        else dq_head = j;
        dq_tail = j;
        plat_mutex_unlock(&dq_lock);
    }
    vk_completion_add(fence, fb_done, j);
    static int said;
    if (!said++) rt_log("vk: fence readback: small render targets go back to their storage at their fence (first: %d parts)", n);
    return rec;
}

// ---------------------------------------------------------------- samplers (LAS textures.c)
static VkCompareOp compare_op(uint32_t f) {
    switch (f) {
    case NVN_COMPARE_FUNC_NEVER: return VK_COMPARE_OP_NEVER;
    case NVN_COMPARE_FUNC_LESS: return VK_COMPARE_OP_LESS;
    case NVN_COMPARE_FUNC_EQUAL: return VK_COMPARE_OP_EQUAL;
    case NVN_COMPARE_FUNC_LEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case NVN_COMPARE_FUNC_GREATER: return VK_COMPARE_OP_GREATER;
    case NVN_COMPARE_FUNC_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case NVN_COMPARE_FUNC_GEQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static struct { int init, mirror_clamp_edge, minmax, border_no_format; } sf;
static float material_aniso;

static void sampler_features(VkCtx *c) {
    if (sf.init) return;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                            .pNext = c->has_custom_border_color ? &cbf : NULL};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    sf.mirror_clamp_edge = f12.samplerMirrorClampToEdge;
    sf.minmax = f12.samplerFilterMinmax;
    sf.border_no_format = c->has_custom_border_color && cbf.customBorderColors && cbf.customBorderColorWithoutFormat;
    sf.init = 1;
}

static VkSampler make_sampler(VkCtx *c, const samp_par *d) {
    sampler_features(c);
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    uint32_t mf = d->minf;
    si.minFilter = mf == 1 || mf == 3 || mf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode modes[3];
    const uint8_t wrap[3] = {d->ws, d->wt, d->wr};
    int border = 0;
    for (int i = 0; i < 3; i++) {
        switch (wrap[i]) {
        case NVN_WRAP_MODE_CLAMP: case NVN_WRAP_MODE_CLAMP_TO_EDGE: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE:
            modes[i] = sf.mirror_clamp_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            break;
        case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: case NVN_WRAP_MODE_CLAMP_TO_BORDER:
            modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            border = 1;
            break;
        case NVN_WRAP_MODE_MIRRORED_REPEAT: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
    si.addressModeU = modes[0], si.addressModeV = modes[1], si.addressModeW = modes[2];
    si.minLod = d->lod_min;
    si.maxLod = d->lod_max > 1000 ? VK_LOD_CLAMP_NONE : d->lod_max;
    if (mf <= NVN_MIN_FILTER_LINEAR) si.maxLod = si.minLod + 0.25f;   // not mipmapped: stay on the base level
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    float lim = c->props.limits.maxSamplerLodBias;
    si.mipLodBias = fmaxf(-lim, fminf(lim, d->lod_bias));
    if (d->aniso > 1) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = d->aniso;
    }
    if (d->cmp_mode) {
        si.compareEnable = VK_TRUE;
        si.compareOp = compare_op(d->cmp_func);
        // A depth-compare sampler set to nearest returns 0 or 1 per tap, and the game's shaders smooth shadow edges by
        // a few taps spaced for its own shadow map texels. With REKARINA_SHADOW_SCALE those taps cover a fraction of the
        // area and the edges turn hard and stepped; linear filtering makes every tap a 2x2 PCF instead. OoT's sampler
        // of its cascades is linear already, and its other depth maps (D32F, single-layer D16) keep their size, so
        // their samplers stay as the game set them.
#ifndef GAME_OOT
        if (vk_shadow_scale() > 1) si.magFilter = si.minFilter = VK_FILTER_LINEAR;
#endif
    }
    VkSamplerCustomBorderColorCreateInfoEXT cb = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rm = {.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    const void **tail = &si.pNext;
    if (border) {
        const float *b = d->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (sf.border_no_format) {
            memcpy(cb.customBorderColor.float32, b, 16);
            si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            *tail = &cb;
            tail = &cb.pNext;
        } else si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    if (d->reduction && sf.minmax && !si.compareEnable) {
        rm.reductionMode = d->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        *tail = &rm;
    }
    VkSampler s;
    VkResult r = vkCreateSampler(c->device, &si, NULL, &s);
    if (r != VK_SUCCESS) rt_fatal("vk: vkCreateSampler failed (%d)", (int)r);
    return s;
}

void vk_sampler_register(int id, const samp_par *s) {
    if (id < 0 || id >= (1 << 12)) rt_fatal("vk: sampler slot %d out of range", id);
    lock_off_core();
    if (id >= nsslots) {
        int n = nsslots ? nsslots : 256;
        while (n <= id) n *= 2;
        SampSlot *p = realloc(sslots, sizeof *p * (size_t)n);
        if (!p) rt_fatal("vk: out of memory for sampler slots");
        memset(p + nsslots, 0, sizeof *p * (size_t)(n - nsslots));
        sslots = p, nsslots = n;
    }
    sslots[id].p = *s;
    sslots[id].s = sslots[id].sp = sslots[id].sc = VK_NULL_HANDLE;
    sslots[id].used = 1;
    if (id >= 0 && id < 4096) samp_reg[id] = 1;
    samp_gen++;
    unlock();
}

// DLSS upscaling (vk_sampler_dlss_bias): the scene's material textures, the filtered mipmapped ones that repeat, are
// sampled for the output's resolution; clamped ones (post effects, LUTs, the UI) keep the game's bias.
static float dlss_bias;

static int dlss_biased(const samp_par *k) {
    int rep_s = k->ws == NVN_WRAP_MODE_REPEAT || k->ws == NVN_WRAP_MODE_MIRRORED_REPEAT;
    int rep_t = k->wt == NVN_WRAP_MODE_REPEAT || k->wt == NVN_WRAP_MODE_MIRRORED_REPEAT;
    static int all = -1;   // DEBUG-BIAS: REKARINA_DLSS_BIAS_ALL=1 also takes the clamped trilinear samplers
    if (all < 0) { const char *e = cfg_get("REKARINA_DLSS_BIAS_ALL"); all = !e || atoi(e) > 0; }
    return !k->cmp_mode && k->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR && (rep_s || rep_t || all);
}

// The sharpness slider's hold on the game's own texture sharpening (vk_sampler_sharpness): the game samples its scene
// textures half a mip level sharp (LOD bias -0.5 on its repeating material samplers, 4,145 of a village frame's sampler
// uses), made for a scene of 0.75 of the picture that its scaling enlarges. At the port's full-size scene, with no
// temporal anti-aliasing, that reads as hard, shimmering detail, and at a native picture no other sharpening runs for
// the slider to take back. A positive b moves those negative biases towards 0, never past it, and touches no other
// sampler.
static float sharp_bias;

void vk_sampler_anisotropy(void) {
    const char *value = cfg_value("REKARINA_ANISO");
    float level = (float)atoi(value);
    lock();
    if (level != material_aniso) {
        material_aniso = level;
        for (int i = 0; i < nsslots; i++) sslots[i].s = sslots[i].sp = sslots[i].sc = VK_NULL_HANDLE;
        samp_gen++;
        rt_log("vk: anisotropic filtering: %s", value);
    }
    unlock();
}

void vk_sampler_sharpness(float b) {
    lock();
    if (b != sharp_bias) {
        sharp_bias = b;
        for (int i = 0; i < nsslots; i++) sslots[i].s = sslots[i].sp = sslots[i].sc = VK_NULL_HANDLE;   // looked up again (the key moves)
        samp_gen++;
    }
    unlock();
}

void vk_sampler_dlss_bias(float b) {
    lock();
    if (b != dlss_bias) {
        dlss_bias = b;
        for (int i = 0; i < nsslots; i++) sslots[i].s = sslots[i].sp = sslots[i].sc = VK_NULL_HANDLE;   // looked up again (the key moves)
        samp_gen++;
    }
    unlock();
}

// Caller holds the lock.
static VkSampler sampler_for(VkCtx *c, const samp_par *s) {
    samp_par k;
    memset(&k, 0, sizeof k);
    k = *s;
    // Leave clamped screen buffers alone: extra taps can reach across an object's edge.
    if (material_aniso > k.aniso && !k.cmp_mode && k.magf == NVN_MAG_FILTER_LINEAR &&
        (k.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || k.minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR) &&
        (k.ws == NVN_WRAP_MODE_REPEAT || k.ws == NVN_WRAP_MODE_MIRRORED_REPEAT) &&
        (k.wt == NVN_WRAP_MODE_REPEAT || k.wt == NVN_WRAP_MODE_MIRRORED_REPEAT))
        k.aniso = material_aniso;
    k.aniso = fminf(k.aniso, c->props.limits.maxSamplerAnisotropy);
    if (sharp_bias != 0 && k.lod_bias < 0 && dlss_biased(&k)) k.lod_bias = fminf(0.0f, k.lod_bias + sharp_bias);
    if (dlss_bias != 0 && dlss_biased(&k)) k.lod_bias += dlss_bias;   // part of the key: samplers live until shutdown
    VkSampler v = (VkSampler)nvn_hm_get(&samplers, &k);
    if (!v) {
        v = make_sampler(c, &k);
        nvn_hm_put(&samplers, &k, (void *)v);
    }
    return v;
}

VkSampler vk_sampler_for(const samp_par *s) {
    VkCtx *c = vk_ctx();
    if (!c || !s) return VK_NULL_HANDLE;
    lock();
    VkSampler v = sampler_for(c, s);
    unlock();
    return v;
}

// Samplers live until shutdown, so a slot keeps the one its parameters map to until it is registered again.
VkSampler vk_sampler_by_id_locked(int id) {
    VkCtx *c = vk_ctx();
    if (!c) return VK_NULL_HANDLE;
    VkSampler v = VK_NULL_HANDLE;
    if (id >= 0 && id < nsslots && sslots[id].used) {
        SampSlot *s = &sslots[id];
        if (!s->s) s->s = sampler_for(c, &s->p);
        v = s->s;
    }
    return v;
}

// DEBUG-SAMP3D: REKARINA_SAMP3D=1 logs the filters and wrap modes of each sampler paired with a 3D texture, once per sampler.
void vk_debug_samp3d_locked(int id, int separate, int format) {
    static uint8_t seen[4096];
    if (id < 0 || id >= nsslots || id >= 4096 || seen[id] || !sslots[id].used || !cfg_env("REKARINA_SAMP3D")) return;
    seen[id] = 1;
    const samp_par *p = &sslots[id].p;
    rt_log("SAMP3D sampler %d separate %d format %d: min %u mag %u wrap %u %u %u", id, separate, format, p->minf, p->magf, p->ws, p->wt, p->wr);
}

// DEBUG-IMGWHERE: where the image of texture slot id lives (own memory or bound into pool memory, the memory type's flags).
void vk_debug_image_where_locked(int id, const char *what) {
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked(id, &v, &hv);
    if (!img) return rt_log("IMGWHERE %s slot %d: no image", what, id);
    const Tex *e = (const Tex *)((const char *)img - offsetof(Tex, pub));
    VmaAllocation a = e->alloc ? e->alloc : e->bound_to;
    VkMemoryPropertyFlags f = 0;
    VmaAllocationInfo info = {0};
    if (a) vmaGetAllocationInfo(vk_ctx()->vma, a, &info), vmaGetMemoryTypeProperties(vk_ctx()->vma, info.memoryType, &f);
    rt_log("IMGWHERE %s addr %#llx %ux%ux%u vk fmt %d levels %u usage %#x mirrored %d %s memtype %u flags %#x offset %#llx", what,
           (unsigned long long)img->addr, img->width, img->height, img->depth, (int)img->format, img->levels, (unsigned)img->usage,
           img->mirrored, e->alloc ? "own memory" : e->bound_to ? "pool memory" : "none", info.memoryType, (unsigned)f,
           (unsigned long long)info.offset);
}

// A depth-compare read through a sampler without a compare is undefined in Vulkan. NVIDIA's driver compares anyway; AMD's
// returns the raw depth, so the game's shadow tests come out wrong there. Such a read gets the slot's sampler with the
// compare on: the game's function when it set a valid one, LEQUAL otherwise. REKARINA_SHADOW_COMPARE=0: the game's sampler.
VkSampler vk_sampler_compare_by_id_locked(int id) {
    VkCtx *c = vk_ctx();
    if (!c || id < 0 || id >= nsslots || !sslots[id].used) return VK_NULL_HANDLE;
    SampSlot *s = &sslots[id];
    static int on = -1;
    if (on < 0) on = rt_knob("SHADOW_COMPARE", "1");
    if (s->p.cmp_mode || !on) return vk_sampler_by_id_locked(id);
    if (!s->sc) {
        samp_par k = s->p;
        k.cmp_mode = 1;
        if (k.cmp_func < NVN_COMPARE_FUNC_NEVER || k.cmp_func > NVN_COMPARE_FUNC_ALWAYS) k.cmp_func = NVN_COMPARE_FUNC_LEQUAL;
        s->sc = sampler_for(c, &k);
    }
    return s->sc;
}

VkSampler vk_sampler_point_by_id_locked(int id) {
    VkCtx *c = vk_ctx();
    if (!c || id < 0 || id >= nsslots || !sslots[id].used) return VK_NULL_HANDLE;
    SampSlot *s = &sslots[id];
    if (!s->sp) {
        samp_par k = s->p;
        k.magf = NVN_MAG_FILTER_NEAREST;
        k.minf = k.minf == NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST :
                 k.minf > NVN_MIN_FILTER_LINEAR ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : k.minf;
        k.aniso = 1;
        s->sp = sampler_for(c, &k);
    }
    return s->sp;
}

VkSampler vk_sampler_by_id(int id) {
    lock();
    VkSampler v = vk_sampler_by_id_locked(id);
    unlock();
    return v;
}

int vk_sampler_params(int id, samp_par *out) {
    lock();
    int ok = id >= 0 && id < nsslots && sslots[id].used;
    if (ok) *out = sslots[id].p;
    unlock();
    return ok;
}

// ---------------------------------------------------------------- statistics and shutdown
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads) {
    lock();
    if (images) *images = by_key.n;
    if (uploads) *uploads = n_uploads;
    if (reuploads) *reuploads = n_reuploads;
    unlock();
}

void vk_texture_census(VkTexCensus *s) {
    VkCtx *c = vk_ctx();
    memset(s, 0, sizeof *s);
    lock();
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e) continue;
        uint64_t bytes = 0;
        if (e->alloc && c) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(c->vma, e->alloc, &ai);
            bytes = ai.size;
        }
        if (e->released) s->released++, s->own_released += e->alloc != NULL, s->bytes_released += bytes;
        else s->live++, s->own_live += e->alloc != NULL, s->bytes_live += bytes;
    }
    unlock();
}

void vk_texture_shutdown(void) {
    VkCtx *c = vk_ctx();
    pm_gone(0, ~0ull, NULL, 0);
    lock();
    Tex **list = malloc(sizeof *list * (size_t)(by_key.n + 1));
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++)
        if (by_key.vals[i]) list[n++] = by_key.vals[i];
    for (int i = 0; i < n; i++) destroy_tex(list[i]);
    free(list);
    park_clear();
    ma_expire(~0ull);   // the images made ahead that nobody took (REKARINA_TEX_MAKE_AHEAD)
    for (uint32_t i = 0; i < samplers.cap; i++)
        if (samplers.vals[i] && c) vkDestroySampler(c->device, (VkSampler)samplers.vals[i], NULL);
    free(by_key.keys), free(by_key.vals), free(by_addr.keys), free(by_addr.vals), free(samplers.keys), free(samplers.vals);
    free(cold_keys.keys), free(cold_keys.vals);
    cold_keys = (nvn_hmap){.klen = sizeof(TexKey)}, n_cold = 0;
    by_key = (nvn_hmap){.klen = sizeof(TexKey)};
    by_addr = (nvn_hmap){.klen = 8};
    samplers = (nvn_hmap){.klen = sizeof(samp_par)};
    free(tslots), free(sslots);
    tslots = NULL, sslots = NULL, ntslots = nsslots = 0, nreleased = 0, pool_slots = 0;
    rel_head = rel_tail = NULL;
    if (sinfo) memset(sinfo, 0, sizeof *sinfo << 20);   // no slot is registered: tokens 0
    unlock();
}

// DEBUG-LIVECHECK: what slot id is and resolves to (the caller holds the texture lock or not: it is recursive)
void vk_texture_slot_log(int id) {
    lock();
    if (id < 0 || id >= ntslots || !tslots[id].used) {
        rt_log("LIVE CHECK: slot %d is not registered", id);
        unlock();
        return;
    }
    const TexSlot *s = &tslots[id];
    const Tex *e = s->ce;
    rt_log("LIVE CHECK: slot %d: texture at %#llx %ux%u fmt %#x levels %u%s, serial %u, resolved under generation %u (now %u); "
           "entry %p%s%s image %#llx addr %#llx %ux%u, bound into pool memory %d, gpu-owned %d, uploaded %d",
           id, (unsigned long long)(uintptr_t)s->t.mem, s->t.d.width, s->t.d.height, s->t.d.format, s->t.d.levels,
           s->has_view ? " (a view)" : "", (uint32_t)(sinfo[id] >> 16), s->cgen, agen[sinfo[id] & 0xffff], (const void *)e,
           e && e->released ? " RELEASED" : "", e && e->pub.gone ? " GONE" : "",
           e ? (unsigned long long)(uintptr_t)e->pub.image : 0ull, e ? (unsigned long long)e->key.addr : 0ull,
           e ? e->pub.width : 0, e ? e->pub.height : 0, e ? e->bound_to != NULL : 0, e ? e->pub.gpu_owned : 0, e ? e->uploaded : 0);
    unlock();
}

// DEBUG-IMGWATCH: REKARINA_IMGWATCH=1: the reaper calls this right before it destroys an image (vk_memory.c reap_one).
// A live or released entry, or the image cache, that still names the image then is a reader the destruction leaves
// pointing at freed memory: the pool's next update gives its slot a view of the dead image (a lost device when a
// shader samples it). Logged with how and when the entry got the image and who handed it to the deferral. The lock is
// only tried (the reaper must not wait for the texture lock); what it misses is counted.
static int imgwatch_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_get("REKARINA_IMGWATCH") != NULL;
    return on;
}
static uint64_t iw_checked, iw_late, iw_hits;
// tex_lock held: the scan for one destroyed image (frame: when the reaper destroyed it)
static void iw_check_locked(uint64_t image, const void *site, int64_t site_off, uint64_t dframe, int late) {
    int vk_dd_reborn(uint64_t h);
    if (late && vk_dd_reborn(image)) return;   // destroyed meanwhile and a new image has the handle: not this one
    iw_checked++;
    uint64_t now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || (uint64_t)(uintptr_t)e->pub.image != image) continue;
        if (e->img_frame > dframe) continue;   // took the handle after the destruction: a new image with the old value
        if (iw_hits++ < 300) {
            int ns = 0, first = -1;
            for (int id = 0; id < ntslots; id++)
                if (tslots[id].used && tslots[id].ce == e) { if (first < 0) first = id; ns++; }
            rt_log("IMGWATCH frame %llu: image %#llx destroyed at frame %llu (handed over by %p, vk_defer_image%+lld) while "
                   "entry %p names it%s: texture %#llx %ux%u fmt %#x levels %u%s%s, got that image %s at frame %llu, released "
                   "at %llu, last lookup %llu, last draw %llu; %d slots resolve to it (first %d)", (unsigned long long)now,
                   (unsigned long long)image, (unsigned long long)dframe, site, (long long)site_off, (const void *)e,
                   late ? " (checked after the destruction)" : "", (unsigned long long)e->key.addr, e->key.d.width, e->key.d.height, e->key.d.format, e->key.d.levels,
                   e->released ? " RELEASED" : " live", e->pub.gone ? " GONE" : "", e->img_how ? e->img_how : "made for it",
                   (unsigned long long)e->img_frame, (unsigned long long)(e->released ? e->rel_frame : 0),
                   (unsigned long long)e->use_frame, (unsigned long long)e->draw_frame, ns, first);
            vk_addr_label_print(image);
        }
    }
    for (int i = 0; i < nparked; i++)
        if ((uint64_t)(uintptr_t)parked[i].image == image && parked[i].frame <= dframe && iw_hits++ < 300)
            rt_log("IMGWATCH frame %llu: image %#llx destroyed at frame %llu (handed over by %p) while the image cache holds "
                   "it (parked at frame %llu, kept for a move %d)", (unsigned long long)now, (unsigned long long)image,
                   (unsigned long long)dframe, site, (unsigned long long)parked[i].frame, (int)parked[i].moved);
    if (!(iw_checked & 65535))
        rt_log("IMGWATCH: %llu image destructions checked (%llu of them later, the lock being held), %llu found named",
               (unsigned long long)iw_checked, (unsigned long long)iw_late, (unsigned long long)iw_hits);
}
// Destructions the reaper could not check at once (the lock was held): checked at the next sweep (iw_drain).
#define IW_RING 32768
static struct { uint64_t image, frame; const void *site; int64_t off; } iw_ring[IW_RING];
static volatile uint32_t iw_head, iw_tail;
static PlatMutex iw_lock = PLAT_MUTEX_INIT;
void vk_texture_reap_check(uint64_t image, const void *site, int64_t site_off) {
    if (!imgwatch_on() || !image) return;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (plat_rmutex_trylock(&tex_lock)) {
        iw_check_locked(image, site, site_off, f, 0);
        plat_rmutex_unlock(&tex_lock);
        return;
    }
    plat_mutex_lock(&iw_lock);
    if (iw_head - iw_tail < IW_RING) iw_ring[iw_head % IW_RING] = (typeof(iw_ring[0])){image, f, site, site_off}, iw_head++;
    plat_mutex_unlock(&iw_lock);
}
// DEBUG-IMGWATCH: at the hand-over itself (tex_lock held; the four places below that give an image of this module to the
// deferral): an image that the image cache still holds, or that a live or released entry other than `owner` names,
// is about to be destroyed under a holder. The source line names the path.
static void iw_defer_check(VkImage img, const Tex *owner, int line) {
    if (!imgwatch_on() || !img) return;
    static uint64_t n;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int i = 0; i < nparked; i++)
        if (parked[i].image == img && n++ < 300)
            rt_log("IMGWATCH frame %llu: vk_texture.c:%d hands image %#llx to the deferral while the image cache holds it (parked at "
                   "frame %llu)", (unsigned long long)f, line, (unsigned long long)(uintptr_t)img, (unsigned long long)parked[i].frame);
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || e == owner || e->pub.image != img || n++ >= 300) continue;
        rt_log("IMGWATCH frame %llu: vk_texture.c:%d hands image %#llx to the deferral while entry %p names it (texture %#llx "
               "%ux%u fmt %#x%s, got it %s at frame %llu)", (unsigned long long)f, line, (unsigned long long)(uintptr_t)img,
               (const void *)e, (unsigned long long)e->key.addr, e->key.d.width, e->key.d.height, e->key.d.format,
               e->released ? " RELEASED" : " live", e->img_how ? e->img_how : "made for it", (unsigned long long)e->img_frame);
        vk_addr_label_print((uint64_t)(uintptr_t)img);
    }
}

static void iw_drain(void) {   // tex_lock held
    if (!imgwatch_on()) return;
    for (;;) {
        plat_mutex_lock(&iw_lock);
        if (iw_tail == iw_head) { plat_mutex_unlock(&iw_lock); return; }
        typeof(iw_ring[0]) r = iw_ring[iw_tail % IW_RING];
        iw_tail++;
        plat_mutex_unlock(&iw_lock);
        iw_late++;
        iw_check_locked(r.image, r.site, r.off, r.frame, 1);
    }
}

// DEBUG-WHONAMES (the device fault report): entries, kept images and slots that name the image
void vk_texture_who_names(uint64_t image) {
    lock();
    const Tex *found[8];
    int nf = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        const Tex *e = by_key.vals[i];
        if (!e || (uint64_t)(uintptr_t)e->pub.image != image) continue;
        if (nf < 8) found[nf++] = e;
        rt_log("  entry %p names it: addr %#llx %ux%u fmt %#x%s%s uploaded %d gpu_owned %d bound %d wr %llu", (const void *)e,
               (unsigned long long)e->key.addr, e->key.d.width, e->key.d.height, e->key.d.format, e->released ? " RELEASED" : " live",
               e->pub.gone ? " GONE" : "", e->uploaded, e->pub.gpu_owned, e->bound_to != NULL, (unsigned long long)e->wr_serial);
    }
    for (int i = 0; i < nparked; i++)
        if ((uint64_t)(uintptr_t)parked[i].image == image) rt_log("  the image cache holds it (kept for a move: %d)", (int)parked[i].moved);
    int ns = 0;
    for (int id = 0; id < ntslots; id++) {
        if (!tslots[id].used || !tslots[id].ce) continue;
        for (int k = 0; k < nf; k++)
            if (tslots[id].ce == found[k] && ns++ < 8)
                rt_log("  slot %d is resolved to entry %p (resolved under generation %u, now %u)", id, (const void *)found[k],
                       tslots[id].cgen, agen[sinfo[id] & 0xffff]);
    }
    rt_log("  texture module: %d entries name it, %d slots resolved to them", nf, ns);
    uint64_t now = plat_now_ns(), fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    uint32_t have = dt_ring ? (dt_next < DT_N ? dt_next : DT_N) : 0;
    for (uint32_t k = 0, shown = 0; k < have && shown < 4; k++) {
        const typeof(*dt_ring) *d = &dt_ring[(dt_next - 1 - k) % DT_N];
        if (d->image != image) continue;
        shown++;
        rt_log("  destroyed %.1f ms (%llu frames) before the report by the caller at destroy_tex%+lld: entry %#llx %ux%u fmt %#x, "
               "%s (released at frame %llu), bound into pool memory %d, gpu-owned %d, last lookup frame %llu, last draw frame %llu, "
               "destroyed at frame %llu", (now - d->ns) / 1e6, (unsigned long long)(fr - d->frame), (long long)d->ra,
               (unsigned long long)d->addr, d->w, d->h, d->fmt, d->released ? "released" : "LIVE", (unsigned long long)d->rel_frame,
               (int)d->bound, (int)d->gpu, (unsigned long long)d->use_frame, (unsigned long long)d->draw_frame,
               (unsigned long long)d->frame);
        if (shown > 1) continue;
        int nsl = 0;   // the slots registered at that storage now: what a shader indexing them gets
        for (int id = 0; id < ntslots && nsl < 8; id++) {
            const TexSlot *s = &tslots[id];
            if (!s->used || (uint64_t)(uintptr_t)s->t.mem != d->addr) continue;
            nsl++;
            rt_log("  slot %d is registered at that storage: %ux%u fmt %#x target %d, resolved to %p (image %#llx%s%s)", id,
                   s->t.d.width, s->t.d.height, s->t.d.format, s->t.d.target, (const void *)s->ce,
                   s->ce ? (unsigned long long)(uintptr_t)s->ce->pub.image : 0ull, s->ce && s->ce->released ? ", released" : "",
                   s->ce && s->ce->pub.gone ? ", GONE" : "");
        }
        rt_log("  %d slots registered at that storage", nsl);
    }
    unlock();
}
