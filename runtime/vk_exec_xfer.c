// Transfers of the Vulkan op-stream replay: buffer and texture copies, texture
// clears, event signals, counter reports and conditional rendering. vk_exec.c calls vk_exec_op_xfer for these ops.
//
// Buffers are guest memory: vk_mem_resolve maps an address to the VkBuffer behind its pool, textures are the vk_texture.c images
// (linear texel layout, GENERAL layout; block-linear storage is only converted when a mirrored texture is uploaded). So a
// copy is one Vulkan transfer command and the pitch of a buffer side comes from the payload (OP_COPY_STRIDE or the
// tight row). Every copy ends the open pass and starts with a full barrier. Destinations: an image is marked
// GPU-written at record time; a CPU-visible buffer range is reported to vk_gpu_written once the GPU is done (write
// watch does not see GPU writes, textures stored there re-upload).
//
// Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
// vk_completion_add, in submit order, as gfx_null.c does: event WRITE / ADD, counter {0, timestamp} for every type but
// samples passed. A timestamp report puts 0 in the counter word: TOTK gates its clears on two such words being unequal.
//
// Samples passed (NVN_COUNTER_TYPE_SAMPLES_PASSED) is a running count on the queue: RESET_COUNTER zeroes it, every
// draw after it adds the samples that passed the depth and stencil tests, REPORT_COUNTER writes the running value.
// TOTK resets and reports it 3 to 8 times a frame in the field and its CPU reads the results back (f_70010c3298); a 0
// there culls what the query tested. The replay counts with Vulkan occlusion queries ("occlusion counts" below).
//
// Conditional rendering is a CPU-side skip, not VK_EXT_conditional_rendering: the extension compares one 32-bit word
// with zero, NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded
// from guest memory; a counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp,
// as unknown if it is samples passed. An unknown count renders: the report usually sits in the command buffer being
// recorded, so its value cannot be waited for, and the word in memory still holds an older query's count. Rendering
// costs at most the work the condition would have saved (MacNativeGames nvn_mtl.m decides the same); in the field
// TOTK conditions only on timestamp reports. The condition sets VkExec.render_disabled, which vk_exec.c honours for
// clears and vk_exec_draw.c for draws and dispatches. A condition that cannot be read (address outside every pool)
// renders.
// Copies and CLEAR_TEX are not conditional (as in MacNativeGames, nvn_mtl.m).
//
// The copy semantics (region and layer arithmetic, row and image strides, whole-level clears, copy / blit / resolve
// choice) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10080-10368, 10618-10642, 1700-1725
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou) and LAS Recomp runtime/nvn/vulkan/exec.c:3045-3235 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "ftrace.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"

enum { W_FORMAT, W_REGION, W_BUFFER, W_OVERLAP, W_COPY, W_CLEAR, W_QUERY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static uint32_t max1(uint32_t v) { return v ? v : 1; }

// ---- completion callbacks (completion thread)

typedef struct { uint64_t addr, size; int owned; } Range;   // owned: a VRAM mirror wrote it and copies it back whole

static void guest_written_cb(void *arg) {
    Range *r = arg;
    if (r->owned) vk_gpu_written_mirror(r->addr, r->size);
    else vk_gpu_written(r->addr, r->size);
    free(r);
}

// The GPU wrote [addr, addr+size) of CPU-visible memory: tell the texture cache once the fence is reached.
static void note_gpu_write(VkExec *ex, uint64_t addr, uint64_t size, int owned) {
    vk_usnap_drop(addr, size);
    if (nvn_gpu_only(addr)) return;
    Range *r = malloc(sizeof *r);
    if (!r) rt_fatal("vk: out of memory");
    *r = (Range){addr, size, owned};
    vk_mem_gpu_pending(addr, size);
    vk_completion_add(ex->fence, guest_written_cb, r);
}

typedef struct { volatile uint32_t *val; uint32_t mode, value; } EventWrite;

// GPU signals of an event recorded and not applied yet (signal_event .. event_cb), per event word: a wait for such an
// event is ordered by the queue (the signalling submission runs first), not by the CPU.
#define EVP_N 256u
static struct { volatile uint32_t *val; int32_t n; } evp[EVP_N];
static PlatMutex evp_lock = PLAT_MUTEX_INIT;
static void evp_add(volatile uint32_t *v, int32_t d) {
    plat_mutex_lock(&evp_lock);
    uint32_t h = (uint32_t)(((uintptr_t)v >> 2) * 2654435761u) & (EVP_N - 1);
    for (uint32_t k = 0; k < EVP_N; k++, h = (h + 1) & (EVP_N - 1)) {
        if (evp[h].val == v) { evp[h].n += d; break; }
        if (!evp[h].val) { evp[h].val = v, evp[h].n = d; break; }
    }
    plat_mutex_unlock(&evp_lock);
}
static int evp_pending(volatile uint32_t *v) {
    int n = 0;
    plat_mutex_lock(&evp_lock);
    uint32_t h = (uint32_t)(((uintptr_t)v >> 2) * 2654435761u) & (EVP_N - 1);
    for (uint32_t k = 0; k < EVP_N && evp[h].val; k++, h = (h + 1) & (EVP_N - 1))
        if (evp[h].val == v) { n = evp[h].n; break; }
    plat_mutex_unlock(&evp_lock);
    return n > 0;
}

// DEBUG-EVLAG: REKARINA_EVENT_LAG=<k>: a GPU signal reaches the event word only k presents after the frame it was
// recorded in (the console's GPU a frame behind its CPU), in order
typedef struct EvLag { struct EvLag *next; void *ev; uint64_t frame; } EvLag;
static EvLag *evl_head, *evl_tail;
static PlatMutex evl_lock = PLAT_MUTEX_INIT;
static int ev_lag = -1;
static void event_apply(void *arg);
void vk_event_lag_tick(uint64_t frames) {
    if (ev_lag <= 0) return;
    plat_mutex_lock(&evl_lock);
    while (evl_head && evl_head->frame + (uint64_t)ev_lag <= frames) {
        EvLag *l = evl_head;
        evl_head = l->next;
        if (!evl_head) evl_tail = NULL;
        event_apply(l->ev);
        free(l);
    }
    plat_mutex_unlock(&evl_lock);
}
static void event_cb(void *arg);
static void event_apply(void *arg) {
    EventWrite *e = arg;
    mem_result_release((uint64_t)(uintptr_t)e->val, sizeof *e->val);   // REKARINA_RESULT_WATCH: the page opens for the write
    if (e->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, e->value, __ATOMIC_RELEASE);
    else __atomic_store_n(e->val, e->value, __ATOMIC_RELEASE);
    evp_add(e->val, -1);
    free(e);
}
typedef struct { EventWrite w; uint64_t frame; } EventLagged;
static void event_cb(void *arg) {   // completion thread
    if (ev_lag <= 0) { event_apply(arg); return; }
    EventLagged *x = arg;
    EvLag *l = malloc(sizeof *l);
    if (!l) { event_apply(arg); return; }
    l->ev = arg, l->frame = x->frame, l->next = NULL;
    plat_mutex_lock(&evl_lock);
    if (evl_tail) evl_tail->next = l;
    else evl_head = l;
    evl_tail = l;
    plat_mutex_unlock(&evl_lock);
    vk_event_lag_tick(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
}

// OP_WAIT_EVENT (nvnCommandBufferWaitEvent). The console's GPU stops there until the event's word satisfies the mode;
// here the replay, which takes the submission's copies of guest memory (uniform blocks, vk_usnap.c) as it records,
// stops instead, until the CPU's nvnEventSignal: OoT records such a wait at the start of each frame's
// lighting, for an event it sets once it has written that frame's late constants (the light grid of the clustered
// lighting: before it, the scene block holds a 1x1x1 placeholder, which put header words of the light lists where
// light numbers belong, NaN in the skin shaders and a white frame with frames in flight, docs/oot-notes.md). A word a
// GPU signal earlier in the queue will set needs no wait, nor does a word the CPU has never signalled (the game's
// per-frame words that GPU work of another of its queues sets: one queue here, in submission order). A wait that runs
// out (REKARINA_EVENT_WAIT_MS, default 100: a hitch of the signalling thread) is logged; after three in a row that word is
// not waited for again, as its signal may come from the thread that is replaying.
// Returns 1 when it waited, so that the caller takes the copies again for the rest of the submission.
static int ev_met_v(volatile uint32_t *val, uint32_t mode, uint32_t value) {
    uint32_t v = __atomic_load_n(val, __ATOMIC_ACQUIRE);
    return mode == NVN_EVENT_WAIT_MODE_EQUAL ? v == value : (int32_t)(v - value) >= 0;
}
static int ev_met(const p_wait_event *p) { return ev_met_v(p->val, p->mode, p->value); }
// The value the newest recorded GPU signal gives each event word (recorded, not necessarily run): a wait it meets is
// ordered by the queue, one it does not meet waits for a signal not recorded yet (vk_exec.c "Queues interleaved").
// Replay thread only.
#define RS_N 512u
static struct { volatile uint32_t *val; uint32_t v; uint64_t cp; } rsig[RS_N];   // cp: async compute value of the signal
static int rsig_index(volatile uint32_t *val, int add) {
    uint32_t h = (uint32_t)(((uintptr_t)val >> 2) * 2654435761u) & (RS_N - 1);
    for (uint32_t k = 0; k < RS_N; k++, h = (h + 1) & (RS_N - 1)) {
        if (rsig[h].val == val) return (int)h;
        if (!rsig[h].val) {
            if (!add) return -1;
            rsig[h].val = val, rsig[h].v = __atomic_load_n(val, __ATOMIC_ACQUIRE), rsig[h].cp = 0;
            return (int)h;
        }
    }
    return -1;
}
static uint32_t *rsig_slot(volatile uint32_t *val, int add) {
    int i = rsig_index(val, add);
    return i < 0 ? NULL : &rsig[i].v;
}
uint64_t vk_exec_wait_cp;
// -1: no GPU signal of the word recorded so far, 1: the newest one meets the wait, 0: it does not
static int rsig_state(const p_wait_event *p) {
    const uint32_t *v = rsig_slot(p->val, 0);
    if (!v) return -1;
    return p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? *v == p->value : (int32_t)(*v - p->value) >= 0;
}
static struct { uint64_t waits, met, met_late, pre, gpu, ns, max_ns, timeouts, subs, deferred, def_met, held; } evw;
// A wait the CPU's signal met before the replay reached it needs the copies again all the same when that signal came
// after they were taken. The submission's uniform snapshot is taken when its replay starts and a mirror is synced at
// its first use, and the game thread that writes the frame's late constants (OoT: the light grid of the
// clustered lighting and its light lists) runs beside the replay: when it wrote and signalled while the commands
// before the wait were being replayed, the wait found its word met, nothing was taken again, and the lighting after it
// read the lists as they stood before. Standing still that is last frame's lists and shows nothing; with the camera
// turning, tiles of the screen lost their lights for one to four frames, and the sun patches on the ground with them
// (the player's "shadows drawn again" and flickering grass in the forest, 5 to 12 times in 30 s of turning,
// 2026-11-14). vk_exec_copies_ns is when the copies the commands read now were begun (vk_exec.c: the submission's
// start, or the last wait that took them again). REKARINA_EVENT_LATE_MET=0: as before.
uint64_t vk_exec_copies_ns;
static struct { volatile uint32_t *val; int fails; } ev_skip[16];   // words whose waits ran out three times in a row
static int ev_nskip;
static uint64_t ev_limit_ns;
// The wait for the CPU itself: 1 when the word is met, 0 when it ran out (si: the word's ev_skip slot, or ev_nskip).
static int ev_wait(volatile uint32_t *val, uint32_t mode, uint32_t value, int si) {
    uint64_t t0 = plat_now_ns(), spins = 0;
    while (!ev_met_v(val, mode, value)) {
        uint64_t t = plat_now_ns();
        if (t - t0 > ev_limit_ns) {
            evw.timeouts++;
            if (si == ev_nskip && ev_nskip < 16) ev_skip[ev_nskip++] = (typeof(ev_skip[0])){val, 0};
            int fails = si < ev_nskip ? ++ev_skip[si].fails : 0;
            rt_log("vk: a command buffer's wait for event word %p (mode %u, value %u, now %u) was not met after %.0f ms%s",
                   (void *)val, mode, value, *val, (t - t0) / 1e6, fails >= 3 ? ": three in a row, no longer waited for" : "");
            return 0;
        }
        if (++spins < 64) plat_yield();
        else plat_sleep_ns(50000);
    }
    uint64_t dt = plat_now_ns() - t0;
    if (si < ev_nskip) ev_skip[si].fails = 0;
    evw.waits++, evw.ns += dt;
    if (dt > evw.max_ns) evw.max_ns = dt;
    return 1;
}
// A deferred wait (REKARINA_EVENT_DEFER=1, off by default). On the console the GPU stops at the command and the submitting thread
// goes on; here the replay runs on that thread, and stopping it there cost ~2.5 ms in nearly half the frames of the
// forest porch (OoT's signal comes from another thread while the game thread records the rest). Instead
// the replay records past the wait: the commands after it get a new uniform arena at once (vk_usnap_resnap), and the
// wait happens in vk_exec_wait_resolve, right before the first part after it goes to the GPU (the next split or the
// end of the submission), by when the signal has mostly come. Then the arena takes the CPU's bytes again
// (vk_usnap_refresh) and the mirrors synced since the wait are synced again (vk_mem_resync_accessed): the part reads
// memory as of the signal, as it did when the replay waited at the command. One wait is pending at a time.
// Measured 2026-11-08 on the porch: no wait left (all 144 per 3600 met before the part went), 54.7 -> 55.3 fps while
// GPU-bound, but one frame in 300 drew the shadows of another frame: the replay reads more guest memory as it records
// than the uniform blocks and mirrors refreshed here (texture and sampler descriptors are the suspects), so it stays off.
static struct { int on, si; volatile uint32_t *val; uint32_t mode, value; } ev_def;
static int ev_defer = -1;
static int wait_event_(VkExec *ex, const p_wait_event *p);
int vk_exec_wait_event(VkExec *ex, const p_wait_event *p) {
    FT(FT_WE0, (uintptr_t)p->val, p->value, ex->fence);
    int r = wait_event_(ex, p);
    FT(FT_WE1, r, p->val ? *p->val : 0, 0);
    return r;
}
static int wait_event_(VkExec *ex, const p_wait_event *p) {
    static int on = -1;
    (void)ex;
    if (on < 0) {
        on = rt_knob("EVENT_WAIT", "1");
        const char *e = cfg_get("REKARINA_EVENT_WAIT_MS");
        ev_limit_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 100) * 1000000ull;
        ev_defer = rt_knob("EVENT_DEFER", "0");
    }
    if (!on || !p->val) return 0;
    {   // DEBUG-EVLOG2
        static int lon = -1;
        static uint64_t lf0 = 900;
        if (lon < 0) { const char *e = cfg_get("REKARINA_EVLOG2"); lon = e != NULL; if (e && atoll(e) > 1) lf0 = (uint64_t)atoll(e); }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (lon && f >= lf0 && f <= lf0 + 5)
            rt_log("EVLOG2 wait word %p mode %u value %u fence %llu frame %llu now %u pending %d cpu %d rec %llu pass %u",
                   (void *)p->val, p->mode, p->value, (unsigned long long)ex->fence, (unsigned long long)f, *p->val,
                   evp_pending(p->val), nvn_event_cpu_signalled(p->val), (unsigned long long)ex->recorded,
                   (unsigned)ex->pass_serial);
    }
    if (++evw.subs >= 3600) {
        rt_log("vk: command buffer event waits, last %llu: %llu waited for the CPU (%llu of them before the submission's "
               "copies; %.3f ms each, %.3f ms at most), %llu "
               "met already (%llu of them signalled after the submission's copies were taken: taken again), %llu ordered by "
               "a GPU signal, %llu timed out; %llu deferred, %llu of them met by the next part; %llu held for another "
               "queue's signal",
               (unsigned long long)evw.subs, (unsigned long long)evw.waits, (unsigned long long)evw.pre,
               evw.waits ? evw.ns / 1e6 / (double)evw.waits : 0.0, evw.max_ns / 1e6, (unsigned long long)evw.met,
               (unsigned long long)evw.met_late, (unsigned long long)evw.gpu, (unsigned long long)evw.timeouts,
               (unsigned long long)evw.deferred, (unsigned long long)evw.def_met, (unsigned long long)evw.held);
        memset(&evw, 0, sizeof evw);
    }
    if (ev_met(p)) {
        static int late = -1;
        if (late < 0) late = rt_knob("EVENT_LATE_MET", "0");
        evw.met++;
        // signalled while this submission was being replayed: what the CPU wrote before it is not in the copies
        if (late && nvn_event_cpu_signal_ns(p->val) > vk_exec_copies_ns) return evw.met_late++, 1;
        return 0;
    }
    {   // a word GPU work signals: ordered by a recorded signal, or held for one another queue records later
        int rs = rsig_state(p);
        if (rs == 1) {
            int i = rsig_index(p->val, 0);   // a compute rest's signal: the graphics work waits for that rest
            if (i >= 0 && rsig[i].cp > vk_sync_ac_covered() && !vk_exec_ac_value) return vk_exec_wait_cp = rsig[i].cp, evw.gpu++, 4;
            return evw.gpu++, 0;
        }
        if (rs == 0 && !nvn_event_cpu_signalled(p->val) && vk_exec_hold_ok()) return evw.held++, 3;
    }
    if (evp_pending(p->val) || !nvn_event_cpu_signalled(p->val)) return evw.gpu++, 0;
    int si = 0;
    while (si < ev_nskip && ev_skip[si].val != p->val) si++;
    if (si < ev_nskip && ev_skip[si].fails >= 3) return 0;
    if (ev_defer) {
        vk_exec_wait_resolve();   // an earlier one first
        ev_def.on = 1, ev_def.si = si, ev_def.val = p->val, ev_def.mode = p->mode, ev_def.value = p->value;
        evw.deferred++;
        return 2;
    }
    uint64_t t0 = plat_now_ns();
    int ev_ok = ev_wait(p->val, p->mode, p->value, si);
    vk_spike.ev += plat_now_ns() - t0;   // DEBUG-SPIKE
    if (!ev_ok) return 0;
    {   // DEBUG-WAITLOG: REKARINA_WAITLOG=1 (nvn_api_core.c nvnEventSignal)
        double vk_listlog_ms(void);
        static int on = -1;
        if (on < 0) on = rt_knob("WAITLOG", "0");
        uint64_t f = on ? __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) : 0;
        if (on && f >= 899 && f <= 902)
            rt_log("WAITLOG replay waited %.3f ms for event %p (value %u) in fence %llu at %.3f", (plat_now_ns() - t0) / 1e6,
                   (void *)p->val, p->value, (unsigned long long)ex->fence, vk_listlog_ms());
    }
    return 1;
}

// Before a submission's copies are taken (vk_exec.c, the start of its replay): its waits for a word the CPU signals
// are waited for here, so the uniform snapshot and the mirrors hold what the CPU wrote before its signal and nothing
// later. Taking the copies again when the wait is reached does the same only when the replay gets there first; when
// the signal came while the commands before the wait were being replayed, the copies were older than it (above
// vk_exec_copies_ns), and taking them again then is too late the other way: the game is writing the next frame's
// blocks by that time (2026-11-14: objects black or missing for up to 19 frames). The replay of the commands before
// the wait no longer overlaps the thread that signals: in the frames where that thread is late, the submission
// starts later by as much. REKARINA_EVENT_PREWAIT=0: off. Returns the number of waits waited for.
static int prewait_ops(const op_hdr *h, int depth) {
    int waited = 0;
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_) return waited;
        const void *a = nvn_op_payload(h);
        if (h->op == OP_END) return waited;
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (h->op == OP_CALL && depth < 8) waited += prewait_ops(nvn_op_first(((const p_jump *)a)->target), depth + 1);
        if (h->op == OP_WAIT_EVENT) {
            const p_wait_event *p = a;
            if (p->val && !ev_met(p) && nvn_event_cpu_signalled(p->val) && !evp_pending(p->val) && rsig_state(p) != 1) {
                int si = 0;
                while (si < ev_nskip && ev_skip[si].val != p->val) si++;
                if (!(si < ev_nskip && ev_skip[si].fails >= 3)) {
                    uint64_t t0 = plat_now_ns();
                    ev_wait(p->val, p->mode, p->value, si);
                    vk_spike.ev += plat_now_ns() - t0;   // DEBUG-SPIKE
                    evw.pre++, waited++;
                }
            }
        }
        h = nvn_op_next(h);
    }
}
int vk_exec_prewait(const uint64_t *handles, int n) {
    static int on = -1;
    if (on < 0) on = rt_knob("EVENT_PREWAIT", "1") && rt_knob("EVENT_WAIT", "1");
    if (!on) return 0;
    if (!ev_limit_ns) {
        const char *e = cfg_get("REKARINA_EVENT_WAIT_MS");
        ev_limit_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 100) * 1000000ull;
    }
    int waited = 0;
    for (int i = 0; i < n; i++)
        if (handles[i]) waited += prewait_ops(nvn_op_first(handles[i]), 0);
    return waited;
}

int vk_exec_wait_ready(const void *payload) {
    const p_wait_event *p = payload;
    return ev_met(p) || rsig_state(p) == 1 || nvn_event_cpu_signalled(p->val);
}

// 1 while a deferred wait's word is not met yet: the parts after it are not split off (vk_exec.c maybe_split) and the
// recording goes on in one part; the next split point after the signal submits it.
int vk_exec_wait_deferred(void) { return ev_def.on && !ev_met_v(ev_def.val, ev_def.mode, ev_def.value); }

void vk_exec_wait_resolve(void) {
    if (!ev_def.on) return;
    ev_def.on = 0;
    if (ev_met_v(ev_def.val, ev_def.mode, ev_def.value)) evw.def_met++;
    else ev_wait(ev_def.val, ev_def.mode, ev_def.value, ev_def.si);   // ran out: the copies below are still the newest
    vk_usnap_refresh();
    vk_mem_resync_accessed();
}

// NVNcounterData = {counter, timestamp}; timestamps in nvn_counter_ticks's unit (nvn_api.h: Tegra 19.2 MHz ticks, the
// base of GetSystemTick; GAME_NS2: the GPU's 614.4 MHz ticks).
// DEBUG-RPLAG: REKARINA_REPORT_LAG=<k>: counter reports reach guest memory k presents after their completion, in order
typedef struct RpLag { struct RpLag *next; uint64_t addr, due, rec[2]; } RpLag;
static RpLag *rpl_head, *rpl_tail;
static PlatMutex rpl_lock = PLAT_MUTEX_INIT;
static int rp_lag = -1;
void vk_report_lag_tick(uint64_t frames) {
    if (rp_lag <= 0) return;
    plat_mutex_lock(&rpl_lock);
    while (rpl_head && rpl_head->due <= frames) {
        RpLag *l = rpl_head;
        rpl_head = l->next;
        if (!rpl_head) rpl_tail = NULL;
        mem_result_release(l->addr, sizeof l->rec);
        memcpy((void *)(uintptr_t)l->addr, l->rec, sizeof l->rec);
        free(l);
    }
    plat_mutex_unlock(&rpl_lock);
}
static void write_counter(uint64_t addr, uint64_t counter) {
    uint64_t rec[2] = {counter, nvn_counter_ticks(kern_now_ns())};   // nvn_api.h: the unit per console
    if (rp_lag < 0) { const char *l = cfg_get("REKARINA_REPORT_LAG"); rp_lag = l ? atoi(l) : 0; }
    if (rp_lag > 0) {
        RpLag *l = malloc(sizeof *l);
        if (l) {
            l->addr = addr, l->rec[0] = rec[0], l->rec[1] = rec[1], l->next = NULL;
            l->due = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + (uint64_t)rp_lag;
            plat_mutex_lock(&rpl_lock);
            if (rpl_tail) rpl_tail->next = l;
            else rpl_head = l;
            rpl_tail = l;
            plat_mutex_unlock(&rpl_lock);
            return;
        }
    }
    mem_result_release(addr, sizeof rec);   // REKARINA_RESULT_WATCH
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
}

static volatile int dbg_oq_lines;   // DEBUG-DRAWDBG
// Reports are numbered in record order (report_seq, replay thread); the completion thread, which runs their callbacks in
// submit order, sets report_written to the number of the last one whose value is in guest memory. A RENDER_COND
// reading a counter must wait for that, not for the fence: vk_sync_fence_done moves before the completion thread has
// run the callbacks of the submission, and a condition recorded in that gap read the counter the game had cleared
// (OoT: a white frame, every shading draw of the scene skipped, about a third of the frames while moving).
static uint64_t report_seq;
static volatile uint64_t report_written;

// GPU timestamps (GAME_NS2). A timestamp report took the time its completion callback ran, so every report of a
// submission, and of submissions the completion thread finished together, carried about the same time. OoT's
// frame controller measures the GPU's frame time from such reports and saw almost none: it kept interval 1 at
// 15-30 real fps and the game ran in slow motion. Each timestamp report now writes a Vulkan timestamp query
// (bottom of pipe) and its callback converts the GPU tick to the host clock with a calibrated (GPU tick, host time)
// pair (VK_KHR/EXT_calibrated_timestamps, refreshed every 250 ms), then to the console's unit (nvn_counter_ticks).
// Without calibration, an unavailable query or a full ring, the callback time stays. REKARINA_GPU_TIMESTAMPS=0 turns it off.
#ifdef GAME_NS2
#define TSQ_N 4096
static VkQueryPool tsq_pool;
static int tsq_state;                    // 0 not set up, 1 on, -1 off
static double tsq_period;                // ns per tick
static uint64_t tsq_mask;
static volatile uint8_t tsq_busy[TSQ_N];
static uint32_t tsq_next;                // replay
static PFN_vkGetCalibratedTimestampsKHR tsq_cal_fn;
static int64_t tsq_cal_tick, tsq_cal_kns, tsq_cal_at;   // completion thread: GPU tick and kern_now_ns time together

static int tsq_ready(void) {
    if (tsq_state) return tsq_state > 0;
    tsq_state = -1;
    VkCtx *c = vk_ctx();
    if (!rt_knob("GPU_TIMESTAMPS", "1")) return 0;
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    if (!f12.hostQueryReset || !bits || c->props.limits.timestampPeriod <= 0 || !c->has_calibrated) {
        rt_log("vk: timestamp reports keep the completion time (host reset %d, valid bits %u, calibration %d)",
               (int)f12.hostQueryReset, bits, c->has_calibrated);
        return 0;
    }
    tsq_cal_fn = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(
        c->device, c->has_calibrated == 2 ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT");
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = TSQ_N};
    if (!tsq_cal_fn || vkCreateQueryPool(c->device, &qi, NULL, &tsq_pool) != VK_SUCCESS) return 0;
    vkResetQueryPool(c->device, tsq_pool, 0, TSQ_N);
    tsq_period = c->props.limits.timestampPeriod;
    tsq_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    tsq_state = 1;
    rt_log("vk: timestamp reports carry GPU times (timestamp queries, %.2f ns a tick)", tsq_period);
    return 1;
}

// Completion thread: the calibrated pair, refreshed every 250 ms. 0: none.
static int tsq_calibrate(void) {
    int64_t now = (int64_t)kern_now_ns();
    if (tsq_cal_at && now - tsq_cal_at < 250000000) return 1;
    VkCtx *c = vk_ctx();
    VkCalibratedTimestampInfoKHR ci[2] = {{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
#ifdef _WIN32
                                          {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR}};
#else
                                          {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
#endif
    uint64_t v[2], dev;
    int64_t k0 = (int64_t)kern_now_ns(), p0 = (int64_t)plat_now_ns();
    if (tsq_cal_fn(c->device, 2, ci, v, &dev) != VK_SUCCESS) return tsq_cal_at != 0;
    int64_t host_ns = (int64_t)((double)v[1] * 1e9 / (double)plat_counter_hz());
    tsq_cal_tick = (int64_t)(v[0] & tsq_mask);
    tsq_cal_kns = host_ns + (k0 - p0);   // kern_now_ns's clock: the same counter, another epoch
    tsq_cal_at = now;
    return 1;
}
#endif

typedef struct { uint64_t addr, seq; int q; } CounterWrite;
static void counter_cb(void *arg) {
    CounterWrite *w = arg;
    uint64_t ns = 0;
#ifdef GAME_NS2
    if (w->q >= 0) {
        VkDevice dev = vk_ctx()->device;
        uint64_t v[2] = {0, 0};
        VkResult r = vkGetQueryPoolResults(dev, tsq_pool, (uint32_t)w->q, 1, sizeof v, v, sizeof v,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if ((r == VK_SUCCESS) && v[1] && tsq_calibrate()) {
            int64_t t = tsq_cal_kns + (int64_t)((double)((int64_t)(v[0] & tsq_mask) - tsq_cal_tick) * tsq_period);
            int64_t now = (int64_t)kern_now_ns();
            if (t > 0 && t <= now) ns = (uint64_t)t;   // a GPU time after the callback would be a calibration error
        }
        vkResetQueryPool(dev, tsq_pool, (uint32_t)w->q, 1);
        __atomic_store_n(&tsq_busy[w->q], 0, __ATOMIC_RELEASE);
    }
#endif
    if (ns) {
        uint64_t rec[2] = {0, nvn_counter_ticks(ns)};
        mem_result_release(w->addr, sizeof rec);   // REKARINA_RESULT_WATCH
        memcpy((void *)(uintptr_t)w->addr, rec, sizeof rec);
    } else {
        write_counter(w->addr, 0);
    }
    __atomic_store_n(&report_written, w->seq, __ATOMIC_RELEASE);
    free(w);
}

// Reports queued by the replay that the GPU has not finished. The condition of RENDER_COND reads a timestamp one as 0
// and a samples-passed one as unknown. The ring holds every report of the submissions in flight: OoT
// brackets each object of its depth prepass with two samples-passed reports (about 400 a frame) and draws the object's
// shading only if they differ (SetRenderEnableConditional, mode 1), while the lit pass adds about 350 more. A ring of
// 256 dropped the prepass reports before the lit pass tested them, the test read guest memory instead (zeros from the
// game's reset of the area) and skipped the draws: once the camera moved, most of the hero's house was depth in the prepass
// but never shaded, and the sky drawn under it showed through (2026-11-06). The lookup stops at the first report
// whose value is in guest memory (report_written), since the ring is in record order.
#define PENDING 16384
static struct { uint64_t addr, fence, seq; int unknown; uint32_t pred_off; uint8_t pred, why; } pending[PENDING];
// pending[].why (REKARINA_TRACE=perf): why a samples-passed report has no GPU predicate: 1 not exact (no reset since
// the last report), 2 a lost draw, 3 several queries, 4 no predicate buffer


// Replay thread: the ring slot of the newest report to each address, open addressing; a slot another report took since
// is told apart by its address. Stale keys are overwritten in place (the table only grows a key's slot once).
#define PIDX_N 65536u
// An entry holds the ring position its report was queued at (pending_head then, not wrapped): the ring slot still
// holds that report while fewer than PENDING reports followed it, and the newest report to an address always owns the
// address's entry, so that tells a stale entry without reading the ring (a second cache miss per report and per
// condition, 1,340 reports and 1,300 condition words a frame in forest village).
static struct { uint64_t addr; unsigned at; } pidx[PIDX_N];
static unsigned pending_head;
// The game's counters lie side by side in an array of its own, eight bytes apart, and are reported in that order:
// slots in address order keep a frame's 2,200 reports and 1,300 condition lookups within a few cache lines each, where
// a mixing hash sent every one to its own line of the 1 MiB table (0.28 ms a frame in the forest, 2026-11-12).
static unsigned pidx_slot(uint64_t addr) { return (unsigned)(addr >> 3) & (PIDX_N - 1); }
static int pidx_live(unsigned h) { return pending_head - pidx[h].at <= PENDING; }   // pending_head is past the entry
static void pidx_note(uint64_t addr, unsigned at) {
    unsigned h = pidx_slot(addr);
    for (unsigned n = 0; n < PIDX_N; n++, h = (h + 1) & (PIDX_N - 1))
        if (!pidx[h].addr || pidx[h].addr == addr || !pidx_live(h)) {   // empty, same, stale
            pidx[h].addr = addr, pidx[h].at = at;
            return;
        }
}
// DEBUG-TS-PENDING: the fence of the newest timestamp report recorded at each address (replay thread writes, the
// game's threads read through vk_ts_report_fence; a torn pair only mislabels one read)
#define TSP_N 1024
static struct { volatile uint64_t addr, fence; } tsp[TSP_N];
void vk_ts_report_note(uint64_t addr, uint64_t fence) {
    unsigned h = (unsigned)((addr * 0x9E3779B97F4A7C15ull) >> 54);
    for (unsigned n = 0; n < 16; n++, h = (h + 1) & (TSP_N - 1))
        if (!tsp[h].addr || tsp[h].addr == addr) {
            __atomic_store_n(&tsp[h].fence, fence, __ATOMIC_RELEASE);
            __atomic_store_n(&tsp[h].addr, addr, __ATOMIC_RELEASE);
            return;
        }
}
uint64_t vk_ts_report_fence(uint64_t addr) {
    unsigned h = (unsigned)((addr * 0x9E3779B97F4A7C15ull) >> 54);
    for (unsigned n = 0; n < 16; n++, h = (h + 1) & (TSP_N - 1)) {
        uint64_t a = __atomic_load_n(&tsp[h].addr, __ATOMIC_ACQUIRE);
        if (!a) return 0;
        if (a == addr) return __atomic_load_n(&tsp[h].fence, __ATOMIC_ACQUIRE);
    }
    return 0;
}

static int pidx_find(uint64_t addr) {
    unsigned h = pidx_slot(addr);
    for (unsigned n = 0; n < 64 && pidx[h].addr; n++, h = (h + 1) & (PIDX_N - 1))
        if (pidx[h].addr == addr) return pidx_live(h) ? (int)(pidx[h].at % PENDING) : -1;
    return -1;
}
enum { PRED_NONE, PRED_QUEUED, PRED_READY, PRED_ZERO };   // pending[].pred: the GPU predicate of a samples-passed count
static unsigned pending_head;

// ---- occlusion counts (samples passed)
//
// After a SAMPLES_PASSED reset, the first draw of a pass begins an occlusion query (precise when the device has
// occlusionQueryPrecise); the query ends with the pass, or at the next report or reset of the counter, and the next
// draw begins another. A Vulkan query begun inside a render pass must end inside it. TOTK always resets, draws the
// tested proxies, reports, so counting stops at a report and restarts at the next reset: queries cover only the draws
// between the two, not every pass of the frame. Once the game reports twice without a reset in between, counting stays
// on for good (cumulative NVN semantics; that one report misses the draws since the previous one). The queries closed since the last
// report or reset form a segment, handed to the completion thread with that report or reset: in submit order it sums
// the segment's results into the running count (a reset drops the sum and zeroes the count) and writes the count of a
// report into guest memory. Slots come from one ring; the completion thread reads and host-resets them
// (hostQueryReset, Vulkan 1.2), so the replay records no reset commands. A full ring (no slot read back yet) leaves the
// draw uncounted and the segment's count at least 1, as does a query still unavailable after its fence. Without hostQueryReset the counts stay 0.
#define OQ_N 4096
#define LOG_ONCE(...) do { static volatile char once_; if (!__atomic_exchange_n(&once_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__); } while (0)
static VkQueryPool oq_pool;
static int oq_state;                     // 0 not set up, 1 counting available, -1 unavailable
static VkQueryControlFlags oq_flags;
static volatile uint8_t oq_busy[OQ_N];   // from begin until the completion thread read and reset the slot
static uint32_t oq_next;                 // replay: next ring slot
static int oq_counting;                  // replay: draws are counted (a reset since the last report, or oq_keep)
static int oq_reported;                  // replay: a samples-passed report since the last reset
static int oq_keep;                      // replay: the game reported twice without a reset: count every draw
static int oq_open = -1;                 // replay: the slot whose query is active in the open pass
static uint32_t *oq_seg;                 // replay: slots closed since the last report or reset
static uint32_t oq_nseg, oq_capseg;
static uint32_t oq_last_n, oq_last_q;    // replay: the slot count (~0: a slot was lost) and first slot of the last segment
static int oq_since_reset;               // replay: no samples-passed report since the last reset
static int oq_lost;                      // replay: the segment missed a draw (ring full)
static uint64_t oq_count;                // completion thread: samples since the last reset

typedef struct { uint64_t addr, seq; int reset, lost; uint32_t n; uint32_t q[]; } OqSegment;
static void oq_batch_flush(uint64_t fence);

int vk_oqlog_on(void);
static int oq_ready(void) {
    if (oq_state) return oq_state > 0;
    VkCtx *v = vk_ctx();
    // vk_device.c enables every feature the device reports, so supported means enabled.
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(v->phys, &f2);
    oq_state = -1;
    if (!f12.hostQueryReset) {
        rt_log("vk: no hostQueryReset: samples-passed counters report 0");
        return 0;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                .queryCount = OQ_N};
    VkResult r = vkCreateQueryPool(v->device, &qi, NULL, &oq_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: occlusion query pool creation failed (%d): samples-passed counters report 0", (int)r);
        return 0;
    }
    vkResetQueryPool(v->device, oq_pool, 0, OQ_N);
    oq_flags = f2.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    oq_state = 1;
    return 1;
}

// Draw stretches on worker threads (vk_exec.c). oq_scan: the replay thread walks a stretch whose draws other command
// buffers record: slots are handed out and closed in the bookkeeping as before, but no query command is recorded here
// (vk_exec_oq_scan_draw gives the slot of the query a draw begins). OW: this thread records a chunk of such a stretch:
// it begins the query of the slot its draw was given and ends it at the next reset, report or the chunk's end. A draw
// with a slot that is not recorded after all (no pipeline yet, an empty scissor) leaves the slot to the next draw of
// its segment, or the query is begun and ended empty: the slot's result must exist, the copy into its predicate word
// waits for it.
static int oq_scan;
static _Thread_local struct { int on, pend, open; } OW;

void vk_exec_oq_scan(int on) { oq_scan = on; }
int vk_exec_oq_scan_open(void) { return oq_open >= 0; }

static void oq_close(VkExec *ex) {
    if (oq_open < 0) return;
    if (!oq_scan) {
        vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)oq_open);
        ex->recorded++;
    }
    if (oq_nseg == oq_capseg) {
        oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
        oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg);
        if (!oq_seg) rt_fatal("vk: out of memory");
    }
    oq_seg[oq_nseg++] = (uint32_t)oq_open;
    oq_open = -1;
}

int vk_exec_oq_scan_draw(void) {
    if (!oq_counting || oq_open >= 0) return -1;
    uint32_t q = oq_next % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        oq_lost = 1;
        return -1;
    }
    oq_busy[q] = 1;
    oq_next++;
    oq_open = (int)q;
    return (int)q;
}

void vk_exec_oq_scan_cut(VkExec *ex) { oq_close(ex); }

void vk_exec_oq_worker(int on) { OW.on = on, OW.pend = OW.open = -1; }
void vk_exec_oq_worker_slot(int q) { if (q >= 0) OW.pend = q; }
void vk_exec_oq_worker_close(VkExec *ex) {
    if (OW.pend >= 0 && OW.open < 0) {
        vkCmdBeginQuery(ex->cb, oq_pool, (uint32_t)OW.pend, oq_flags);
        ex->recorded++;
        OW.open = OW.pend;
    }
    OW.pend = -1;
    if (OW.open < 0) return;
    vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)OW.open);
    ex->recorded++;
    OW.open = -1;
}

void vk_exec_oq_draw(VkExec *ex) {
    if (OW.on) {
        if (OW.pend >= 0 && OW.open < 0) {
            vkCmdBeginQuery(ex->cb, oq_pool, (uint32_t)OW.pend, oq_flags);
            ex->recorded++;
            OW.open = OW.pend, OW.pend = -1;
        }
        return;
    }
    if (oq_counting && vk_oqlog_on())   // DEBUG-OQLOG
        rt_log("OQ draw f %llu open %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), oq_open);
    if (!oq_counting || oq_open >= 0) return;
    uint32_t q = oq_next % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        oq_lost = 1;
        return;
    }
    oq_busy[q] = 1;
    oq_next++;
    vkCmdBeginQuery(ex->cb, oq_pool, q, oq_flags);
    ex->recorded++;
    oq_open = (int)q;
}

void vk_exec_oq_pass_end(VkExec *ex) {
    if (OW.on) vk_exec_oq_worker_close(ex);
    else oq_close(ex);
}

void vk_exec_oq_draw_lost(VkExec *ex) {
    (void)ex;
    if (OW.on) return;   // the chunk's slots were given out by the replay thread's walk
    if (oq_counting) oq_lost = 1;   // as a full ring: the segment's count is at least 1 and not exact
}

// DEBUG-OQLOG: REKARINA_OQLOG=<frame>: for three frames from <frame>, every samples-passed reset and report, every render
// condition with how it was decided, and every draw a condition skipped while samples were counted.
int vk_oqlog_on(void) {
    static int on = -1;
    static uint64_t from, n = 3;   // REKARINA_OQLOG=<frame>[:<frames>]
    if (on < 0) {
        const char *e = cfg_get("REKARINA_OQLOG");
        on = e && *e;
        char *end = NULL;
        from = on ? strtoull(e, &end, 0) : 0;
        if (on && end && *end == ':') n = strtoull(end + 1, NULL, 0);
    }
    if (!on) return 0;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    return f >= from && f < from + n;
}

void vk_oqlog_note(const char *what, int a, int b) {   // a draw the replay dropped while samples were counted
    if (!vk_oqlog_on() || !oq_counting) return;
    rt_log("OQ %s f %llu %d %d", what, (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), a, b);
}

void vk_oqlog_skip(VkExec *ex) {   // a draw skipped by its condition
    if (!vk_oqlog_on() || !oq_counting) return;
    rt_log("OQ skipdraw f %llu cond %#llx mode %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
           (unsigned long long)ex->render_cond.addr, (int)ex->render_cond.mode);
}

void vk_exec_xfer_shutdown(void) {
    if (oq_state > 0) vkDestroyQueryPool(vk_ctx()->device, oq_pool, NULL);
#ifdef GAME_NS2
    if (tsq_state > 0) vkDestroyQueryPool(vk_ctx()->device, tsq_pool, NULL);
    tsq_pool = VK_NULL_HANDLE, tsq_state = 0, tsq_next = 0, tsq_cal_at = 0;
    memset((void *)tsq_busy, 0, sizeof tsq_busy);
#endif
    oq_pool = VK_NULL_HANDLE, oq_state = 0, oq_next = 0, oq_counting = oq_reported = oq_keep = 0, oq_open = -1;
    oq_nseg = 0, oq_lost = 0, oq_count = 0;
    memset((void *)oq_busy, 0, sizeof oq_busy);
}

// OoT brackets every object of its depth pre-pass with a reset and a report: about 6,700 segments a frame. Each
// used to read its slots and host-reset them with driver calls of its own, 0.8 ms on the completion thread before the
// frame's fence could be reported done, which the present waits for. Now the first segment handled after a submission's
// flush reads the whole pool once (the fence has passed: every query of the submission is final), the segments take
// their counts from that copy, and the slots read go back to the ring together, in runs, when the submission's flush
// callback comes (vk_exec_oq_submit_end queues it behind the submission's segments).
static uint64_t oq_res[OQ_N][2];   // completion thread: count and availability per slot, as of the read
static int oq_res_ok;
static uint8_t oq_read[OQ_N];      // completion thread: slots read and not yet reset
static uint32_t oq_nread;
static uint32_t oq_queued;         // replay: segments queued since the last flush callback

// DEBUG-OQSTAT (REKARINA_OQSTAT=1): per submission, how many samples-passed reports came out zero and how many queries
// did, with the frame counter: a frame in which objects lose their shading for want of a count shows as a jump.
static int oqstat_on = -1;
static uint32_t oqstat_rep, oqstat_rep0, oqstat_q, oqstat_q0, oqstat_na;

static void oq_flush(void) {
    if (oqstat_on < 0) oqstat_on = cfg_get("REKARINA_OQSTAT") != NULL;
    if (oqstat_on && oqstat_rep) {
        rt_log("OQSTAT frame %llu reports %u zero %u queries %u zero %u unavailable %u",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), oqstat_rep, oqstat_rep0, oqstat_q,
               oqstat_q0, oqstat_na);
        oqstat_rep = oqstat_rep0 = oqstat_q = oqstat_q0 = oqstat_na = 0;
    }
    VkDevice dev = vk_ctx()->device;
    for (uint32_t i = 0; i < OQ_N && oq_nread;) {
        if (!oq_read[i]) { i++; continue; }
        uint32_t j = i;
        while (j < OQ_N && oq_read[j]) j++;
        vkResetQueryPool(dev, oq_pool, i, j - i);
        for (uint32_t k = i; k < j; k++) oq_read[k] = 0, oq_nread--, __atomic_store_n(&oq_busy[k], 0, __ATOMIC_RELEASE);
        i = j;
    }
    oq_res_ok = 0;
}
static void oq_flush_cb(void *arg) { (void)arg; oq_flush(); }

// vk_exec.c, at the end of a submission's recording: the slots its segments read are reset behind them.
void vk_exec_oq_submit_end(uint64_t fence) {
    oq_batch_flush(fence);
    if (!oq_queued) return;
    oq_queued = 0;
    vk_completion_add_result(fence, oq_flush_cb, NULL);
}

static void oq_seg_done(const OqSegment *s) {
    uint64_t sum = 0;
    if (s->n && !oq_res_ok) {
        // The fence has passed, so a query that is still unavailable will not become available: waiting would block the
        // completion thread, and with it every later callback, for good. It counts as visible (one sample).
        VkResult r = vkGetQueryPoolResults(vk_ctx()->device, oq_pool, 0, OQ_N, sizeof oq_res, oq_res, sizeof oq_res[0],
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY) {
            vk_device_fault_report(r);
            rt_fatal("vk: occlusion query results failed (%d)", (int)r);
        }
        oq_res_ok = 1;
    }
    for (uint32_t i = 0; i < s->n; i++) {
        uint32_t q = s->q[i];
        if (oqstat_on > 0) oqstat_q++, oqstat_q0 += oq_res[q][1] && !oq_res[q][0], oqstat_na += !oq_res[q][1];
        if (oq_res[q][1]) sum += oq_res[q][0];
        else {
            sum += 1;
            LOG_ONCE("vk: occlusion query slot %u not available after its fence, counted as visible", q);
        }
        if (!oq_read[q]) oq_read[q] = 1, oq_nread++;
    }
    if (oq_nread > OQ_N / 2) oq_flush();   // a submission without its flush callback must not starve the ring
    if (s->reset) oq_count = 0;
    else {
        oq_count += sum;
        if (s->lost && !oq_count) oq_count = 1;
        if (oqstat_on > 0 && s->addr) oqstat_rep++, oqstat_rep0 += !oq_count;
        {   // DEBUG: REKARINA_OQ_FORCE=<n>: the game reads at least n samples for every report (its CPU culling sees all visible)
            static long force = -1;
            if (force < 0) { const char *e = cfg_env("REKARINA_OQ_FORCE"); force = e && *e ? atol(e) : 0; }
            if (s->addr) write_counter(s->addr, force && oq_count < (uint64_t)force ? (uint64_t)force : oq_count);
        }
        if (s->seq) __atomic_store_n(&report_written, s->seq, __ATOMIC_RELEASE);
        if (s->addr && dbg_oq_lines && dbg_oq_lines < 3000)   // DEBUG-DRAWDBG
            dbg_oq_lines++, rt_log("DRAWDBG OQWRITE addr %#llx count %llu (segment %u queries, sum %llu)", (unsigned long long)s->addr,
                                   (unsigned long long)oq_count, s->n, (unsigned long long)sum);
    }
}

// The segments reach the completion thread in batches: records back to back, one callback for all the replay closed
// since the last batch went. A callback each was two allocations and two queue entries under the completion lock per
// object of OoT's prepass and lit pass, about 3,100 a frame in the forest, with as many frees on the completion
// thread before the frame's results were done. A batch goes before any other result callback is queued (an event's,
// a timestamp's: they run in the order recorded), at the results point and at the submission's end
// (vk_exec_oq_submit_end).
typedef struct { uint32_t bytes, cap; uint8_t data[]; } OqBatch;
static OqBatch *oq_batch;                // replay: segments not handed over yet
static OqBatch *volatile oq_batch_spare; // a finished batch's memory, taken again by the replay

static size_t oq_rec_size(uint32_t n) { return (sizeof(OqSegment) + sizeof(uint32_t) * n + 7) & ~(size_t)7; }

static void oq_batch_cb(void *arg) {
    OqBatch *b = arg;
    for (uint32_t off = 0; off < b->bytes;) {
        const OqSegment *s = (const OqSegment *)(const void *)(b->data + off);
        oq_seg_done(s);
        off += (uint32_t)oq_rec_size(s->n);
    }
    b->bytes = 0;
    OqBatch *old = __atomic_exchange_n(&oq_batch_spare, b, __ATOMIC_ACQ_REL);
    free(old);
}

static void oq_batch_flush(uint64_t fence) {
    if (!oq_batch) return;
    vk_completion_add_result(fence, oq_batch_cb, oq_batch);
    oq_batch = NULL;
}

// Hands the segment to the completion thread with a report (addr; 0: no CPU-visible memory) or a reset.
static void oq_segment(VkExec *ex, uint64_t addr, int reset, uint64_t seq) {
    oq_close(ex);
    size_t need = oq_rec_size(oq_nseg);
    if (!oq_batch) {
        oq_batch = __atomic_exchange_n(&oq_batch_spare, NULL, __ATOMIC_ACQ_REL);
        if (oq_batch && oq_batch->cap < need) free(oq_batch), oq_batch = NULL;
        if (!oq_batch) {
            uint32_t cap = need > 256 * 1024 ? (uint32_t)need : 256 * 1024;
            oq_batch = malloc(sizeof *oq_batch + cap);
            if (!oq_batch) rt_fatal("vk: out of memory");
            oq_batch->cap = cap;
        }
        oq_batch->bytes = 0;
    }
    if (oq_batch->bytes + need > oq_batch->cap) {
        uint32_t cap = oq_batch->cap * 2 + (uint32_t)need;
        oq_batch = realloc(oq_batch, sizeof *oq_batch + cap);
        if (!oq_batch) rt_fatal("vk: out of memory");
        oq_batch->cap = cap;
    }
    OqSegment *s = (OqSegment *)(void *)(oq_batch->data + oq_batch->bytes);
    oq_batch->bytes += (uint32_t)need;
    s->addr = addr, s->seq = seq, s->reset = reset, s->lost = oq_lost, s->n = oq_nseg;
    oq_last_n = oq_lost ? ~0u : oq_nseg, oq_last_q = oq_nseg ? oq_seg[0] : 0;
    memcpy(s->q, oq_seg, sizeof *s->q * oq_nseg);
    oq_nseg = 0, oq_lost = 0;
    oq_queued++;
    (void)ex;
}

static void reset_counter(VkExec *ex, const p_reset_counter *p) {
    if (p->type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) return;
    if (vk_oqlog_on()) rt_log("OQ reset f %llu", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));   // DEBUG-OQLOG
    oq_segment(ex, 0, 1, 0);
    oq_counting = 1, oq_reported = 0, oq_since_reset = 1;
}

// A samples-passed report: the segment goes with it (addr 0: no CPU-visible memory) and counting stops until the next
// reset, unless reports came without resets before.
static void oq_report(VkExec *ex, uint64_t addr, uint64_t seq) {
    oq_segment(ex, addr, 0, seq);
    if (oq_reported && !oq_keep) {
        oq_keep = 1;
        rt_log("vk: samples passed reported twice without a reset: counting every draw from now on");
    }
    oq_reported = 1;
    if (!oq_keep) oq_counting = 0;
}

// REKARINA_RESULT_WATCH_NOREPORTS and REKARINA_DEPTH_PROBE, read once: getenv walks the whole environment, and these were
// asked per counter report and per depth copy, ~20 % of the replay thread on the forest porch.
static int rw_noreports(void) {
    static int v = -1;
    if (v < 0) v = cfg_env("REKARINA_RESULT_WATCH_NOREPORTS") != NULL;
    return v;
}
static int depth_probe(void) {
    static int v = -1;
    if (v < 0) v = cfg_env("REKARINA_DEPTH_PROBE") != NULL;
    return v;
}

static void signal_event(VkExec *ex, const p_signal_event *p) {
    if (!rw_noreports()) mem_result_watch((uint64_t)(uintptr_t)p->val, sizeof *p->val, ex->fence);   // REKARINA_RESULT_WATCH
    if (ev_lag < 0) { const char *l = cfg_get("REKARINA_EVENT_LAG"); ev_lag = l ? atoi(l) : 0; }
    EventLagged *ex_ = malloc(sizeof *ex_);
    if (!ex_) rt_fatal("vk: out of memory");
    ex_->frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    EventWrite *e = &ex_->w;
    *e = (EventWrite){p->val, p->mode, p->value};
    evp_add(p->val, 1);
    int ri = rsig_index(p->val, 1);
    if (ri >= 0) {
        rsig[ri].v = p->mode == NVN_EVENT_SIGNAL_MODE_ADD ? rsig[ri].v + p->value : p->value;
        rsig[ri].cp = vk_exec_ac_value;   // recorded by a compute rest (async compute), else 0
    }
    {   // DEBUG-EVLOG2: REKARINA_EVLOG2=1: GPU signals and waits of event words, frames 900-905
        static int on = -1;
        static uint64_t f0 = 900;
        if (on < 0) { const char *e = cfg_get("REKARINA_EVLOG2"); on = e != NULL; if (e && atoll(e) > 1) f0 = (uint64_t)atoll(e); }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= f0 && f <= f0 + 5)
            rt_log("EVLOG2 gpu signal word %p mode %u value %u fence %llu frame %llu now %u rec %llu pass %u", (void *)p->val,
                   p->mode, p->value, (unsigned long long)ex->fence, (unsigned long long)f, *p->val,
                   (unsigned long long)ex->recorded, (unsigned)ex->pass_serial);
    }
    oq_batch_flush(ex->fence);   // the segments recorded before it run before it
    vk_completion_add_result(ex->fence, event_cb, e);
}

#ifdef GAME_NS2
// ---- GPU conditions (NVN2)
// OoT brackets each object of its depth prepass with a samples-passed reset and report, and shades it in
// the lit pass of the same submission only if that count is not zero (SetRenderEnableConditional, mode 1, against a
// zero word at addr + 16). The replay records those draws before the GPU ran the report, so the CPU-side condition
// below can only render them (unknown) or read a stale word: rendering them all turned the frame white (conditional
// draws meant for nothing visible), reading memory left the house unshaded. Such a count is therefore a GPU
// predicate: the report's occlusion query result is copied into a word of pred_buf (vkCmdCopyQueryPoolResults, once
// the pass that holds the query ends: vk_exec_oq_after_pass), and the draws, dispatches and attachment clears under
// the condition run inside VK_EXT_conditional_rendering on that word (inverted for mode 0). Only an exact count
// qualifies: a report right after a reset whose segment holds one query (or none: a known 0), compared with a zero
// word the CPU can read; any other condition stays on the CPU. REKARINA_GPU_COND=0 turns it off.
#define PRED_N 65536u
static VkBuffer pred_buf;
static VmaAllocation pred_alloc;
static int pred_state;                   // 0 not set up, 1 usable, -1 not available
static uint32_t pred_next;
static PFN_vkCmdBeginConditionalRenderingEXT p_cond_begin;
static PFN_vkCmdEndConditionalRenderingEXT p_cond_end;
typedef struct { uint32_t q, off; unsigned pi; uint64_t seq; } PredCopy;
static PredCopy *pcq;                    // replay: copies waiting for the end of the pass that holds their query
static int npcq, cpcq;

static int pred_ready(void) {
    if (pred_state) return pred_state > 0;
    pred_state = -1;
    VkCtx *c = vk_ctx();
    if (!c->has_conditional_rendering || !rt_knob("GPU_COND", "1")) {
        rt_log("vk: samples-passed conditions on the CPU only (%s)",
               c->has_conditional_rendering ? "REKARINA_GPU_COND=0" : "no VK_EXT_conditional_rendering");
        return 0;
    }
    p_cond_begin = (PFN_vkCmdBeginConditionalRenderingEXT)vkGetDeviceProcAddr(c->device, "vkCmdBeginConditionalRenderingEXT");
    p_cond_end = (PFN_vkCmdEndConditionalRenderingEXT)vkGetDeviceProcAddr(c->device, "vkCmdEndConditionalRenderingEXT");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PRED_N * 4,
                             .usage = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (!p_cond_begin || !p_cond_end || vmaCreateBuffer(c->vma, &bi, &ai, &pred_buf, &pred_alloc, NULL) != VK_SUCCESS) {
        rt_log("vk: conditional rendering predicates could not be set up: the CPU decides");
        return 0;
    }
    rt_log("vk: samples-passed conditions of the same submission run on the GPU (VK_EXT_conditional_rendering)");
    pred_state = 1;
    return 1;
}

static void pred_flush(VkExec *ex) {
    if (!npcq) return;
    // runs of consecutive slots into consecutive words go in one copy (the prepass's ~750 a frame are mostly one run)
    for (int i = 0, j; i < npcq; i = j) {
        for (j = i + 1; j < npcq && pcq[j].q == pcq[j - 1].q + 1 && pcq[j].off == pcq[j - 1].off + 4; j++) {}
        vkCmdCopyQueryPoolResults(ex->cb, oq_pool, pcq[i].q, (uint32_t)(j - i), pred_buf, pcq[i].off, 4,
                                  VK_QUERY_RESULT_WAIT_BIT);
        ex->recorded++;
    }
    for (int i = 0; i < npcq; i++) {
        unsigned k = pcq[i].pi;
        if (pending[k].seq == pcq[i].seq) pending[k].pred = PRED_READY;
    }
    npcq = 0;
    vk_exec_barrier(ex);   // the words before the conditional rendering that reads them
}

void vk_exec_oq_after_pass(VkExec *ex) { pred_flush(ex); }

// After a samples-passed report entered pending[k]: its predicate, when the count is exact.
static void pred_note(VkExec *ex, unsigned k, int exact) {
    pending[k].why = !exact ? 1 : oq_last_n == ~0u ? 2 : oq_last_n > 1 ? 3 : !pred_ready() ? 4 : 0;
    if (!exact || oq_last_n == ~0u || !pred_ready()) return;
    if (!oq_last_n) {
        // A segment without queries is a known 0: the game drew nothing between the reset and the report (an object it
        // culled itself), and the lit draws conditioned on it are skipped. Rendering them all cost the forest two
        // thirds of its frame rate (41.5 -> 14.6 fps). A draw the replay skipped (vk_exec_oq_draw_lost) makes the
        // segment not exact instead, so it never ends here. REKARINA_PRED_ZERO=0 renders them (comparisons).
        static int zero = -1;
        if (zero < 0) zero = rt_knob("PRED_ZERO", "1");
        if (zero) pending[k].pred = PRED_ZERO;
        else pending[k].pred = PRED_ZERO + 1;   // DEBUG-OQLOG: marks it for cond_enabled_ (rendered as unknown)
        return;
    }
    if (oq_last_n != 1) return;
    pending[k].pred_off = (pred_next++ % PRED_N) * 4;
    pending[k].pred = PRED_QUEUED;
    if (npcq == cpcq) {
        cpcq = cpcq ? cpcq * 2 : 256;
        pcq = realloc(pcq, sizeof *pcq * (size_t)cpcq);
        if (!pcq) rt_fatal("vk: out of memory");
    }
    pcq[npcq++] = (PredCopy){oq_last_q, pending[k].pred_off, k, pending[k].seq};
    if (!ex->in_pass) pred_flush(ex);
}

void vk_exec_cond_begin(VkExec *ex) {
    if (ex->gpu_cond)
        rec_cond_begin(p_cond_begin, ex->cb, ex->gpu_cond_buf ? ex->gpu_cond_buf : pred_buf, ex->gpu_cond_off,
                       ex->gpu_cond_flags), ex->recorded++;
}
void vk_exec_cond_end(VkExec *ex) {
    if (ex->gpu_cond) rec_cond_end(p_cond_end, ex->cb), ex->recorded++;
}
#endif

static void report_counter(VkExec *ex, const p_report_counter *p) {
    int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && oq_ready();
    {   // DEBUG-DRAWDBG
        int vk_dump_active_dbg(void);
        if (vk_dump_armed && cfg_env("REKARINA_DRAWDBG") && vk_dump_active_dbg()) {
            dbg_oq_lines = 1;
            rt_log("DRAWDBG REPORT type %u addr %#llx samples %d fence %llu depth %#llx keep %d counting %d", p->type,
                   (unsigned long long)p->addr, samples, (unsigned long long)ex->fence,
                   ex->depth.img ? (unsigned long long)ex->depth.img->addr : 0ull, oq_keep, oq_counting);
        }
    }
    if (nvn_gpu_only(p->addr)) {   // CPU-invisible pool: no host memory behind it
        if (samples) oq_report(ex, 0, 0);
        return;
    }
    VkMemSpan m;
    if (!vk_mem_resolve(p->addr, sizeof(NVNcounterData), &m))
        rt_fatal("vk: counter report at %#llx is in no memory pool", (unsigned long long)p->addr);
    if (!rw_noreports()) mem_result_watch(p->addr, sizeof(NVNcounterData), ex->fence);   // REKARINA_RESULT_WATCH
    vk_mem_count(m.cls, VK_MUSE_GPU_WRITE, sizeof(NVNcounterData));
    uint64_t seq = ++report_seq;
    int exact = oq_since_reset;   // the count is the segment's sum
    if (samples) oq_report(ex, p->addr, seq), oq_since_reset = 0;
    else {
        CounterWrite *w = malloc(sizeof *w);
        if (!w) rt_fatal("vk: out of memory");
        *w = (CounterWrite){p->addr, seq, -1};
#ifdef GAME_NS2
        if (p->type == NVN_COUNTER_TYPE_TIMESTAMP) vk_ts_report_note(p->addr, ex->fence);   // DEBUG-TS-PENDING
        if (p->type == NVN_COUNTER_TYPE_TIMESTAMP && tsq_ready()) {
            uint32_t q = tsq_next % TSQ_N;
            if (!__atomic_load_n(&tsq_busy[q], __ATOMIC_ACQUIRE)) {
                tsq_busy[q] = 1, tsq_next++;
                vkCmdWriteTimestamp(ex->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsq_pool, q);
                ex->recorded++;
                w->q = (int)q;
            }
        }
#endif
        oq_batch_flush(ex->fence);   // the segments recorded before it run before it
        vk_completion_add_result(ex->fence, counter_cb, w);
    }
    unsigned at = pending_head++, k = at % PENDING;
    pending[k] = (typeof(pending[0])){p->addr, ex->fence, seq, samples, 0, PRED_NONE};
    pidx_note(p->addr, at);
#ifdef GAME_NS2
    if (samples) pred_note(ex, k, exact);
    if (samples && vk_oqlog_on())   // DEBUG-OQLOG
        rt_log("OQ report f %llu addr %#llx n %d exact %d pred %d off %u", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
               (unsigned long long)p->addr, oq_last_n == ~0u ? -1 : (int)oq_last_n, exact, (int)pending[k].pred, pending[k].pred_off);
#else
    (void)exact;
#endif
}

// ---- conditional rendering

// 1: *v read, 0: not readable, 2: a samples-passed report not done yet (unknown; *pi: its pending entry)
static int counter_word(uint64_t addr, uint64_t *v, int *pi) {
    uint64_t written = __atomic_load_n(&report_written, __ATOMIC_ACQUIRE);
    if (pi) *pi = -1;
    // The newest report to addr decides (pidx_find: the replay's index of it); when its value is in guest memory, so
    // are the older ones. A linear walk of the ring took ~15 % of the replay's CPU time on the porch (700 conditions a
    // frame against several hundred reports in flight).
    int i = pidx_find(addr);
    if (i >= 0 && pending[i].seq > written) {
        if (pi) *pi = i;
        return *v = 0, pending[i].unknown ? 2 : 1;
    }
    VkMemSpan m;
    if (nvn_gpu_only(addr) || !vk_mem_resolve(addr, sizeof *v, &m)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    vk_mem_count(m.cls, VK_MUSE_CPU_READ, sizeof *v);
    return 1;
}

// DEBUG-OQLOG: the predicate words of the GPU-decided conditions of the logged frames, read back a few frames later
static struct { uint64_t frame; uint32_t idx, off; } pl[8192];
static int npl, pl_done;
static VkBuffer pl_buf;
static VmaAllocation pl_alloc;
static const uint32_t *pl_map;
static void pl_report(void *arg) {
    (void)arg;
    for (int i = 0; i < npl; i++)
        rt_log("OQ predval f %llu idx %u off %u value %u", (unsigned long long)pl[i].frame, pl[i].idx, pl[i].off,
               pl_map[pl[i].off / 4]);
}
static void pl_snapshot(VkExec *ex) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = PRED_N * 4,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pl_buf, &pl_alloc, &info) != VK_SUCCESS) return;
    pl_map = info.pMappedData;
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    VkBufferCopy cp = {0, 0, PRED_N * 4};
    vkCmdCopyBuffer(ex->cb, pred_buf, pl_buf, 1, &cp);
    ex->recorded++;
    vk_completion_add(ex->fence, pl_report, NULL);
}

static int cond_enabled_(VkExec *ex, const p_render_cond *c, const char **why);
static uint64_t unk_why[8];     // unknown conditions: report why (pending[].why 0-4), 5 y pending, 6 y not a zero word, 7 no index
static uint64_t mem_class[5];   // memory conditions: y zero & x pending, y zero & x done, y nonzero & x pending,
                                // y nonzero & x done, y pending (REKARINA_TRACE=perf)
static int cond_enabled(VkExec *ex, const p_render_cond *c) {
    const char *why = "mem";
    int r = cond_enabled_(ex, c, &why);
    {   // per frame for the flight recorder (cond_* columns)
        int k = ex->gpu_cond ? 0 : !strcmp(why, "zero") ? -1 : !strcmp(why, "unknown") ? 1 : r ? 2 : 3;
        if (k >= 0) perf_count(PERF_CNT_COND_GPU + k);
    }
    if (rt_trace & RT_TRACE_PERF) {   // decisions by kind, logged every 300 frames: gpu, zero, unknown, memory true/false
        static uint64_t n[5], lastf;
        int kind = ex->gpu_cond ? 0 : !strcmp(why, "zero") ? 1 : !strcmp(why, "unknown") ? 2 : r ? 3 : 4;
        n[kind]++;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f / 300 != lastf / 300 && f) {
            double fr = f > lastf ? (double)(f - lastf) : 1.0;
            rt_log("perf: frame %llu: render conditions per frame: gpu %.1f, known zero %.1f, unknown (rendered) %.1f, "
                   "memory true %.1f, memory false %.1f; memory words: y0 x-pending %.1f, y0 x-done %.1f, y!=0 x-pending "
                   "%.1f, y!=0 x-done %.1f, y-pending %.1f", (unsigned long long)f, n[0] / fr, n[1] / fr, n[2] / fr, n[3] / fr,
                   n[4] / fr, mem_class[0] / fr, mem_class[1] / fr, mem_class[2] / fr, mem_class[3] / fr, mem_class[4] / fr);
            rt_log("perf: frame %llu: unknown conditions per frame by cause: predicate missing %.1f, not exact %.1f, lost draw "
                   "%.1f, several queries %.1f, no buffer %.1f, y pending %.1f, y not zero %.1f, no index %.1f",
                   (unsigned long long)f, unk_why[0] / fr, unk_why[1] / fr, unk_why[2] / fr, unk_why[3] / fr, unk_why[4] / fr,
                   unk_why[5] / fr, unk_why[6] / fr, unk_why[7] / fr);
            memset(unk_why, 0, sizeof unk_why);
            memset(n, 0, sizeof n);
            memset(mem_class, 0, sizeof mem_class);
            lastf = f;
        }
    }
    {   // DEBUG-OQLOG
        static uint64_t lastf;
        static uint32_t idx;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != lastf) lastf = f, idx = 0;
        idx++;
        if (vk_oqlog_on() && ex->gpu_cond && npl < 8192) pl[npl++] = (typeof(pl[0])){f, idx, ex->gpu_cond_off};
        else if (npl && !pl_done && !vk_oqlog_on()) pl_done = 1, pl_snapshot(ex);
    }
    ex->dbg_zero = !strcmp(why, "zero");   // DEBUG-OQLOG
    if (vk_oqlog_on())   // DEBUG-OQLOG
        rt_log("OQ cond f %llu addr %#llx mode %d %s -> %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
               (unsigned long long)c->addr, (int)c->mode, ex->gpu_cond ? "gpu" : why, r);
    return r;
}
static int cond_enabled_(VkExec *ex, const p_render_cond *c, const char **why) {
    {   // DEBUG: REKARINA_COND=always renders whatever the condition says
        static int always = -1;
        if (always < 0) { const char *e = cfg_env("REKARINA_COND"); always = e && !strcmp(e, "always"); }
        if (always) return 1;
    }
    // DEBUG: REKARINA_COND_ALT=1: on odd frames the conditions decided on the CPU as false (known-zero segments and words
    // read from memory) render, for side-by-side shots of consecutive frames
    static int alt = -1;
    if (alt < 0) alt = cfg_env("REKARINA_COND_ALT") != NULL;
    int alt_now = alt && (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) & 1);
    uint64_t x, y;
    int ix;
    int rx = counter_word(c->addr, &x, &ix), ry = counter_word(c->addr + 16, &y, NULL);
#ifdef GAME_NS2
    ex->gpu_cond = 0;
    if (rx == 2 && ry == 1 && !y && ix >= 0) {   // the count against zero: on the GPU, or known zero
        if (pending[ix].pred == PRED_ZERO) return *why = "zero", alt_now ? 1 : c->mode == 0;
        if (pending[ix].pred == PRED_ZERO + 1) *why = "zero";   // DEBUG-OQLOG: rendered (REKARINA_PRED_ZERO=0)
        static int pred_alt = -1;   // DEBUG: REKARINA_PRED_ALT=1: on odd frames GPU predicates render (side-by-side shots)
        if (pred_alt < 0) pred_alt = cfg_env("REKARINA_PRED_ALT") != NULL;
        if (pending[ix].pred == PRED_READY && pred_alt && (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) & 1))
            return *why = "gpu-alt", 1;
        static int pred_always = -1;   // DEBUG: REKARINA_PRED_ALWAYS=1: what a GPU predicate would decide is rendered
        if (pred_always < 0) pred_always = cfg_env("REKARINA_PRED_ALWAYS") != NULL;
        if (pending[ix].pred == PRED_READY && pred_always) return *why = "gpu-always", 1;
        if (pending[ix].pred == PRED_READY) {
            ex->gpu_cond = 1, ex->gpu_cond_off = pending[ix].pred_off, ex->gpu_cond_buf = VK_NULL_HANDLE;
            ex->gpu_cond_flags = c->mode == 0 ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0;
            return 1;
        }
    }
#else
    (void)ex, (void)ix;
#endif
    if (!rx || !ry) {
        warn_once(W_BUFFER, "conditional render counters at %#llx are not readable: rendering", (unsigned long long)c->addr);
        return 1;
    }
#ifdef GAME_NS2
    // Words in memory, not reports of the replay: the game also conditions on counts that GPU copies and shaders write.
    // Read on the CPU at record time they were often not written yet (the GPU a frame behind, REKARINA_PRESENT_SYNC=n:
    // passes skipped and white frames, 50-60 in 2,500 frames). Against a zero word, as the console's GPU does it: the
    // conditional rendering reads x where the GPU reads that memory, in GPU order (REKARINA_MEM_COND=0: the CPU decides).
    if (rx == 1 && ry == 1) {
        // REKARINA_MEM_COND: 0 (default) the CPU decides from memory; 2 a condition against a zero word the GPU has not
        // been told to write runs on the GPU on x (its real value in GPU order). Never "render" an unwritten word: a
        // condition rendered by default draws what was meant for nothing visible and the scene goes white.
        static int mem_cond = -1;
        if (mem_cond < 0) mem_cond = rt_knob("MEM_COND", "0");
        int xpend = vk_mem_gpu_write_pending(c->addr, 8), ypend = vk_mem_gpu_write_pending(c->addr + 16, 8);
        if (rt_trace & RT_TRACE_PERF) {   // classification, logged with the decision counts
            int k = ypend ? 4 : y ? (xpend ? 2 : 3) : (xpend ? 0 : 1);
            mem_class[k]++;
        }
        if (mem_cond == 2 && !ypend && !y && pred_ready()) {
            VkMemSpan m;
            if (vk_mem_resolve(c->addr, 4, &m) && !(m.offset & 3)) {
                VK_MEM_PICK(m);   // the buffer GPU reads of this memory use (its VRAM mirror, synced for this submission)
                ex->gpu_cond = 1, ex->gpu_cond_buf = m.buffer, ex->gpu_cond_off = (uint32_t)m.offset;
                ex->gpu_cond_flags = c->mode == 0 ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0;
                *why = "gpu-mem";
                return 1;
            }
        }
    }
#endif
    extern int vk_dbg_cond_unknown, vk_dbg_cond_why;   // DEBUG-SKIPLOG
    vk_dbg_cond_unknown = rx == 2 || ry == 2;
    vk_dbg_cond_why = rx != 2 ? 1 : ry != 1 ? 2 : y ? 3 : ix < 0 ? 4 : 10 + pending[ix].pred;
    if ((rx == 2 || ry == 2) && (rt_trace & RT_TRACE_PERF)) {   // why unknown (logged with the decision counts)
        int r = ry == 2 ? 5 : ry != 1 || y ? 6 : ix < 0 ? 7 : pending[ix].why;
        if (r >= 0 && r < 8) unk_why[r]++;
    }
    if (rx == 2 || ry == 2) return *why = strcmp(*why, "zero") ? "unknown" : *why, 1;   // a samples-passed count not done yet
    {   // DEBUG-SKIPLOG: conditions decided from memory
        static int on = -1;
        static unsigned long long lo, hi;
        if (on < 0) { const char *e = cfg_env("REKARINA_SKIPLOG"); on = e && sscanf(e, "%llu-%llu", &lo, &hi) == 2; }
        unsigned long long f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= lo && f <= hi)
            rt_log("SKIPLOG memcond frame %llu addr %#llx mode %d x %llu y %llu -> %d", f, (unsigned long long)c->addr, c->mode,
                   (unsigned long long)x, (unsigned long long)y, c->mode == 0 ? x == y : x != y);
    }
    if (alt_now) return 1;
    return c->mode == 0 ? x == y : x != y;
}

// ---- textures

typedef struct {
    const VkTexImage *img;
    VkImageAspectFlags aspect;
    uint32_t level, layer0;
    int bpb, bw, bh;
    int32_t x, y, z;           // region origin
    uint32_t w, h, nz;         // region extent; nz: slices (3D) or layers
} Sub;

// Resolves tex, its view and a copy region to image coordinates. 0 (warned) when the copy cannot be replayed.
static int sub_resolve(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level,
                       Sub *s) {
    memset(s, 0, sizeof *s);
    if (!t) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || f.vk != f.ideal || f.expand_rgb || f.astc_bw) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", t->d.format);
        return 0;
    }
    s->bpb = nvn_fmt_bpb(t->d.format, &s->bw, &s->bh);
    s->img = vk_texture_get(t);
    if (!s->img || !s->bpb) {
        warn_once(W_FORMAT, "copy of format %#x texture without an image: skipped", t->d.format);
        return 0;
    }
    const VkTexImage *img = s->img;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    else if (img->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) s->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else {
        warn_once(W_FORMAT, "copy of depth-stencil format %#x texture: skipped", t->d.format);
        return 0;
    }
    s->level = has_v && v->set_levels ? v->base_level : 0;
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    int k = img->scale > 1 ? img->scale : 1;   // a larger shadow map: the game's region in its texels
    s->x = r->xoffset * k, s->y = r->yoffset * k, s->z = r->zoffset;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    s->w = r->width > 0 ? (uint32_t)r->width * k : defaults_to_level ? (s->x < (int32_t)lw ? lw - (uint32_t)s->x : 0) : 0;
    s->h = r->height > 0 ? (uint32_t)r->height * k : defaults_to_level ? (s->y < (int32_t)lh ? lh - (uint32_t)s->y : 0) : 0;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t layers = is3d ? 1 : (uint32_t)s->z + s->nz, first = is3d ? 0 : s->layer0;
    if (s->level >= img->levels || s->x < 0 || s->y < 0 || s->z < 0 || !s->w || !s->h ||
        (uint64_t)s->x + s->w > lw || (uint64_t)s->y + s->h > lh || first + layers > img->layers ||
        (is3d && (uint64_t)s->z + s->nz > max1(img->depth >> s->level))) {
        warn_once(W_REGION, "copy region %d,%d,%d %ux%ux%u outside level %u of a %ux%u texture: skipped", s->x, s->y,
                  s->z, s->w, s->h, s->nz, s->level, img->width, img->height);
        return 0;
    }
    return 1;
}

static VkImageSubresourceLayers sub_layers(const Sub *s) {
    int is3d = s->img->type == VK_IMAGE_TYPE_3D;
    return (VkImageSubresourceLayers){s->aspect, s->level, is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
}

static VkOffset3D sub_offset(const Sub *s) {
    return (VkOffset3D){s->x, s->y, s->img->type == VK_IMAGE_TYPE_3D ? s->z : 0};
}

static VkExtent3D sub_extent(const Sub *s) {
    return (VkExtent3D){s->w, s->h, s->img->type == VK_IMAGE_TYPE_3D ? s->nz : 1};
}

// Buffer side of a texture <-> buffer copy: bytes covered, pitches, Vulkan buffer span (a VRAM mirror when the pool
// has one).
static int buffer_side(const VkExec *ex, const Sub *s, uint64_t addr, uint64_t *size, VkBufferImageCopy *c,
                       VkMemSpan *span) {
    uint32_t wb = (s->w + (uint32_t)s->bw - 1) / (uint32_t)s->bw, hb = (s->h + (uint32_t)s->bh - 1) / (uint32_t)s->bh;
    uint64_t row = ex->copy_stride > 0 ? (uint64_t)ex->copy_stride : (uint64_t)wb * (uint64_t)s->bpb;
    if (row % (uint64_t)s->bpb || row < (uint64_t)wb * (uint64_t)s->bpb) {
        warn_once(W_REGION, "buffer copy row stride %llu does not fit %u blocks of %d bytes: skipped",
                  (unsigned long long)row, wb, s->bpb);
        return 0;
    }
    uint64_t img = row * hb;
#ifdef GAME_NS2
    // NVN2 SetCopyImageStride: the bytes between the 2D images of a 3D or array copy; Vulkan takes them in rows
    if (ex->copy_image_stride > 0) {
        uint64_t is = (uint64_t)ex->copy_image_stride;
        if (is % row || is < img) {
            warn_once(W_REGION, "buffer copy image stride %llu is not whole rows of %llu bytes past the image: skipped",
                      (unsigned long long)is, (unsigned long long)row);
            return 0;
        }
        img = is;
    }
#endif
    *size = img * (s->nz - 1) + row * (hb - 1) + (uint64_t)wb * (uint64_t)s->bpb;
    VkMemSpan m;
    if (!addr || !vk_mem_resolve(addr, *size, &m) || m.avail < *size) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)addr, (unsigned long long)*size);
        return 0;
    }
    VK_MEM_PICK(m);
    *span = m;
    *c = (VkBufferImageCopy){.bufferOffset = m.offset, .bufferRowLength = (uint32_t)(row / (uint64_t)s->bpb) * (uint32_t)s->bw,
#ifdef GAME_NS2
                             .bufferImageHeight = (uint32_t)(img / row) * (uint32_t)s->bh,
#else
                             .bufferImageHeight = hb * (uint32_t)s->bh,
#endif
                             .imageSubresource = sub_layers(s),
                             .imageOffset = sub_offset(s), .imageExtent = sub_extent(s)};
    return 1;
}

static void copy_buffer_image(VkExec *ex, const p_copy_b2t *p, int to_image) {
    Sub s;
    if (!sub_resolve(p->tex, &p->view, p->has_view, &p->r, 1, &s)) return;
    if (s.img->scale > 1) {   // the buffer holds the game's texel count, the image more
        warn_once(W_REGION, "buffer copy of a shadow map enlarged by REKARINA_SHADOW_SCALE: skipped");
        return;
    }
    uint64_t size;
    VkBufferImageCopy c;
    VkMemSpan m;
    if (!buffer_side(ex, &s, p->src, &size, &c, &m)) return;
    { void vk_dbg_r8(const char *, const tex_t *, long, long); vk_dbg_r8(to_image ? "copy-b2t" : "copy-t2b", p->tex, (long)c.bufferRowLength, (long)ex->copy_stride); }   // DEBUG-R8LOG
    vk_mem_count(m.cls, to_image ? VK_MUSE_COPY_SRC : VK_MUSE_GPU_WRITE, size);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (to_image) vkCmdCopyBufferToImage(ex->cb, m.buffer, s.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    else vkCmdCopyImageToBuffer(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, m.buffer, 1, &c);
    ex->recorded++;
    if (to_image) {
        vk_interface_coverage_write(s.img->image, s.img->addr);
        vk_texture_gpu_written(p->tex);
    }
    else note_gpu_write(ex, p->src, size, vk_mem_gpu_wrote(&m, size));
}

static void copy_t2t(VkExec *ex, const p_copy_t2t *p) {
    Sub s, d;
    if (!sub_resolve(p->src, &p->sv, p->has_sv, &p->sr, 1, &s)) return;
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, &d)) return;
    { void vk_dbg_r8(const char *, const tex_t *, long, long); vk_dbg_r8("copy-t2t-dst", p->dst, (long)p->src->d.format, (long)p->src->d.width); vk_dbg_r8("copy-t2t-src", p->src, 0, 0); }   // DEBUG-R8LOG
    d.nz = s.nz;
    int same_size = s.w == d.w && s.h == d.h;
    int same_format = s.img->format == d.img->format;
    if (s.img->samples != d.img->samples && !(s.img->samples > 1 && d.img->samples == 1 && same_size && same_format)) {
        warn_once(W_COPY, "texture copy between %u and %u samples: skipped", s.img->samples, d.img->samples);
        return;
    }
    if (s.aspect != d.aspect || (s.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !same_format)) {
        warn_once(W_COPY, "texture copy between formats %#x and %#x: skipped", p->src->d.format, p->dst->d.format);
        return;
    }
    VkImageSubresourceLayers sl = sub_layers(&s), dl = sub_layers(&d);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (s.img->samples > 1 && d.img->samples == 1) {
        VkImageResolve rv = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdResolveImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
    } else if (same_size && same_format && s.aspect == VK_IMAGE_ASPECT_DEPTH_BIT && depth_probe()) {
        // DEBUG-PROBE: the copy of the prepass depth is replaced by a clear to far, so EQUAL draws into it (made LESS with
        // writes, vk_pipeline.c) leave their own depth there for the dump
        VkClearDepthStencilValue cv = {1.0f, 0};
        VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_DEPTH_BIT, dl.mipLevel, 1, dl.baseArrayLayer, dl.layerCount};
        vkCmdClearDepthStencilImage(ex->cb, d.img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rr);
    } else if (same_size && same_format) {
        VkImageCopy cp = {sl, sub_offset(&s), dl, sub_offset(&d), sub_extent(&s)};
        vkCmdCopyImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    } else {   // scaled or converting copy: nearest, or linear when asked for (colour only)
        int is3d = s.img->type == VK_IMAGE_TYPE_3D;
        VkImageBlit bl = {.srcSubresource = sl, .dstSubresource = dl};
        bl.srcOffsets[0] = sub_offset(&s);
        bl.srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, is3d ? s.z + (int32_t)s.nz : 1};
        bl.dstOffsets[0] = sub_offset(&d);
        bl.dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, is3d ? d.z + (int32_t)d.nz : 1};
        int linear = (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdBlitImage(ex->cb, s.img->image, VK_IMAGE_LAYOUT_GENERAL, d.img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                       linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    ex->recorded++;
    vk_interface_coverage_copy(s.img->image, d.img->image, d.img->addr,
        same_size && same_format && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT && s.img->samples == 1 && d.img->samples == 1 &&
        !s.x && !s.y && !s.z && !s.level && !s.layer0 && s.nz == 1 && s.w == s.img->width && s.h == s.img->height &&
        !d.x && !d.y && !d.z && !d.level && !d.layer0 && d.nz == 1 && d.w == d.img->width && d.h == d.img->height);
    vk_texture_gpu_written(p->dst);
    // A GPU result copied into a tiny texture the CPU reads (a readback): it reaches guest memory with the console's
    // latency, not as soon as our GPU is done (vk_texture.c vk_texture_readback_hold).
    if (p->dst->mem && !nvn_gpu_only((uintptr_t)p->dst->mem) && p->dst->L.size <= 1024 &&
        (p->dst->pool_flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED)))
        vk_texture_readback_hold(p->dst);
}

// ---- buffer to buffer

static void copy_b2b(VkExec *ex, const p_copy_b2b *p) {
    uint64_t src = p->src, dst = p->dst, left = p->size;
    int started = 0, owned = 1;
    while (left) {
        VkMemSpan s, d;
        if (!vk_mem_resolve(src, 1, &s) || !vk_mem_resolve(dst, 1, &d)) {
            warn_once(W_BUFFER, "buffer copy %#llx -> %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < s.avail ? left : s.avail;
        if (n > d.avail) n = d.avail;
        VK_MEM_PICK(s);
        VK_MEM_PICK(d);
        if (s.buffer == d.buffer && s.offset < d.offset + n && d.offset < s.offset + n) {
            warn_once(W_OVERLAP, "buffer copy %#llx -> %#llx overlaps within one buffer: skipped",
                      (unsigned long long)p->src, (unsigned long long)p->dst);
            break;
        }
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VkBufferCopy bc = {s.offset, d.offset, n};
        vkCmdCopyBuffer(ex->cb, s.buffer, d.buffer, 1, &bc);
        owned &= vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(s.cls, VK_MUSE_COPY_SRC, n);
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        src += n, dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->dst, p->size - left, owned);
}

// ---- buffer fills (ClearBuffer, and ReportValue as a 4-byte one): vkCmdFillBuffer in GPU order after a full
// barrier, so the value lands after earlier work, as NVN's semaphore release does; CPU-visible ranges go to
// vk_guest_written at the fence, as copies do. Works in CPU-invisible pools.

static void clear_buf(VkExec *ex, const p_clear_buf *p) {
    if ((p->addr | p->size) & 3) {
        warn_once(W_BUFFER, "buffer fill at %#llx of %llu bytes is not 4-byte aligned: skipped", (unsigned long long)p->addr,
                  (unsigned long long)p->size);
        return;
    }
    uint64_t dst = p->addr, left = p->size;
    int started = 0, owned = 1;
    while (left) {
        VkMemSpan d;
        if (!vk_mem_resolve(dst, 1, &d) || d.offset % 4) {
            warn_once(W_BUFFER, "buffer fill at %#llx (%llu bytes) leaves backed memory: rest skipped",
                      (unsigned long long)p->addr, (unsigned long long)p->size);
            break;
        }
        uint64_t n = left < d.avail ? left : d.avail & ~3ull;
        if (!n) break;
        if (!started) vk_exec_pass_end(ex), vk_exec_barrier(ex), started = 1;
        VK_MEM_PICK(d);
        vkCmdFillBuffer(ex->cb, d.buffer, d.offset, n, p->value);
        owned &= vk_mem_gpu_wrote(&d, n);
        ex->recorded++;
        vk_mem_count(d.cls, VK_MUSE_GPU_WRITE, n);
        dst += n, left -= n;
    }
    if (started) note_gpu_write(ex, p->addr, p->size - left, owned);
}

// ---- texture clears

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return 1;
    default: return 0;
    }
}

// Whole levels only (vkCmdClear*Image has no rectangle); partial regions, channel masks and a view that reinterprets
// the format are skipped (MacNativeGames does not replay them either, nvn_mtl.m).
static void clear_tex(VkExec *ex, const p_clear_tex *p) {
    const tex_t *t = p->tex;
    if (!t) return;
    Sub s;
    if (!sub_resolve(t, &p->view, p->has_view, &p->r, 1, &s)) return;
    const VkTexImage *img = s.img;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t lw = max1(img->width >> s.level), lh = max1(img->height >> s.level);
    int reformat = p->has_view && p->view.set_format && p->view.format != t->d.format;
    if (s.x || s.y || s.w != lw || s.h != lh || (p->mask & 0xf) != 0xf || reformat ||
        (is3d && (s.z || s.nz != max1(img->depth >> s.level)))) {
        warn_once(W_CLEAR, "texture clear of region %d,%d %ux%u mask %#x (level %ux%u, view format change %d): skipped",
                  s.x, s.y, s.w, s.h, p->mask, lw, lh, reformat);
        return;
    }
    VkImageSubresourceRange rng = {img->aspect, s.level, 1, is3d ? 0 : s.layer0 + (uint32_t)s.z, is3d ? 1 : s.nz};
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkClearColorValue cv;
        if (int_format(img->format) != (p->is_uint != 0)) {
            warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint, (int)img->format);
            return;
        }
        memcpy(&cv, &p->c, sizeof cv);
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearColorImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    } else {
        int both = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        VkClearDepthStencilValue dv = {0, 0};
        if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) dv.depth = p->c.f[0];
        if (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) dv.stencil = both ? p->c.u[1] : p->c.u[0];
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearDepthStencilImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rng);
    }
    ex->recorded++;
    vk_interface_coverage_write(img->image, img->addr);
    vk_texture_gpu_written(t);
}

void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_RENDER_ENABLE && op != OP_SIGNAL_EVENT) vk_contact_flush(ex);
    switch (op) {
    case OP_COPY_B2B: copy_b2b(ex, payload); break;
    case OP_COPY_B2T: copy_buffer_image(ex, payload, 1); break;
    case OP_COPY_T2B: copy_buffer_image(ex, payload, 0); break;   // p_copy_b2t: src holds the destination address
    case OP_COPY_T2T: copy_t2t(ex, payload); break;
    case OP_CLEAR_TEX: clear_tex(ex, payload); break;
    case OP_SIGNAL_EVENT: signal_event(ex, payload); break;
    case OP_REPORT_COUNTER: report_counter(ex, payload); break;
    case OP_RESET_COUNTER: reset_counter(ex, payload); break;
    case OP_CLEAR_BUF: clear_buf(ex, payload); break;
    case OP_RENDER_ENABLE:
        ex->render_disabled = !((const p_flag *)payload)->value;
#ifdef GAME_NS2
        ex->gpu_cond = 0, ex->cond_active = 0;
#endif
        break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        ex->render_disabled = !cond_enabled(ex, payload);
#ifdef GAME_NS2
        ex->cond_active = 1, ex->cond_cpu = !ex->render_disabled;
#endif
        break;
#ifdef GAME_NS2
    case OP_FENCE_SYNC: {   // CommandBufferFenceSync: done when this submission is (nvn_api_ns2.c)
        sync_t *sy = ((const p_fence_sync *)payload)->sync;
        if (sy) sy->fence = ex->fence, __atomic_store_n(&sy->fenced, 1, __ATOMIC_RELEASE);
        break;
    }
#endif
    default: rt_fatal("vk: op %s is not a transfer", nvn_op_name(op));
    }
}
