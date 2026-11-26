// The just-in-time frame start (REKARINA_FRAME_START=jit, 1 or on; off by default).
//
// Where the GPU is the limit, the game's main thread runs its update early and then waits for the presenting thread,
// which is still busy with the frame before (its replay and the present sync, a wait for the GPU): before its acquire,
// in nvnWindowAcquireTexture, or when it hands the frame over. The input its update's first jobs sampled sat idle for
// that wait before the picture could go on: input latency and nothing else. This file holds the main thread's update
// start back by that wait, so its frame ends when the presenting thread can take it: the picture reaches the screen at
// the same moment, the input is newer.
//
// The anchor: the return of the main thread's wait on the hand-over queue. The main thread is the thread that calls
// nvnWindowAcquireTexture; the presenting thread the one that calls nvnQueuePresentTexture. The queue is learned from
// the two threads waking each other: the address of a SignalToAddress by which the presenting thread wakes the main
// thread (it took a frame and the main thread waited for room) or the main thread wakes the presenting thread (it waited
// for a frame), the most frequent one. After its acquire, the main thread's first WaitForAddress or SignalToAddress
// within FS_QSPAN bytes of that address is its hand-over; on return from that call it is held (kernel.c calls
// frame_start_kwait after the SVC, with no kernel lock held), with its virtual core given up (kern_core_away).
// Roles and the learned address, no code addresses. A frame whose hand-over is not seen (the queue had room and nobody
// waited) has no hold and counts as a frame without wait. In the title and loads the game reads the pad on the main
// thread itself; in play another thread reads it (DEBUG-PADCENSUS shows which).
//
// The hold's size: per cycle (update start to update start) the main thread's natural wait = its waits that the GPU or
// the presenting thread ended (kernel waits the presenting thread signalled: kernel.c wt_gpu; the acquire's and
// nvnSyncWait's waits) plus the hold that cycle started with (a held cycle waits that much less), and the hold is at
// most the presenting thread's own present-sync wait for the GPU (frame_start_present_wait). Where
// the GPU is the limit that wait is the game's own wait for the presenting thread just before its acquire (a frame
// trace of the village at 2880 lines: 5.9 ms of a 17.9 ms cycle, ending at the acquire), well after the input job.
// The hold = the 10th percentile of the last FS_N natural waits minus
// FS_MARGIN_NS, at most the same percentile of the last FS_N present-sync waits minus FS_MARGIN_NS. A frame that waited less than FS_SEEN_NS was late or had no wait: its natural wait counts as the hold
// minus the margin, and FS_LATE_RUN such frames in a row put the hold to 0 at once and start the record over (the waits
// vanished: a CPU-bound place). Where the main thread never waits (the frame rate is the game's own cycle) the hold is 0
// by construction. None under DLSS frame generation (Reflex paces the frame there: vk_dlssg.c), while an arrival holds
// frames unshown, or after a wait over FS_LONG_NS (a load): those start the record over.
//
// REKARINA_PACELOG gets per frame the update start (PL_UPD, after the hold), the pad read (PL_PAD: the first one on any
// thread after the update start, else the newest before it), the hold, the natural wait's measured part and the
// hand-over wait itself (PLD_*); the log has a summary line every FS_LOG_FRAMES frames.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "frame_start.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "pacelog.h"
#include "platform/platform.h"

#define FS_N 48                       // about half a second of frames
#define FS_PCT 10                     // percentile of the natural waits
static int fs_pct = FS_PCT;
#define FS_MARGIN_NS 500000ull        // kept below that percentile
#define FS_SEEN_NS 50000ull           // a smaller wait counts as none (the frame was late or never waited)
#define FS_LATE_RUN 3                 // frames in a row without a wait: the hold goes to 0 at once
#define FS_MAX_NS 25000000ull         // no hold over 25 ms
#define FS_LONG_NS 100000000ull       // a wait this long is a load: start over
#define FS_SPIN_NS (2ull * PLAT_WAIT_LATE_NS)   // the end of a hold is waited for on the clock (the OS timer is late)
#define FS_LOG_FRAMES 1200
#define FS_QSPAN 512                  // the queue's addresses: within this many bytes of the presenting thread's wait
#define FS_QLEARN 64                  // wakes between the two threads per look at the table

extern int kern_wait_meas;                          // kernel.c
uint64_t kern_self_wait_ns(uint64_t *last_end);     // kernel.c
uint64_t kern_self_gpu_wait_ns(uint64_t *last_end); // kernel.c
extern volatile uint32_t kern_fs_main_tid, kern_fs_present_tid;   // kernel.c
void kern_self_name(char *name, int n);             // kernel.c
int vk_dlssg_active(void);                          // vk_dlssg.c
int vk_reflex_active(void);                         // vk_reflex.c
int vk_arrive_holding(void);                        // vk_arrive.c

static int fs_mode = -1;          // 0 off, 1 jit (REKARINA_FRAME_START)
static int fs_meas;               // the mode or the pacing log reads the waits
static volatile uint32_t fs_tid;  // the thread that acquires: the game's main thread
static volatile uint32_t fs_ptid; // the thread that presents
static volatile uint64_t fs_qaddr;   // the hand-over queue's address (learned), 0 unknown
// main thread only from here
static int fs_open;               // an acquire, and no update start since
static uint64_t fs_frame;         // the frame number the last acquire started (pacing log numbering)
static uint64_t fs_wait;          // its waits so far: the acquire's and nvnSyncWait's (host waits, not in the kernel's)
static uint64_t fs_kw0;           // the thread's kernel wait total at the acquire
static uint64_t fs_hold;          // the hold the current frame started with
static uint64_t fs_upd_ns;        // the current frame's update start (after its hold), 0 none seen
static uint64_t fs_pad_before;    // the newest pad read before it
static volatile uint64_t fs_pad_last, fs_pad_after;   // any thread: the newest pad read, the first after fs_upd_ns
static uint64_t fs_cyc;           // the main thread's current cycle began here (its last update start), 0 none yet
static uint64_t fs_g0, fs_hw, fs_hw_end;   // the cycle's GPU-paced kernel waits at its start; its host waits for the GPU
static uint64_t fs_acquire_wait;
static uint32_t fs_nat[FS_N];
static unsigned fs_n, fs_late;
// The presenting thread's present sync waits for the GPU (frame_start_present_wait), the last FS_N: the hold never
// exceeds their FS_PCT percentile less the margin, so it is zero wherever the presenting thread does not wait for the
// GPU (the title, loading screens, menus, CPU-bound play), whatever else the main thread waits for there (the game's own
// pacing, a host thread's event). Written by the presenting thread, read by the main thread: single words.
static volatile uint32_t fs_ps[FS_N];
static volatile uint32_t fs_ps_n;
static uint64_t fs_s_capped;   // summary: frames whose hold the present sync's waits cut
// summary
static uint64_t fs_s_frames, fs_s_held, fs_s_hold, fs_s_late, fs_s_noq, fs_s_q;
static uint32_t fs_s_holds[FS_LOG_FRAMES], fs_s_waits[FS_LOG_FRAMES];

static void fstart_init(void) {
    const char *e = cfg_get("REKARINA_FRAME_START");
    // Off unless asked for: at the title the holds reached 16 ms and a 20 s start showed 207 frames over 25 ms against
    // 3 without it (2026-11-18, the first start of the build that had it on); the measured places were all in play.
    fs_mode = e && (!strcmp(e, "jit") || !strcmp(e, "1") || !strcmp(e, "on"));
    {   // DEBUG-FSPCT: REKARINA_FRAME_START_PCT=<1..50>: the percentile of the natural waits (an A/B aid)
        const char *q = cfg_get("REKARINA_FRAME_START_PCT");
        if (q && atoi(q) >= 1 && atoi(q) <= 50) fs_pct = atoi(q);
    }
    fs_meas = fs_mode || pacelog_on();
    if (fs_meas) kern_wait_meas = 1;
    rt_log("frame start: %s (REKARINA_FRAME_START=%s)%s", fs_mode ? "just in time: the main thread's update start waits "
           "out its recent waits at the frame's end" : "as the game starts it", e && *e ? e : "unset",
           fs_meas ? ", waits measured" : "");
}

static int cmp32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
static uint32_t pct(const uint32_t *v, unsigned n, int p) {
    uint32_t s[FS_LOG_FRAMES];
    if (!n) return 0;
    memcpy(s, v, n * sizeof *s);
    qsort(s, n, sizeof *s, cmp32);
    return s[(uint64_t)n * (unsigned)p / 100 < n ? (uint64_t)n * (unsigned)p / 100 : n - 1];
}

// What the hooks cost the game's threads (the main and the presenting thread; the others only compare an id in kernel.c):
// their time without the holds, summed over all calls, reported per frame in the summary line.
static volatile uint64_t fs_cost_ns, fs_cost_n, fs_hold_ns_total;
static double cost_frame_us(void) {   // the hooks' time a frame, the holds taken out
    uint64_t c = fs_cost_ns, h = fs_hold_ns_total;
    return (c > h ? c - h : 0) / 1e3 / FS_LOG_FRAMES;
}
static void cost_add(uint64_t ns) { __atomic_fetch_add(&fs_cost_ns, ns, __ATOMIC_RELAXED), __atomic_fetch_add(&fs_cost_n, 1, __ATOMIC_RELAXED); }

static void summary(uint64_t wait) {
    fs_s_waits[fs_s_frames % FS_LOG_FRAMES] = (uint32_t)(wait < 0xffffffffull ? wait : 0xffffffffull);
    fs_s_holds[fs_s_frames % FS_LOG_FRAMES] = (uint32_t)fs_hold;
    fs_s_held += fs_hold > 0;
    fs_s_hold += fs_hold;
    if (++fs_s_frames % FS_LOG_FRAMES) return;
    rt_log("frame start: %u frames: main's waits for the GPU or the presenting thread a cycle p10/50/90 %.2f/%.2f/%.2f ms; held %llu, hold mean %.2f "
           "p50/p90 %.2f/%.2f ms; late or no wait %llu; hand-over seen %llu, not seen %llu (queue %#llx); hooks %.1f us a frame "
           "in %.0f calls; hold cut to the present sync's waits %llu",
           FS_LOG_FRAMES, pct(fs_s_waits, FS_LOG_FRAMES, 10) / 1e6, pct(fs_s_waits, FS_LOG_FRAMES, 50) / 1e6,
           pct(fs_s_waits, FS_LOG_FRAMES, 90) / 1e6, (unsigned long long)fs_s_held, fs_s_hold / 1e6 / FS_LOG_FRAMES,
           pct(fs_s_holds, FS_LOG_FRAMES, 50) / 1e6, pct(fs_s_holds, FS_LOG_FRAMES, 90) / 1e6,
           (unsigned long long)fs_s_late, (unsigned long long)fs_s_q, (unsigned long long)fs_s_noq,
           (unsigned long long)fs_qaddr, cost_frame_us(), (double)fs_cost_n / FS_LOG_FRAMES, (unsigned long long)fs_s_capped);
    fs_cost_ns = fs_cost_n = fs_hold_ns_total = 0, fs_s_capped = 0;
    fs_s_held = fs_s_hold = fs_s_late = fs_s_noq = fs_s_q = 0;
}

static void hold_for(uint64_t ns) {
    int away = kern_core_away();
    uint64_t end = kern_now_ns() + ns;
    for (;;) {
        uint64_t now = kern_now_ns();
        if (now >= end) break;
        if (end - now > FS_SPIN_NS) plat_wait_ns(end - now - FS_SPIN_NS);
        else plat_yield();
    }
    kern_core_back(&away);
}

// The frame that just ended: its waits since its acquire (wait), whether its hand-over was seen; returns the next hold.
static uint64_t frame_end(uint64_t wait, int seen) {
    summary(wait);
    if (seen) fs_s_q++;
    else fs_s_noq++;
    uint64_t prev_hold = seen ? fs_hold : 0, nat;   // without a hand-over the hold never happened
    if (wait > FS_LONG_NS || vk_dlssg_active() || vk_reflex_active() || vk_arrive_holding()) {
        fs_n = fs_late = 0;   // not a frame to learn from: start over
        return 0;
    }
    if (wait < FS_SEEN_NS) {
        fs_s_late++;
        fs_late++;
        nat = prev_hold > FS_MARGIN_NS ? prev_hold - FS_MARGIN_NS : 0;
    } else {
        fs_late = 0;
        nat = wait + prev_hold;
    }
    fs_nat[fs_n++ % FS_N] = (uint32_t)nat;
    if (fs_late >= FS_LATE_RUN) { fs_n = 0; return 0; }   // the waits are gone: start the record over
    if (fs_mode <= 0 || fs_n < 16) return 0;
    uint64_t p = pct(fs_nat, fs_n < FS_N ? fs_n : FS_N, fs_pct), hold = p > FS_MARGIN_NS ? p - FS_MARGIN_NS : 0;
    {   // the GPU's share: the presenting thread's waits for it (REKARINA_FRAME_START_PSYNC=0: no cap, the old rule)
        static int cap = -1;
        if (cap < 0) cap = rt_knob("FRAME_START_PSYNC", "1");
        if (cap) {
            uint32_t ps[FS_N], n = __atomic_load_n(&fs_ps_n, __ATOMIC_ACQUIRE);
            if (n > FS_N) n = FS_N;
            for (uint32_t i = 0; i < n; i++) ps[i] = __atomic_load_n(&fs_ps[i], __ATOMIC_RELAXED);
            uint64_t q = n >= 16 ? pct(ps, n, fs_pct) : 0, lim = q > FS_MARGIN_NS ? q - FS_MARGIN_NS : 0;
            if (hold > lim) hold = lim, fs_s_capped++;
        }
    }
    return hold > FS_MAX_NS ? FS_MAX_NS : hold;
}

// nvnWindowAcquireTexture, after its waits (waited_ns: the pacer's and the fence's). frame: the pacing log's number of
// the frame the game starts.
static void frame_start_acquire_(uint64_t waited_ns, uint64_t frame) {
    if (__builtin_expect(fs_mode < 0, 0)) fstart_init();
    if (!fs_meas) return;
    uint32_t me = plat_thread_id();
    if (fs_tid != me) {
        if (fs_tid) rt_log("frame start: the acquiring thread is now %u (was %u)", me, fs_tid);
        fs_tid = me, fs_n = fs_late = 0, fs_open = 0;
        fs_cyc = fs_acquire_wait = fs_hw = fs_hw_end = 0;
        kern_fs_main_tid = me;
    }
    if (waited_ns) fs_acquire_wait += waited_ns, fs_hw += waited_ns, fs_hw_end = kern_now_ns();
    if (fs_open) {   // the frame before ended without a hand-over seen: no hold, its waits as they were
        uint64_t kw = kern_self_wait_ns(NULL);
        pacelog_dur(PLD_MWAIT, fs_frame, fs_wait + (kw - fs_kw0));
        frame_end(0, 0);   // its hold would have no place to happen: the next frame has none
        fs_cyc = 0;        // and its cycle has no start: the next one is measured from its own
        fs_hold = 0;
        fs_upd_ns = 0;
    }
    if (fs_upd_ns) {   // this frame's update start and input (seen before the frame had its number)
        uint64_t pad = fs_pad_after ? fs_pad_after : fs_pad_before;
        pacelog_mark(PL_UPD, frame, fs_upd_ns);
        if (pad) pacelog_mark(PL_PAD, frame, pad);
        pacelog_dur(PLD_HOLD, frame, fs_hold);
        fs_upd_ns = 0;
    }
    fs_open = 1;
    fs_frame = frame;
    fs_wait = waited_ns;
    fs_kw0 = kern_self_wait_ns(NULL);
}

// nvnSyncWait on any thread: a host wait for the GPU (ns)
static void frame_start_host_wait_(uint64_t ns) {
    if (fs_meas <= 0 || plat_thread_id() != fs_tid) return;
    if (fs_open) fs_wait += ns;
    fs_hw += ns, fs_hw_end = kern_now_ns();
}

// nvnQueuePresentTexture: who presents
static void frame_start_present_(void) {
    if (fs_meas > 0) {
        uint32_t me = plat_thread_id();
        if (fs_ptid != me) fs_ptid = me, fs_qaddr = 0, kern_fs_present_tid = me;
    }
}

// nvnQueuePresentTexture's present sync: ns it waited for the GPU (0: it did not wait). Presenting thread.
void frame_start_present_wait(uint64_t ns) {
    if (fs_meas <= 0) return;
    uint32_t i = __atomic_load_n(&fs_ps_n, __ATOMIC_RELAXED);
    __atomic_store_n(&fs_ps[i % FS_N], (uint32_t)(ns < 0xffffffffull ? ns : 0xffffffffull), __ATOMIC_RELAXED);
    __atomic_store_n(&fs_ps_n, i + 1, __ATOMIC_RELEASE);
}

// A wake between the main and the presenting thread: the address most of them are on is the hand-over queue.
static void learn(uint64_t addr) {
    static struct { uint64_t a; unsigned n; } t[8];
    static unsigned total;
    int i = 0;
    while (i < 8 && t[i].n && t[i].a != addr) i++;
    if (i == 8) i = 7;   // a full table: the last row takes the newcomer
    if (t[i].a != addr) t[i].a = addr, t[i].n = 0;
    t[i].n++;
    if (++total < FS_QLEARN) return;
    int best = 0;
    for (int k = 1; k < 8; k++) best = t[k].n > t[best].n ? k : best;
    uint64_t q = t[best].n * 4 > total ? t[best].a : 0;
    if (q != fs_qaddr)
        rt_log("frame start: hand-over queue %#llx (%u of %u wakes between the main thread %u and the presenting thread %u)",
               (unsigned long long)q, t[best].n, total, fs_tid, fs_ptid);
    fs_qaddr = q;
    memset(t, 0, sizeof t);
    total = 0;
}

// kernel.c, after SVC WaitForAddress (0x34; arg: how long the call blocked, ns) or SignalToAddress (0x35; arg: the host
// thread id of the first thread it woke, 0 none) returned, no kernel lock held. On the main thread the first one on the
// hand-over queue after its acquire is its update start.
static void frame_start_kwait_(int svc, uint64_t addr, uint64_t arg) {
    if (fs_meas <= 0) return;
    uint32_t me = plat_thread_id();
    if (svc == 0x35 && arg && ((me == fs_ptid && arg == fs_tid) || (me == fs_tid && arg == fs_ptid))) learn(addr);
    uint64_t q = fs_qaddr, waited_ns = svc == 0x34 ? arg : 0;
    if (me != fs_tid || !fs_open || !q || addr + FS_QSPAN < q || addr > q + FS_QSPAN) return;
    fs_open = 0;
    uint64_t kw = kern_self_wait_ns(NULL), gend, g = kern_self_gpu_wait_ns(&gend);
    pacelog_dur(PLD_QWAIT, fs_frame, waited_ns);
    pacelog_dur(PLD_MWAIT, fs_frame, fs_wait + (kw - fs_kw0));
    // The cycle that ends here (update start to update start): its waits that the GPU or the presenting thread ended
    // (the game's wait for the presenting thread before its acquire, a hand-over that found the queue full, the
    // acquire's and nvnSyncWait's waits) are the frame's slack; with no cycle start seen yet there is no measure.
    uint64_t gw = 0;
    if (fs_cyc) {
        uint64_t last = gend > fs_hw_end ? gend : fs_hw_end;
        gw = g - fs_g0 + fs_hw;
        pacelog_dur(PLD_GWAIT, fs_frame, gw);
        pacelog_dur(PLD_GLAST, fs_frame, last > fs_cyc ? last - fs_cyc : 0);
        pacelog_dur(PLD_ACQUIRE_WAIT, fs_frame, fs_acquire_wait);
        pacelog_dur(PLD_SYNC_WAIT, fs_frame, fs_hw - fs_acquire_wait);
        pacelog_dur(PLD_WAKE_WAIT, fs_frame, g - fs_g0);
    }
    uint64_t hold = fs_cyc ? frame_end(gw, 1) : 0;
    fs_hold = hold;
    if (hold) {
        uint64_t h0 = kern_now_ns();
        hold_for(hold);
        __atomic_fetch_add(&fs_hold_ns_total, kern_now_ns() - h0, __ATOMIC_RELAXED);
    }
    fs_upd_ns = fs_cyc = kern_now_ns();
    fs_g0 = kern_self_gpu_wait_ns(NULL), fs_acquire_wait = fs_hw = fs_hw_end = 0;
    fs_pad_before = fs_pad_last;
    fs_pad_after = 0;
}

// nn::hid::GetNpadStates (ov_input.c) answered with the live controller, on any thread, before the input is sampled.
static void frame_start_pad_(void) {
    if (__builtin_expect(fs_mode < 0, 0)) fstart_init();
    if (!fs_meas) return;
    uint64_t now = kern_now_ns();
    if (fs_upd_ns && !fs_pad_after && now > fs_upd_ns) fs_pad_after = now;
    fs_pad_last = now;
}

// DEBUG-PADCENSUS: which threads call nn::hid::GetNpadStates with which style and id, and whether the call was answered
// with the live controller; a table in the log every 4,000 calls while the pacing log or the mode is on.
void frame_start_pad_census(uint32_t want, int live, uint32_t id) {
    if (!pacelog_on()) return;
    static PlatMutex m = PLAT_MUTEX_INIT;
    static struct { uint32_t tid, key; uint64_t n; char name[24]; } t[16];
    static uint64_t calls;
    uint32_t tid = plat_thread_id(), key = want | (uint32_t)live << 8 | id << 16;
    plat_mutex_lock(&m);
    int i = 0;
    while (i < 16 && t[i].n && !(t[i].tid == tid && t[i].key == key)) i++;
    if (i < 16 && !t[i].n) kern_self_name(t[i].name, sizeof t[i].name);
    if (i < 16) t[i].tid = tid, t[i].key = key, t[i].n++;
    if (!(++calls % 4000)) {
        char line[1024];
        int o = snprintf(line, sizeof line, "frame start: pad census at frame %llu (acquiring thread %u):",
                         (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), fs_tid);
        for (int k = 0; k < 16 && t[k].n && o < (int)sizeof line - 64; k++)
            o += snprintf(line + o, sizeof line - o, " [thread %u %s style %u live %u id %#x: %llu]", t[k].tid, t[k].name,
                          t[k].key & 0xff, t[k].key >> 8 & 1, t[k].key >> 16, (unsigned long long)t[k].n);
        rt_log("%s", line);
        memset(t, 0, sizeof t);
    }
    plat_mutex_unlock(&m);
}

// The hooks: each one timed for the summary line (cost_add) while the frame start or the pacing log is on.
void frame_start_acquire(uint64_t waited_ns, uint64_t frame) {
    uint64_t t0 = fs_meas > 0 ? kern_now_ns() : 0;
    frame_start_acquire_(waited_ns, frame);
    if (t0) cost_add(kern_now_ns() - t0);
}

void frame_start_host_wait(uint64_t ns) {
    uint64_t t0 = fs_meas > 0 ? kern_now_ns() : 0;
    frame_start_host_wait_(ns);
    if (t0) cost_add(kern_now_ns() - t0);
}

void frame_start_present(void) {
    uint64_t t0 = fs_meas > 0 ? kern_now_ns() : 0;
    frame_start_present_();
    if (t0) cost_add(kern_now_ns() - t0);
}

void frame_start_kwait(int svc, uint64_t addr, uint64_t arg) {
    uint64_t t0 = fs_meas > 0 ? kern_now_ns() : 0;
    frame_start_kwait_(svc, addr, arg);
    if (t0) cost_add(kern_now_ns() - t0);
}

void frame_start_pad(void) {
    uint64_t t0 = fs_meas > 0 ? kern_now_ns() : 0;
    frame_start_pad_();
    if (t0) cost_add(kern_now_ns() - t0);
}
