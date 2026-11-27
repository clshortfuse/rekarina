// Adapted from MacNativeGames games/totk/runtime/game_patches.c @ ebe621d (MIT, Copyright (c) 2026 MacNativeGames
// contributors): runtime_frame_step_update (2338bb0: no 1/120 s floor) and the speed survey's player chain
// (world_survey.c player_actor).
// Frame rate: the frame limit and the real-time frame step, the host half of the patch sets fps60, realtime and cloth
// (build tables patches-v121.txt, patches-v143.txt).
//
// TotK's adaptive present-interval controller (main+0x8f26c8) turns the main thread's measured work into a number of
// 60 Hz ticks clamped to [minimum, maximum]; shipping data sets the minimum to 2 (30 fps), the set fps60 lowers it to
// 1. The controller still raises the interval when a frame does not fit. Shipping code then steps the game by whole
// vsyncs (VFR step main+0x7ede6c: ratio = interval * 0.5, seconds = interval / 60; global ratio at main+0x8f2394), so
// game speed is right only when frames take exactly their interval. With the set realtime both read the patch page
// instead: +0 ratio = seconds * 30, +4 seconds (1.4.3: +8 interval = seconds * 60, the producer's nominal interval
// before its half-step coefficient), the moving average of the last 5 display intervals, at most 1/15 s
// (slower frames slow the game down, as on the console) and at least 1/240 s, so a higher frame-rate cap only changes
// the pacer (frame_step_cap_ns, REKARINA_FPS_LIMIT). The set cloth gives hair and cloth the same clock and needs realtime.
//
// The floor: with a shorter step (ratio below 0.125) 1.4.3 aborts within seconds of some save loads (issue #241, a
// 280 Hz monitor): Havok's hkDefaultMemoryImpl reports "Out of memory" for the buffer the physics pre-solve (profiler
// scope "TtPreSolve") resizes in main+0x1261630. Forced steps on the Depths save: 1/240 s plays, 1/244 s and shorter
// abort; the field, sky island and fight saves play at 1/280 s, the town save aborts after about 800 frames. The
// same short steps also crash a physics worker reading that buffer's contacts: an access violation in main+0x1292560
// or its caller main+0x1286870 (narrowphase contact setup, reached from the grid task main+0x12f1ea0), reading a
// Havok body's shape at body+0x78 as null or garbage (issues #29 and #92, 240 Hz monitors with fps_limit = display).
// Steps drawn from 1/252..1/230 s crash the Depths load there in seconds; 1/240..1/230 s plays. The frame limit
// stops at 240 too (LIMIT_MAX_HZ), so the floor only catches the average dipping under 1/240 s at that limit: a
// present late by less than a period keeps the pacer's phase (nvn_api_obj.c, pace), so the next display interval is
// shorter than the period. No limit makes the game run slower than the clock.
//
// The sample is the display time the pacer gives each present (nvnQueuePresentTexture), so the step follows the
// frames as shown, pacing waits included. With the step on, the pacer ignores the game's interval and holds a frame
// only to the frame limit (nvn_api_obj.c present_period): the controller's 2 or 3 vsyncs would only add latency. The
// vsync event stays at 60 Hz whatever the limit: the game does not wait for it each frame (1.4.3 shows its title at 144
// fps with the event at 60 Hz).
// One 64-bit and one 32-bit store per present; the guest reads the floats on its own threads, and each stays finite
// and in range whichever store it sees.
//
// The world clock (frame_step_world_us) sums the steps from the first present that finds the player actor away from
// the origin, where the title screen keeps it: the end of the first save load. Recorded and scripted input
// (input.c) runs on it, so a replay presses at the same game time at any frame rate and after a first load of any
// length; a later load (fast travel) still shifts what follows by its length.
//
// REKARINA_TRACE=perf adds a line per 120 presents: fps, the game seconds stepped against wall seconds, and with
// REKARINA_SPEED_PROBE=1 the player's position and ground speed (an independent check: the hero runs at the same speed at
// any frame rate when game time follows wall time).
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "pacelog.h"   // DEBUG-PACELOG
#include "layout.h"
#include "platform/platform.h"

#define STEP_SAMPLES 5
#ifdef GAME_OOT
// OoT on the console steps 1/60 s, or 1/30 s when its frame controller drops to 30 fps, never more. With
// steps up to 1/15 s (the forest house at 15-25 fps) 11 of 13 runs crashed at the camera turn (main+0xc938d0, an
// empty resource slot); capped at 1/30 s, three runs passed it (REKARINA_PRESENT_SYNC=c on, no white frame either). A
// slower frame slows the game down instead, as the console's would.
#define STEP_MAX_DEFAULT (1.0 / 30.0)
#else
#define STEP_MAX_DEFAULT (1.0 / 15.0)
#endif
static double STEP_MAX_S = STEP_MAX_DEFAULT;   // longest step; a longer frame slows the game down
#define STEP_RESET_S 0.25          // a gap this long (loading, a debugger) restarts the average
#define STEP_FIRST_S (1.0 / 60.0)
#define STEP_MIN_S (1.0 / 240.0)   // shortest step; a shorter one makes the game's physics run out of memory

static _Atomic uint64_t *step_word;   // patch page +0: ratio (low float), seconds (high float)
static double step_now = STEP_FIRST_S;   // the step the patch page holds: what the game steps each frame
static _Atomic uint32_t *step_interval;   // patch page +8: seconds * 60 (patches-v143.txt realtime)
static uint64_t last_ns;
static double samples[STEP_SAMPLES];
static unsigned sample_count, sample_next;

static void step_store(float seconds) {
    step_now = seconds;
    float ratio = seconds * 30.0f, interval = seconds * 60.0f;
    uint32_t lo, hi, iv;
    memcpy(&lo, &ratio, 4);
    memcpy(&hi, &seconds, 4);
    memcpy(&iv, &interval, 4);
    atomic_store_explicit(step_word, (uint64_t)lo | (uint64_t)hi << 32, memory_order_release);
    atomic_store_explicit(step_interval, iv, memory_order_release);
}

static void frame_limit_live(void);

void frame_step_init(void) {
    int fps60 = rt_patch_set("fps60"), realtime = rt_patch_set("realtime"), cloth = rt_patch_set("cloth");
    if (realtime < 0) {
        rt_log("frame rate: no patch table, shipping 30 fps whole-vsync steps");
        return;
    }
    if (!rt_patch_enabled(realtime) && rt_patch_enabled(cloth)) {
        rt_patch_disable(cloth);
        rt_log("frame rate: cloth clock off, it needs the real-time step (REKARINA_REALTIME=0)");
    }
    {   // DEBUG-STEPMAX: REKARINA_STEP_MAX_FPS=<rate>: the longest step is 1/rate s (A/B of STEP_MAX_DEFAULT)
        const char *e = cfg_get("REKARINA_STEP_MAX_FPS");
        if (e && atoi(e) > 0) STEP_MAX_S = 1.0 / atoi(e), rt_log("frame rate: longest step 1/%d s (REKARINA_STEP_MAX_FPS)", atoi(e));
    }
    if (rt_patch_enabled(realtime)) {
        step_word = (_Atomic uint64_t *)rt_patch_page();
        step_interval = (_Atomic uint32_t *)((uint8_t *)rt_patch_page() + 8);
        step_store((float)STEP_FIRST_S);
    }
    frame_limit_live();   // REKARINA_FPS_LIMIT decides the set fps60 from the start
    char how[96] = "whole-vsync steps";
    if (step_word)
        snprintf(how, sizeof how, "real-time step (5-frame average of display intervals, at most 1/%.0f s)", 1.0 / STEP_MAX_S);
    rt_log("frame rate: present interval %s, %s",
           fps60 < 0 ? "the game's own" : rt_patch_enabled(fps60) ? "1..3 (above 30 fps)" : "2..3 (shipping 30 fps)", how);
}

int frame_step_realtime(void) { return step_word != NULL; }
double frame_step_now(void) { return step_now; }   // DEBUG-PACELOG: the step the patch page holds

// The frame limit (menu, REKARINA_FPS_LIMIT, live): 30, 60, 90, 120, 144, any whole rate from 30 to 240, or display (the
// refresh rate of the window's monitor, at most 240). A higher number gives 240, the rate of the shortest step the
// game survives (STEP_MIN_S). Unset, the older REKARINA_FPS60=0 still means 30. A limit of 30 switches the set
// fps60 off (the shipping minimum interval 2), every other limit switches it on; the controller reads its minimum
// interval from the patched sites each frame. With the real-time step the pacer holds frames to the limit; above 60
// only the pacer changes, as the step's floor is the limit's ceiling. Without it the game steps whole 60 Hz vsyncs and the pacer
// follows the game's interval, so the limit stops at 60.
#define LIMIT_MIN_HZ 30
#define LIMIT_MAX_HZ 240   // 1 / STEP_MIN_S
static int fps60_set = -2;
static unsigned limit_seen;
static int limit_read;
static int limit_hz = -1;   // 0: display; -1 before the first read

static int limit_parse(void) {
    const char *e = cfg_get("REKARINA_FPS_LIMIT"), *old = cfg_get("REKARINA_FPS60");
#ifdef GAME_OOT
    if (!e || !*e) return 0;   // OoT: unset is the monitor's rate (at most 240), cfg.c's default
#endif
    if (!e || !*e) return old && !strcmp(old, "0") ? 30 : 60;
    if (!strcmp(e, "display")) return 0;
    char *end;
    long n = strtol(e, &end, 10);
    if (!*end && n > LIMIT_MAX_HZ) {
        rt_log("frame rate: REKARINA_FPS_LIMIT=%s is above %d, where the game's physics runs out of memory; %d", e,
               LIMIT_MAX_HZ, LIMIT_MAX_HZ);
        return LIMIT_MAX_HZ;
    }
    if (!*end && n >= LIMIT_MIN_HZ) return (int)n;
    rt_log("frame rate: REKARINA_FPS_LIMIT=%s is not %d..%d or display; 60", e, LIMIT_MIN_HZ, LIMIT_MAX_HZ);
    return 60;
}

static void frame_limit_live(void) {
    unsigned v = cfg_version();
    if (limit_read && v == limit_seen) return;
    limit_read = 1;
    limit_seen = v;
    int hz = limit_parse();
    if (hz == limit_hz) return;
    limit_hz = hz;
    if (fps60_set == -2) fps60_set = rt_patch_set("fps60");
    if (fps60_set >= 0) rt_patch_switch(fps60_set, hz != 30);
    if ((hz > 60 || hz == 0) && !step_word)
        rt_log("frame rate: limits above 60 need the real-time step (REKARINA_REALTIME=1); 60 fps");
    else if (hz) rt_log("frame rate: limit %d fps", hz);
    else if (gfx_window_refresh_hz() > 0) rt_log("frame rate: limit at the monitor's refresh rate (%.3f Hz now)", gfx_window_refresh_hz());
    else rt_log("frame rate: limit at the monitor's refresh rate (60 until the window measures it)");
}

int vk_fg_generating(void);   // vk_fg.cpp

// REKARINA_MOVIE_FPS (default 60, 0 off): the frame rate while a movie plays. A movie's event script counts the game's updates
// as 60 Hz frames whatever the step: above 60 fps its narration lines start early, over one another, and the closing
// fade to black comes long before the movie ends and hides it. The movies are 60 fps, so the cap shows the same pictures.
__attribute__((weak)) int codec_movie_live(void) { return 0; }
static int movie_cap_hz(void) {
    static int hz = -1;
    if (hz < 0) {
        const char *e = cfg_get("REKARINA_MOVIE_FPS");
        hz = e && *e ? atoi(e) : 60;
        if (hz && hz < LIMIT_MIN_HZ) hz = 60;
    }
    if (!hz || !codec_movie_live()) return 0;
    static int logged;
    if (logged++ < 3) rt_log("frame rate: a movie plays: at most %d fps until it ends (REKARINA_MOVIE_FPS)", hz);
    return hz;
}

uint64_t frame_step_cap_ns(void) {
    frame_limit_live();
    double hz = limit_hz ? limit_hz : gfx_window_refresh_hz();
    if (hz < LIMIT_MIN_HZ) hz = 60;   // display rate unknown (no window yet)
    if (hz > LIMIT_MAX_HZ) hz = LIMIT_MAX_HZ;
    int mhz = movie_cap_hz();
    if (mhz && hz > mhz) hz = mhz;
    if (vk_fg_generating()) hz /= 2;   // frame generation shows two frames for each the game draws
    return (uint64_t)(1e9 / hz + 0.5);
}

// The frame rate the limit gives, for the performance overlay: the setting, the monitor's rate for "display", 60 at
// most without the real-time step; 0 before the first frame read the setting.
double frame_step_limit_hz(void) {
    int hz = limit_hz;
    if (hz < 0) return 0;
    double r = hz ? hz : gfx_window_refresh_hz();
    if (r < LIMIT_MIN_HZ) r = 60;
    return !step_word && r > 60 ? 60 : r > LIMIT_MAX_HZ ? LIMIT_MAX_HZ : r;
}

// The scene's scale of the game's 1440-line picture while the fixed path runs (recomp/patches-oot-v111.txt, set res_fixed): the
// DLSS mode's share when DLSS or FSR 3.1 runs, else the whole picture (render_scale and REKARINA_SCENE_SCALE lower it).
int vk_dlaa_available(void);
static double scene_share(void) {
    // The mode's share only where an upscaler runs (vk_dlaa_available: DLSS or FSR loaded at start). On a card
    // without DLSS and taa = dlss, or with taa = fsr3 and no AMD library, the scene stayed at 67 % or less with only
    // the game's own scaler to bring it back: a soft picture the menu gave no reason for.
    if (vk_dlaa_available() && cfg_bool("REKARINA_DLAA")) {
        const char *m = cfg_value("REKARINA_DLSS");
        return !strcmp(m, "quality") ? 2.0 / 3 : !strcmp(m, "balanced") ? 0.58 : !strcmp(m, "performance") ? 0.5
             : !strcmp(m, "ultra_performance") ? 1.0 / 3 : 1.0;
    }
    // Without DLSS the scene has the picture's size, 2560x1440: at the 0.625 the console mostly picks (1600x900) the
    // game's own scaler left the base picture soft. render_scale = 50..100 in rekarina.ini lowers it for slower cards;
    // REKARINA_SCENE_SCALE (a share, 0.25..1) is the test override.
    const char *e = cfg_get("REKARINA_SCENE_SCALE"), *r = cfg_value("REKARINA_RENDER_SCALE");
    double s = e && *e ? atof(e) : r && *r >= '0' && *r <= '9' ? atof(r) / 100.0 : 1.0;
    return s < 0.25 ? 0.25 : s > 1.0 ? 1.0 : s;
}

// The scene's share of the game's picture where the fixed path sets it (OoT), else 0: the overlay's "3D
// scene" line, for which res_sizes has only placeholders in a build without scene patch sets.
double res_fixed_scene_share(void) {
    static int set = -2;
    if (set == -2) set = rt_patch_set("res_fixed");
    return set < 0 ? 0.0 : scene_share();
}

// Each present. The set goes on once the controller has filled the fields the fixed path keeps.
#define RES_FIXED_AFTER 120
void res_fixed_present(void) {
    static int set = -2;
    static uint64_t presents;
    static unsigned seen = ~0u;
    static int seen_upscaler = -1;
    if (set == -2) set = rt_patch_set("res_fixed");
    if (set < 0) return;
    unsigned v = cfg_version();
    const int upscaler = vk_dlaa_available();   // readiness can change without an edit to the settings
    if (v != seen || upscaler != seen_upscaler) {
        seen = v;
        seen_upscaler = upscaler;
        float s = (float)(scene_share() * wide_scale_k());
        memcpy((uint8_t *)rt_patch_page() + 240, &s, 4);
    }
    if (++presents == RES_FIXED_AFTER && rt_knob("FIXED_SCENE", "1")) {
        rt_patch_switch(set, 1);
        rt_log("resolution: the game's dynamic resolution off: the scene at %.0f %% of its picture", scene_share() * 100);
    }
}

// ---- REKARINA_TRACE=perf speed lines
static uint64_t win_t0, win_frames, win_hist[4];   // win_hist: frames by present interval 1, 2, 3+ ([0] unused)
static double win_game;
static int probe = -1;
static float last_pos[3];
static uint64_t last_pos_ns;

static int read_guest(uint64_t a, void *out, size_t n) { return a && plat_mem_read(a, out, n); }
static uint64_t read_ptr(uint64_t a) {
    uint64_t p = 0;
    return read_guest(a, &p, 8) ? p : 0;
}
// 1.2.1: [main+0x4728538] (1.1.0: [main+0x4630a68], 1.4.3: [main+0x3add238]) +0x1e8 scene, its component 13 lists the
// resident actors; the one named "Player" has its position at +0x2b4 (MacNativeGames world_survey.c, game_version.h
// REKARINA_GAME_V121/V143). The 1.1.0 slot was measured from its own dereference chain: 181 scene accesses in that build
// read it, and one of them reproduces this walk (the 14-component check, component 13, the 0x70 stride and the name
// compare) instruction for instruction.
static int player_pos(float xyz[3]) {
#ifdef GAME_NS2
    (void)xyz;   // the walk above is TotK's: no player position, the world clock does not start
    return 0;
#endif
    uint64_t module = read_ptr(MOD_MAIN_BASE + (RT_V143 ? 0x3add238 : RT_V110 ? 0x4630a68 : 0x4728538)),
             scene = module ? read_ptr(module + 0x1e8) : 0;
    int32_t count = 0;
    if (!scene || !read_guest(scene + 0x50, &count, 4) || count < 14 || count > 256) return 0;
    uint64_t comps = read_ptr(scene + 0x58), resident = comps ? read_ptr(comps + 13 * 8) : 0;
    if (!resident || !read_guest(resident + 0x20, &count, 4) || count < 1 || count > 64) return 0;
    uint64_t links = read_ptr(resident + 0x28);
    for (int i = 0; links && i < count; i++) {
        uint64_t data = read_ptr(links + (uint64_t)i * 0x70 + 8), actor = data ? read_ptr(data + 0x40) : 0;
        uint64_t name = actor ? read_ptr(actor + 0x218) : 0;
        char text[7];
        if (name && read_guest(name, text, 7) && !memcmp(text, "Player", 7))
            return read_guest(actor + 0x2b4, xyz, 12) && isfinite(xyz[0]) && isfinite(xyz[2]);
    }
    return 0;
}

static void speed_report(uint64_t now, double step, int interval) {
    if (!win_t0) {
        win_t0 = now;
        return;
    }
    win_frames++;
    win_game += step;
    win_hist[interval < 1 ? 1 : interval > 3 ? 3 : interval]++;
    if (win_frames < 120) return;
    if (probe < 0) {
        const char *e = cfg_get("REKARINA_SPEED_PROBE");
        probe = e && !strcmp(e, "1");
    }
    double wall = (double)(now - win_t0) * 1e-9;
    char pos[160] = "";
    float p[3];
    if (probe && player_pos(p)) {
        double d = last_pos_ns ? hypot(p[0] - last_pos[0], p[2] - last_pos[2]) : 0,
               s = last_pos_ns ? (double)(now - last_pos_ns) * 1e-9 : 0;
        snprintf(pos, sizeof pos, "; player %.1f %.1f %.1f, %.2f m in %.3f s = %.2f m/s", p[0], p[1], p[2], d, s,
                 s > 0 ? d / s : 0);
        memcpy(last_pos, p, sizeof p);
        last_pos_ns = now;
    }
    if (step_word)
        rt_log("fps: %llu frames in %.3f s (%.1f fps), %.3f s game stepped, speed %.3f; interval 1/2/3+ %llu/%llu/%llu%s",
               (unsigned long long)win_frames, wall, (double)win_frames / wall, win_game, win_game / wall,
               (unsigned long long)win_hist[1], (unsigned long long)win_hist[2], (unsigned long long)win_hist[3], pos);
    else
        rt_log("fps: %llu frames in %.3f s (%.1f fps), whole-vsync steps; interval 1/2/3+ %llu/%llu/%llu%s",
               (unsigned long long)win_frames, wall, (double)win_frames / wall, (unsigned long long)win_hist[1],
               (unsigned long long)win_hist[2], (unsigned long long)win_hist[3], pos);
    win_t0 = now;
    win_frames = 0;
    win_game = 0;
    memset(win_hist, 0, sizeof win_hist);
}

static _Atomic int64_t world_us = -1;
static _Atomic uint64_t game_us;   // game time stepped since the first present (frame_step_ticks)
static uint64_t presents;

int64_t frame_step_world_us(void) { return atomic_load_explicit(&world_us, memory_order_relaxed); }
uint64_t frame_step_ticks(void) { return atomic_load_explicit(&game_us, memory_order_relaxed) * 60 / 1000000; }

static void world_clock(double dt, double step) {
    presents++;
    int64_t w = atomic_load_explicit(&world_us, memory_order_relaxed);
    if (w < 0) {
        float p[3];
        if (!player_pos(p) || fabsf(p[0]) + fabsf(p[2]) < 1.0f) return;   // a few guest reads, until the load ends
        rt_log("frame step: world clock starts at present %llu, %.2f s after the guest started (the player at %.1f %.1f %.1f)",
               (unsigned long long)presents, rt_guest_start_ns ? (kern_now_ns() - rt_guest_start_ns) / 1e9 : 0.0, p[0],
               p[1], p[2]);
        atomic_store_explicit(&world_us, 0, memory_order_relaxed);
        return;
    }
    if (!step_word) step = dt > STEP_MAX_S ? STEP_MAX_S : dt;   // whole-vsync steps: the display time stands in
    atomic_store_explicit(&world_us, w + (int64_t)(step * 1e6 + 0.5), memory_order_relaxed);
}

// ---- DEBUG-PACELOG: REKARINA_PACELOG=<file>, CSV at exit: frame, the moments in microseconds from the first acquire
// (acquire, call, return, flip, gpu, visible, blit, GPU timestamp; 0: not seen; pacelog.h), the step in microseconds. A store or two per moment.
#define PL_CAP (1u << 18)   // about 45 minutes at 95 fps; later frames overwrite the oldest
static struct PlRec { uint64_t t[PL_N]; uint64_t frame; float step, d[PLD_N]; } *pl;
static int pl_state = -1;
static uint64_t pl_t0;

int pacelog_on(void) {
    if (__builtin_expect(pl_state < 0, 0)) {
        const char *e = cfg_get("REKARINA_PACELOG");
        pl_state = e && *e && (pl = calloc(PL_CAP, sizeof *pl)) != NULL;
        if (pl_state) rt_log("frame rate: pacing log on (REKARINA_PACELOG=%s)", e);
    }
    return pl_state;
}

static struct PlRec *pl_rec(uint64_t frame) {
    struct PlRec *r = &pl[frame % PL_CAP];
    if (r->frame != frame) memset(r, 0, sizeof *r), r->frame = frame;
    return r;
}

void pacelog_mark(int what, uint64_t frame, uint64_t ns) {
    if (!pacelog_on() || what < 0 || what >= PL_N || !frame) return;
    if (!pl_t0 && what != PL_VIS) pl_t0 = ns;
    pl_rec(frame)->t[what] = ns;
}

void pacelog_dur(int what, uint64_t frame, uint64_t ns) {
    if (!pacelog_on() || what < 0 || what >= PLD_N || !frame) return;
    pl_rec(frame)->d[what] = (float)(ns / 1e3);
}

void pacelog_step(uint64_t frame, double step_s) {
    if (!pacelog_on() || !frame) return;
    pl_rec(frame)->step = (float)(step_s * 1e6);
}

void pacelog_write(void) {
    if (pl_state <= 0) return;
    FILE *f = fopen(cfg_get("REKARINA_PACELOG"), "w");
    if (!f) return;
    fprintf(f, "frame,acq_us,call_us,ret_us,flip_us,gpu_us,vis_us,blit_us,gts_us,step_us,pad_us,upd_us,hold_us,mwait_us,qwait_us,psync_us,gwait_us,glast_us,acquire_wait_us,sync_wait_us,wake_wait_us\n");
    uint64_t lo = ~0ull, hi = 0;
    for (uint32_t i = 0; i < PL_CAP; i++)
        if (pl[i].frame) lo = pl[i].frame < lo ? pl[i].frame : lo, hi = pl[i].frame > hi ? pl[i].frame : hi;
    for (uint64_t fr = lo; hi && fr <= hi; fr++) {
        const struct PlRec *r = &pl[fr % PL_CAP];
        if (r->frame != fr) continue;
        fprintf(f, "%llu", (unsigned long long)fr);
        for (int k = 0; k < PL_N; k++) {   // the step after PL_GTS, so the columns before it keep their places
            if (k == PL_PAD) fprintf(f, ",%.1f", r->step);
            fprintf(f, ",%.1f", k == PL_VIS ? r->t[k] / 1e3   // the presentation engine's clock (it may not be ours): as it is
                                : r->t[k] && r->t[k] >= pl_t0 ? (r->t[k] - pl_t0) / 1e3 : 0.0);
        }
        for (int k = 0; k < PLD_N; k++) fprintf(f, ",%.1f", r->d[k]);
        fprintf(f, "\n");
    }
    fclose(f);
    rt_log("frame rate: pacing log of frames %llu-%llu written", (unsigned long long)lo, (unsigned long long)hi);
    pl_state = 0;
}

// vk_arrive.c: while a scene's first frames are held unshown, each steps the game by the shortest step, so its opening
// animation and fade barely move before the first frame is shown; the average starts again after.
static int step_hold;
void frame_step_hold(int on) { step_hold = on; }

void frame_step_present(uint64_t display_ns, int interval) {
    frame_limit_live();
    double step = 0, dt = last_ns ? (double)(display_ns - last_ns) * 1e-9 : 0;
    if (step_word && step_hold) {
        sample_count = sample_next = 0;
        step = STEP_MIN_S;
        step_store((float)step);
    } else if (step_word && last_ns) {
        if (dt > STEP_RESET_S || dt < 0.001) {
            sample_count = sample_next = 0;
        } else {
            if (dt > STEP_MAX_S) dt = STEP_MAX_S;
            samples[sample_next++ % STEP_SAMPLES] = dt;
            if (sample_count < STEP_SAMPLES) sample_count++;
            double sum = 0;
            for (unsigned i = 0; i < sample_count; i++) sum += samples[i];
            step = sum / sample_count;
            if (step < STEP_MIN_S) step = STEP_MIN_S;
            step_store((float)step);
        }
    }
    last_ns = display_ns;
    // The game clock of untimed script lines: what the game stepped this frame, the patch page's step with the
    // real-time step (kept through a reset), whole vsyncs without it.
    double game = step_word ? step_now : (interval > 0 ? interval : 1) / 60.0;
    atomic_store_explicit(&game_us, atomic_load_explicit(&game_us, memory_order_relaxed) + (uint64_t)(game * 1e6 + 0.5),
                          memory_order_relaxed);
    world_clock(dt, step);
    if (rt_trace & RT_TRACE_PERF) speed_report(display_ns, step, interval);
}
