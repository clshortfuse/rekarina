// Interface of the host runtime: guest memory, modules, kernel, IPC, file system and boot.
//
// Guest addresses are host addresses. Every guest-visible region lies below 2^39 (IPC buffer descriptors carry 39
// address bits). Fixed parts of the guest address map:
//   modules           MOD_<X>_BASE from layout.h, from 0x70_0000_0000
//   kernel region     RT_KERNEL_BASE, 256 MiB: TLR, TLS blocks, host objects the guest reads (mem_kernel_alloc)
//   host functions    RT_HOSTFN_BASE, 1 MiB, never accessible: synthetic addresses of host functions
// The stack, heap and alias regions are placed at start-up (mem_layout).
#pragma once
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "dispatch.h"

// ---- logging and results
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
// DEBUG-HEAPCHECK: where a growing array of the suspects lives now (perf.c heap_check names the one a broken heap
// entry lies in or after). what: a string literal, the key.
void rt_heap_note(const char *what, const void *p, size_t bytes);   // "[host] <msg>\n" to the log, one write
// The log is stderr plus, in a game run whose stderr is not a file, logs\totk.log next to rekarina.exe (main.c log_open).
// rt_logf and rt_log_raw write to both without the prefix and the newline: crash reports, fatal errors, guest output.
void rt_logf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void rt_vlogf(const char *fmt, va_list ap);
void rt_log_raw(const char *p, size_t n);
const char *rt_log_path(void);                  // the log file, NULL when this run writes none
enum { RT_TRACE_SVC = 1, RT_TRACE_HOSTCALL = 2, RT_TRACE_IPC = 4, RT_TRACE_FS = 8, RT_TRACE_GFX = 16, RT_TRACE_PERF = 32,
       RT_TRACE_SCHED = 64, RT_TRACE_AUDIO = 128, RT_TRACE_PRESENT = 256 };
extern unsigned rt_trace;                       // from REKARINA_TRACE=svc,host,ipc,fs,gfx,perf,sched,audio,present (or all)
// prof_win.c: REKARINA_PROFILE=<path> sampling profiler; REKARINA_TRACE=perf frame timing (waits of the render thread)
// PERF_TEX: texture uploads on the render thread (deswizzle, decode, staging), PERF_ASTC included
enum { PERF_PACE, PERF_FENCE, PERF_DRAIN, PERF_PIPE, PERF_XLATE, PERF_TEX, PERF_ASTC, PERF_N };
void prof_start(void);
void prof_write(void);                          // idempotent; boot.c calls it before _exit
uint64_t perf_begin(void);                      // start time of a wait; pair with perf_end
void perf_end(int kind, uint64_t t0);           // counts render-thread time only
void perf_replay(uint64_t t0);                  // replay (NVN -> Vulkan recording) time on any thread, waits inside it included
void perf_replay_thread(void);                  // the calling thread replays for the render thread: perf_end counts its waits
void perf_main_thread(void);                    // the calling thread is the guest main thread (REKARINA_PROFILE_MAIN)
// REKARINA_SECTIONS=<file>: sums of the replay's sections per present, a row of CSV each with the interval and the render
// thread's waits (perf.c), plus a line every 60 frames of what the load registers and creates. A section is a clock read
// at each end and two additions; off, sec_on (set once at start) is the only cost. The _T/_D/_P kinds split work by
// what asked for the image: a render target or vk_texture_get, a draw's binding, the texture pool's update. SUB_*
// partition vk_sync_submit (SUB_OFFCPU, beside them: of the submission's wall time, what the replay thread spent not
// running, preempted or blocked; PAR_RUN / PAR_SPIN, inside SUB_REST: the chunk join running left-over chunks itself /
// waiting for the other threads' chunks; RL_*: the replay thread's waits for a lock another thread holds: the
// texture lock, the upload lock, the pool lock), OP_* the ops inside it (REKARINA_SPIKE_MS also on), MB_* vk_mem_mirror_begin.
// TS_* (vk_texture.c, "the texture stream"): the replay thread's storage hashes and texture cache file reads, its waits
// for a stream worker, and the counts of textures not ready at their first use and of pool slots left for later.
// The clock is the CPU's cycle counter, so a section around a draw stays under 20 ns.
#define SEC_LIST(X) \
    X(SUB_TOTAL) X(SUB_LOCK) X(SUB_EV) X(SUB_USNAP) X(SUB_MIRB) X(SUB_MIRE) X(SUB_GMEM) X(SUB_TEX) X(SUB_REC) X(SUB_UPL) \
    X(SUB_PRO) X(SUB_QUE) X(SUB_REST) X(SUB_OFFCPU) X(PAR_RUN) X(PAR_SPIN) X(RL_TEX) X(RL_UPL) X(RL_POOL) \
    X(OP_DRAW) X(OP_DISPATCH) X(OP_CLEAR) X(OP_REPORT) X(OP_BARRIER) X(OP_PASS) X(OP_OTHER) \
    X(BIND_RES) X(BIND_CS) X(SET_DYN) X(SHADER_MISS) X(PIPE_MISS) X(SHADER_MESH) X(PIPE_MESH) X(SIM8_SHADER) X(SIM8_PIPE) \
    X(TP_UPDATE) X(TP_RESOLVE) X(TP_VIEWS) X(TP_PUT) X(TP_SLOTS) X(TP_DESC) X(TP_BACKLOG) X(TP_ARRAY) X(TP_OLD) X(TP_NEW) \
    X(IC_CREATE_T) X(IC_CREATE_D) X(IC_CREATE_P) X(IC_BIND_ALIAS_T) X(IC_BIND_ALIAS_D) X(IC_BIND_ALIAS_P) \
    X(IC_BIND_SUB_T) X(IC_BIND_SUB_D) X(IC_BIND_SUB_P) X(IC_VMA_T) X(IC_VMA_D) X(IC_VMA_P) \
    X(IC_REQ) X(IC_RESOLVE) X(IC_ALLOC_SUB) X(IC_ZERO) X(MADE_T) X(MADE_D) X(MADE_P) \
    X(REP_ALIAS_NEW) X(REP_ALIAS_REPEAT) X(REP_OWN_NEW) X(REP_OWN_REPEAT) \
    X(UP_TOTAL) X(DESW_S) X(DESW_M) X(DESW_L) X(DESW_XL) X(STAGING) X(UP_FLUSH) \
    X(ASTC_REC) X(ASTC_SCRATCH) X(ASTC_GLOCK) X(ASTC_TLOCK) \
    X(MIR_CREATE) X(MIR_SYNC) \
    X(MB_TOTAL) X(MB_AWB) X(MB_DEMOTE) X(MB_LOCKS) X(MB_SNAP) X(MB_FREEZE) X(FZ_QUERY) X(FZ_LOCK) X(FZ_STATE) X(FZ_PROTECT) \
    X(FZ_OK) X(FZ_REFUSED) \
    X(TEXREG) X(TEXREG_LOCK) X(VKALLOC_R) X(VKALLOC_O) X(MR_BUDGET) X(GPU_UPLOAD) X(GPU_PROLOGUE) \
    X(PK_STATE) X(PK_BIND) X(PK_DRAW) X(PK_XFER) X(PK_BARRIER) X(PK_OTHER) \
    X(TS_HASH) X(TS_READ) X(TS_WAIT) X(TS_MISS) X(TS_DEFER) X(ROW)
enum {
#define SEC_X(n) SEC_##n,
    SEC_LIST(SEC_X)
#undef SEC_X
    SEC_N
};
extern int sec_on;
#if defined(__x86_64__)
static inline uint64_t sec_clock(void) { return __builtin_ia32_rdtsc(); }
#else
uint64_t plat_now_ns(void);
static inline uint64_t sec_clock(void) { return plat_now_ns(); }
#endif
void sec_add(int kind, uint64_t ticks);                      // any thread
void sec_add_ns(int kind, uint64_t ns);
int sec_replay_thread(void);   // the calling thread replays (the render thread or the asynchronous replay's)                      // a duration from the ns clock
void sec_count(int kind, int64_t n);                         // a count without a time
void sec_peak(int kind, uint64_t v);                         // the largest value of this present (counts, e.g. the pool backlog)
double sec_ticks_ms(uint64_t ticks);
void sec_init(void);                                         // reads REKARINA_SECTIONS; repeat calls do nothing
void sec_alloc_note(uint64_t size, uint32_t type, uint64_t ticks);   // VMA's vkAllocateMemory wrapper (vk_device.c)
// The long-frame report (perf.c "Long frames"): running totals of what the back end did, taken per frame. Any thread.
enum { LF_UP_BYTES, LF_ADOPT, LF_DROP, LF_EVICT, LF_EVICT_BYTES, LF_POOL_RES, LF_POOL_STANDIN, LF_N };
void lf_count(int kind, int64_t n);
int perf_scene_frames_ago(void);   // presents since a submission last drew a scene (weak: 0; vk_texture.c has it)
extern _Thread_local int sec_thread_off;   // this thread's sections are not timed (the chunk threads of vk_exec.c)
static inline uint64_t sec_begin(void) { return __builtin_expect(sec_on, 0) && !sec_thread_off ? sec_clock() : 0; }
static inline void sec_end(int kind, uint64_t t0) { if (__builtin_expect(t0 != 0, 0)) sec_add(kind, sec_clock() - t0); }
// Events per 60-frame window: graphics pipelines finished by the background workers, draws skipped while pending,
// first texture uploads and re-uploads after CPU writes, draws skipped while a shader translates in the background,
// draws that waited for their pending pipeline and got it, draws into a target the previous frame did not render that
// waited for their pending shader or pipeline instead of being skipped (vk_exec_draw.c prepare_draw). The rest feed only
// the flight recorder (flight.c): graphics pipelines created on any thread, texture images created, images created at an
// address where a live image of another descriptor already starts, write-backs done and skipped because the CPU
// rewrote the storage after the GPU render (vk_texture.c vk_texture_write_back), mesh and geometry draws and compute
// dispatches left out while their shader is translated or their pipeline built beside the replay (vk_exec_draw.c
// left_out).
enum { PERF_CNT_PIPE_ASYNC, PERF_CNT_PIPE_SKIP, PERF_CNT_TEX_UP, PERF_CNT_TEX_REUP, PERF_CNT_SHADER_SKIP,
       PERF_CNT_PIPE_WAIT, PERF_CNT_ONESHOT_WAIT, PERF_CNT_PIPE_BUILT, PERF_CNT_TEX_NEW, PERF_CNT_TEX_ALIAS,
       PERF_CNT_WB, PERF_CNT_WB_SKIP, PERF_CNT_COND_GPU, PERF_CNT_COND_UNKNOWN, PERF_CNT_COND_MEM_TRUE,
       PERF_CNT_COND_MEM_FALSE, PERF_CNT_MESH_SKIP, PERF_CNT_CS_SKIP, PERF_CNT_N };
void perf_count(int kind);
void perf_present(uint64_t frame);
void perf_summary_flush(void);                  // prints the frame-time summary of the running period (exit); the 60 s lines come from perf_present
void gfx_exit_log(void);                        // back end's lines for the exit summary (weak no-op in prof_win.c)
void perf_gpu_wait(uint64_t t0);                // a wait for a GPU fence ended (any thread; the render thread's also counts as PERF_FENCE)
// GPU side of the 60-frame perf line, filled by the graphics back end when it set perf_gpu_stats. gpu_ns: GPU busy time
// since the previous call (timestamp queries, overlaps counted once), have_gpu 0 when the device has no timestamps.
typedef struct {
    int have_gpu, have_vram, vram_exact;
    uint64_t gpu_ns, gpu_parts, gpu_missed;     // busy time, submission parts timed, parts left untimed
    uint64_t vram_used, vram_budget;
    int have_idle;                              // calibrated timestamps: the idle split below is filled
    uint64_t idle_ns, starved_ns, queued_ns;    // GPU idle between timed parts; of it, before the part was submitted; submit to start
    uint64_t idle_gaps, max_gap_ns;             // gaps over 0.5 ms; the longest
} PerfGpu;
extern void (*perf_gpu_stats)(PerfGpu *out);
// The performance overlay (perfmon.c): mode 0 off, 1 fps, 2 full. While it is on, perf_present hands every frame to
// perfmon_frame; full also takes the event counts (perf_count) and the GPU timestamps as REKARINA_TRACE=perf does.
void perf_set_overlay(int mode);
int perf_gpu_timing(void);                      // GPU timestamps wanted: REKARINA_TRACE=perf or the full overlay
void perf_gpu_busy(uint64_t ns);                // GPU busy time of a timed submission part (vk_sync.c, any thread)
uint64_t perf_gpu_busy_total(void);             // its running total (the overlay's GPU time per frame)
#define RT_RESULT(module, desc) ((uint32_t)(module) | ((uint32_t)(desc) << 9))
void rt_sha256(const void *p, size_t n, uint8_t out[32]);   // BCrypt; fatal if the provider fails

// ---- guest address map (fixed part)
#define RT_ADDRESS_LIMIT (1ull << 39)
#define RT_KERNEL_BASE 0x7010000000ull
#define RT_KERNEL_SIZE 0x10000000ull
#define RT_HOSTFN_BASE 0x7020000000ull
#define RT_HOSTFN_SIZE 0x100000ull

// ---- AAPCS64 argument access for host functions (section 2.2)
static inline uint64_t arg_x(const Cpu *c, int i) { return i < 8 ? c->x[i] : LD64(c->sp + 8 * (uint64_t)(i - 8)); }
static inline void *arg_p(const Cpu *c, int i) { return (void *)(uintptr_t)arg_x(c, i); }
static inline float arg_s(const Cpu *c, int i) { uint32_t b = (uint32_t)c->v[i]; float f; memcpy(&f, &b, 4); return f; }
static inline double arg_d(const Cpu *c, int i) { uint64_t b = (uint64_t)c->v[i]; double d; memcpy(&d, &b, 8); return d; }
static inline void ret_x(Cpu *c, uint64_t v) { c->x[0] = v; }
static inline void ret_s(Cpu *c, float f) { uint32_t b; memcpy(&b, &f, 4); c->v[0] = b; }
static inline void ret_d(Cpu *c, double d) { uint64_t b; memcpy(&b, &d, 8); c->v[0] = b; }

// ---- host functions (hostfn.c)
// Every host function is a GuestFn at a synthetic address RT_HOSTFN_BASE + 16*i. It runs on the calling guest thread,
// may clobber x0-x18, v0-v7, v16-v31, must preserve x19-x29, sp, v8-v15, tpidr, and returns to the guest BL site with
// a C return. On entry c->pc holds the function's own synthetic address.
typedef struct { const char *name; GuestFn fn; } RtHostEntry;
#define RT_CAT2(a, b) a##b
#define RT_CAT(a, b) RT_CAT2(a, b)
// File-scope declarations; fn must be declared before. The loader binds them by symbol name. The entries of every
// object file meet in one section per kind, which hostfn.c walks: on PE/COFF ".ovr$m", which the linker sorts between
// hostfn.c's ".ovr$a" and ".ovr$z" markers; on ELF "totk_ovr", which the linker brackets with __start_totk_ovr and
// __stop_totk_ovr.
#ifdef _WIN32
#define RT_HOST_SECTION(kind) section("." #kind "$m")
#else
#define RT_HOST_SECTION(kind) section("totk_" #kind)
#endif
#define HOST_OVERRIDE(sym, fn) /* replaces sym in every GOT/GLOB_DAT/ABS64 of every module */                  \
    __attribute__((used, RT_HOST_SECTION(ovr))) static const RtHostEntry RT_CAT(rt_ovr_, __COUNTER__) = {sym, fn}
#define HOST_OBSERVER(sym, fn) /* JUMP_SLOT only; fn calls rt_original(c) */                                    \
    __attribute__((used, RT_HOST_SECTION(obs))) static const RtHostEntry RT_CAT(rt_obs_, __COUNTER__) = {sym, fn}
uint64_t hostfn_register(const char *name, GuestFn fn);    // synthetic address; before hostfn_seal only. The same
                                                           // (name, fn) pair returns the same address
GuestFn hostfn_override(const char *name);                 // NULL if none
GuestFn hostfn_observer(const char *name);                 // NULL if none
void hostfn_add_override(const char *name, GuestFn fn);    // an override outside the linker table (probes); before
                                                           // ldr_load
const char *hostfn_name(uint64_t addr);                    // name registered at a synthetic address, NULL if none
void hostfn_seal(void);                                    // build the dispatch table (rt_dispatch_init over every
                                                           // linked module); idempotent; registration ends here
uint64_t hostfn_calls(void);                               // host-function calls so far, all threads (exact while rt_counts_exact)
// 1 until the game's first present (nvn_api_obj.c): host calls and SVCs are counted one by one, for the boot line and
// the probes. In play the counters are shared words every game thread would write thousands of times a frame, so the
// host calls are no longer counted and the SVCs are added in batches per thread.
extern int rt_counts_exact;
void hostfn_report(void);                                  // call count per host function to stderr, busiest first
int hostfn_table_size(int observers);                      // linked HOST_OVERRIDE (0) / HOST_OBSERVER (1) entries
const char *hostfn_table_name(int observers, int i);       // symbol of linked entry i, NULL past the end
uint64_t rt_original(Cpu *c);                              // inside an observer: call the guest definition with the
                                                           // current x0..x7, return x0. Uses c->pc, so call it before
                                                           // anything that changes c->pc (rt_call restores it)
uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...);         // host -> guest, section 2.3. n <= 8 arguments, each read
                                                           // as uint64_t: cast every argument. Seals the dispatch
                                                           // table on first use. Sets x30 = RT_RET_HOST, restores
                                                           // x30 and pc, returns x0. Does not touch sp
#define RT_RET_HOST 0x70200ff000ull                        // x30 sentinel inside the host-function window; a jump to
                                                           // it is a named fatal

// ---- guest memory (memory.c)
typedef struct { uint64_t base, size; } RtRange;
typedef struct { RtRange aslr, stack, heap, alias, kernel, hostfn; uint64_t pool; } RtGuestLayout;
typedef struct { uint64_t base, size; uint32_t state, attr, perm, ipc, device, pad; } RtMemoryInfo;   // Horizon layout
void mem_init(void);                                       // reserve every region; on a collision re-run the process
                                                           // (up to 20 times, REKARINA_REEXEC), then fatal
const RtGuestLayout *mem_layout(void);
uint32_t mem_set_heap_size(uint64_t size, uint64_t *addr);                      // 0x01
uint32_t mem_set_permission(uint64_t a, uint64_t n, uint32_t perm);             // 0x02
uint32_t mem_set_attribute(uint64_t a, uint64_t n, uint32_t mask, uint32_t at); // 0x03
uint32_t mem_map(uint64_t dst, uint64_t src, uint64_t n);                       // 0x04
uint32_t mem_unmap(uint64_t dst, uint64_t src, uint64_t n);                     // 0x05
uint32_t mem_query(uint64_t a, RtMemoryInfo *out);                              // 0x06
uint32_t mem_map_physical(uint64_t a, uint64_t n);                              // 0x2C
uint32_t mem_unmap_physical(uint64_t a, uint64_t n);                            // 0x2D
int mem_info(uint32_t id, uint64_t *out);                  // GetInfo memory ids 2-7, 12-17, 21, 22, 28; 0 if not a memory id
void *mem_kernel_alloc(uint64_t size, uint64_t align);     // zeroed, guest-visible, in the kernel region; align <= 64 KiB
uint64_t mem_written_pages(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max);
                                                           // GetWriteWatch(RESET): fills pages[] (4 KiB addresses)
                                                           // with the pages of [addr, addr+size) written since the
                                                           // last call, returns their count (at most max; a page
                                                           // past max stays reported next time). The range must lie
                                                           // inside the stack/heap block or the alias region
                                                           // (write watch); otherwise returns UINT64_MAX
uint64_t mem_written_peek(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max);
                                                           // the same without the reset: the bits stay for the next
                                                           // mem_written_pages (verify checks only)
int mem_freeze(uint64_t a);                                // makes the 2 MiB region at a (2 MiB aligned, watched,
                                                           // committed PAGE_READWRITE) read-only until a write or
                                                           // mem_thaw thaws it; 1 frozen (or already), 0 refused
int mem_frozen(uint64_t a);                                // the 2 MiB region of a is frozen: no page in it changed
                                                           // since its freeze
void mem_thaw(uint64_t a, uint64_t n);                     // makes the frozen regions over [a, a+n) writable again;
                                                           // before a kernel write (file read) into guest memory
void mem_freeze_stats(uint64_t *frozen, uint64_t *freezes, uint64_t *fault_thaws, uint64_t *other_thaws);
// GPU result watch (REKARINA_RESULT_WATCH=1, a diagnostic for running frames ahead of the GPU): the guest pages that hold a
// result the GPU produces (counter reports, events) are made inaccessible when the replay records the command and
// accessible again when the completion thread writes the value; the first access before that is logged with the
// instruction that made it (a guest function in rekarina.map) and whether the GPU had finished, then the page opens.
int mem_result_watch_on(void);
// DEBUG-TS-PENDING (vk_exec_xfer.c): the fence of the newest timestamp report recorded at addr, 0 when none.
void vk_ts_report_note(uint64_t addr, uint64_t fence);
uint64_t vk_ts_report_fence(uint64_t addr);
// Copy guard: guest ranges a recorded GPU copy reads when the GPU reaches it (vk_memory.c's mirror uploads) stay
// read-only until then; a CPU write waits for exactly that copy (wait), the rest of the frame runs on.
void mem_copy_guard_setup(uint64_t (*done)(void), int (*wait)(uint64_t value));
void mem_copy_guard(uint64_t addr, uint64_t n, uint64_t value);   // after the copy's submission
void mem_copy_guard_sweep(void);                                    // opens the guards whose copy ran
void mem_copy_guard_stats(uint64_t *ranges, uint64_t *faults, uint64_t *waits, uint64_t *wait_ns);
void mem_result_setup(uint64_t (*fence_done)(void), uint64_t (*frame)(void));
void mem_result_watch(uint64_t addr, uint64_t n, uint64_t fence);   // replay thread, at the recording
// The result guard (REKARINA_RESULT_GUARD): guest threads that touch [addr, addr+n) before fence is done wait for it (the
// replay thread, at the end of the recording; the completion releases the pages). wait: blocks until fence is done
// (1) or gives up (0); guest: 1 on a guest thread.
int mem_result_guard_on(void);
// Lazy write-back (vk_memory.c): pages closed until the CPU touches them; the hook runs at that first access.
void mem_lazy_setup(void (*hook)(uint64_t page_addr, int cold));
int mem_lazy_close(uint64_t addr, int cold);
void mem_lazy_close_run(uint64_t addr, uint32_t n, uint8_t *ok);   // n pages as mem_lazy_close(page, 0) each; ok[i] its answer
void mem_lazy_open(uint64_t addr, uint64_t n);
void mem_result_guard(uint64_t addr, uint64_t n, uint64_t fence);
void mem_result_set_wait(int (*wait)(uint64_t fence), int (*guest)(void));
int kern_guest_thread(void);   // 1 on a guest thread (kernel.c)
void mem_result_release(uint64_t addr, uint64_t n);                // completion thread, before writing the value
void mem_result_log(uint64_t frame);                               // the readers seen so far, every few seconds
                                                           // regions frozen now; freezes and thaws since the last call
void mem_kernel_free(void *p, uint64_t size);              // TLR/TLS reuse after thread exit

// ---- modules (loader.c)
typedef struct {
    const char *name; uint64_t base, size, text_end, data_off;   // text_end, data_off: offsets from base
    uint64_t init, init_array, init_array_size, ctor_flag;  // absolute guest addresses; init_array_size in bytes
} RtLoaded;
void ldr_load(const char *exefs_dir);                      // load, relocate (overrides applied), seal; once
int ldr_count(void);
const RtLoaded *ldr_module(int i);                         // order: main, subsdk0, sdk
uint64_t ldr_symbol(const char *name);                     // exported definition, 0 if none
uint64_t ldr_plt_stub(const char *module, const char *name); // PLT stub whose GOT slot holds name, 0 if none
const char *ldr_describe(uint64_t a, char *buf, int n);    // "main+0x1234 (symbol+0x10)", "host <name>", or the address
uint8_t *ldr_read_verified(const char *dir, const char *file, size_t *len);   // whole file, SHA-256 checked against
                                                           // modules.c (fatal on mismatch); free() it

// Written by tools/build_game.py into build/<version>/modules.c.
typedef struct { const char *file, *sha256; } RtFileHash;
extern const RtFileHash rt_exefs_sha256[];                 // tools/data/expected-<version>.json "exefs_sha256"
extern const int rt_exefs_sha256_count;
extern const char rt_default_exefs[], rt_default_romfs[], rt_default_profile[];
extern const int rt_game_version;                          // the game build: 110 (1.1.0), 121 (1.2.1) or 143 (1.4.3)
#ifdef GAME_OOT
#define RT_PORT_NAME "Rekarina"                            // the window, menu and Vulkan application name
#define RT_WINDOW_TITLE "Rekarina"
#else
#define RT_PORT_NAME "TotK Recomp"
#define RT_WINDOW_TITLE "TotK Recomp"
#endif
// The files the player sees beside the game: <stem>.ini, logs/<stem>.log, <stem>_xlate.exe with its .id and
// <stem>_dlssg.dll, and the game's exe as the package names it. RT_OLD_FILE_STEM: the stem those files had in a folder
// made before Rekarina got its own names: cfg.c moves that ini over at the first start, gfx_shader.c and vk_dlssg.c
// still take the translator and the library under it (a build folder has them under the old names as well).
#ifdef GAME_OOT
#define RT_FILE_STEM "rekarina"
#define RT_OLD_FILE_STEM "totk"
#define RT_EXE_STEM "Rekarina"
#else
#define RT_FILE_STEM "totk"
#define RT_EXE_STEM "totk"
#endif
#define RT_INI_FILE RT_FILE_STEM ".ini"
#ifdef _WIN32
#define RT_EXE_FILE RT_EXE_STEM ".exe"
#else
#define RT_EXE_FILE RT_EXE_STEM
#endif
extern const char rt_port_version[];                       // VERSION at the build's root: 0.1.0, 0.1.0+973c626 in a release
const char *rt_game_version_text(void);
// The Switch 2 graphics scale tier reported to the game (ns2_host.c; 0: none, the game draws at the window's size).
int host_graphics_tier(void);                  // "1.1.1": the game update for menus, logs and reports
#ifdef GAME_NS2   // another game (Rekarina): no TotK build's addresses or behaviour apply
#define RT_V110 0
#define RT_V143 0
#else
#define RT_V110 (rt_game_version == 110)                    // 1.1.0: its own boot constants and data slots
#define RT_V143 (rt_game_version >= 143)
#endif

// ---- kernel (kernel.c)
typedef struct RtThread RtThread;
Cpu *kern_cpu(void);                                       // current guest thread's Cpu, NULL on host threads
void kern_free_cores_note(void);                           // for crash reports: logs that free_cores = 1 lifted the
                                                           // virtual core gate, nothing when the gate is on
uint32_t kern_main_thread(Cpu **cpu);                      // main thread object and Cpu for the calling host thread
                                                           // (main.npdm: 1 MiB stack at the top of the stack region,
                                                           // priority 44, core 0); on the main thread: its handle
void kern_main_stack(uint64_t *base, uint64_t *size);      // the main thread's stack, size 0 before kern_main_thread;
                                                           // svc.c reports it to QueryMemory, memory.c does not know it
void kern_run_on_guest_main(void (*fn)(Cpu *c));           // start the guest main host thread, run fn there, join
int kern_core_away(void);                                  // give the virtual core up around a blocking host wait
void kern_core_back(int *away);
int kern_core_away_soft(void);                             // kern_core_away for long host work (the replay): with
                                                           // folded cores the slot is kept until a better thread takes it
int kern_wait_off_core(void);                              // REKARINA_WAIT_OFF_CORE (on unless 0)
// A host lock other threads may hold for milliseconds, from a guest thread that holds no other host lock: 1 when taken,
// with every wait for it made without the virtual core; 0: take it the usual way (not on a core, the switch off, or
// still busy after a few rounds). try_lock takes it (1) or fails at once; wait_free returns once it was free.
int kern_lock_off_core(int (*try_lock)(void *), void (*wait_free)(void *), void *m);
int kern_core_away_counted(void);                          // kern_core_away under REKARINA_WAIT_OFF_CORE, counted
void kern_checkpoint(void);                                // preemption point, at every SVC and indirect call
// A preempt request is pending for the thread (Cpu.thread): its RtThread starts with the flag (kernel.c).
static inline int kern_preempt_pending(const void *thread) { return __atomic_load_n((const int *)thread, __ATOMIC_RELAXED); }
// Set once before guest threads start (REKARINA_CORE_DIAG, REKARINA_ASYNC_PREEMPT): host calls (hostfn.c) and SVCs (svc.c) count their depth in
// the thread's second int, so the kernel knows whether a core owner runs game code at the top level.
extern int kern_host_track;
static inline int *kern_host_depth(void *thread) { return (int *)thread + 1; }
void kern_host_left(void *thread);                         // back at depth 0 with a preempt request pending
int kern_core_diag_on(void);                               // REKARINA_CORE_DIAG
void kern_core_diag_frame(uint64_t frame, uint64_t interval_ns);   // perf_present, every present (also logs the
                                                           // asynchronous preemption counters)
void kern_sched_report(uint64_t frame);                    // REKARINA_TRACE=sched: per-thread time split, on present
uint32_t kern_handle_new(int kind, void *obj);             // kind: KERN_SESSION or KERN_TMEM (ipc.c)
void *kern_handle_get(uint32_t h, int kind);
void kern_handle_close(uint32_t h);                        // CloseHandle from the host side, any thread; fatal on a
                                                           // bad handle
enum { KERN_SESSION = 4, KERN_TMEM = 5 };
uint32_t kern_event_new(void);                             // writable handle
uint32_t kern_event_readable(uint32_t writable);
void kern_event_signal(uint32_t writable);
void kern_bind_system_event(Cpu *c, uint64_t ev, int autoclear);   // guest thread only
void kern_signal_system_event(uint64_t ev);                // any thread
int kern_svc(Cpu *c, uint32_t imm);                        // thread/sync SVCs; 1 if handled
uint64_t kern_now_ns(void);
extern uint64_t rt_guest_start_ns;   // kern_now_ns when the guest started (boot.c), 0 before
typedef struct { uint64_t svcs, threads, host_calls; } RtStats;
RtStats rt_stats(void);                                    // host_calls: hostfn_calls()

// ---- IPC (ipc.c)
int ipc_svc(Cpu *c, uint32_t imm);                         // 0x15, 0x1F, 0x21, 0x22; 1 if handled
void ipc_handle_closed(int kind, void *obj);               // CloseHandle on a KERN_SESSION/KERN_TMEM handle

// ---- file system (romfs.c / fs_host.c)
void fs_init(void);                                        // idempotent; paths from rt_config()

// ---- account (ov_system.c)
void account_uid(uint64_t out[2]);                         // the local user's nn::account::Uid {lo, hi}, persistent in
                                                           // <profile>/account.uid; fs_host.c names save data after it

// ---- movies (ov_movie.c)
void movie_tick(Cpu *c);                                   // at every present, on the presenting guest thread: hands
                                                           // decoded movie frames over when due, signals their events
// ---- camera speed (camera_speed.c): REKARINA_STICK_SENS
float cam_speed_wanted(void);                              // the factor REKARINA_STICK_SENS asks for: TotK 0.1 to 3,
                                                           // OoT 0.5 to 4
float cam_speed_applied(void);                             // the factor in effect: TotK 1 until romfs.c serves the
                                                           // pack; OoT the live factor while patch set stick_sens is on
int cam_pack_build(const void *in, size_t n, const void *zsdic, size_t zsdic_n, void **out, size_t *out_n);
                                                           // Pack/Actor/PlayerCamera.pack.zs with the camera speeds
                                                           // multiplied, malloc'd; 0 = factor 1 or cannot (logged)
void cam_pack_served(void);                                // romfs.c serves the multiplied pack
// ---- the port's scene scale on a mod's Bootup pack (scene_pack.c, OoT)
int scene_pack_build(const void *in, size_t n, const void *zsdic, size_t zsdic_n, void **out, size_t *out_n);
                                                           // the values set (> 0, out: the pack again, malloc'd), 0 when
                                                           // the pack has them already, -1 when it cannot (logged)
// ---- game resolution (resolution.c): REKARINA_OUTPUT_RES, REKARINA_RENDER_SCALE
uint64_t res_heap_extra_bytes(void);                       // added to the application pool (mem_init), 2 MiB steps
uint64_t res_heap_extra_kib(void);                         // added to the HeapDef texture heap and its parent; 0 = none
int res_heapdef_build(const void *in, size_t n, void **out, size_t *out_n);   // the HeapDef table (.byml.zs) with
                                                           // the grown heaps, malloc'd; 0 = cannot (logged)
void res_heapdef_served(void);                             // romfs.c serves the grown table
void res_init(void);                                       // after rt_patches_init: the patch page and the sets
float res_dlss_lod_bias(void);                             // REKARINA_DLSS upscaling: texture LOD bias (<= 0)
int res_dlss_upscales(void);                               // a DLSS mode that scales; 0: DLAA, scene = output
int res_dlss_mode(const char *mode);                       // REKARINA_DLSS changed while running: new scene size
void res_fn_hit(uint64_t addr, uint64_t x0);               // main.c fnprof_hit: catches the renderer at start
void res_present(Cpu *c);                                  // nvnQueuePresentTexture: applies a pending scene size
void res_sizes(unsigned out[2], unsigned scene[2]);      // the output and scene sizes in effect (the game's own
                                                           // until res_init switched the sets on)
int res_is_16x9(unsigned w, unsigned h);                   // within 1% of 16:9 (the test res_init and romfs.c share)
int res_wants_wide_overlay(const unsigned out[2]);         // not 16:9 on a game version with the camera aspect patch
void res_wanted(unsigned out[2]);                          // the output size REKARINA_OUTPUT_RES asks for
void res_fov_live(void);                                   // REKARINA_FOV to the camera (set fov); on a settings change
// ---- code patches (patches.c) and the frame step (frame_rate.c)
void rt_patches_init(void);                                // after ldr_load: switches from the environment, the patch
                                                           // page committed; logs the sets
int rt_patch_set(const char *name);                        // index of a set of generated patches.h, -1 if absent
int rt_patch_enabled(int set);                             // the set is on (0 for -1)
void rt_patch_disable(int set);                            // before the guest starts: a set needs another that is off
void rt_patch_switch(int set, int on);                     // at run time (live settings): each site takes the new code
                                                           // the next time it runs
void *rt_patch_page(void);                                 // the page the patched code reads (NULL without the table)
void frame_step_init(void);                                // after rt_patches_init: the real-time step's first value
int frame_step_realtime(void);                             // the real-time step is on: presents ignore the interval
void wide_init(void);                                      // widescreen.c
int wide_available(void);                                  // the generated C has the widescreen sets
float wide_aspect_ratio(void);                             // active picture aspect ratio (16:9 when widescreen is off)
void wide_picture(unsigned *w, unsigned *h);               // the picture the game draws (its scene is never larger)
unsigned wide_auto_picture_height(void);                   // automatic picture height for the current window settings
uint64_t wide_heap_extra_kib(void);                        // the texture heap's growth (OoT: full scene, widescreen)
uint64_t wide_heap_base_kib(void);                         // the texture heap's shipped size (OoT)
double wide_scale_k(void);                                 // the picture's height over the game's 1440
int res_adjustable(void);                                  // the generated C has the output and scene patch sets
void res_fixed_present(void);                              // frame_rate.c: each present, the fixed scene scale
uint64_t frame_step_cap_ns(void);                          // with the real-time step: the shortest frame in ns
double frame_step_limit_hz(void);                         // the frame rate the limit gives (overlay); 0 unknown
int64_t frame_step_world_us(void);                         // game time since the loaded save's first frame, -1 before
                                                           // (REKARINA_FPS_LIMIT, live)
uint64_t frame_step_ticks(void);                           // game time since the first present in 60 Hz ticks (input scripts)
// In-launch A/B measurement: REKARINA_AB=<seconds> splits the world clock (frame_step_world_us) into blocks of that much
// game time in the order A B B A (repeated); rt_ab() is 1 in the B blocks, 0 in A blocks, before the world clock and
// without REKARINA_AB. rt_knob(name, def) reads the switch
// REKARINA_<name> once per call (def when unset): 0 off, 2 for "ab" (on in the B blocks only), 1 for anything else;
// callers keep it and test it with RT_KNOB_ON. The perf line counts the B frames of its window (tools/ab_summary.py
// compares the windows that are wholly A or wholly B).
int rt_ab(void);
int rt_knob(const char *name, const char *def);
#define RT_KNOB_ON(mode) ((mode) == 2 ? rt_ab() : (mode))
void frame_step_present(uint64_t display_ns, int interval); // nvnQueuePresentTexture: the frame's display time and
                                                           // present interval (60 Hz ticks)
// ---- boot (boot.c, main.c)
typedef struct { const char *exefs, *romfs, *profile; int boot; double seconds; const char *shot, *gpu; } RtConfig;
                                                           // romfs: directory with romfs.bin and index.json;
                                                           // frames: --frames N (0 = no limit); shot: --shot
                                                           // PATH; gpu: REKARINA_GPU
const RtConfig *rt_config(void);
void rt_boot_event(const char *what, Cpu *c, const char *detail);   // checkpoint log; "nvn" ends --boot
void rt_boot(void);                                        // boot.c
void rt_boot_process_priority(int level);                  // the priority class rt_boot takes once the entry step is done
// Modes of rekarina.exe. main.c has run mem_init and parsed the paths; each mode calls ldr_load itself if it needs the
// modules, registers its host functions, then seals (hostfn_seal, or implicitly at the first rt_call). Return the
// process exit code.
int probe_cfg(void);                                       // cfg_probe.c
int probe_dpi(void);                                       // dpi_probe_win.c (Windows only)
int probe_mods(void);                                      // mods_probe.c
void discord_init(void);                                   // discord_rpc.c: the Rich Presence thread, game runs only
int probe_discord(void);                                   // discord_rpc.c
int probe_loader(void), probe_kernel(void), probe_ipc(void), probe_services(void), probe_fs(void);
int probe_gfx_abi(void);                                   // gfx_abi_probe.c
int probe_vkdevice(void);                                 // vk_device.c
int probe_nvn(void), probe_nvn_tex(void);                  // nvn_probe.c, nvn_tex_probe.c
int probe_window(void);                                    // gfx_window_probe.c
int probe_shader(void);                                    // gfx_shader.c
int probe_census_dir(char *dir, size_t n);                 // gfx_shader_probe.c: build/<version>/shaders for the probes
int probe_vkmem(void), probe_vkexec(void), probe_astc(void);                 // tex_probe.c, vk_exec_probe.c, astc_probe.c
int probe_color(void);                                    // color_probe.c
int probe_movie(void);                                     // movie_probe.c: host VP9 movie::Decoder on one RomFS movie
int probe_codec(const char *path);                         // ns2_codec.c (GAME_NS2): one WebM through the host nn::codec
                                                           // video decoder, as OoT's movie library calls it
int vk_pipeline_prepare_shaders(void);                     // vk_pipeline.c: --prepare-shaders (tools/prepare_shaders.py)
int vk_pipeline_warm_shaders(void);                        // vk_pipeline.c: --warm-shaders (tools/build_game.py)
int vk_pipeline_warm_libraries(void);                      // vk_pipeline.c: --warm-libraries (the launcher's background build)
int tex_prepare_scan_main(void);                           // tex_prepare.c: --scan-textures (the entry step's child)
int tex_prepare_main(void);                                // tex_prepare.c: --prepare-textures (tools/oot_prepare_textures.py)
void tex_prepare_entry_start(void);                        // tex_prepare.c: the entry step's textures (vk_pipeline_prewarm)
void tex_prepare_entry_join(void);
