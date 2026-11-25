// GPU breadcrumbs for a lost device (vk.h vk_crumb). VK_AMD_buffer_marker writes a 32-bit code into a buffer from
// the command stream: at TOP_OF_PIPE once the command processor reaches the marker, at BOTTOM_OF_PIPE once every
// command before it has finished. Each crumb writes both, into two words of one host-coherent buffer, so after a
// device loss word 0 names the newest point the GPU reached and word 1 the newest point it had finished: the hang lies
// between them. A code is the crumb kind in the top 4 bits and its number in the low 28. The markers cost two small
// writes per crumb, and crumbs mark submissions, upload batches, ASTC chunks and presents, not draws.
//
// Pass marks (vk.h vk_pass_mark_begin) use words 2 and 3 of the same buffer: the replay marks the start and the end of
// every render pass and dispatch, and a ring of records says what each mark was. The design follows Savage Wind
// Recomp's GPU checkpoints (refs/SavageWindRecomp runtime/nvn/vulkan/exec.c ck_mark and vk_checkpoints_report: a mark
// per pass and dispatch, numbers in a ring of 4096, text only at the device loss); the code is ours.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "report.h"

static PFN_vkCmdWriteBufferMarkerAMD write_marker;
static VkBuffer buf;
static VmaAllocation alloc;
static volatile uint32_t *words;   // 0, 1: crumbs; 2, 3: pass marks

// Pass marks (below)
int vk_pass_marks_on;
static int pm_amd;                          // marks into words 2 and 3
static PFN_vkCmdSetCheckpointNV set_checkpoint;
static void pass_marks_init(VkCtx *c);

static void buffer_init(VkCtx *c) {
    write_marker = (PFN_vkCmdWriteBufferMarkerAMD)vkGetDeviceProcAddr(c->device, "vkCmdWriteBufferMarkerAMD");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (!write_marker || vmaCreateBuffer(c->vma, &bi, &ai, &buf, &alloc, &info) != VK_SUCCESS) {
        rt_log("vk: GPU breadcrumbs unavailable");
        write_marker = NULL;
        return;
    }
    words = info.pMappedData;
    words[0] = words[1] = words[2] = words[3] = 0;
    rt_log("vk: GPU breadcrumbs on (VK_AMD_buffer_marker)");
}

void vk_crumbs_init(VkCtx *c) {
    // DEBUG-NOCRUMBS: REKARINA_CRUMBS=0: no breadcrumbs at all (what the per-part crumbs cost the GPU)
    if (c->has_buffer_marker && rt_knob("CRUMBS", "1")) buffer_init(c);
    pass_marks_init(c);
}

void vk_crumbs_release(void) {
    VkCtx *c = vk_ctx();
    if (buf && c) vmaDestroyBuffer(c->vma, buf, alloc);
    buf = VK_NULL_HANDLE, alloc = NULL, words = NULL, write_marker = NULL;
    vk_pass_marks_on = pm_amd = 0, set_checkpoint = NULL;
}

void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n) {
    if (!write_marker || !cb) return;
    uint32_t code = (uint32_t)kind << 28 | (uint32_t)(n & 0x0fffffff);
    write_marker(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buf, 0, code);
    write_marker(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, buf, 4, code);
}

static void describe(uint32_t code, char *o, size_t size) {
    static const char *const kinds[] = {"nothing", "the start of a part of submission", "the end of submission",
                                        "the start of upload batch", "the end of upload batch",
                                        "ASTC decode chunk", "the start of present",
                                        "the end of present"};
    unsigned k = code >> 28;
    if (k >= VK_CRUMB_CHUNK && k <= VK_CRUMB_AFTER_STRETCH) {   // DEBUG-PARCRUMB
        char d[200];
        vk_exec_par_describe((code & 0x0fffffff) >> 4, d, sizeof d);
        snprintf(o, size, "%s %u of draw stretch %u (%s)", k == VK_CRUMB_CHUNK ? "the start of chunk" : k == VK_CRUMB_CHUNK_END ?
                 "the end of chunk" : "the replay thread's command buffer after chunk count", code & 15, (code & 0x0fffffff) >> 4, d);
        return;
    }
    static const char *const copies[] = {"the start of an alias copy from the texture at page", "the end of an alias copy from the texture at page",
                                         "the start of mesh draw count copies, batch", "the end of mesh draw count copies, batch"};
    if (k == VK_CRUMB_DRAW) {   // DEBUG-DRAWCRUMB
        void vk_exec_draw_crumb_describe(uint32_t n, char *o, size_t size);
        vk_exec_draw_crumb_describe(code & 0x0fffffff, o, size);
        return;
    }
    if (k >= VK_CRUMB_ALIAS && k <= VK_CRUMB_MESHCNT_END) {   // DEBUG-COPYCRUMB
        snprintf(o, size, "%s %#x", copies[k - VK_CRUMB_ALIAS], code & 0x0fffffff);
        return;
    }
    if (!code) snprintf(o, size, "nothing yet");
    else if (k < sizeof kinds / sizeof *kinds) snprintf(o, size, "%s %u", kinds[k], code & 0x0fffffff);
    else snprintf(o, size, "unknown code %#x", code);
}

// DEBUG-LIVECHECK: how far the GPU is by its breadcrumbs: the submission (low 28 bits of its value) of the newest
// command whose end-of-pipe marker ran, and whether that command was the submission's last (its end crumb). 0: the
// newest marker is not one that names a submission (an upload batch, a present), or there are no breadcrumbs.
int vk_crumbs_progress(uint32_t *sub, int *ended, uint32_t *raw) {
    if (!words) return 0;
    uint32_t code = words[1], k = code >> 28, n = code & 0x0fffffff;
    *raw = code;
    if (k == VK_CRUMB_SUBMIT_END) { *sub = n, *ended = 1; return 1; }
    if (k == VK_CRUMB_SUBMIT) { *sub = n, *ended = 0; return 1; }
    if (k == VK_CRUMB_DRAW) {
        uint64_t vk_exec_draw_crumb_fence(uint32_t n);
        uint64_t f = vk_exec_draw_crumb_fence(n);
        if (!f) return 0;
        *sub = (uint32_t)(f & 0x0fffffff), *ended = 0;
        return 1;
    }
    return 0;
}

void vk_crumbs_report(void) {
    if (!words) return;
    char reached[1100], finished[1100];
    describe(words[0], reached, sizeof reached);
    describe(words[1], finished, sizeof finished);
    rt_log("vk: GPU breadcrumbs: the GPU last reached %s", reached);
    rt_log("vk: GPU breadcrumbs: everything before %s had finished", finished);
}

// ---- pass marks

// Cost per pass or dispatch: about twenty stores into the ring and two commands, one at the start and one at the end,
// each a marker write or a checkpoint. Per draw nothing, except that the branch of vk_exec_draw.c prepare_draw that
// records changed pipeline state also notes the shaders of a pass's first draw. A full barrier already precedes every
// pass and dispatch (vk_exec.c vk_exec_pass_begin, vk_exec_draw.c op_dispatch), so the end-of-pipe write adds no wait.
static VkPassMark ring[VK_PASS_MARKS];
static uint32_t seq;                        // marks made; only the replay thread writes it

// REKARINA_PASS_CRUMBS: unset, buffer marks where VK_AMD_buffer_marker exists; 1, also NVIDIA checkpoints; 0, neither.
// After a run whose graphics device was lost (report.c report_last_run_lost_device: logs/rekarina.1.log at this start),
// a player's run (one that writes logs/rekarina.log) marks its passes as REKARINA_PASS_CRUMBS=1 would, unless the
// setting says otherwise: the logs that two players sent of such losses said only which submission's part the GPU was
// in (2026-11-19), and the marks cost about a tenth of the GPU's frame, so not every run pays for them.
static int marks_after_loss(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_PASS_CRUMBS");
        on = (!e || !*e) && rt_log_path() && report_last_run_lost_device();
    }
    return on;
}

int vk_pass_marks_want_nv(void) {
    const char *e = cfg_get("REKARINA_PASS_CRUMBS");
    return (e && *e && strcmp(e, "0") != 0) || marks_after_loss();
}

static void pass_marks_init(VkCtx *c) {
    const char *e = cfg_get("REKARINA_PASS_CRUMBS");
    int off = e && !strcmp(e, "0");
#ifdef GAME_OOT
    // OoT: off unless asked for (REKARINA_PASS_CRUMBS=1, or =amd for the buffer marks alone). A frame of forest
    // village has ~295 passes and dispatches, each marked at its start and at its end, and a mark at the end of the
    // pipe makes the GPU finish what is in flight before it goes on: with the marks the native 2560x1440 scene took
    // 13.5 ms of GPU time a frame, without them 12.2 (68 against 75 fps, standing camera, RTX 5070).
    if ((!e || !*e) && !cfg_get("REKARINA_GPU_PASSTIME") && !marks_after_loss()) off = 1;   // the pass timing needs the marks
#endif
    if (marks_after_loss()) rt_log("vk: the last run lost the graphics device: this one marks its passes for the report");
    set_checkpoint = c->has_nv_checkpoints
                         ? (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(c->device, "vkCmdSetCheckpointNV") : NULL;
    pm_amd = !off && write_marker && words;
    vk_pass_marks_on = pm_amd || set_checkpoint;
    if (vk_pass_marks_on)
        rt_log("vk: pass marks on (%s%s%s; REKARINA_PASS_CRUMBS=0 turns them off%s)", pm_amd ? "VK_AMD_buffer_marker" : "",
               pm_amd && set_checkpoint ? " and " : "", set_checkpoint ? "VK_NV_device_diagnostic_checkpoints" : "",
               set_checkpoint ? "" : ", =1 adds NVIDIA checkpoints where the driver has them");
    else
        rt_log("vk: pass marks off (%s)", off ? "REKARINA_PASS_CRUMBS=0"
                                              : "no VK_AMD_buffer_marker; REKARINA_PASS_CRUMBS=1 asks for NVIDIA checkpoints");
}

static void describe_mark(uint32_t id, char *o, size_t size);

// ---- GPU time per pass (REKARINA_GPU_PASSTIME=<frame>): every pass and dispatch mark of frames <frame> and <frame> + 1
// also writes a timestamp at its start (top of pipe) and end (bottom of pipe); at frame <frame> + 8 the times are read
// and the 40 longest logged with what each mark was. A diagnostic: it adds two timestamp writes per pass.
#define PT_N 4096                            // marks profiled at most (two queries each)
static struct {
    int state;                               // 0 not decided, 1 waiting, 2 recording, 3 recorded, -1 off or done
    uint64_t frame;
    VkQueryPool pool;
    uint32_t id0, id1;                       // marks [id0, id1) were profiled
} PT;

static uint64_t pt_frame(void) { return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED); }

static void pt_report(void) {
    VkCtx *c = vk_ctx();
    uint32_t n = PT.id1 - PT.id0;
    if (n > PT_N) n = PT_N;
    uint64_t *v = calloc((size_t)n * 4, sizeof *v);
    struct { double ms; uint32_t id; } *t = calloc(n ? n : 1, sizeof *t);
    if (!v || !t) return;
    double sum = 0, per = c->props.limits.timestampPeriod / 1e6;
    uint32_t k = 0;
    // Also the shape of the frame: marks by duration and the GPU time between one mark's end and the next one's
    // start (barriers, copies, clears and query copies outside any mark), in record order.
    uint64_t prev_end = 0, first = 0, last = 0;
    double gaps = 0, bucket_ms[4] = {0};
    uint32_t bucket_n[4] = {0};
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = PT.id0 + i, q = (id % PT_N) * 2;
        uint64_t r[4] = {0};
        if (vkGetQueryPoolResults(c->device, PT.pool, q, 2, sizeof r, r, 2 * sizeof r[0],
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) < 0 || !r[1] || !r[3])
            continue;
        double ms = (double)(r[2] - r[0]) * per;
        t[k].ms = ms, t[k].id = id, k++;
        sum += ms;
        if (prev_end && r[0] > prev_end) gaps += (double)(r[0] - prev_end) * per;
        if (!first || r[0] < first) first = r[0];
        if (r[2] > last) last = r[2];
        prev_end = r[2];
        int b = ms < 0.02 ? 0 : ms < 0.1 ? 1 : ms < 0.5 ? 2 : 3;
        bucket_n[b]++, bucket_ms[b] += ms;
    }
    rt_log("GPU_PASSTIME shape: span %.2f ms, gaps between marks %.2f ms; marks under 20 us %u (%.2f ms), 20-100 us %u "
           "(%.2f ms), 0.1-0.5 ms %u (%.2f ms), over 0.5 ms %u (%.2f ms)", first && last > first ? (last - first) * per : 0.0,
           gaps, bucket_n[0], bucket_ms[0], bucket_n[1], bucket_ms[1], bucket_n[2], bucket_ms[2], bucket_n[3], bucket_ms[3]);
    if (cfg_env("REKARINA_GPU_PASSTIME_ALL"))   // DEBUG: every mark of the two frames in record order
        for (uint32_t i = 0; i < k; i++) {
            char d[256];
            describe_mark(t[i].id, d, sizeof d);
            rt_log("GPU_PT_SEQ %4u %6.3f ms  %s", i, t[i].ms, d);
        }
    for (uint32_t i = 1; i < k; i++)   // longest first
        for (uint32_t j = i; j > 0 && t[j].ms > t[j - 1].ms; j--) {
            typeof(*t) x = t[j]; t[j] = t[j - 1]; t[j - 1] = x;
        }
    rt_log("GPU_PASSTIME frames %llu-%llu: %u marks timed, %.2f ms in passes and dispatches (two frames)",
           (unsigned long long)PT.frame, (unsigned long long)PT.frame + 1, k, sum);
    for (uint32_t i = 0; i < k && i < 40; i++) {
        char d[256];
        describe_mark(t[i].id, d, sizeof d);
        rt_log("GPU_PASSTIME %6.3f ms  %s", t[i].ms, d);
    }
    free(v), free(t);
}

// The mark's timestamp, when its frame is profiled. Recorder thread (vk_pass_mark_cmd).
static void pt_cmd(VkCommandBuffer cb, uint32_t id, int end) {
    if (PT.state != 2 && PT.state != 3) return;
    if (id < PT.id0 || (PT.state == 3 && id >= PT.id1) || id - PT.id0 >= PT_N) return;
    // The real command: this runs where the recorder (vk_rec.c) plays its packets into cb, and vk_rec.h routes
    // vkCmdWriteTimestamp to a new packet (written into the stream being played, it broke it: unknown packet).
#undef vkCmdWriteTimestamp
    static int drained = -1;   // DEBUG-PTDRAIN: REKARINA_PASSTIME_DRAINED=1: a mark starts once the work before it is done
    if (drained < 0) drained = cfg_get("REKARINA_PASSTIME_DRAINED") != NULL;
    {   // DEBUG-PTBAR: REKARINA_PASSTIME_BARRIER=1: a full barrier before each stamp of a dispatch or copy mark, so a mark's time is its own work
        static int bar = -1;
        if (bar < 0) bar = cfg_env("REKARINA_PASSTIME_BARRIER") != NULL;
        const VkPassMark *pm = bar ? vk_pass_mark_record(id) : NULL;
        if (pm && pm->kind == VK_MARK_DISPATCH) {
#undef vkCmdPipelineBarrier2
            VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
            VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
            vkCmdPipelineBarrier2(cb, &di);
        }
    }
    vkCmdWriteTimestamp(cb, end || drained ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, PT.pool,
                        (id % PT_N) * 2 + (end != 0));
}

// The profile's state at a new mark id. Replay thread (vk_pass_mark_begin).
static void pt_mark(uint32_t id) {
    if (PT.state < 0) return;
    if (!PT.state) {
        const char *e = cfg_get("REKARINA_GPU_PASSTIME");
        PT.state = -1;
        if (!e || !*e) return;
        PT.frame = strtoull(e, NULL, 0);
        VkCtx *c = vk_ctx();
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                    .queryCount = PT_N * 2};
        if (vkCreateQueryPool(c->device, &qi, NULL, &PT.pool) != VK_SUCCESS) return;
        vkResetQueryPool(c->device, PT.pool, 0, PT_N * 2);
        PT.state = 1;
    }
    uint64_t f = pt_frame();
    if (PT.state == 1 && f >= PT.frame) PT.state = 2, PT.id0 = id;
    else if (PT.state == 2 && f >= PT.frame + 2) PT.state = 3, PT.id1 = id;
    else if (PT.state == 3 && f >= PT.frame + 8) PT.state = -1, pt_report();
}

void vk_pass_mark_cmd(VkCommandBuffer cb, uint32_t id, int end) {
    if (pm_amd)
        write_marker(cb, end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buf,
                     end ? 12 : 8, id);
    if (set_checkpoint) set_checkpoint(cb, (const void *)(uintptr_t)((uint64_t)id << 1 | (end != 0)));
    pt_cmd(cb, id, end);
}

VkPassMark *vk_pass_mark_begin(VkCommandBuffer cb, int kind, uint64_t fence) {
    if (!vk_pass_marks_on || !cb) return NULL;
    uint32_t id = ++seq;
    if (!id) id = ++seq;   // 0 means no mark
    pt_mark(id);
    VkPassMark *m = &ring[id & (VK_PASS_MARKS - 1)];
    m->id = id, m->kind = (uint8_t)kind, m->fence = fence, m->draws = UINT32_MAX;
    rec_pass_mark(cb, id, 0);
    return m;
}

void vk_pass_mark_end(VkCommandBuffer cb, const VkPassMark *m) {
    if (m && cb) rec_pass_mark(cb, m->id, 1);
}

int vk_pass_marks_read(uint32_t *started, uint32_t *finished) {
    if (!pm_amd) return 0;
    *started = words[2], *finished = words[3];
    return 1;
}

const VkPassMark *vk_pass_mark_record(uint32_t id) {
    const VkPassMark *m = &ring[id & (VK_PASS_MARKS - 1)];
    return id && m->id == id ? m : NULL;
}

// The first 8 bytes of the shader's input id (gfx_shader.c), the start of its stored input's name <cache>/in/<id>.in.
static unsigned long long shader_id(const GfxShader *s) {
    if (!s) return 0;
    const uint8_t *iid = gfx_shader_input_id(s);
    unsigned long long v = 0;
    for (int i = 0; i < 8; i++) v = v << 8 | iid[i];
    {   // DEBUG-SHDUMP: REKARINA_SHDUMP_ID=<hex>,<hex>...: this shader's SPIR-V goes to REKARINA_SHDUMP_DIR/<id>.spv once
        const char *e = cfg_env("REKARINA_SHDUMP_ID");
        char hex[17]; snprintf(hex, sizeof hex, "%016llx", v);
        static unsigned long long done[64]; static int nd; int seen = 0;
        for (int i = 0; i < nd; i++) seen |= done[i] == v;
        if (e && strstr(e, hex) && !seen && nd < 64) {
            uint64_t words = 0; const uint32_t *w = gfx_shader_spirv(s, &words); char path[600];
            snprintf(path, sizeof path, "%s/%s.spv", cfg_env("REKARINA_SHDUMP_DIR") ? cfg_env("REKARINA_SHDUMP_DIR") : ".", hex);
            FILE *f = fopen(path, "wb"); if (f && w) fwrite(w, 4, (size_t)words, f); if (f) fclose(f);
            done[nd++] = v;
        }
    }
    return v;
}

static void describe_mark(uint32_t id, char *o, size_t size) {
    const VkPassMark *m = vk_pass_mark_record(id);
    if (!id) {
        snprintf(o, size, "nothing yet");
        return;
    }
    if (!m) {
        snprintf(o, size, "no record (a later mark took its slot)");
        return;
    }
    if (m->kind >= VK_MARK_PROLOGUE) {
        static const char *const k[] = {"mirror prologue", "write-back and merges", "merge dispatches", "write-back copies", "UI HUD-less capture and mask clear"};
        snprintf(o, size, "%s of submission %llu", k[m->kind - VK_MARK_PROLOGUE], (unsigned long long)m->fence);
        return;
    }
    if (m->kind == VK_MARK_DISPATCH) {
        snprintf(o, size, "dispatch %u of submission %llu: %ux%ux%u groups%s, cs %016llx", m->step,
                 (unsigned long long)m->fence, m->w, m->h, m->d, m->w ? "" : " (indirect)", shader_id(m->first[0]));
        return;
    }
    size_t n = (size_t)snprintf(o, size, "pass %u of submission %llu: %ux%u, %u layer%s, colour formats", m->step,
                                (unsigned long long)m->fence, m->w, m->h, m->d, m->d == 1 ? "" : "s");
    for (int i = 0; i < m->ncolor && i < 8 && n < size; i++)
        n += (size_t)snprintf(o + n, size - n, "%s%d", i ? "," : " ", (int)m->color[i]);
    if (!m->ncolor && n < size) n += (size_t)snprintf(o + n, size - n, " none");
    if (n < size) n += (size_t)snprintf(o + n, size - n, ", depth format %d", (int)m->depth);
    if (n >= size) return;
    if (m->draws == UINT32_MAX)
        snprintf(o + n, size - n, ", still recording; first draw vs %016llx fs %016llx", shader_id(m->first[0]),
                 shader_id(m->first[1]));
    else if (!m->draws)
        snprintf(o + n, size - n, ", no draws");
    else
        snprintf(o + n, size - n, ", %u draws; first vs %016llx fs %016llx, last vs %016llx fs %016llx", m->draws,
                 shader_id(m->first[0]), shader_id(m->first[1]), shader_id(m->last[0]), shader_id(m->last[1]));
}

void vk_pass_marks_format(const char *source, uint32_t started, uint32_t finished, char *o, size_t size) {
    char a[640], b[640];
    describe_mark(started, a, sizeof a);
    describe_mark(finished, b, sizeof b);
    snprintf(o, size,
             "vk: pass marks (%s, %u made): the GPU last started mark %u and last finished mark %u\n"
             "vk:   last started %u: %s\n"
             "vk:   last finished %u: %s",
             source, (unsigned)seq, (unsigned)started, (unsigned)finished, (unsigned)started, a, (unsigned)finished, b);
}

static void log_lines(char *t) {
    for (char *line = t, *nl; line && *line; line = nl ? nl + 1 : NULL) {
        nl = strchr(line, '\n');
        if (nl) *nl = 0;
        rt_log("%s", line);
    }
}

// vkGetQueueCheckpointData2NV, else vkGetQueueCheckpointDataNV, on the graphics queue (the replay's): one entry per
// stage with the last checkpoint that stage executed. A checkpoint is the mark's id times 2, plus 1 at its end, so the
// top-of-pipe entry names the last start or end the GPU reached, and the bottom-of-pipe entry the last one before which
// everything had finished: at a start, that is the end of the mark before it.
static void nv_report(void) {
    VkCtx *c = vk_ctx();
    if (!c || !set_checkpoint) return;
    PFN_vkGetQueueCheckpointData2NV get2 =
        (PFN_vkGetQueueCheckpointData2NV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointData2NV");
    PFN_vkGetQueueCheckpointDataNV get1 =
        (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(c->device, "vkGetQueueCheckpointDataNV");
    enum { MAXN = 32 };
    uint64_t stage[MAXN], marker[MAXN];
    uint32_t n = 0;
    if (get2) {
        VkCheckpointData2NV d[MAXN];
        get2(c->gfx_queue, &n, NULL);
        if (n > MAXN) n = MAXN;
        for (uint32_t i = 0; i < n; i++) d[i] = (VkCheckpointData2NV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV};
        get2(c->gfx_queue, &n, d);
        for (uint32_t i = 0; i < n; i++) stage[i] = d[i].stage, marker[i] = (uint64_t)(uintptr_t)d[i].pCheckpointMarker;
    } else if (get1) {
        VkCheckpointDataNV d[MAXN];
        get1(c->gfx_queue, &n, NULL);
        if (n > MAXN) n = MAXN;
        for (uint32_t i = 0; i < n; i++) d[i] = (VkCheckpointDataNV){.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV};
        get1(c->gfx_queue, &n, d);
        for (uint32_t i = 0; i < n; i++) stage[i] = d[i].stage, marker[i] = (uint64_t)(uintptr_t)d[i].pCheckpointMarker;
    } else {
        rt_log("vk: pass marks: no vkGetQueueCheckpointDataNV entry point");
        return;
    }
    uint32_t started = 0, finished = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = (uint32_t)(marker[i] >> 1);
        int end = (int)(marker[i] & 1);
        rt_log("vk: checkpoint of stage %#llx: the %s of mark %u", (unsigned long long)stage[i], end ? "end" : "start", id);
        if (stage[i] == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) started = id;
        if (stage[i] == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) finished = end ? id : id ? id - 1 : 0;
    }
    if (!n) rt_log("vk: pass marks: the driver reported no checkpoints");
    char t[2048];
    vk_pass_marks_format("VK_NV_device_diagnostic_checkpoints", started, finished, t, sizeof t);
    log_lines(t);
}

void vk_pass_marks_report(void) {
    if (!vk_pass_marks_on) return;
    uint32_t started, finished;
    char t[2048];
    if (vk_pass_marks_read(&started, &finished)) {
        vk_pass_marks_format("VK_AMD_buffer_marker", started, finished, t, sizeof t);
        log_lines(t);
    }
    nv_report();
}
