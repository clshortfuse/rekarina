// The problem report (report.h): one zip with what a report of a crash, a lost GPU device or a wrong picture needs,
// made where the player can find it. The archive is a plain ZIP with stored members (no compression: logs are text and
// stay small, and a reader opens it without anything of ours). What goes in is text only; the user's home folder and
// name are replaced in every member, so paths read C:\Users\<user>\... and /home/<user>/....
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "report.h"
#include "sysinfo.h"
#include "ui_text.h"
#include "vk.h"

#define LOG_TAIL (16u << 20)   // the last bytes of a log that go in
#define MAX_MEMBERS 32

// ---- the user's name and home folder, replaced in every member
typedef struct { char from[512]; const char *to; } Scrub;
static Scrub scrubs[8];
static int n_scrubs;

static void scrub_add(const char *from, const char *to) {
    if (!from || strlen(from) < 3 || n_scrubs >= (int)(sizeof scrubs / sizeof *scrubs)) return;
    snprintf(scrubs[n_scrubs].from, sizeof scrubs[n_scrubs].from, "%s", from);
    scrubs[n_scrubs++].to = to;
}

static void scrub_init(void) {
    if (n_scrubs) return;
    char v[512], w[512];
    // the home folder in both slash forms first, then the bare name where a path names it
    if (plat_getenv("USERPROFILE", v, sizeof v) || plat_getenv("HOME", v, sizeof v)) {
        scrub_add(v, "<home>");
        snprintf(w, sizeof w, "%s", v);
        for (char *p = w; *p; p++) *p = *p == '\\' ? '/' : *p;
        scrub_add(w, "<home>");
    }
    if (plat_getenv("USERNAME", v, sizeof v) || plat_getenv("USER", v, sizeof v)) {
        static const char *const frames[] = {"\\Users\\%s", "/Users/%s", "/home/%s", "\\\\Users\\\\%s"};
        static const char *const repl[] = {"\\Users\\<user>", "/Users/<user>", "/home/<user>", "\\\\Users\\\\<user>"};
        for (int i = 0; i < 4; i++) {
            snprintf(w, sizeof w, frames[i], v);
            scrub_add(w, repl[i]);
        }
    }
}

// A copy of data with every scrub applied (case-insensitive on Windows paths); *n is updated. NULL on no memory.
static uint8_t *scrubbed(const uint8_t *data, size_t *n) {
    scrub_init();
    size_t cap = *n + 64, len = 0;
    uint8_t *out = malloc(cap);
    if (!out) return NULL;
    for (size_t i = 0; i < *n;) {
        int hit = -1;
        for (int k = 0; k < n_scrubs && hit < 0; k++) {
            size_t fl = strlen(scrubs[k].from);
            if (i + fl <= *n) {
                size_t j = 0;
                while (j < fl) {
                    char a = (char)data[i + j], b = scrubs[k].from[j];
                    if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
                    if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
                    if (a != b) break;
                    j++;
                }
                if (j == fl) hit = k;
            }
        }
        const uint8_t *src = hit >= 0 ? (const uint8_t *)scrubs[hit].to : data + i;
        size_t sl = hit >= 0 ? strlen(scrubs[hit].to) : 1;
        if (len + sl > cap) {
            cap = cap * 2 + sl;
            uint8_t *g = realloc(out, cap);
            if (!g) { free(out); return NULL; }
            out = g;
        }
        memcpy(out + len, src, sl);
        len += sl;
        i += hit >= 0 ? strlen(scrubs[hit].from) : 1;
    }
    *n = len;
    return out;
}

// ---- a ZIP of stored members
static uint32_t crc32_of(const uint8_t *p, size_t n) {
    static uint32_t table[256];
    if (!table[1])
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    uint32_t crc = ~0u;
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

typedef struct { char name[96]; uint32_t crc, size, offset; } Member;
typedef struct { FILE *f; Member m[MAX_MEMBERS]; int n; uint32_t at; uint16_t dtime, ddate; int ok; } Zip;

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { put16(p, (uint16_t)v), put16(p + 2, (uint16_t)(v >> 16)); }

static void zip_write(Zip *z, const void *p, size_t n) {
    if (z->ok && fwrite(p, 1, n, z->f) != n) z->ok = 0;
    z->at += (uint32_t)n;
}

// A member from memory: scrubbed, then stored.
static void zip_add(Zip *z, const char *name, const uint8_t *data, size_t n) {
    if (z->n >= MAX_MEMBERS || !z->ok) return;
    uint8_t *clean = scrubbed(data, &n);
    if (!clean) { z->ok = 0; return; }
    Member *m = &z->m[z->n++];
    snprintf(m->name, sizeof m->name, "%s", name);
    m->crc = crc32_of(clean, n), m->size = (uint32_t)n, m->offset = z->at;
    uint8_t h[30] = {0};
    put32(h, 0x04034b50), put16(h + 4, 20), put16(h + 6, 0x0800), put16(h + 8, 0);   // UTF-8 names, stored
    put16(h + 10, z->dtime), put16(h + 12, z->ddate), put32(h + 14, m->crc), put32(h + 18, m->size), put32(h + 22, m->size);
    put16(h + 26, (uint16_t)strlen(m->name)), put16(h + 28, 0);
    zip_write(z, h, sizeof h);
    zip_write(z, m->name, strlen(m->name));
    zip_write(z, clean, n);
    free(clean);
}

static void zip_add_text(Zip *z, const char *name, const char *text) { zip_add(z, name, (const uint8_t *)text, strlen(text)); }

// A file on disk; a log only its last LOG_TAIL bytes. Missing files are left out.
static void zip_add_file(Zip *z, const char *name, const char *path, int tail) {
    uint64_t size = 0;
    uint8_t *d = plat_file_read(path, &size);
    if (!d) return;
    size_t from = tail && size > LOG_TAIL ? (size_t)(size - LOG_TAIL) : 0;
    zip_add(z, name, d + from, (size_t)size - from);
    free(d);
}

static void zip_finish(Zip *z) {
    uint32_t start = z->at;
    for (int i = 0; i < z->n; i++) {
        const Member *m = &z->m[i];
        uint8_t c[46] = {0};
        put32(c, 0x02014b50), put16(c + 4, 20), put16(c + 6, 20), put16(c + 8, 0x0800), put16(c + 10, 0);
        put16(c + 12, z->dtime), put16(c + 14, z->ddate), put32(c + 16, m->crc), put32(c + 20, m->size), put32(c + 24, m->size);
        put16(c + 28, (uint16_t)strlen(m->name)), put32(c + 42, m->offset);
        zip_write(z, c, sizeof c);
        zip_write(z, m->name, strlen(m->name));
    }
    uint8_t e[22] = {0};
    put32(e, 0x06054b50), put16(e + 8, (uint16_t)z->n), put16(e + 10, (uint16_t)z->n), put32(e + 12, z->at - start);
    put32(e + 16, start);
    zip_write(z, e, sizeof e);
}

// ---- what goes in
static int game_folder(char *out, size_t n) {
    if (!plat_exe_path(out, n)) return 0;
    char *cut = plat_last_sep(out), *f = strrchr(out, '/');
    if (f > cut) cut = f;
    if (!cut) return 0;
    *cut = 0;
    return 1;
}

int report_folder(char *out, size_t n) {
    char dir[PLAT_PATH_MAX * 3];
    if (!game_folder(dir, sizeof dir)) return 0;
    int k = snprintf(out, n, "%s/reports", dir);
    if (k < 0 || (size_t)k >= n) return 0;
    plat_file_mkdir_p(out);
    return plat_dir_exists(out);
}

typedef struct { char *buf; size_t cap, at; } Text;

static void add(Text *t, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void add(Text *t, const char *fmt, ...) {
    if (t->at >= t->cap) return;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(t->buf + t->at, t->cap - t->at, fmt, ap);
    va_end(ap);
    if (k > 0) t->at += (size_t)k < t->cap - t->at ? (size_t)k : t->cap - t->at - 1;
}

typedef struct { Text *t; uint64_t files, bytes; } Tally;

static int tally(const char *name, int kind, uint64_t size, void *ctx) {
    (void)name;
    Tally *y = ctx;
    if (kind == PLAT_FS_FILE) y->files++, y->bytes += size;
    return 1;
}

// The size of a file or a folder's files (one level), for the shader cache's state.
static void add_cache(Text *t, const char *dir, const char *name) {
    char p[PLAT_PATH_MAX * 3 + 64];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    if (plat_dir_exists(p)) {
        Tally y = {t, 0, 0};
        plat_fs_list(p, tally, &y);
        add(t, "  %-26s folder, %llu files, %.1f MB\n", name, (unsigned long long)y.files, y.bytes / 1048576.0);
    } else {
        int64_t s = plat_file_size(p);
        if (s >= 0) add(t, "  %-26s %.1f MB\n", name, s / 1048576.0);
        else add(t, "  %-26s missing\n", name);
    }
}

static int add_mod(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size;
    if (kind == PLAT_FS_DIR) add(ctx, "  %s\n", name);
    return 1;
}

// The graphics cards Vulkan lists, for a report made without the game running (`--make-report`).
static void add_vulkan_devices(Text *t) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = RT_PORT_NAME,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        add(t, "Vulkan: no instance could be made (no Vulkan driver?)\n");
        return;
    }
    VkPhysicalDevice pd[8];
    uint32_t n = 8;
    if (vkEnumeratePhysicalDevices(inst, &n, pd) < 0) n = 0;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceDriverProperties dp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dp};
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd[i], &mp);
        uint64_t vram = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += mp.memoryHeaps[h].size;
        const VkPhysicalDeviceProperties *p = &p2.properties;
        add(t, "GPU %u: %s (vendor 0x%04x, device 0x%04x), %.1f GB device-local, driver %s %s (0x%08x), Vulkan %u.%u.%u\n", i,
            p->deviceName, p->vendorID, p->deviceID, vram / 1073741824.0, dp.driverName, dp.driverInfo, p->driverVersion,
            VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion));
    }
    if (!n) add(t, "Vulkan: no graphics card listed\n");
    vkDestroyInstance(inst, NULL);
}

int report_make(char *path, size_t n) {
    char dir[PLAT_PATH_MAX * 3], out_dir[PLAT_PATH_MAX * 3 + 16];
    if (!game_folder(dir, sizeof dir) || !report_folder(out_dir, sizeof out_dir)) return 0;
    struct tm tm;
    plat_local_time(&tm);
    int k = snprintf(path, n, "%s/" RT_FILE_STEM "-report-%04d%02d%02d-%02d%02d%02d.zip", out_dir, tm.tm_year + 1900,
                     tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    if (k < 0 || (size_t)k >= n) return 0;
    Zip z = {.f = fopen(path, "wb"), .ok = 1};
    if (!z.f) return 0;
    z.dtime = (uint16_t)(tm.tm_hour << 11 | tm.tm_min << 5 | tm.tm_sec / 2);
    z.ddate = (uint16_t)((tm.tm_year - 80) << 9 | (tm.tm_mon + 1) << 5 | tm.tm_mday);

    enum { CAP = 64 * 1024 };
    char *buf = malloc(CAP), *info = malloc(16384);
    if (!buf || !info) { fclose(z.f); free(buf); free(info); return 0; }
    Text t = {buf, CAP, 0};
    add(&t, RT_PORT_NAME " problem report, made %04d-%02d-%02d %02d:%02d:%02d\n", tm.tm_year + 1900, tm.tm_mon + 1,
        tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    add(&t, "Build: " RT_PORT_NAME " %s, game %s\n", rt_port_version, rt_game_version_text());
    add(&t, "Device profile: %s%s\n", cfg_device_profile(),
        cfg_is_set("REKARINA_DEVICE_PROFILE") ? " (set by hand)" : " (chosen for this machine)");
#ifdef _WIN32
    if (plat_is_wine()) add(&t, "Runs under Wine / Proton\n");
#endif
    if (sysinfo_text(info, 16384) > 0) add(&t, "%s", info);   // OS, CPU, RAM, GPU of this run, settings in effect
    if (!vk_ctx()) add_vulkan_devices(&t);
    add(&t, "Shader and texture caches beside the game:\n");
    static const char *const caches[] = {"shader-cache", "shader-prepared.bin", "pipeline-cache.bin", "pipeline-keys.bin",
                                         "pipeline-keys-mesh.bin", "compute-used.bin", "nv-cache", "mesa-cache",
                                         RT_FILE_STEM "_xlate.id"};
    for (size_t i = 0; i < sizeof caches / sizeof *caches; i++) add_cache(&t, dir, caches[i]);
    {
        char mods[PLAT_PATH_MAX * 3 + 16];
        snprintf(mods, sizeof mods, "%s/mods", dir);
        add(&t, "Mods folder:\n");
        size_t before = t.at;
        plat_fs_list(mods, add_mod, &t);
        if (t.at == before) add(&t, "  none\n");
    }
    zip_add(&z, "system.txt", (const uint8_t *)buf, t.at);

    char p[PLAT_PATH_MAX * 3 + 64], name[64];
    for (int i = 0; i < 5; i++) {   // this run's log (as far as written) and the four before it
        if (i) snprintf(name, sizeof name, "logs/" RT_FILE_STEM ".%d.log", i);
        else snprintf(name, sizeof name, "logs/" RT_FILE_STEM ".log");
        snprintf(p, sizeof p, "%s/%s", dir, name);
        zip_add_file(&z, name, p, 1);
    }
    // the logs of the preparation's other modes (main.c log_open): the launcher's libraries build, the texture walk
    static const char *const child_logs[] = {"logs/" RT_FILE_STEM "-warm-libraries.log",
                                             "logs/" RT_FILE_STEM "-scan-textures.log"};
    for (int i = 0; i < 2; i++) {
        snprintf(p, sizeof p, "%s/%s", dir, child_logs[i]);
        zip_add_file(&z, child_logs[i], p, 1);
    }
    snprintf(p, sizeof p, "%s/" RT_INI_FILE, dir);
    zip_add_file(&z, RT_INI_FILE, p, 0);
    snprintf(p, sizeof p, "%s/" RT_FILE_STEM "-install.json", dir);
    zip_add_file(&z, RT_FILE_STEM "-install.json", p, 0);
    snprintf(p, sizeof p, "%s/memory-test.txt", dir);   // Linux: `--memory-test`'s last result (memtest.c), when run
    zip_add_file(&z, "memory-test.txt", p, 0);
    zip_add_text(&z, "README.txt",
                 RT_PORT_NAME " problem report. Made on this computer and sent nowhere: attach it to your report.\n"
                 "system.txt   the computer (OS, CPU, memory, graphics card and driver), the build, the settings in "
                 "effect and the state of the shader caches\n"
                 "logs/        the logs of the last five runs (the newest is " RT_FILE_STEM ".log; a crash or a lost "
                 "graphics device is written at its end)\n"
                 RT_INI_FILE "  your settings file\n"
                 "No save data and no game files are in it. Your user name and home folder in any path are replaced by "
                 "<user> and <home>.\n");
    zip_finish(&z);
    int ok = z.ok;
    if (fclose(z.f) != 0) ok = 0;
    free(buf);
    free(info);
    if (!ok) plat_file_delete(path);
    rt_log("report: %s %s (%d members)", ok ? "made" : "could not be written:", path, z.n);
    return ok;
}

static int lost_device = -1;   // report_last_run_failed: the previous run's log says its graphics device was lost

int report_last_run_lost_device(void) {
    if (lost_device < 0) report_last_run_failed();
    return lost_device > 0;
}

int report_last_run_failed(void) {
    static int known = -1;
    if (known >= 0) return known;
    known = 0;
    lost_device = 0;
    char dir[PLAT_PATH_MAX * 3], p[PLAT_PATH_MAX * 3 + 64];
    if (!game_folder(dir, sizeof dir)) return 0;
    snprintf(p, sizeof p, "%s/logs/" RT_FILE_STEM ".1.log", dir);
    PlatFile *f = plat_file_open(p);
    if (!f) return 0;
    int64_t size = plat_file_size(p);
    enum { TAIL = 256 * 1024 };
    char *b = malloc(TAIL + 1);
    if (b && size > 0) {
        uint64_t from = size > TAIL ? (uint64_t)size - TAIL : 0;
        int64_t got = plat_file_pread(f, b, (uint32_t)(size - (int64_t)from), from);
        if (got > 0) {
            b[got] = 0;
            for (int64_t i = 0; i < got; i++) if (!b[i]) b[i] = ' ';
            known = strstr(b, "FATAL") != NULL || strstr(b, "vk: device lost") != NULL;
            lost_device = strstr(b, "vk: device lost") != NULL || strstr(b, "the device is lost") != NULL ||
                          strstr(b, "failed (-4") != NULL;
        }
    }
    free(b);
    plat_file_close(f);
    if (known) rt_log("report: the previous run ended with an error (%s): the menu offers a problem report", p);
    return known;
}

static volatile int32_t box_shown;   // the one message box of a run's end (report_crash_notice, report_fatal_box)

static int box_wanted(void) {
    const char *bg = cfg_get("REKARINA_BACKGROUND_WINDOW"), *nobox = cfg_get("REKARINA_NO_CRASH_BOX");
    return rt_log_path() && !(bg && atoi(bg)) && !(nobox && atoi(nobox));   // test runs: no box to wait on
}

void report_crash_notice(void) {
    if (!box_wanted() || plat_xchg32(&box_shown, 1)) return;
    char text[1024];
    snprintf(text, sizeof text, menu_text(UI_REPORT_CRASHED), rt_log_path());
    rt_log("report: the player is told the game stopped (a message box)");
    plat_message_box(RT_PORT_NAME, text);
}

void report_fatal_box(const char *text) {
    if (!box_wanted() || plat_xchg32(&box_shown, 1)) return;
    rt_log("report: the player is told what stopped the game and what to do (a message box)");
    plat_message_box(RT_PORT_NAME, text);
}

// ---- frame generation that stopped the game
// AMD's frame generation (vk_fg.cpp) and NVIDIA's (vk_dlssg.c, through Streamline) run in libraries this process cannot
// catch a crash in, and they are set up at the start, before the settings menu could switch them off: 1.2.2's AMD
// swapchain crashed at every start with a fullscreen window, and the player had to edit rekarina.ini to play again.
// So a crash from the start of their set-up (fg_guard_arm) until they have generated FG_GUARD_FRAMES frames
// (fg_guard_running), within ten minutes of it, leaves logs/framegen-stopped.txt with this build's version and the
// setting (fg_guard_crashed, from main.c's crash handler). The next run blocks that selection on the same build.
// OoT keeps the other backend available for live selection. Choosing frame generation in the menu removes the file
// (fg_guard_clear) for the next start; the saved setting is not rewritten.
// A crash can interrupt any thread. Read the backend, setting, progress and start time as one snapshot.
static uint64_t fg_guard_state;
enum { FG_GUARD_FRAMES = 300, FG_COUNT_MASK = 511, FG_BACKEND_SHIFT = 9, FG_SETTING_SHIFT = 11, FG_TIME_SHIFT = 13 };

static FgGuardBackend fg_guard_backend(uint64_t state) {
    return (FgGuardBackend)((state >> FG_BACKEND_SHIFT) & 3);
}

static const char *fg_guard_name(FgGuardBackend backend) {
    return backend == FG_GUARD_DLSS ? "dlss" : "fsr3";
}

static int fg_guard_path(char *out, size_t n) {
    char dir[PLAT_PATH_MAX * 3];
    if (!game_folder(dir, sizeof dir)) return 0;
    snprintf(out, n, "%s/logs/framegen-stopped.txt", dir);
    return 1;
}

static FgGuardBackend fg_guard_blocked_setting;
#ifdef GAME_OOT
static int fg_guard_marker_cleared;
#endif

static void fg_guard_read_marker(void *unused) {
    (void)unused;
    char p[PLAT_PATH_MAX * 3 + 64], want[96];
    if (!fg_guard_path(p, sizeof p)) return;
    uint64_t size = 0;
    char *b = (char *)plat_file_read(p, &size);
    if (!b) return;
    for (FgGuardBackend backend = FG_GUARD_DLSS; backend <= FG_GUARD_FSR; ++backend) {
        const char *setting = fg_guard_name(backend);
#ifndef GAME_OOT
        if (strcmp(setting, cfg_value("REKARINA_FRAMEGEN"))) continue;
#endif
        int n = snprintf(want, sizeof want, "%s %s", rt_port_version, setting);
        if (n > 0 && n < (int)sizeof want && size >= (uint64_t)n && !memcmp(b, want, (size_t)n) &&
            (size == (uint64_t)n || b[n] == '\n')) {
            fg_guard_blocked_setting = backend;
            break;
        }
    }
    free(b);
    if (fg_guard_blocked_setting)
        rt_log("fg: frame generation (%s) blocked for this run: the last run stopped during setup (%s; "
               "choosing frame generation again in the settings menu retries on the next start)",
               fg_guard_name(fg_guard_blocked_setting), p);
    else plat_file_delete(p);
}

int fg_guard_blocked(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, fg_guard_read_marker, NULL);
#ifdef GAME_OOT
    return fg_guard_blocked_setting && !strcmp(cfg_value("REKARINA_FRAMEGEN"), fg_guard_name(fg_guard_blocked_setting));
#else
    return fg_guard_blocked_setting != 0;
#endif
}

#ifdef GAME_OOT
int fg_guard_retry_pending(void) {
    return __atomic_load_n(&fg_guard_marker_cleared, __ATOMIC_ACQUIRE) && fg_guard_blocked();
}
#endif

void fg_guard_arm(FgGuardBackend backend) {
    if (fg_guard_backend(__atomic_load_n(&fg_guard_state, __ATOMIC_RELAXED)) == backend) return;
    const FgGuardBackend setting = !strcmp(cfg_value("REKARINA_FRAMEGEN"), "dlss") ? FG_GUARD_DLSS : backend;
    const uint64_t state = (plat_now_ns() / 1000000ull << FG_TIME_SHIFT) |
                          ((uint64_t)setting << FG_SETTING_SHIFT) | ((uint64_t)backend << FG_BACKEND_SHIFT);
    __atomic_store_n(&fg_guard_state, state, __ATOMIC_RELAXED);
}

void fg_guard_running(FgGuardBackend backend) {
    uint64_t state = __atomic_load_n(&fg_guard_state, __ATOMIC_RELAXED);
    if (fg_guard_backend(state) == backend && (state & FG_COUNT_MASK) < FG_GUARD_FRAMES)
        __atomic_compare_exchange_n(&fg_guard_state, &state, state + 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}

void fg_guard_disarm(FgGuardBackend backend) {
    uint64_t state = __atomic_load_n(&fg_guard_state, __ATOMIC_RELAXED);
    while (fg_guard_backend(state) == backend &&
           !__atomic_compare_exchange_n(&fg_guard_state, &state, 0, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

void fg_guard_clear(void) {
    char p[PLAT_PATH_MAX * 3 + 64];
    if (!fg_guard_path(p, sizeof p)) return;
#ifdef GAME_OOT
    // Keep the startup block for this process even if the menu clears the file before the first guard check.
    fg_guard_blocked();
#endif
    plat_file_delete(p);
#ifdef GAME_OOT
    if (plat_file_exists(p)) rt_log("fg: could not clear the startup crash marker (%s)", p);
    else __atomic_store_n(&fg_guard_marker_cleared, 1, __ATOMIC_RELEASE);
#endif
}

void fg_guard_crashed(void) {
    const uint64_t state = __atomic_load_n(&fg_guard_state, __ATOMIC_RELAXED);
    const unsigned frames = (unsigned)(state & FG_COUNT_MASK);
    if (!state || frames >= FG_GUARD_FRAMES || plat_now_ns() / 1000000ull - (state >> FG_TIME_SHIFT) > 600000)
        return;
    char p[PLAT_PATH_MAX * 3 + 64], line[96];
    if (!fg_guard_path(p, sizeof p)) return;
    const char *setting = fg_guard_name((FgGuardBackend)((state >> FG_SETTING_SHIFT) & 3));
    int n = snprintf(line, sizeof line, "%s %s\n", rt_port_version, setting);
    if (plat_file_write(p, line, (size_t)n))
        rt_log("fg: the game stopped while frame generation (%s) was being set up (%llu frames generated): the next start "
               "runs without it (%s)", fg_guard_name(fg_guard_backend(state)), (unsigned long long)frames, p);
}

int report_main(void) {
    char path[PLAT_PATH_MAX * 3 + 96];
    if (!report_make(path, sizeof path)) {
        fprintf(stderr, "report: could not be written\n");
        return 1;
    }
    printf("%s\n", path);
    char dir[PLAT_PATH_MAX * 3 + 16];
    if (!cfg_get("REKARINA_REPORT_NO_OPEN") && report_folder(dir, sizeof dir) && plat_open_folder(dir))
        plat_wait_ns(1500000000ull);   // the file manager is asked on a thread of its own: not cut off by the exit
    return 0;
}
