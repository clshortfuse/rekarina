#include <windows.h>
#include <sddl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "host.h"
#include "renderer_mcp.h"

#define REQUEST_MAX 4096
#define RESPONSE_MAX (REQUEST_MAX + 2048)

typedef struct { unsigned iw, ih, ow, oh; } Extents;
typedef struct {
    unsigned width, height, tonemap_width, tonemap_height;
    int hdr10, route, present_result, presented, evaluated, tonemapped, dlss_ok;
    RendererMcpDlss dlss_route;
    uint32_t dlss_result;
    Extents dlss, bloom;
    char bloom_route[40];
    SceneReplay scene;
    int hudless, coverage;
} Snapshot;

static Snapshot current;
static atomic_bool enabled;
static SRWLOCK snapshot_lock = SRWLOCK_INIT;

static int publish(void) {
    return atomic_load_explicit(&enabled, memory_order_relaxed) && TryAcquireSRWLockExclusive(&snapshot_lock);
}

void renderer_mcp_scene(const SceneReplay *scene, int hudless, int coverage) {
    if (!publish()) return;
    current.scene = *scene;
    current.hudless = hudless;
    current.coverage = coverage;
    ReleaseSRWLockExclusive(&snapshot_lock);
}

void renderer_mcp_present(unsigned width, unsigned height, int hdr10, int route, int result) {
    if (!publish()) return;
    current.width = width;
    current.height = height;
    current.hdr10 = hdr10;
    current.route = route;
    current.present_result = result;
    current.presented = 1;
    ReleaseSRWLockExclusive(&snapshot_lock);
}

void renderer_mcp_dlss(RendererMcpDlss route, unsigned iw, unsigned ih, unsigned ow, unsigned oh,
                       int ok, uint32_t result) {
    if (!publish()) return;
    current.dlss = (Extents){iw, ih, ow, oh};
    current.dlss_route = route;
    current.dlss_ok = ok;
    current.dlss_result = result;
    current.evaluated = 1;
    ReleaseSRWLockExclusive(&snapshot_lock);
}

void renderer_mcp_bloom(const char *route, unsigned iw, unsigned ih, unsigned ow, unsigned oh) {
    if (!publish()) return;
    current.bloom = (Extents){iw, ih, ow, oh};
    // Only internal route labels are accepted, never arbitrary JSON text.
    size_t i = 0;
    for (; route[i] && i < sizeof current.bloom_route - 1; ++i) {
        char c = route[i];
        current.bloom_route[i] = (c >= 'a' && c <= 'z') || c == '_' || c == '-' ? c : '_';
    }
    current.bloom_route[i] = 0;
    ReleaseSRWLockExclusive(&snapshot_lock);
}

void renderer_mcp_tonemap(unsigned width, unsigned height) {
    if (!publish()) return;
    current.tonemap_width = width;
    current.tonemap_height = height;
    current.tonemapped = 1;
    ReleaseSRWLockExclusive(&snapshot_lock);
}

typedef struct { const char *p, *end; } Json;

static const char *space(const char *p) { return p + strspn(p, " \r\n\t"); }
static int digit(char c) { return c >= '0' && c <= '9'; }
static int hex(char c) {
    if (digit(c)) return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static const char *string_end(const char *p) {
    if (*p++ != '"') return NULL;
    while ((unsigned char)*p >= 32) {
        if (*p == '"') return p + 1;
        if (*p++ != '\\') continue;
        if (*p == 'u') {
            ++p;
            for (int i = 0; i < 4; ++i) if (hex(*p++) < 0) return NULL;
        } else if (*p && strchr("\"\\/bfnrt", *p)) ++p;
        else return NULL;
    }
    return NULL;
}

static const char *value_end(const char *p, unsigned depth) {
    if (depth == 16) return NULL;
    if (*p == '"') return string_end(p);
    if (*p == '{' || *p == '[') {
        int object = *p == '{';
        char end = object ? '}' : ']';
        p = space(p + 1);
        if (*p == end) return p + 1;
        for (;;) {
            if (object) {
                p = string_end(p);
                if (!p || *(p = space(p)) != ':') return NULL;
                p = space(p + 1);
            }
            p = value_end(p, depth + 1);
            if (!p) return NULL;
            p = space(p);
            if (*p == end) return p + 1;
            if (*p != ',') return NULL;
            p = space(p + 1);
        }
    }
    if (!strncmp(p, "true", 4) || !strncmp(p, "null", 4)) return p + 4;
    if (!strncmp(p, "false", 5)) return p + 5;
    if (*p == '-') ++p;
    if (!digit(*p)) return NULL;
    if (*p == '0') ++p;
    else while (digit(*p)) ++p;
    if (*p == '.') {
        if (!digit(*++p)) return NULL;
        while (digit(*p)) ++p;
    }
    if (*p == 'e' || *p == 'E') {
        ++p;
        if (*p == '-' || *p == '+') ++p;
        if (!digit(*p)) return NULL;
        while (digit(*p)) ++p;
    }
    return p;
}

static int equals(Json j, const char *s) {
    if (!j.p || *j.p != '"') return 0;
    const char *p = j.p + 1;
    while (p < j.end - 1 && *s) {
        unsigned c = (unsigned char)*p++;
        if (c == '\\') {
            c = (unsigned char)*p++;
            if (c == 'u') {
                c = 0;
                for (int i = 0; i < 4; ++i) c = 16 * c + hex(*p++);
            } else if (c != '/' && c != '"' && c != '\\') return 0;
        }
        if (c != (unsigned char)*s++) return 0;
    }
    return p == j.end - 1 && !*s;
}

static Json field(Json obj, const char *name) {
    if (!obj.p || *obj.p != '{') return (Json){0};
    for (const char *p = space(obj.p + 1); *p != '}'; p = space(p + 1)) {
        const char *k = string_end(p);
        const char *v = space(space(k) + 1);
        const char *e = value_end(v, 0);
        if (equals((Json){p, k}, name)) return (Json){v, e};
        p = space(e);
        if (*p != ',') break;
    }
    return (Json){0};
}

static const char tool_list[] = "{\"tools\":[{\"name\":\"renderer_snapshot\","
    "\"description\":\"Read latest renderer observations (not necessarily from one frame).\","
    "\"inputSchema\":{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false},"
    "\"annotations\":{\"readOnlyHint\":true,\"idempotentHint\":true,\"openWorldHint\":false}}]}";

static void snapshot_result(char *result, size_t size) {
    AcquireSRWLockShared(&snapshot_lock);
    Snapshot s = current;
    ReleaseSRWLockShared(&snapshot_lock);
    char present[160] = "null", dlss[160] = "null", bloom[160] = "null", tonemap[80] = "null";
    if (s.presented) snprintf(present, sizeof present,
        "{\"extent\":[%u,%u],\"hdr10\":%s,\"route\":\"%s\",\"vk_result\":%d}",
        s.width, s.height, s.hdr10 ? "true" : "false", s.route == 1 ? "streamline" : s.route == 2 ? "fsr" : "vulkan",
        s.present_result);
    if (s.evaluated) snprintf(dlss, sizeof dlss,
        "{\"route\":\"%s\",\"input\":[%u,%u],\"output\":[%u,%u],\"ok\":%s,\"result\":%u}",
        s.dlss_route == RENDERER_MCP_NGX ? "ngx" : "streamline",
        s.dlss.iw, s.dlss.ih, s.dlss.ow, s.dlss.oh, s.dlss_ok ? "true" : "false", s.dlss_result);
    if (*s.bloom_route) snprintf(bloom, sizeof bloom,
        "{\"route\":\"%s\",\"input\":[%u,%u],\"output\":[%u,%u]}",
        s.bloom_route, s.bloom.iw, s.bloom.ih, s.bloom.ow, s.bloom.oh);
    if (s.tonemapped) snprintf(tonemap, sizeof tonemap, "{\"hdr_capture_extent\":[%u,%u]}",
                               s.tonemap_width, s.tonemap_height);
    char scene[512] = "null";
    if (s.scene.key.frame) snprintf(scene, sizeof scene,
        "{\"frame\":%llu,\"view\":\"%llx\",\"route\":%d,\"render\":[%u,%u],\"reconstruction\":[%u,%u],"
        "\"bloom\":[%u,%u],\"completed\":%d,\"light\":%u,\"tonemap\":%u,\"packed\":%u,\"output\":%u,"
        "\"window\":%u,\"hudless_eligible\":%d,\"coverage_eligible\":%d}",
        (unsigned long long)s.scene.key.frame, (unsigned long long)s.scene.key.view, (int)s.scene.route,
        s.scene.rw, s.scene.rh, s.scene.uw, s.scene.uh, s.scene.bw, s.scene.bh, s.scene.completed,
        s.scene.light, s.scene.tonemap, s.scene.packed, s.scene.output, s.scene.window, s.hudless, s.coverage);
    snprintf(result, size, "{\"content\":[{\"type\":\"text\",\"text\":\"Latest renderer observations\"}],"
             "\"structuredContent\":{\"dlss\":%s,\"bloom\":%s,\"tonemap\":%s,\"present\":%s,\"scene\":%s}}",
             dlss, bloom, tonemap, present, scene);
}

static int dispatch(const char *request, char *out, size_t size) {
    char buf[1536];
    const char *result = NULL, *error = NULL;
    int code = 0;
    const char *p = space(request), *e = value_end(p, 0);
    Json id = {0};
    if (!e || *space(e)) error = "Parse error", code = -32700;
    else {
        Json obj = {p, e}, method = field(obj, "method"), params = field(obj, "params");
        id = field(obj, "id");
        if (!equals(field(obj, "jsonrpc"), "2.0") || !method.p || *method.p != '"' ||
            (id.p && *id.p != '"' && *id.p != 'n' && *id.p != '-' && !digit(*id.p))) {
            id = (Json){0};
            error = "Invalid request", code = -32600;
        } else if (!id.p) return 0;
        else if (equals(method, "initialize")) {
            const char *version = equals(field(params, "protocolVersion"), "2025-06-18") ? "2025-06-18" : "2025-11-25";
            snprintf(buf, sizeof buf, "{\"protocolVersion\":\"%s\",\"capabilities\":{\"tools\":{}},"
                     "\"serverInfo\":{\"name\":\"rekarina-renderer\",\"version\":\"1\"}}", version);
            result = buf;
        } else if (equals(method, "ping")) result = "{}";
        else if (equals(method, "tools/list")) result = tool_list;
        else if (equals(method, "tools/call")) {
            if (!equals(field(params, "name"), "renderer_snapshot")) error = "Unknown tool", code = -32602;
            else {
                Json args = field(params, "arguments");
                if (args.p && (*args.p != '{' || *space(args.p + 1) != '}'))
                    error = "Expected no arguments", code = -32602;
                else snapshot_result(buf, sizeof buf), result = buf;
            }
        } else error = "Method not found", code = -32601;
    }
    if (!id.p) id = (Json){"null", NULL};
    int len = id.end ? (int)(id.end - id.p) : 4;
    if (error) return snprintf(out, size, "{\"jsonrpc\":\"2.0\",\"id\":%.*s,\"error\":{\"code\":%d,\"message\":\"%s\"}}\n",
                               len, id.p, code, error);
    return snprintf(out, size, "{\"jsonrpc\":\"2.0\",\"id\":%.*s,\"result\":%s}\n", len, id.p, result);
}

static int complete(HANDLE pipe, OVERLAPPED *ov, BOOL done, DWORD timeout, DWORD *count) {
    if (!done) {
        if (GetLastError() != ERROR_IO_PENDING) return 0;
        if (WaitForSingleObject(ov->hEvent, timeout) != WAIT_OBJECT_0) {
            CancelIoEx(pipe, ov);
            GetOverlappedResult(pipe, ov, count, TRUE);
            return 0;
        }
    }
    return GetOverlappedResult(pipe, ov, count, FALSE);
}

static void serve(HANDLE pipe, HANDLE event) {
    char request[REQUEST_MAX + 1], response[RESPONSE_MAX];
    DWORD used = 0;
    for (;;) {
        DWORD count;
        OVERLAPPED ov = {.hEvent = event};
        if (!complete(pipe, &ov, ReadFile(pipe, request + used, REQUEST_MAX - used, NULL, &ov), 30000, &count) ||
            !count || memchr(request + used, 0, count)) return;
        used += count;
        char *start = request, *nl;
        while ((nl = memchr(start, '\n', used - (DWORD)(start - request)))) {
            *nl = 0;
            int n = dispatch(start, response, sizeof response);
            if (n < 0 || (size_t)n >= sizeof response) return;
            if (n) {
                ov = (OVERLAPPED){.hEvent = event};
                if (!complete(pipe, &ov, WriteFile(pipe, response, n, NULL, &ov), 5000, &count) || count != (DWORD)n)
                    return;
            }
            start = nl + 1;
        }
        used -= (DWORD)(start - request);
        if (used == REQUEST_MAX) return;
        memmove(request, start, used);
    }
}

static DWORD WINAPI worker(void *arg) {
    HANDLE pipe = arg, event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (event) {
        for (;;) {
            DWORD count;
            OVERLAPPED ov = {.hEvent = event};
            BOOL connected = ConnectNamedPipe(pipe, &ov);
            if (!connected && GetLastError() != ERROR_PIPE_CONNECTED && !complete(pipe, &ov, FALSE, INFINITE, &count)) break;
            serve(pipe, event);
            DisconnectNamedPipe(pipe);
        }
        CloseHandle(event);
    }
    atomic_store_explicit(&enabled, 0, memory_order_relaxed);
    CloseHandle(pipe);
    return 0;
}

static PSECURITY_DESCRIPTOR user_acl(void) {
    HANDLE token;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return NULL;
    union { TOKEN_USER user; char data[sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE]; } info;
    DWORD size;
    BOOL ok = GetTokenInformation(token, TokenUser, &info, sizeof info, &size);
    CloseHandle(token);
    LPSTR sid;
    if (!ok || !ConvertSidToStringSidA(info.user.User.Sid, &sid)) return NULL;
    char sddl[192];
    int n = snprintf(sddl, sizeof sddl, "D:P(A;;GRGW;;;%s)", sid);
    LocalFree(sid);
    PSECURITY_DESCRIPTOR desc = NULL;
    if (n > 0 && (size_t)n < sizeof sddl)
        ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1, &desc, NULL);
    return desc;
}

void renderer_mcp_start(void) {
    char opt[2], name[80];
    if (GetEnvironmentVariableA("REKARINA_RENDERER_MCP", opt, sizeof opt) != 1 || opt[0] != '1') return;
    snprintf(name, sizeof name, "\\\\.\\pipe\\rekarina-renderer-%lu", (unsigned long)GetCurrentProcessId());
    PSECURITY_DESCRIPTOR desc = user_acl();
    if (!desc) { rt_log("renderer MCP: could not build a user-only pipe ACL"); return; }
    SECURITY_ATTRIBUTES sa = {sizeof sa, desc, FALSE};
    HANDLE pipe = CreateNamedPipeA(name, PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, RESPONSE_MAX, REQUEST_MAX, 0, &sa);
    LocalFree(desc);
    if (pipe == INVALID_HANDLE_VALUE) { rt_log("renderer MCP: could not create pipe"); return; }
    atomic_store_explicit(&enabled, 1, memory_order_relaxed);
    HANDLE thread = CreateThread(NULL, 0, worker, pipe, 0, NULL);
    if (!thread) {
        atomic_store_explicit(&enabled, 0, memory_order_relaxed);
        CloseHandle(pipe);
        rt_log("renderer MCP: could not start pipe worker");
        return;
    }
    CloseHandle(thread);  // Process lifetime; the worker owns the pipe and does not use renderer objects.
    rt_log("renderer MCP: listening on %s", name);
}
