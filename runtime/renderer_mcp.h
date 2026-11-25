#ifndef RENDERER_MCP_H
#define RENDERER_MCP_H

#include <stdint.h>
#include "scene_replay.h"

#ifdef _WIN32
void renderer_mcp_start(void);
void renderer_mcp_scene(const SceneReplay *scene, int hudless, int coverage);
void renderer_mcp_present(unsigned width, unsigned height, int hdr10, int route, int result);
typedef enum { RENDERER_MCP_NGX, RENDERER_MCP_STREAMLINE } RendererMcpDlss;
void renderer_mcp_dlss(RendererMcpDlss route, unsigned input_width, unsigned input_height,
                       unsigned output_width, unsigned output_height, int ok, uint32_t result);
void renderer_mcp_bloom(const char *route, unsigned input_width, unsigned input_height,
                        unsigned output_width, unsigned output_height);
void renderer_mcp_tonemap(unsigned width, unsigned height);
#else
#define renderer_mcp_scene(...) ((void)0)
#define renderer_mcp_start() ((void)0)
#define renderer_mcp_present(...) ((void)0)
#define renderer_mcp_dlss(...) ((void)0)
#define renderer_mcp_bloom(...) ((void)0)
#define renderer_mcp_tonemap(...) ((void)0)
#endif

#endif
