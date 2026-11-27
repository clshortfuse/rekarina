// Widescreen (OoT 1.1.1): the game draws a picture of another shape than 16:9, with the camera seeing more
// of the world at the sides (Hor+: the vertical view stays the game's) instead of a stretched 16:9 view.
//
// The game presents a 2560x1440 picture at every console mode and renders its scene at 1920x1080, which its upscaler
// takes to the picture's size; both sizes are constants in its code. The patch sets of recomp/patches-v111.txt read
// them from the patch page instead (wide_out: the picture, wide_scene: the scene), wide_aspect replaces a 16:9 ratio of
// sead's perspective projection (every 16:9 camera of the game) with the picture's, wide_hud keeps the HUD and menus
// in their 16:9 shape in the middle of the picture (the UI's full-screen projection squeezed horizontally; fades and
// the menu backgrounds that redraw a capture of the whole picture stay full width), and wide_movie shows pre-rendered
// movies in the same middle 16:9. The heights stay the game's (1440 and 1080), so only the width grows, by the shape's
// ratio to 16:9; the present scales the picture to the window keeping its shape (gfx_present.c, REKARINA_ASPECT=fit).
//
// Setting REKARINA_GAME_ASPECT, read at start: "16:9" (the game's own, every set off), "window" (the shape of the window,
// or of the monitor in fullscreen), or a fixed shape "21:9" (2560x1080 monitors, 64:27), "32:9". A shape within 1% of
// 16:9 or narrower than it keeps the game's own sizes; the widest is 32:9.
//
// The page's layout is described at the top of recomp/patches-v111.txt.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "layout.h"
#include "gfx_wsys.h"
#include "host.h"
#include "ui/menu.h"
#include "vk_dlaa.h"
#include "wide_ui.h"

#define GAME_W 2560   // the picture, main+0x1202ee0 and main+0x31f39f4
#define GAME_H 1440
#define SCENE_H 1080
#define GAME_DTA_KIB 312320.0     // the texture heap made for that picture (wide_heap_extra_kib)

// The window's size, or the monitor's in fullscreen; 0 when unknown.
static int screen_size(int *w, int *h) {
    *w = *h = 0;
    if (gfx_cfg_fullscreen()) return gfx_window_monitor_size(w, h) && *h > 0;
    gfx_cfg_window_size(w, h);
    return *h > 0;
}

// The shape the setting asks for as width / height; 0 for the game's own.
static double wanted_ratio(void) {
    const char *s = cfg_value("REKARINA_GAME_ASPECT");
    if (!strcmp(s, "21:9")) return 64.0 / 27.0;
    if (!strcmp(s, "32:9")) return 32.0 / 9.0;
    if (strcmp(s, "window")) return 0;
    int w, h;
    if (!screen_size(&w, &h)) {
        rt_log("widescreen: the screen's size is unknown; the game's own 16:9");
        return 0;
    }
    return (double)w / h;
}

// REKARINA_GAME_RES: the picture's height, 720 up to 4320 ("window": the window's height, or the monitor's in
// fullscreen, never below the game's 1440). 720, 900 and 1080 are below the game's own: lighter pictures for slower
// GPUs, their scenes and the DLSS modes' shares of them as of any other picture (scale_k). On a small GPU much of the
// frame's cost follows the picture, not the scene (post-processing, the UI, the scaler): a Steam Deck shows a 16:9
// picture in 1280x720 of its screen, so the game's 2560x1440 drew four times the pixels it can show.
#define PICTURE_MIN_H 720
unsigned wide_auto_picture_height(void) {
    int w, h;
    if (!screen_size(&w, &h) || h < GAME_H) return GAME_H;
    return h > 4320 ? 4320 : (unsigned)h;
}

static unsigned picture_height(void) {
    const char *s = cfg_value("REKARINA_GAME_RES");
    if (!strcmp(s, "window")) return wide_auto_picture_height();
    int h = atoi(s);
    return h < PICTURE_MIN_H ? PICTURE_MIN_H : h > 4320 ? 4320 : (unsigned)h;
}



// The game turns the fixed scale into a scene size against the smaller of its own 1440-line picture and the picture
// it draws. Measured by the scene targets it makes (2026-11-14): a 2160-line picture with scale 1.0 gets a 2560x1440
// scene and with 1.5 a 3840x2160 one; a 1080-line picture with 1.0 gets 1920x1080 and with 0.75 gets 1440x810. So a
// share of a taller picture goes there times the picture's height over 1440, and a share of the 1080-line picture as
// it is: wide_init sets the factor (frame_rate.c res_fixed_present). Until then the factor was the height over 1440
// for every picture, and at 1080 the scene had 810 lines where the setting said 1080, DLSS quality 540 for 720.
static double scale_k = 1.0;
double wide_scale_k(void) { return scale_k; }

static float active_aspect = (float)GAME_W / (float)GAME_H;
float wide_aspect_ratio(void) { return active_aspect; }

// The picture the settings ask for, *ow x *oh; 0 when it is the game's own 2560x1440 (every set stays off). Read at the
// start, before the window exists: the monitor's size is known then.
static int picture_size(unsigned *ow, unsigned *oh, int quiet) {
    double r = wanted_ratio(), game = (double)GAME_W / GAME_H;
    *oh = picture_height();
    if (r <= 0 || r < game * 1.01) {
        if (r > 0 && !quiet) rt_log("widescreen: a %.4f picture is not wider than 16:9; the game's 16:9", r);
        r = game;
    }
    if (r == game && *oh == GAME_H) return 0;
    if (r > 32.0 / 9.0) r = 32.0 / 9.0;
    *ow = (unsigned)lround(*oh * r / 8) * 8;   // whole multiples of 8 pixels
    return 1;
}

// The texture heap: the engine's render targets live in "DynamicTextureAllocatorMem" of the HeapDef table
// (System/HeapSize/Product.Nin_Ounce_NVN2.HeapDef.byml.zs; 312320 KiB, carved from "Graphics"), made for the 2560x1440
// picture. Larger pictures' targets did not fit: the render-target manager's search for a free block (main+0x3ed00)
// fell to its out-of-memory path and read through a null block when the pause menu's blur needed one at 5120x2160.
// The heap and its parent grow by the shipped size times the picture's pixels over 2560x1440 (its targets are the
// picture's size or a share of it), in whole MiB; resolution.c rewrites the table (res_heapdef_build) and memory.c adds
// the amount to the application pool, so no other heap shrinks.
//
// The 16:9 picture needs more than the shipped size too. The console's dynamic resolution keeps its scene at 3/4 of
// the picture at most (1920x1080) and lowers it where the GPU is busy; here the scene has the whole picture
// (frame_rate.c scene_share), and square depth targets report padded storage (nvn_api_obj.c storage_size). Where the
// game asks for many square shadow maps a frame, a new side each frame while the camera moves (the room of pots and
// torches beside the Market's gate: 6 to 17 a frame, 386 to 1918 texels), the shipped heap ran out with the same null
// read, as the hero walked in. SCENE_ROOM_KIB more holds them. REKARINA_TEX_HEAP_MIB = n (test) makes the heap n MiB.
#define SCENE_ROOM_KIB (256.0 * 1024)
uint64_t wide_heap_base_kib(void) { return (uint64_t)GAME_DTA_KIB; }
uint64_t wide_heap_extra_kib(void) {
    const char *e = cfg_env("REKARINA_TEX_HEAP_MIB");
    if (e && *e) {
        double want = atof(e) * 1024 - GAME_DTA_KIB;
        return want > 0 ? (uint64_t)ceil(want / 1024) * 1024 : 0;
    }
    double grow = 0;
    unsigned ow, oh;
    if (wide_available() && picture_size(&ow, &oh, 1)) grow = (double)ow * oh / ((double)GAME_W * GAME_H) - 1;
    double kib = (grow > 0 ? GAME_DTA_KIB * grow : 0) + SCENE_ROOM_KIB;
    return (uint64_t)ceil(kib / 1024) * 1024;
}

void wide_init(void) {
    int so = rt_patch_set("wide_out"), ss = rt_patch_set("wide_scene"), sa = rt_patch_set("wide_aspect"),
        sh = rt_patch_set("wide_hud"), sm = rt_patch_set("wide_movie");
    wide_ui_init();   // active at 16:9 too so Custom HUD placement works live without restart
    // The scene's share of the picture is not a size in this set: the game's GPU-time dynamic resolution sets it each
    // frame (3/4 of the picture at most), and writing other sizes into wide_scene's constant or the 1920x1080 render
    // sizes left the light target where the controller put it. DLSS's modes fix the share through res_fixed instead
    // (wide_present).
    double game = (double)GAME_W / GAME_H, f = 0.75;
    unsigned ow, oh;
    if (!picture_size(&ow, &oh, 0)) return;   // the game's own sizes: every set stays off
    if (so < 0 || ss < 0 || sa < 0 || sh < 0 || sm < 0) {
        rt_log("widescreen: the generated C has no widescreen patch sets (gen.py --patches recomp/patches-v111.txt); "
               "the game's own 2560x1440 picture");
        return;
    }
    // The scene 3/4 of both sides, its width in whole multiples of 8 pixels and its height even.
    unsigned sw = (unsigned)lround(ow * f / 8) * 8;
    unsigned sch = (unsigned)lround(oh * f / 2) * 2;
    uint64_t out = (uint64_t)oh << 32 | ow, scene = (uint64_t)sch << 32 | sw;
    float match = 16.0f / 9.0f, ratio = (float)ow / oh, tolerance = 0.01f, squeeze = match / ratio;
    active_aspect = ratio;
    float ui[2] = {2.0f / 3840.0f, 2.0f / 2160.0f}, squeeze4[4] = {squeeze, squeeze, squeeze, squeeze};
    float margin = (1.0f - squeeze) / 2;
    // the screens whose UI keeps the whole picture (recomp/patches-v111.txt, set wide_hud): the item-wheel, pause and
    // system menu backgrounds, the event and loading fades
    uint64_t keep[2] = {MOD_MAIN_BASE + 0x4bed098, MOD_MAIN_BASE + 0x4be4fb0},
             keep2[2] = {MOD_MAIN_BASE + 0x4be61c8, MOD_MAIN_BASE + 0x4be7b30};
    uint8_t *page = rt_patch_page();
    memcpy(page + 96, &out, 8);
    memcpy(page + 104, &scene, 8);
    memcpy(page + 112, &match, 4);
    memcpy(page + 116, &ratio, 4);
    memcpy(page + 120, &tolerance, 4);
    memcpy(page + 144, ui, 8);
    memcpy(page + 160, keep, 16);
    memcpy(page + 176, squeeze4, 16);
    memcpy(page + 192, &margin, 4);
    memcpy(page + 208, keep2, 16);
    rt_patch_switch(so, 1);
    rt_patch_switch(ss, 1);
    rt_patch_switch(sa, 1);
    rt_patch_switch(sh, 1);
    rt_patch_switch(sm, 1);
    scale_k = oh > GAME_H ? (double)oh / GAME_H : 1.0;
    {   // DEBUG-SCALEK: REKARINA_SCALE_K=<factor> replaces it (a test of what the game takes the fixed scale against)
        const char *e = cfg_env("REKARINA_SCALE_K");
        if (e && *e) scale_k = atof(e);
    }
    rt_log("widescreen: picture %ux%u, scene %ux%u (%.0f%% of it), camera aspect ratio %.4f (16:9 is %.4f), UI and movies "
           "in the middle %.1f%%", ow, oh, sw, sch, f * 100, ratio, game, squeeze * 100);
}

// The picture the game draws: the widescreen one wide_init set up, else the game's own 2560x1440. DLSS's largest mode
// (dlaa) renders the scene at this size.
void wide_picture(unsigned *w, unsigned *h) {
    static unsigned pw, ph;
    if (!pw) {
        if (!wide_available() || !picture_size(&pw, &ph, 1)) pw = GAME_W, ph = GAME_H;
    }
    *w = pw, *h = ph;
}

// The generated C has the widescreen patch sets (the menu shows the row only then).
int wide_available(void) {
    static const char *const names[] = {"wide_out", "wide_scene", "wide_aspect", "wide_hud", "wide_movie"};
    for (int i = 0; i < 5; i++)
        if (rt_patch_set(names[i]) < 0) return 0;
    return 1;
}
