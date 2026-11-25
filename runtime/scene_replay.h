#ifndef SCENE_REPLAY_H
#define SCENE_REPLAY_H

#include <stdint.h>

typedef struct { uint64_t frame, view; } SceneReplayKey;
typedef enum { SCENE_ROUTE_ORIGINAL, SCENE_ROUTE_BLOOM, SCENE_ROUTE_NO_BLOOM, SCENE_ROUTE_LATE } SceneReplayRoute;

typedef struct {
    SceneReplayKey key;
    SceneReplayRoute route;
    unsigned rw, rh, uw, uh, bw, bh;
    uint32_t light, tonemap, packed, output, window;
    uint32_t direct_composite, direct_window;
    unsigned direct_w, direct_h;
    int bloom_seen, bloom_written, completed;
} SceneReplay;

static inline int scene_replay_matches(SceneReplayKey a, SceneReplayKey b) {
    return a.frame && a.frame == b.frame && a.view == b.view;
}

static inline int scene_replay_direct_present(const SceneReplay *scene, uint32_t window) {
    return !scene->completed && !scene->packed && scene->key.frame && scene->key.view && scene->tonemap &&
        scene->direct_composite && window && scene->direct_window == window && scene->direct_w && scene->direct_h;
}

typedef enum { SCENE_UI_ALPHA_NONE, SCENE_UI_ALPHA_EMPTY, SCENE_UI_ALPHA_DRAWN } SceneUiAlpha;

static inline SceneUiAlpha scene_replay_ui_alpha(int hudless, int rejected, int output_unchanged, int coverage_matches) {
    if (!hudless || rejected) return SCENE_UI_ALPHA_NONE;
    if (coverage_matches) return SCENE_UI_ALPHA_DRAWN;
    return output_unchanged ? SCENE_UI_ALPHA_EMPTY : SCENE_UI_ALPHA_NONE;
}

static inline SceneReplayRoute scene_replay_early_route(int bloom_consumer, int bloom_seen) {
    return bloom_consumer ? SCENE_ROUTE_BLOOM : bloom_seen ? SCENE_ROUTE_ORIGINAL : SCENE_ROUTE_NO_BLOOM;
}

#endif
