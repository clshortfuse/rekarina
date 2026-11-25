#include <assert.h>
#include "scene_replay.h"

int main(void) {
    SceneReplayKey gameplay = {42, 7};
    assert(scene_replay_matches(gameplay, gameplay));
    assert(!scene_replay_matches(gameplay, (SceneReplayKey){42, 8}));
    assert(!scene_replay_matches(gameplay, (SceneReplayKey){41, 7}));
    assert(!scene_replay_matches((SceneReplayKey){0, 7}, (SceneReplayKey){0, 7}));
    assert(scene_replay_early_route(1, 0) == SCENE_ROUTE_BLOOM);
    assert(scene_replay_early_route(0, 0) == SCENE_ROUTE_NO_BLOOM);
    assert(scene_replay_early_route(0, 1) == SCENE_ROUTE_ORIGINAL);
    SceneReplay direct = {.key = gameplay, .tonemap = 11, .direct_composite = 12,
                          .direct_window = 13, .direct_w = 2560, .direct_h = 1440};
    assert(scene_replay_direct_present(&direct, 13));
    assert(!scene_replay_direct_present(&direct, 12));
    direct.key.frame = 0;
    assert(!scene_replay_direct_present(&direct, 13));
    direct.key = gameplay;
    direct.packed = 14;
    assert(!scene_replay_direct_present(&direct, 13));
    direct.packed = direct.direct_composite = 0;
    assert(!scene_replay_direct_present(&direct, 13));
    assert(scene_replay_ui_alpha(1, 0, 1, 0) == SCENE_UI_ALPHA_EMPTY);
    assert(scene_replay_ui_alpha(1, 0, 0, 1) == SCENE_UI_ALPHA_DRAWN);
    assert(scene_replay_ui_alpha(1, 0, 0, 0) == SCENE_UI_ALPHA_NONE);
    assert(scene_replay_ui_alpha(1, 1, 1, 0) == SCENE_UI_ALPHA_NONE);
    assert(scene_replay_ui_alpha(0, 0, 1, 0) == SCENE_UI_ALPHA_NONE);
    return 0;
}
