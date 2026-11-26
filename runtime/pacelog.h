// DEBUG-PACELOG (frame_rate.c, REKARINA_PACELOG=<file>): per presented frame (gfx_counters.frames numbering, synchronous
// replay) the moments of its life: the game's acquire (frame start), its present call, the present's return, the host
// present and the GPU's completion of its blit, and the step the next frame is simulated with. CSV at exit.
#pragma once
#include <stdint.h>
// PL_VIS: first pixel visible (REKARINA_PRESENT_TIMING); PL_BLIT: the present thread has its swap chain image (the fence
// and acquire waits behind it) and submits the blit; PL_GTS: the GPU's own clock at the blit's end (a timestamp query,
// calibrated to the host clock): PL_GPU minus PL_GTS is how late the completion thread saw it.
// PL_PAD: the frame's pad read (frame_start.c: the first nn::hid::GetNpadStates on any thread after the frame's update
// start, else the newest before it); PL_UPD: the main thread's update start (its return from the hand-over of the frame
// before, after any frame start hold).
enum { PL_ACQ, PL_CALL, PL_RET, PL_FLIP, PL_GPU, PL_VIS, PL_BLIT, PL_GTS, PL_PAD, PL_UPD, PL_N };
// Durations per frame in microseconds (frame_start.c): the hold before its update start, the main thread's waits from
// its acquire to the next update start (the acquire's own and its kernel waits), its wait on the hand-over queue alone,
// and the present's wait for the GPU (nvn_api_obj.c nvnQueuePresentTexture, the present sync).
// PLD_GWAIT / PLD_GLAST: over the main thread's whole cycle (its update start to the next), its waits that the GPU
// (a host thread's signal, the acquire's fence, nvnSyncWait) or the presenting thread ended, and the end of the last of
// them counted from the update start.
// ACQUIRE_WAIT, SYNC_WAIT and WAKE_WAIT split GWAIT into acquire, nvnSyncWait and guest kernel wake time.
enum { PLD_HOLD, PLD_MWAIT, PLD_QWAIT, PLD_PSYNC, PLD_GWAIT, PLD_GLAST, PLD_ACQUIRE_WAIT, PLD_SYNC_WAIT, PLD_WAKE_WAIT, PLD_N };
void pacelog_dur(int what, uint64_t frame, uint64_t ns);
void gfx_blit_frame(uint64_t frame);          // gfx_present.c: the frame the next blit presents (vk_backend.c present_main)
uint64_t gfx_blit_gpu_ns(uint64_t frame);     // its GPU end on the host clock (kern_now_ns), 0 if unknown; once (also the
                                              // frame clock's shown moment, perf.c)
int pacelog_on(void);
void pacelog_mark(int what, uint64_t frame, uint64_t ns);
void pacelog_step(uint64_t frame, double step_s);
void pacelog_write(void);    // at exit (perf.c perf_summary_flush)
double frame_step_now(void); // the step the patch page holds now (seconds)
void gfx_present_frame(uint64_t frame);   // gfx_present.c: the frame the next flip presents (present timing ids)
