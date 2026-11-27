// The problem report (runtime/report.c): one zip a player can attach to a report, made by the menu's "Make a problem
// report", by `rekarina --make-report` (the launcher's button) and offered after a run that ended in an error. Nothing is
// uploaded.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Writes <game folder>/reports/rekarina-report-<date>-<time>.zip: system.txt (OS, CPU, RAM, GPU and driver, the device
// profile, the build and game versions, the settings in effect, the shader cache's state), the logs of the last five
// runs (each its last 16 MiB), rekarina.ini, the setup's record, the mods folder's names, and README.txt saying what is
// inside. No save data and no game files; the user's name and home folder in any path are replaced. path (n bytes)
// receives the zip's path. 1 on success.
int report_make(char *path, size_t n);
// The folder the reports go into (<game folder>/reports, made if missing). 0 on failure.
int report_folder(char *out, size_t n);
// 1 when the previous run's log (logs/rekarina.1.log at this start) ended with an error: a crash (FATAL), a lost GPU
// device or a failed start. Read once.
int report_last_run_failed(void);
// 1 when that log says the graphics device was lost (vk_crumbs.c turns the pass marks on for this run). Read once.
int report_last_run_lost_device(void);
// `rekarina --make-report`: makes the report without starting the game, prints its path, shows the folder; exit code.
int report_main(void);
// At the end of a crash or a fatal error of a game run that writes logs/rekarina.log (a player's run, not a test run
// with its own log file): a message box that says the game stopped and how to make a report. Once; Windows only.
void report_crash_notice(void);
// A fatal error the player can put right himself (romfs.c: the dump is not where rekarina.ini says): text in the
// message box instead of the crash notice, which then shows nothing more. Shown for the same runs as the crash notice.
void report_fatal_box(const char *text);

// Frame generation that stopped the game while it was being set up (report.c, "frame generation that stopped the
// game"). AMD's and NVIDIA's run in their own libraries; a crash there from the start of their set-up to their first
// few hundred generated frames leaves logs/framegen-stopped.txt, and the next start of the same build with the same
// setting runs without frame generation, so the game still starts.
typedef enum { FG_GUARD_DLSS = 1, FG_GUARD_FSR } FgGuardBackend;
void fg_guard_arm(FgGuardBackend backend);
void fg_guard_running(FgGuardBackend backend);
void fg_guard_disarm(FgGuardBackend backend);
int fg_guard_blocked(void);               // Whether the startup marker blocks the selected setting (startup setting on TotK).
#ifdef GAME_OOT
int fg_guard_retry_pending(void);          // The selected backend is blocked in this process but can retry after restart.
#endif
void fg_guard_clear(void);                // the player chose frame generation again: the next start tries it
void fg_guard_crashed(void);              // the crash handler (main.c): writes the file when the set-up was running
#define FG_GUARD_WHY "it stopped the game at the last start, so this start runs without it (choose it again here to try once more)"

#ifdef __cplusplus
}
#endif
