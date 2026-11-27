// Settings pages in two panes (ui/option_rows.cpp): option rows on the left, the selected (controller) or hovered (mouse)
// row's description on the right. Each row shows a setting of the cfg table (cfg.h) and changes it through cfg_set: A or
// a click steps it (toggle, next value), left and right step it. A row marks the default value, says "after restart" when
// the setting is read at start, and is greyed out with a note when an environment variable fixes it.
//
//   opt_begin("##screen", scale);
//   opt_toggle("REKARINA_FULLSCREEN", UI_MENU_FULLSCREEN, UI_MENU_FULLSCREEN_DESC);
//   opt_end();
//
// key: the setting's REKARINA_ name, which must have a cfg_meta row. Present path only (ImGui).
#pragma once
#include "imgui.h"

// The ImGui keys the gamepad's X and Y arrive as in the menu (menu.cpp pad_keys): keys ImGui's own navigation does not
// read, so these buttons act on the menu's rows only (option_rows.cpp)
#define MENU_PAD_X ImGuiKey_GamepadR3
#define MENU_PAD_Y ImGuiKey_GamepadL3

extern "C" {
#include "ui_text.h"
}

void opt_begin(const char *id, float scale);
void opt_end(void);
// The description pane's content, drawn by the menu beside the rows: the label and the description (with its notes) of
// the hovered row, else the selected one, else the last shown. opt_desc_clear: nothing shown until a row is selected.
const char *opt_desc_label(void);
const char *opt_desc_text(void);
const char *opt_desc_key(void);   // the setting (or row id) of the row described, "" when none
void opt_desc_clear(void);
// The cursor of the game's menus around [a, b]: a white frame with a pale glow outside it.
void opt_select_frame(ImDrawList *dl, ImVec2 a, ImVec2 b, float scale);
// The grain of the menu's surfaces over [a, b] (ui/menu.cpp: a square of the font atlas); alpha 0..255 is its strength.
void opt_grain(ImDrawList *dl, ImVec2 a, ImVec2 b, int alpha);
// The height the next pages leave free under their panes (the menu's bottom row).
void opt_reserve_bottom(float h);
// The height opt_reserve_bottom keeps free, for a tab that lays out its own panes (ui/amiibo_page.cpp).
float opt_bottom(void);
// The next page's first row takes the controller's cursor (opening the menu, switching tabs).
void opt_focus_first(void);
// The row of key (or button id) takes the cursor, scrolled into view, when it is next drawn.
void opt_focus_key(const char *key);
// The key of the row the cursor is on in this frame's page (as opt_focus_key takes it), "" when none is.
const char *opt_focus_row(void);
// The next row is greyed out like one the environment fixes, with note under its label and in its description.
void opt_disable_next(UiTextId note);
void opt_disable_next_text(const char *note);   // note must outlive the frame

// Page text that is not a row: a group heading in the accent colour, a wrapped paragraph (warn: the warning colour).
void opt_group(UiTextId title);
void opt_text(UiTextId text, bool warn = false);

// The return value of every row: true when it changed the setting this frame.
// inverted: the row shows "on" while the key is 0 (a setting whose key names the opposite).
bool opt_toggle(const char *key, UiTextId label, UiTextId desc, bool inverted = false);
// One name per value of the key's cfg_meta values list, in its order. When the key has no fixed default (an empty
// def), names[0] is the automatic choice (the key unset) and names[1..] follow the values. note: a short line under
// the value in place of the "default" mark (NULL: none).
bool opt_choice(const char *key, UiTextId label, const UiTextId *names, int n, UiTextId desc,
                const char *note = nullptr);
// A choice over a list the caller builds (devices, window sizes): values[i] is written, names[i] shown; "" means the key
// unset. cur: the index in effect, -1 to find cfg_value(key) in values (no match: the value shown as it is).
bool opt_choice_list(const char *key, UiTextId label, const char *const *values, const char *const *names, int n,
                     UiTextId desc, int cur = -1, const char *note = nullptr);
// A row that runs an action: true on A or a click. id: unique on the page and not a setting ("##copy_info"); value:
// shown on the right ("" none).
bool opt_button(const char *id, UiTextId label, UiTextId desc, const char *value = "");
// A row whose label, value, note (under the value) and description are text, not catalog ids: a mod's folder name.
// Returns 2 (A / click), -1 / +1 (left / right), 3 / 4 (move up / down: the top / left face buttons, Ctrl+Up /
// Ctrl+Down, the drawn arrows; movable rows only) or 0.
int opt_text_row(const char *id, const char *label, const char *value, const char *note, const char *desc,
                 bool movable);
int opt_text_choice(const char *id, UiTextId label, const char *value, const char *note, UiTextId desc);
// Rows and text that are not settings of the cfg table: a label and description of the caller's own (the cheats of
// the cheat files). opt_toggle_text shows on / off (value: shown instead, NULL for On / Off) and returns true on A, a
// click, left or right; the caller changes its state. id: unique on the page, the environment check uses it.
bool opt_toggle_text(const char *id, const char *label, const char *desc, bool on, const char *value = nullptr);
void opt_group_text(const char *title);
void opt_text_str(const char *text, bool warn = false);
// lo..hi in steps of step (left and right held for a second go ten times as fast); fmt formats the value ("%.0f %%").
// note: a short line under the value in place of the "default" mark (NULL: none). fallback: the setting that applies
// while key is unset; the row shows its value then, and a change sets key. A key with no table default of its own takes
// the "default" mark from the fallback's default, and a step onto the fallback's value unsets key again. log_scale: the
// bar's places follow the value's logarithm (a multiplier such as the mouse speed: 0.1, 1 and 10 equally far apart);
// a range from 0 keeps 0 at the left end. Scale and offset convert stored values to slider units, or a pair of
// conversion functions can map the stored value to slider units and back.
bool opt_slider(const char *key, UiTextId label, float lo, float hi, UiTextId desc, float step = 1,
                const char *fmt = "%g", const char *note = nullptr, const char *fallback = nullptr,
                bool log_scale = false, double scale = 1, double offset = 0,
                double (*to_slider)(double) = nullptr, double (*from_slider)(double) = nullptr);

// The value bar of the slider row the cursor is on, as last drawn (screen pixels); false when the cursor is on no slider
// row since the menu opened. For the menu's test script (REKARINA_MENU_SCRIPT dragbar).
bool opt_focused_bar(ImVec2 *a, ImVec2 *b);

// Binding rows (remapping on the Controls tab): a label and three cells, each selectable on its own (left / right move between them).
// opt_bind_header: the column titles above them. opt_bind_row: key is the setting's name without REKARINA_ (BIND_A; the
// description names it when the environment sets it); a disabled cell shows its text greyed and takes no cursor; note
// (NULL none) follows the description. Returns the cell A / Enter / a click activated (0..2), 10 + the cell Delete,
// Backspace or X cleared, or -1. opt_focus_key("<key>#<cell>") puts the cursor on a cell.
void opt_bind_header(UiTextId c0, UiTextId c1, UiTextId c2);
int opt_bind_row(const char *key, const char *label, UiTextId desc, const char *const cells[3], const bool enabled[3],
                 const char *note = nullptr);

// rekarina.ini is written a moment after the last change (a held slider would rewrite it every frame) and when the menu
// closes. opt_save_tick: on every frame the menu lays out; force: write now if anything is pending. 1 while a write is
// pending.
int opt_save_tick(int force);
// 1 while a changed setting waits for its write (any thread).
int opt_save_pending(void);
// A change made outside a row (Reset, F10): rekarina.ini is written as after a row's change.
void opt_save_soon(void);
