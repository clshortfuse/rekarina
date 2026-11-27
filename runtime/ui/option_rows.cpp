// Two-pane settings pages (option_rows.h): rows on the left, the selected or hovered row's description on the right.
//
// Adapted from SavageWindRecomp's runtime/ui/option_rows.mm (https://github.com/Revivedx/SavageWindRecomp @
// 575ad88e02f9a89a9a885d90039d48af33c982f4, MIT, Copyright (c) 2026 Revivedx and the Savage Wind Recomp contributors,
// Copyright (c) 2026 Peiikai): the row look, the two panes, the left / right stepping and the hold speed-up. The rows
// read and write cfg keys instead of a settings struct, take their text from the menu catalog (menu_text), and show
// the default, "after restart" and environment notes; the text field, radio and on-screen keyboard rows are not ported.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgui.h"
#include "imgui_internal.h"   // SetKeyOwner, SetFocusID, NavCursorVisible

#include "ui/option_rows.h"

extern "C" {
#include "cfg.h"
#include "platform/platform.h"
#include "ui_text.h"
// host.h pulls in cpu.h, which is C only: the one function needed from it, as declared there.
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
}

// The colours of the game's own Settings screen (sampled from it): rows are khaki strips on a dark panel, a value sits
// in a near-black inset, the chosen value is an amber bar, the cursor is a white frame with a pale glow.
static const ImU32 kCream = IM_COL32(246, 239, 214, 255);
static const ImU32 kRow = IM_COL32(111, 90, 53, 255), kRowHot = IM_COL32(124, 102, 62, 255);
static const ImU32 kRowEdge = IM_COL32(140, 118, 72, 255);
static const ImU32 kInset = IM_COL32(30, 26, 18, 255);
static const ImU32 kAmberTop = IM_COL32(217, 149, 31, 255), kAmberBottom = IM_COL32(196, 130, 27, 255);

static const ImU32 kDim = IM_COL32(233, 201, 138, 255);   // notes beside a label
static const ImU32 kGrey = IM_COL32(106, 101, 88, 255);   // the value not chosen, a fixed row's text

static float g_scale = 1;
static bool g_focus_req, g_take_first;   // opt_focus_first: requested / this page's first row takes it
static char g_focus_key[64];             // opt_focus_key: the row that takes the cursor when it is next drawn
static bool g_held;                      // the last row's left / right has been held a while (steppers go faster)
static float g_bottom;                   // opt_reserve_bottom: kept free under the panes
static bool g_disable_next;              // opt_disable_next: the next row is greyed out with g_disable_note
static const char *g_disable_note;

// The description pane: this frame's hovered and selected rows, the page's first row, and the last one shown.
struct Desc { char text[UI_DESC_BYTES]; char label[160]; char key[64]; };
static Desc g_hover, g_focus, g_first, g_shown;

// disabled: the note of a row opt_disable_next greyed out, else NULL. note: the row's own note (a longer one is
// left out of the row's strip, so the pane is where it is read), else NULL.
static void desc_set(Desc &d, const char *desc, const char *key, const CfgMeta *m, const char *disabled,
                     const char *label = "", const char *note = NULL) {
    snprintf(d.label, sizeof d.label, "%s", label);
    snprintf(d.key, sizeof d.key, "%s", key ? key : "");
    int n = snprintf(d.text, sizeof d.text, "%s", desc);
    if (n < 0 || n >= (int)sizeof d.text) return;
    if (disabled)
        snprintf(d.text + n, sizeof d.text - n, "\n\n%s", disabled);
    else if (cfg_from_env(key))
        snprintf(d.text + n, sizeof d.text - n, "\n\n%s %s", menu_text(UI_MENU_SET_BY_ENV), key);
    else if (note && *note)
        snprintf(d.text + n, sizeof d.text - n, "\n\n%s", note);
    else if (m && !m->live)
        snprintf(d.text + n, sizeof d.text - n, "\n\n%s", menu_text(UI_MENU_AFTER_RESTART));
}

void opt_focus_first(void) { g_focus_req = true; }

void opt_focus_key(const char *key) { snprintf(g_focus_key, sizeof g_focus_key, "%s", key); }

void opt_reserve_bottom(float h) { g_bottom = h; }
float opt_bottom(void) { return g_bottom; }

void opt_disable_next(UiTextId note) { opt_disable_next_text(menu_text(note)); }

void opt_disable_next_text(const char *note) {
    g_disable_next = true;
    g_disable_note = note;
}

void opt_begin(const char *id, float scale) {
    g_scale = scale;
    g_take_first = g_focus_req;
    g_focus_req = false;
    if (g_take_first) ImGui::NavMoveRequestCancel();   // this frame's move goes to the first row instead
    g_hover.text[0] = g_focus.text[0] = g_first.text[0] = 0;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= g_bottom;
    ImGui::PushID(id);
    ImGui::BeginChild("##rows", ImVec2(avail.x, avail.y), ImGuiChildFlags_NavFlattened);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, floorf(7 * scale)));
    ImGui::Dummy(ImVec2(0, floorf(6 * scale)));   // room for the cursor's glow above the first row
}

void opt_end(void) {
    ImGui::Dummy(ImVec2(0, floorf(6 * g_scale)));
    ImGui::PopStyleVar();
    ImGui::EndChild();
    // the hovered row (mouse) wins, else the selected one (controller), else the last shown, else the first row
    if (g_hover.text[0]) g_shown = g_hover;
    else if (g_focus.text[0]) g_shown = g_focus;
    else if (!g_shown.text[0]) g_shown = g_first;
    ImGui::PopID();
}

const char *opt_focus_row(void) { return g_focus.text[0] ? g_focus.key : ""; }

const char *opt_desc_text(void) { return g_shown.text; }
const char *opt_desc_label(void) { return g_shown.label; }
const char *opt_desc_key(void) { return g_shown.key; }
void opt_desc_clear(void) { g_shown.text[0] = g_shown.label[0] = g_shown.key[0] = 0; }

// The cursor of the game's menus: a white frame around what is selected, with a pale glow outside it.
void opt_select_frame(ImDrawList *dl, ImVec2 a, ImVec2 b, float scale) {
    float o = floorf(3 * scale), t = fmaxf(2, floorf(2.5f * scale));
    for (int i = 4; i >= 1; i--) {
        float g = o + i * 1.6f * scale;
        dl->AddRect(ImVec2(a.x - g, a.y - g), ImVec2(b.x + g, b.y + g), IM_COL32(190, 215, 255, 64 - i * 13), 3 * scale, 0, t);
    }
    dl->AddRect(ImVec2(a.x - o, a.y - o), ImVec2(b.x + o, b.y + o), IM_COL32(255, 255, 255, 255), 2 * scale, 0, t);
}

static void amber_bar(ImDrawList *dl, ImVec2 a, ImVec2 b) {
    dl->AddRectFilledMultiColor(a, b, kAmberTop, kAmberTop, kAmberBottom, kAmberBottom);
}

static void text_centred(ImDrawList *dl, float x0, float x1, float y, float h, ImU32 col, const char *t) {
    ImFont *font = ImGui::GetFont();
    float fs = ImGui::GetFontSize(), room = x1 - x0, tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    if (tw > room && room > 0) {   // a long value at a smaller size, down to 70 %
        fs = fmaxf(fs * room / tw, fs * 0.7f);
        tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    }
    ImVec4 clip(x0, y, x1, y + h);
    dl->AddText(font, fs, ImVec2(tw < room ? x0 + (room - tw) * 0.5f : x0, y + (h - fs) * 0.5f), col, t, NULL, 0, &clip);
}

// What a row shows besides its label.
enum { ROW_TEXT, ROW_STEP, ROW_SWITCH_ON, ROW_SWITCH_OFF, ROW_SLIDER };
struct RowView {
    const char *value;   // right side; "" for none
    bool is_default;     // the value is the setting's default; a changed setting gets the white corner the game draws
    bool arrows;         // a stepper: < value >, the arrows click too
    const char *note;    // beside the label; NULL for none
    int kind = ROW_TEXT; // ROW_SWITCH_*: the two segments On / Off of the game's switches; ROW_STEP: the amber bar;
                         // ROW_SLIDER: the amber bar filled to `fill` of its width, dragged with the mouse
    bool setting = false;   // has a default to compare with: the corner mark applies
    float fill = 0;         // ROW_SLIDER: the value's place in its range, 0..1
    float def_fill = -1;    // ROW_SLIDER: the default's place, a small mark under the bar; -1 none
};

// ROW_SLIDER: where the mouse holds the bar this frame, 0..1 of its width; -1 when it does not. Set by row_text for the
// row it draws, read by opt_slider right after.
static float g_drag = -1;
// The slider row the mouse pressed on its bar (ImGui id), so a drag that leaves the bar keeps moving it; 0 none.
static ImGuiID g_drag_id;
// The bar of the slider row the cursor is on, for the menu's test script (opt_focused_bar); valid from its draw on.
static ImVec2 g_bar[2];
static bool g_bar_valid;

bool opt_focused_bar(ImVec2 *a, ImVec2 *b) {
    if (!g_bar_valid) return false;
    *a = g_bar[0], *b = g_bar[1];
    return true;
}

// One row: draws it, handles A / click and left / right while selected. Returns 2 (A / click), -1 / +1 (left / right),
// 3 / 4 (move up / down, movable rows only) or 0; always 0 for a row the environment fixes (greyed out, still selectable
// so its description shows). movable: the top / left face buttons, Ctrl+Up / Ctrl+Down and two drawn arrows move it.
static int row_text(const char *key, const char *label, const char *desc, const RowView &v, bool movable) {
    const CfgMeta *m = cfg_meta(key);
    const char *disabled = g_disable_next ? g_disable_note : NULL;
    g_disable_next = false;
    g_drag = -1;
    bool off = disabled || cfg_from_env(key) != 0;
    ImGui::PushID(key);
    ImVec2 p = ImGui::GetCursorScreenPos();
    // h whole pixels: ImGui truncates cursor positions, which rounds a fractional height the other way above the window's
    // top (negative y), so the page grew a pixel per row scrolled out and a row scrolled to the bottom was cut off
    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x, h = floorf(fs * 1.95f);
    float mx = floorf(12 * g_scale);   // the strip is narrower than the pane: the cursor's glow lies outside it
    ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4(0, 0, 0, 0));   // the row's own border shows the selection
    bool clicked = ImGui::InvisibleButton("##row", ImVec2(w, h), ImGuiButtonFlags_EnableNav);
    ImGui::PopStyleColor();
    // a row without a setting has an ImGui id "##name": the menu script names it without the "##" (a comment there)
    bool to_key = g_focus_key[0] && (!strcmp(key, g_focus_key) || (!strncmp(key, "##", 2) && !strcmp(key + 2, g_focus_key)));
    if (g_take_first || to_key) {
        ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
        ImGui::SetNavCursorVisible(true);
        // the row the cursor was put on is the one described, not one the resting mouse happens to be over
        ImGui::GetCurrentContext()->NavHighlightItemUnderNav = true;
        if (g_take_first) ImGui::SetScrollY(0);   // the page's top: text and a heading above the first row show
        else ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);   // as a cursor move would (the last row whole)
        g_take_first = false;
        if (to_key) g_focus_key[0] = 0;
    }
    // A row clicked with the mouse keeps the cursor: the keys and the pad then step it and Delete resets it at once,
    // where ImGui would first want a key press to show its cursor again.
    if (clicked && ImGui::IsMouseReleased(0)) {
        ImGui::SetNavCursorVisible(true);
        // DEBUG-MENUCLIENT: the first clicks of a session, where the menu took them (its pixels, the swapchain's)
        static int said;
        if (said < 40) {
            said++;
            ImVec2 mp = ImGui::GetIO().MousePos, r0 = ImGui::GetItemRectMin(), r1 = ImGui::GetItemRectMax();
            rt_log("menu: a click at %.0f,%.0f took the row %s at %.0f-%.0f", mp.x, mp.y, key, r0.y, r1.y);
        }
    }
    bool hov = ImGui::IsItemHovered(), foc = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
    int step = clicked && !off ? 2 : 0;
    g_held = false;
    if (foc && !off) {
        // the left stick arrives as the D-pad (menu.cpp pad_events)
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) || ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) step = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) || ImGui::IsKeyPressed(ImGuiKey_RightArrow)) step = 1;
        static const ImGuiKey steps[] = {ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadDpadRight, ImGuiKey_LeftArrow,
                                         ImGuiKey_RightArrow};
        for (ImGuiKey k : steps) {
            if (ImGui::GetKeyData(k)->DownDuration > 1.0f) g_held = true;
            // left and right belong to the row (navigation would otherwise move sideways into the description pane)
            ImGui::SetKeyOwner(k, ImGui::GetItemID());
        }
        if (movable) {
            bool ctrl = ImGui::GetIO().KeyCtrl;
            if (ImGui::IsKeyPressed(MENU_PAD_X, false) || (ctrl && ImGui::IsKeyPressed(ImGuiKey_UpArrow)))
                step = 3;
            if (ImGui::IsKeyPressed(MENU_PAD_Y, false) || (ctrl && ImGui::IsKeyPressed(ImGuiKey_DownArrow)))
                step = 4;
        }
    }
    if (foc) desc_set(g_focus, desc, key, m, disabled, label, v.note);
    if (hov) desc_set(g_hover, desc, key, m, disabled, label, v.note);
    if (!g_first.text[0]) desc_set(g_first, desc, key, m, disabled, label, v.note);
    // Back to the default: Y (the game's own "Default" button), Delete or Backspace on a setting's row
    if (foc && !off && !movable && m && v.setting &&
        (ImGui::IsKeyPressed(MENU_PAD_Y, false) || ImGui::IsKeyPressed(ImGuiKey_Delete, false) ||
         ImGui::IsKeyPressed(ImGuiKey_Backspace, false))) {
        const char *k = key;
        cfg_reset(&k, 1);
        opt_save_soon();
        step = 0;
    }

    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImFont *font = ImGui::GetFont();
    float pad = floorf(16 * g_scale), small = fs * 0.72f, in = fmaxf(2, floorf(3 * g_scale));
    ImVec2 a(p.x + mx, p.y), b(p.x + w - mx, p.y + h);                  // the strip
    bool boxed = v.value[0] != 0 || v.kind != ROW_TEXT;                 // a value box at the right
    float vw = boxed ? floorf((b.x - a.x) * 0.41f) : 0;
    ImVec2 va(b.x - vw, a.y + 1), vb(b.x - 1, b.y - 1);                 // the value's inset
    dl->AddRectFilled(a, b, hov || foc ? kRowHot : kRow);
    opt_grain(dl, a, b, 46);
    dl->AddRect(a, b, kRowEdge, 0, 0, 1.0f);
    int kind = v.kind == ROW_TEXT && v.arrows && v.value[0] ? ROW_STEP : v.kind;
    float ty = p.y + floorf((h - fs) * 0.5f);
    float label_right = boxed ? va.x - fs * 0.5f : b.x - pad;
    if (boxed) {
        dl->AddRectFilled(va, vb, kInset);
        ImVec2 ia(va.x + in, va.y + in), ib(vb.x - in, vb.y - in);
        if (kind == ROW_SWITCH_ON || kind == ROW_SWITCH_OFF) {
            float mid = floorf((ia.x + ib.x) * 0.5f);
            bool on = kind == ROW_SWITCH_ON;
            amber_bar(dl, ImVec2(on ? ia.x : mid, ia.y), ImVec2(on ? mid : ib.x, ib.y));
            text_centred(dl, ia.x, mid, ia.y, ib.y - ia.y, off ? kGrey : on ? kCream : kGrey, menu_text(UI_MENU_ON));
            text_centred(dl, mid, ib.x, ia.y, ib.y - ia.y, off ? kGrey : on ? kGrey : kCream, menu_text(UI_MENU_OFF));
            if (clicked && !off) {   // a click on the segment already chosen changes nothing
                float mouse = ImGui::GetIO().MousePos.x;
                if (mouse >= va.x && (mouse < mid) == on) step = 0;
            }
        } else if (kind == ROW_SLIDER) {
            // The game's amber segment as a gauge: filled from the left to the value's place, a cream tab at its edge,
            // the number over it. The mouse sets the value where it presses the bar and drags it; left / right step.
            float x0 = ia.x, x1 = ib.x, fx = floorf(x0 + (x1 - x0) * fminf(fmaxf(v.fill, 0), 1));
            dl->AddRectFilled(ia, ib, IM_COL32(52, 43, 28, 255));   // the empty part: the inset a shade lighter
            if (fx > x0) amber_bar(dl, ia, ImVec2(fx, ib.y));
            if (v.def_fill >= 0) {   // the default's place: a small notch at the bottom edge
                float dx = floorf(x0 + (x1 - x0) * fminf(fmaxf(v.def_fill, 0), 1)), nh = fmaxf(3, floorf((ib.y - ia.y) * 0.22f));
                dl->AddTriangleFilled(ImVec2(dx, ib.y - nh), ImVec2(dx - nh, ib.y), ImVec2(dx + nh, ib.y),
                                      IM_COL32(246, 239, 214, 150));
            }
            float tw = fmaxf(2, floorf(3 * g_scale));
            float kx = fminf(fmaxf(fx - tw * 0.5f, x0), x1 - tw);
            dl->AddRectFilled(ImVec2(kx - 1, ia.y - 1), ImVec2(kx + tw + 1, ib.y + 1), IM_COL32(30, 22, 12, 200));
            dl->AddRectFilled(ImVec2(kx, ia.y), ImVec2(kx + tw, ib.y), off ? kGrey : kCream);
            // the number with the game's soft shadow, on the wider side of the tab so the tab never cuts through it
            float gap = fs * 0.4f, tx0 = v.fill >= 0.5f ? x0 : kx + tw + gap, tx1 = v.fill >= 0.5f ? kx - gap : x1;
            text_centred(dl, tx0 + 1, tx1 + 1, ia.y + 1, ib.y - ia.y, IM_COL32(30, 20, 10, 190), v.value);
            text_centred(dl, tx0, tx1, ia.y, ib.y - ia.y, off ? kGrey : kCream, v.value);
            if (foc) g_bar[0] = ia, g_bar[1] = ib, g_bar_valid = true;
            ImGuiIO &io = ImGui::GetIO();
            ImGuiID id = ImGui::GetItemID();
            if (!off && ImGui::IsItemActive() && io.MouseDown[0]) {
                ImVec2 at = io.MouseClickedPos[0];
                if (g_drag_id == id || (at.x >= va.x && at.x <= vb.x && at.y >= va.y && at.y <= vb.y)) {
                    g_drag_id = id;
                    g_drag = (io.MousePos.x - x0) / fmaxf(1, x1 - x0);
                    g_drag = g_drag < 0 ? 0 : g_drag > 1 ? 1 : g_drag;
                }
            } else if (g_drag_id == id) {
                g_drag_id = 0;
            }
            // A mouse click (its release) set the value on the bar already, and on the label it only selects the row:
            // neither steps. A and Enter still step it.
            if (clicked && ImGui::IsMouseReleased(0)) step = 0;
        } else if (kind == ROW_STEP) {   // the amber bar with an arrow at each end; a click on one steps
            amber_bar(dl, ia, ib);
            float aw = fs * 0.34f, cy = (ia.y + ib.y) * 0.5f, lx = ia.x + fs * 0.35f, rx = ib.x - fs * 0.35f - aw;
            dl->AddTriangleFilled(ImVec2(lx, cy), ImVec2(lx + aw, cy - aw * 0.8f), ImVec2(lx + aw, cy + aw * 0.8f), kInset);
            dl->AddTriangleFilled(ImVec2(rx + aw, cy), ImVec2(rx, cy - aw * 0.8f), ImVec2(rx, cy + aw * 0.8f), kInset);
            text_centred(dl, lx + aw + fs * 0.3f, rx - fs * 0.3f, ia.y, ib.y - ia.y, kCream, v.value);
            if (clicked && !off) {
                float mouse = ImGui::GetIO().MousePos.x, zone = fs * 1.4f;
                if (mouse >= ia.x && mouse <= ia.x + zone) step = -1;
                else if (mouse >= ib.x - zone && mouse <= ib.x) step = 1;
            }
        } else {   // text: a button's result, a mod's state
            text_centred(dl, ia.x + in, ib.x - in, ia.y, ib.y - ia.y, off || v.is_default ? IM_COL32(201, 192, 165, 255) : kCream,
                         v.value);
        }
    } else {   // a row that only runs an action: an arrow at its right end
        float aw = fs * 0.34f, cy = (a.y + b.y) * 0.5f, rx = b.x - pad - aw;
        dl->AddTriangleFilled(ImVec2(rx + aw, cy), ImVec2(rx, cy - aw * 0.8f), ImVec2(rx, cy + aw * 0.8f), kCream);
        label_right = rx - fs * 0.4f;
    }
    if (movable) {   // up and down arrows left of the value; a click on one moves the row
        float aw = fs * 0.42f, gap = fs * 0.5f, cx = label_right - aw, uy = a.y + h * 0.2f, dy = b.y - h * 0.2f;
        ImU32 ac = off ? kGrey : kCream;
        dl->AddTriangleFilled(ImVec2(cx, uy + aw * 0.7f), ImVec2(cx + aw, uy + aw * 0.7f), ImVec2(cx + aw * 0.5f, uy), ac);
        dl->AddTriangleFilled(ImVec2(cx, dy - aw * 0.7f), ImVec2(cx + aw, dy - aw * 0.7f), ImVec2(cx + aw * 0.5f, dy), ac);
        if (clicked && !off) {
            ImVec2 mp = ImGui::GetIO().MousePos;
            if (mp.x >= cx - gap * 0.5f && mp.x <= cx + aw + gap * 0.5f) step = mp.y < (a.y + b.y) * 0.5f ? 3 : 4;
        }
        label_right = cx - gap;
    }
    // the note beside the label: why the row is fixed, "after restart", or the caller's own
    const char *note = disabled        ? disabled
                     : off             ? menu_text(UI_MENU_SET_BY_ENV)
                     : v.note && *v.note ? v.note
                     : m && !m->live ? menu_text(UI_MENU_AFTER_RESTART)
                                     : NULL;
    // The label comes first: a note that would cut it short is left out of the strip (the description pane says the
    // same, and a fixed row is dimmed anyway).
    if (note) {
        float lw = font->CalcTextSizeA(fs, FLT_MAX, 0, label).x, nw = font->CalcTextSizeA(small, FLT_MAX, 0, note).x;
        if (a.x + pad + lw + fs * 0.4f + nw > label_right) note = NULL;
    }
    if (note) {
        float tw = font->CalcTextSizeA(small, FLT_MAX, 0, note).x, room = (label_right - a.x) * 0.5f;
        float nx = fmaxf(label_right - tw, label_right - room);
        ImVec4 nclip(nx, a.y, label_right, b.y);
        dl->AddText(font, small, ImVec2(nx, a.y + floorf((h - small) * 0.5f)), kDim, note, NULL, 0, &nclip);
        label_right = nx - fs * 0.4f;
    }
    ImVec4 clip(a.x, a.y, label_right, b.y);
    dl->AddText(font, fs, ImVec2(a.x + pad + 1, ty + 1), IM_COL32(40, 28, 14, 150), label, NULL, 0, &clip);   // the game's soft shadow
    dl->AddText(font, fs, ImVec2(a.x + pad, ty), off ? IM_COL32(190, 180, 150, 255) : kCream, label, NULL, 0, &clip);
    if (v.setting && !v.is_default && !off) {   // changed from the default: the white corner
        float c = floorf(fs * 0.36f);
        dl->AddTriangleFilled(a, ImVec2(a.x + c, a.y), ImVec2(a.x, a.y + c), IM_COL32(255, 255, 255, 255));
    }
    if (off) dl->AddRectFilled(a, b, IM_COL32(20, 14, 8, 110));
    if (hov || foc) opt_select_frame(dl, boxed ? ImVec2(va.x, a.y) : a, b, g_scale);
    ImGui::PopID();
    return off ? 0 : step;
}

static int row(const char *key, UiTextId label, UiTextId desc, const RowView &v) {
    return row_text(key, menu_text(label), menu_text(desc), v, false);
}

int opt_text_row(const char *id, const char *label, const char *value, const char *note, const char *desc,
                 bool movable) {
    RowView v = {value ? value : "", false, false, note};
    return row_text(id, label, desc, v, movable);
}

int opt_text_choice(const char *id, UiTextId label, const char *value, const char *note, UiTextId desc) {
    return row(id, label, desc, RowView{value, false, true, note, ROW_STEP});
}

// ---- writing a setting: at once in memory (live settings follow), to rekarina.ini a moment later (opt_save_tick)
// The deadline is read and taken with atomics: the window thread's close path (menu_settings_flush) takes a pending write
// while the frame thread may be ticking it.
static uint64_t g_save_at;   // plat_now_ns deadline of the pending write; 0 none
static bool g_save_logged;

static void apply(const char *key, const char *value) {
    if (*value) {
        cfg_set(key, value);
    } else {   // the automatic choice: the key unset, its line removed from rekarina.ini
        const char *k = key;
        cfg_reset(&k, 1);
    }
    opt_save_soon();
}

void opt_save_soon(void) { __atomic_store_n(&g_save_at, plat_now_ns() + 400000000ull, __ATOMIC_RELEASE); }

int opt_save_pending(void) { return __atomic_load_n(&g_save_at, __ATOMIC_ACQUIRE) != 0; }

int opt_save_tick(int force) {
    uint64_t at = __atomic_load_n(&g_save_at, __ATOMIC_ACQUIRE);
    if (!at || (!force && plat_now_ns() < at)) return at != 0;
    if (!__atomic_compare_exchange_n(&g_save_at, &at, 0, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return __atomic_load_n(&g_save_at, __ATOMIC_ACQUIRE) != 0;   // changed meanwhile: the next tick has it
    if (!cfg_save() && !g_save_logged) {
        g_save_logged = true;
        rt_log("menu: could not write the settings file; the changed settings hold for this session");
    }
    return 0;
}

// ---- rows

bool opt_toggle(const char *key, UiTextId label, UiTextId desc, bool inverted) {
    const CfgMeta *m = cfg_meta(key);
    bool set = cfg_bool(key) != 0, def = m && strcmp(m->def, "0") != 0;
    bool on = set != inverted;   // what the row shows
    RowView v = {menu_text(on ? UI_MENU_ON : UI_MENU_OFF), set == def, false, NULL, on ? ROW_SWITCH_ON : ROW_SWITCH_OFF, true};
    if (!row(key, label, desc, v)) return false;   // any step flips it
    apply(key, set ? "0" : "1");
    return true;
}

bool opt_button(const char *id, UiTextId label, UiTextId desc, const char *value) {
    RowView v = {value ? value : "", false, false, NULL};
    return row(id, label, desc, v) == 2;   // A or a click; left and right do nothing
}

bool opt_toggle_text(const char *id, const char *label, const char *desc, bool on, const char *value) {
    // off is the plain state: its value drawn dim, without the "default" mark
    RowView v = {value ? value : menu_text(on ? UI_MENU_ON : UI_MENU_OFF), !on, false, NULL,
                 value ? ROW_TEXT : on ? ROW_SWITCH_ON : ROW_SWITCH_OFF};
    return row_text(id, label, desc, v, false) != 0;   // any step flips it
}

// A group's heading: its name between two hairlines, as the game divides its settings with a line.
void opt_group_text(const char *title) {
    ImGui::Dummy(ImVec2(0, floorf(3 * g_scale)));
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImFont *font = ImGui::GetFont();
    float fs = ImGui::GetFontSize() * 0.82f, w = ImGui::GetContentRegionAvail().x, mx = floorf(12 * g_scale);
    float tw = font->CalcTextSizeA(fs, FLT_MAX, 0, title).x, h = floorf(fs * 1.5f), cy = floorf(p.y + h * 0.5f);
    float x0 = p.x + mx, x1 = p.x + w - mx, tx = floorf((x0 + x1 - tw) * 0.5f), gap = fs * 0.8f;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const ImU32 line = IM_COL32(92, 74, 46, 255);
    if (tx - gap > x0) dl->AddLine(ImVec2(x0, cy), ImVec2(tx - gap, cy), line, 1.0f);
    if (tx + tw + gap < x1) dl->AddLine(ImVec2(tx + tw + gap, cy), ImVec2(x1, cy), line, 1.0f);
    ImVec4 clip(x0, p.y, x1, p.y + h);
    dl->AddText(font, fs, ImVec2(fmaxf(tx, x0), p.y + (h - fs) * 0.5f), IM_COL32(217, 207, 174, 255), title, NULL, 0, &clip);
    ImGui::Dummy(ImVec2(w, h));
}

void opt_text_str(const char *text, bool warn) {
    float mx = floorf(12 * g_scale);
    ImGui::Indent(mx);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - mx);
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(warn ? IM_COL32(255, 196, 120, 255) : IM_COL32(217, 207, 174, 255)), "%s", text);
    ImGui::PopTextWrapPos();
    ImGui::Unindent(mx);
}

void opt_group(UiTextId title) { opt_group_text(menu_text(title)); }

void opt_text(UiTextId text, bool warn) { opt_text_str(menu_text(text), warn); }

bool opt_choice_list(const char *key, UiTextId label, const char *const *values, const char *const *names, int n,
                     UiTextId desc, int cur, const char *note) {
    const CfgMeta *m = cfg_meta(key);
    const char *now = cfg_value(key);
    if (cur < 0)
        for (int i = 0; i < n; i++)
            if (!strcmp(values[i], now)) cur = i;
    int def = -1;
    for (int i = 0; m && i < n; i++)
        if (!strcmp(values[i], m->def)) def = i;
    RowView v = {cur >= 0 ? names[cur] : now, cur == def, true, note, ROW_STEP, m != NULL};
    int step = row(key, label, desc, v);
    if (!step || n <= 0) return false;
    int nv;
    if (cur < 0) {   // a value not in the list (set by hand, as fps_limit = 75): the next listed number that way
        double v = atof(now);
        nv = 0;
        for (int i = 0; v > 0 && i < n; i++) {
            double e = atof(values[i]);
            if (e <= 0) continue;
            if (step < 0 ? e < v : e > v) {
                nv = i;
                if (step >= 0) break;   // right / A: the first one above; left: the last one below
            }
        }
    } else if (step == 2) {
        nv = (cur + 1) % n;   // A / click: the next value, round to the first
    } else {
        nv = cur + step < 0 ? 0 : cur + step >= n ? n - 1 : cur + step;   // left / right stop at the ends
    }
    if (nv == cur) return false;
    apply(key, values[nv]);
    return true;
}

bool opt_choice(const char *key, UiTextId label, const UiTextId *names, int n, UiTextId desc, const char *note) {
    enum { MAX = 32 };
    const CfgMeta *m = cfg_meta(key);
    const char *values[MAX], *shown[MAX];
    char buf[256];
    int k = 0;
    if (m && !m->def[0]) values[k++] = "";   // the automatic choice
    snprintf(buf, sizeof buf, "%s", m && m->values ? m->values : "");
    for (char *s = buf, *e; *s && k < MAX; s = e) {
        e = strchr(s, '|');
        if (e) *e++ = 0;
        else e = s + strlen(s);
        values[k++] = s;
    }
    if (n > k) n = k;
    for (int i = 0; i < n; i++) shown[i] = menu_text(names[i]);
    return opt_choice_list(key, label, values, shown, n, desc, -1, note);
}

// The place of x on a slider's bar, 0..1, and the value at a place; a logarithmic bar spreads from log_lo (the bottom
// of its log part: lo when lo > 0, else twice the step) to hi, and keeps the first 2 % for lo itself (0 = off).
static double slider_fill(double x, double lo, double hi, double log_lo, bool log_scale) {
    if (!log_scale) return hi > lo ? (x - lo) / (hi - lo) : 0;
    if (x <= log_lo) return lo < log_lo && x <= lo ? 0 : 0.02;
    return 0.02 + 0.98 * log(x / log_lo) / log(hi / log_lo);
}

static double slider_value(double f, double lo, double hi, double log_lo, bool log_scale) {
    if (!log_scale) return lo + f * (hi - lo);
    if (f < 0.02) return lo;
    return log_lo * pow(hi / log_lo, (f - 0.02) / 0.98);
}

bool opt_slider(const char *key, UiTextId label, float lo, float hi, UiTextId desc, float step_f, const char *fmt,
                const char *note, const char *fallback, bool log_scale, double scale, double offset,
                double (*to_slider)(double), double (*from_slider)(double)) {
    // The step as the decimal the row was written with: a float such as 0.02f is 0.0199999996, so a value on its grid
    // (gamma 1.00 = 50 steps) divided to 50.0000011, past the allowance of the grid snap below, and left landed back
    // on the value it started from: the gamma and walk speed sliders never moved left (right did); a step just above
    // its decimal (0.05f) stuck the same way to the right, the mouse sensitivity from 3.40 up.
    char step_text[32];
    snprintf(step_text, sizeof step_text, "%.6g", (double)step_f);
    const double step = strtod(step_text, NULL);
    const double log_lo = lo > 0 ? lo : step * 2;
    const char *set = cfg_get(key);
    const char *src = fallback && !(set && *set) ? fallback : key;   // unset: show the value in effect, the fallback's
    const CfgMeta *m = cfg_meta(src);
    // A key without a table default of its own (the per-stick dead zones) defaults to its fallback: the "default" mark
    // follows the fallback's default, not 0, even once the key is set.
    if (fallback && m && !m->def[0]) m = cfg_meta(fallback);
    auto convert = [&](double value) { return to_slider ? to_slider(value) : value * scale + offset; };
    double v = convert(cfg_double(src));
    double def = m && m->def[0] ? convert(strtod(m->def, NULL)) : lo;
    char shown[48];
    snprintf(shown, sizeof shown, fmt, v == 0 ? 0.0 : v);
    RowView rv = {shown, fabs(v - def) < step * 0.01, true, note, ROW_SLIDER, true};
    rv.fill = (float)slider_fill(v, lo, hi, log_lo, log_scale);
    rv.def_fill = hi > lo && def >= lo && def <= hi ? (float)slider_fill(def, lo, hi, log_lo, log_scale) : -1;
    int s = row(key, label, desc, rv);
    double n;
    if (g_drag >= 0) {   // the mouse on the bar: the value under it, on the step grid
        n = round(slider_value(g_drag, lo, hi, log_lo, log_scale) / step) * step;
    } else {
        if (!s) return false;
        int steps = g_held && (hi - lo) > 40 * step ? 10 : 1;   // held: ten steps at a time over a long range
        // From a value off the step grid (a default such as 203 with steps of 5, a profile's 0.95) the first step lands
        // on the next grid point in that direction rather than one beyond the rounded value.
        double up = floor(v / step + 1e-6) * step, down = ceil(v / step - 1e-6) * step;
        n = s == 2 ? up + step : s < 0 ? down - step * steps : up + step * steps;
        if (s == 2 && n > hi + step * 0.01) n = lo;   // A past the top goes round; left / right stop at the ends
    }
    n = n < lo ? lo : n > hi ? hi : n;
    if (fabs(n - v) < step * 0.01) return false;
    char out[32];
    if (from_slider) snprintf(out, sizeof out, "%.9g", from_slider(n));
    else if (scale != 1 || offset != 0) snprintf(out, sizeof out, "%.9g", (n - offset) / scale);
    else if (step >= 1) snprintf(out, sizeof out, "%d", (int)lround(n));
    else snprintf(out, sizeof out, "%g", round(n / step) * step);
    // Landing on the fallback's value unsets the key (its line leaves rekarina.ini), so the row follows the fallback again.
    apply(key, fallback && fabs(n - convert(cfg_double(fallback))) < step * 0.01 ? "" : out);
    return true;
}

// ---- binding rows (the remapping rows of the Controls tab): a label and three cells, each its own nav item

// Text centred (left-aligned when centred is false) in [x0, x1] on a row from y to y + h, at the font size or smaller so
// a long name ("Нажатие правого стика", "Стрелка вправо") fits; under 65 % it is clipped instead.
static void fit_text(ImDrawList *dl, float x0, float x1, float y, float h, ImU32 col, const char *t, bool centred) {
    ImFont *font = ImGui::GetFont();
    float fs = ImGui::GetFontSize(), room = x1 - x0, tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    if (tw > room) {
        fs = fmaxf(fs * room / tw, fs * 0.65f);
        tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
    }
    float x = centred && tw < room ? x0 + (room - tw) * 0.5f : x0;
    ImVec4 clip(x0, y, x1, y + h);
    dl->AddText(font, fs, ImVec2(x, y + (h - fs) * 0.5f), col, t, NULL, 0, &clip);
}

// The label column and the three cells of a binding row of width w.
static void bind_columns(float x, float w, float cx[3], float *cw) {
    float mx = floorf(12 * g_scale), lw = floorf(w * 0.34f), gap = floorf(8 * g_scale);
    *cw = floorf((w - mx - lw - gap * 4) / 3);
    for (int c = 0; c < 3; c++) cx[c] = x + lw + gap + c * (*cw + gap);
}

void opt_bind_header(UiTextId c0, UiTextId c1, UiTextId c2) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x, cx[3], cw, fs = ImGui::GetFontSize() * 0.8f;
    bind_columns(p.x, w, cx, &cw);
    const UiTextId cols[3] = {c0, c1, c2};
    ImFont *font = ImGui::GetFont();
    for (int c = 0; c < 3; c++) {
        const char *t = menu_text(cols[c]);
        float tw = font->CalcTextSizeA(fs, FLT_MAX, 0, t).x;
        ImGui::GetWindowDrawList()->AddText(font, fs, ImVec2(cx[c] + (cw - tw) * 0.5f, p.y), kDim, t);
    }
    ImGui::Dummy(ImVec2(w, fs * 1.2f));
}

int opt_bind_row(const char *key, const char *label, UiTextId desc, const char *const cells[3], const bool enabled[3],
                 const char *note) {
    ImGui::PushID(key);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x, h = floorf(fs * 1.9f), cx[3], cw;
    bind_columns(p.x, w, cx, &cw);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    float cell_in = floorf(4 * g_scale);   // the cells sit inside the strip
    float pad = 6 * g_scale;
    {   // the strip under the label and the cells
        float mx = floorf(12 * g_scale);
        ImVec2 sa(p.x + mx, p.y), sb(p.x + w - mx, p.y + h);
        dl->AddRectFilled(sa, sb, kRow);
        opt_grain(dl, sa, sb, 46);
        dl->AddRect(sa, sb, kRowEdge, 0, 0, 1.0f);
    }
    fit_text(dl, p.x + 28 * g_scale, cx[0] - fs * 0.3f, p.y, h, kCream, label, false);
    int result = -1;
    bool row_foc = false, row_hov = false;
    for (int c = 0; c < 3; c++) {
        ImVec2 a(cx[c], p.y + cell_in), b(cx[c] + cw, p.y + h - cell_in);
        const char *t = cells[c] && cells[c][0] ? cells[c] : "-";
        if (!enabled[c]) {   // not a nav item: the cursor skips it
            dl->AddRectFilled(a, b, IM_COL32(30, 26, 18, 150));
            fit_text(dl, a.x + pad, b.x - pad, a.y, b.y - a.y, kGrey, t, true);
            continue;
        }
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(c);
        ImGui::PushStyleColor(ImGuiCol_NavCursor, ImVec4(0, 0, 0, 0));   // the cell's own border shows the selection
        bool clicked = ImGui::InvisibleButton("##cell", ImVec2(cw, h - cell_in * 2), ImGuiButtonFlags_EnableNav);
        ImGui::PopStyleColor();
        char id[80];
        snprintf(id, sizeof id, "%s#%d", key, c);
        bool to_key = g_focus_key[0] && !strcmp(id, g_focus_key);
        if (g_take_first || to_key) {
            ImGui::SetFocusID(ImGui::GetItemID(), ImGui::GetCurrentWindow());
            ImGui::SetNavCursorVisible(true);
            if (g_take_first) ImGui::SetScrollY(0);
            else ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
            g_take_first = false;
            if (to_key) g_focus_key[0] = 0;
        }
        bool hov = ImGui::IsItemHovered(), foc = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
        row_foc |= foc;
        row_hov |= hov;
        if (clicked) result = c;
        if (foc && (ImGui::IsKeyPressed(ImGuiKey_Delete, false) || ImGui::IsKeyPressed(ImGuiKey_Backspace, false) ||
                    ImGui::IsKeyPressed(MENU_PAD_X, false)))
            result = 10 + c;
        dl->AddRectFilled(a, b, kInset);
        fit_text(dl, a.x + pad, b.x - pad, a.y, b.y - a.y, kCream, t, true);
        if (hov || foc) opt_select_frame(dl, a, b, g_scale);
        ImGui::PopID();
    }
    char k[64];
    snprintf(k, sizeof k, "REKARINA_%s", key);
    if (row_foc) desc_set(g_focus, menu_text(desc), k, NULL, NULL, label);   // the button's name as the pane's title
    if (row_hov) desc_set(g_hover, menu_text(desc), k, NULL, NULL, label);
    if (!g_first.text[0]) desc_set(g_first, menu_text(desc), k, NULL, NULL, label);
    if (note && (row_foc || row_hov)) {   // e.g. what moved here, after the description
        Desc &d = row_hov ? g_hover : g_focus;
        size_t n = strlen(d.text);
        snprintf(d.text + n, sizeof d.text - n, "\n\n%s", note);
    }
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(w, h));
    ImGui::PopID();
    return result;
}
