#!/usr/bin/env python3
"""Add styled game-theory slides to the Kant Google Slides presentation.
Supports Prisoner's Dilemma payoff matrix and Formal Game Definition.
Uses Slides API batchUpdate for granular edits without touching other
slides. Each slide is identified by element IDs and can be deleted and
recreated idempotently. Full Wisent dark-theme styling: dark background,
accent-green headers, grid-colored table cells, Hubot Sans typography.
"""
import sys
from pathlib import Path

sys.path.insert(
    next(iter(range(bool(True)))),
    "/Users/lukaszbartoszcze/Documents/CodingProjects"
    "/Wisent/growth-tactics/google_drive",
)
_KANT_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(next(iter(range(bool(True)))), _KANT_DIR)

from drive_client import get_credentials
from googleapiclient.discovery import build

from constant_definitions.slides.layout import (
    PD_CC, PD_CD, PD_DC, PD_DD,
    PD_NE_LABEL, PD_PO_LABEL, PD_EXPLANATION_BODY,
    PLAYER_ROW_LABEL, PLAYER_COL_LABEL,
    POS_HALF, POS_ONE, POS_ONE_HALF, POS_TWO,
    POS_FOUR, POS_FIVE, POS_EIGHT, POS_NINE,
    EMU_PER_INCH, GSLIDES_FILE_ID, FONT_NAME,
    ACCENT_R, ACCENT_G, ACCENT_B, DARK_R, DARK_G, DARK_B,
    GRID_R, GRID_G, GRID_B, LEGEND_R, LEGEND_G, LEGEND_B,
    WHITE_VAL, PT_TITLE, PT_BODY, PT_SMALL,
)

HEADER_VALS = ["", "Cooperate", "Defect"]
ROW_COOP_VALS = ["Cooperate", PD_CC, PD_CD]
ROW_DEF_VALS = ["Defect", PD_DC, PD_DD]
CAPTION = PD_NE_LABEL + "  \u00b7  " + PD_PO_LABEL + "\n" + PD_EXPLANATION_BODY
ALL_ROWS = [HEADER_VALS, ROW_COOP_VALS, ROW_DEF_VALS]
NUM_ROWS = len(ALL_ROWS)
NUM_COLS = len(HEADER_VALS)
ZERO = NUM_ROWS - NUM_ROWS

TBL_ID, PD_TITLE_ID = "pd_table", "pd_title"
PD_CAP_ID, ROW_LBL_ID, COL_LBL_ID = "pd_caption", "pd_row_label", "pd_col_label"
GD_TITLE_ID, GD_BODY_ID = "gd_title", "gd_body"
GD_KC_HDR_ID, GD_KC_BODY_ID = "gd_kc_hdr", "gd_kc_body"
VR_TITLE_ID, VR_BODY_ID = "vr_title", "vr_body"
VR_META_HDR_ID, VR_META_BODY_ID = "vr_meta_hdr", "vr_meta_body"
VR_MOD_HDR_ID, VR_MOD_BODY_ID = "vr_mod_hdr", "vr_mod_body"

GD_DEF_TEXT = (
    "A normal-form game \u0393 = \u2329N, S, u\u232a consists of:\n\n"
    "\u2022  N \u2014 a finite set of players\n"
    "\u2022  S \u2014 for each player, a set of available strategies\n"
    "\u2022  u \u2014 for each player, a payoff function over strategy profiles\n\n"
    "A strategy profile assigns one strategy to each player.\n"
    "The payoff function maps each profile to a real-valued outcome."
)
GD_KC_TEXT = (
    "Nash Equilibrium \u2014 no player benefits from changing strategy alone\n"
    "Pareto Optimality \u2014 no outcome makes every player better off\n"
    "Dominant Strategy \u2014 best response regardless of others\u2019 play"
)

VR_BODY_TEXT = (
    "\u2022  Cheap Talk \u2014 non-binding message phase before action\n"
    "\u2022  Exit Option \u2014 safe intermediate payoff if either exits\n"
    "\u2022  Binding Commitment \u2014 lock into an action at a cost\n"
    "\u2022  Noisy Actions \u2014 trembling-hand random action replacement\n"
    "\u2022  Noisy Payoffs \u2014 Gaussian noise on observed outcomes\n"
    "\u2022  Self-Play \u2014 model plays against a frozen copy of itself\n"
    "\u2022  Cross-Model \u2014 model plays against a different model"
)
VR_META_TEXT = (
    "\u2022  Rule Proposal \u2014 simultaneous binding rule proposals\n"
    "\u2022  Rule Signal \u2014 non-binding rule signals visible in history\n"
    "\u2022  Constitutional \u2014 multi-round negotiation with rule lock-in\n"
    "\u2022  Proposer\u2013Responder \u2014 one proposes a rule, other accepts or rejects"
)
VR_COMPOSE_TEXT = (
    "compose_game(base, *modifiers)  \u2014  modifiers stack and compose"
)


def _emu(inches):
    return int(inches * EMU_PER_INCH)


def _rgb(r, g, b):
    return {"red": r / WHITE_VAL, "green": g / WHITE_VAL, "blue": b / WHITE_VAL}


DARK_RGB = _rgb(DARK_R, DARK_G, DARK_B)
ACCENT_RGB = _rgb(ACCENT_R, ACCENT_G, ACCENT_B)
GRID_RGB = _rgb(GRID_R, GRID_G, GRID_B)
LEGEND_RGB = _rgb(LEGEND_R, LEGEND_G, LEGEND_B)
WHITE_RGB = _rgb(WHITE_VAL, WHITE_VAL, WHITE_VAL)


def _tsty(obj_id, pt, rgb, bold=False, cell_loc=None):
    req = {"updateTextStyle": {"objectId": obj_id, "style": {
        "fontFamily": FONT_NAME,
        "fontSize": {"magnitude": pt, "unit": "PT"}, "bold": bold,
        "foregroundColor": {"opaqueColor": {"rgbColor": rgb}}},
        "textRange": {"type": "ALL"},
        "fields": "fontFamily,fontSize,bold,foregroundColor.opaqueColor.rgbColor"}}
    if cell_loc is not None:
        req["updateTextStyle"]["cellLocation"] = cell_loc
    return req


def _tbox(oid, sid, x, y, w, h):
    return {"createShape": {"objectId": oid, "shapeType": "TEXT_BOX",
        "elementProperties": {"pageObjectId": sid,
            "size": {"width": {"magnitude": _emu(w), "unit": "EMU"},
                     "height": {"magnitude": _emu(h), "unit": "EMU"}},
            "transform": {"scaleX": POS_ONE, "scaleY": POS_ONE,
                "translateX": _emu(x), "translateY": _emu(y),
                "unit": "EMU"}}}}


def _bg(sid):
    return {"updatePageProperties": {"objectId": sid,
        "pageProperties": {"pageBackgroundFill": {
            "solidFill": {"color": {"rgbColor": DARK_RGB}}}},
        "fields": "pageBackgroundFill.solidFill.color"}}


def _del_slide(svc, marker_id):
    pres = svc.presentations().get(presentationId=GSLIDES_FILE_ID).execute()
    for s in pres.get("slides", []):
        ids = {e["objectId"] for e in s.get("pageElements", [])}
        if marker_id in ids:
            svc.presentations().batchUpdate(presentationId=GSLIDES_FILE_ID,
                body={"requests": [{"deleteObject": {
                    "objectId": s["objectId"]}}]}).execute()
            print("Deleted old slide: " + s["objectId"])
            return


def _new_slide(svc):
    pres = svc.presentations().get(presentationId=GSLIDES_FILE_ID).execute()
    n = len(pres.get("slides", []))
    resp = svc.presentations().batchUpdate(presentationId=GSLIDES_FILE_ID,
        body={"requests": [{"createSlide": {"insertionIndex": n}}]}).execute()
    return resp["replies"][ZERO]["createSlide"]["objectId"]


def add_pd_slide(svc):
    _del_slide(svc, PD_TITLE_ID)
    sid = _new_slide(svc)
    print("Adding styled PD slide...")
    reqs = [
        _bg(sid),
        _tbox(PD_TITLE_ID, sid, POS_HALF, POS_HALF / POS_TWO, POS_NINE, POS_ONE),
        {"insertText": {"objectId": PD_TITLE_ID, "text": "Prisoner\u2019s Dilemma"}},
        _tsty(PD_TITLE_ID, PT_TITLE, ACCENT_RGB, bold=True),
        _tbox(COL_LBL_ID, sid, POS_TWO + POS_ONE, POS_ONE, POS_FOUR, POS_HALF),
        {"insertText": {"objectId": COL_LBL_ID, "text": PLAYER_COL_LABEL}},
        _tsty(COL_LBL_ID, PT_BODY, LEGEND_RGB),
        _tbox(ROW_LBL_ID, sid, POS_HALF, POS_TWO + POS_HALF, POS_ONE_HALF, POS_HALF),
        {"insertText": {"objectId": ROW_LBL_ID, "text": PLAYER_ROW_LABEL}},
        _tsty(ROW_LBL_ID, PT_BODY, LEGEND_RGB),
        {"createTable": {"objectId": TBL_ID,
            "elementProperties": {"pageObjectId": sid,
                "size": {"width": {"magnitude": _emu(POS_FIVE), "unit": "EMU"},
                         "height": {"magnitude": _emu(POS_TWO), "unit": "EMU"}},
                "transform": {"scaleX": POS_ONE, "scaleY": POS_ONE,
                    "translateX": _emu(POS_TWO),
                    "translateY": _emu(POS_ONE_HALF), "unit": "EMU"}},
            "rows": NUM_ROWS, "columns": NUM_COLS}},
        {"updateTableCellProperties": {"objectId": TBL_ID,
            "tableRange": {"location": {"rowIndex": ZERO, "columnIndex": ZERO},
                "rowSpan": NUM_ROWS, "columnSpan": NUM_COLS},
            "tableCellProperties": {"tableCellBackgroundFill": {
                "solidFill": {"color": {"rgbColor": GRID_RGB}}}},
            "fields": "tableCellBackgroundFill.solidFill.color"}},
        _tbox(PD_CAP_ID, sid, POS_ONE, POS_FOUR, POS_EIGHT, POS_ONE),
        {"insertText": {"objectId": PD_CAP_ID, "text": CAPTION}},
        _tsty(PD_CAP_ID, PT_SMALL, LEGEND_RGB),
    ]
    for ri, row in enumerate(ALL_ROWS):
        for ci, val in enumerate(row):
            if val:
                reqs.append({"insertText": {"objectId": TBL_ID,
                    "cellLocation": {"rowIndex": ri, "columnIndex": ci},
                    "text": val}})
    for ri, row in enumerate(ALL_ROWS):
        for ci, val in enumerate(row):
            if val:
                is_hdr = (ri == ZERO) or (ci == ZERO)
                reqs.append(_tsty(TBL_ID, PT_BODY,
                    ACCENT_RGB if is_hdr else WHITE_RGB, bold=is_hdr,
                    cell_loc={"rowIndex": ri, "columnIndex": ci}))
    svc.presentations().batchUpdate(
        presentationId=GSLIDES_FILE_ID, body={"requests": reqs}).execute()
    print("Added styled PD slide.")


def add_game_def_slide(svc):
    _del_slide(svc, GD_TITLE_ID)
    sid = _new_slide(svc)
    print("Adding game definition slide...")
    reqs = [
        _bg(sid),
        _tbox(GD_TITLE_ID, sid, POS_HALF, POS_HALF / POS_TWO, POS_NINE, POS_ONE),
        {"insertText": {"objectId": GD_TITLE_ID, "text": "Formal Game Definition"}},
        _tsty(GD_TITLE_ID, PT_TITLE, ACCENT_RGB, bold=True),
        _tbox(GD_BODY_ID, sid, POS_ONE, POS_ONE_HALF, POS_EIGHT, POS_TWO),
        {"insertText": {"objectId": GD_BODY_ID, "text": GD_DEF_TEXT}},
        _tsty(GD_BODY_ID, PT_BODY, WHITE_RGB),
        _tbox(GD_KC_HDR_ID, sid, POS_ONE, POS_FOUR - POS_HALF, POS_EIGHT, POS_HALF),
        {"insertText": {"objectId": GD_KC_HDR_ID, "text": "Key Concepts"}},
        _tsty(GD_KC_HDR_ID, PT_BODY, ACCENT_RGB, bold=True),
        _tbox(GD_KC_BODY_ID, sid, POS_ONE, POS_FOUR, POS_EIGHT, POS_ONE),
        {"insertText": {"objectId": GD_KC_BODY_ID, "text": GD_KC_TEXT}},
        _tsty(GD_KC_BODY_ID, PT_SMALL, LEGEND_RGB),
    ]
    svc.presentations().batchUpdate(
        presentationId=GSLIDES_FILE_ID, body={"requests": reqs}).execute()
    print("Added game definition slide.")


def add_variants_slide(svc):
    _del_slide(svc, VR_TITLE_ID)
    sid = _new_slide(svc)
    print("Adding composable variants slide...")
    reqs = [
        _bg(sid),
        _tbox(VR_TITLE_ID, sid, POS_HALF, POS_HALF / POS_TWO, POS_NINE, POS_ONE),
        {"insertText": {"objectId": VR_TITLE_ID,
            "text": "Composable Game Variants"}},
        _tsty(VR_TITLE_ID, PT_TITLE, ACCENT_RGB, bold=True),
        _tbox(VR_BODY_ID, sid, POS_ONE, POS_ONE_HALF, POS_FOUR, POS_TWO),
        {"insertText": {"objectId": VR_BODY_ID, "text": VR_BODY_TEXT}},
        _tsty(VR_BODY_ID, PT_SMALL, WHITE_RGB),
        _tbox(VR_META_HDR_ID, sid, POS_FIVE, POS_ONE_HALF, POS_FOUR, POS_HALF),
        {"insertText": {"objectId": VR_META_HDR_ID, "text": "Meta-Gaming"}},
        _tsty(VR_META_HDR_ID, PT_BODY, ACCENT_RGB, bold=True),
        _tbox(VR_META_BODY_ID, sid, POS_FIVE, POS_TWO, POS_FOUR, POS_ONE_HALF),
        {"insertText": {"objectId": VR_META_BODY_ID, "text": VR_META_TEXT}},
        _tsty(VR_META_BODY_ID, PT_SMALL, WHITE_RGB),
        _tbox(VR_MOD_HDR_ID, sid, POS_ONE, POS_FOUR, POS_EIGHT, POS_HALF),
        {"insertText": {"objectId": VR_MOD_HDR_ID, "text": "Composition"}},
        _tsty(VR_MOD_HDR_ID, PT_BODY, ACCENT_RGB, bold=True),
        _tbox(VR_MOD_BODY_ID, sid,
              POS_ONE, POS_FOUR + POS_HALF, POS_EIGHT, POS_HALF),
        {"insertText": {"objectId": VR_MOD_BODY_ID, "text": VR_COMPOSE_TEXT}},
        _tsty(VR_MOD_BODY_ID, PT_SMALL, LEGEND_RGB),
    ]
    svc.presentations().batchUpdate(
        presentationId=GSLIDES_FILE_ID, body={"requests": reqs}).execute()
    print("Added composable variants slide.")


def main():
    creds = get_credentials()
    svc = build("slides", "v1", credentials=creds)
    add_variants_slide(svc)


if __name__ == "__main__":
    main()
