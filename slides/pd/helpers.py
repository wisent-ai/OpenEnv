"""Shared helpers for adding styled slides to the Kant Google Slides
presentation via the Slides API. Provides text styling, text box
creation, background fill, slide creation and deletion utilities.
All styling follows the Wisent dark theme: dark background, accent
green headers, grid-colored table cells, and Hubot Sans typography.
These helpers are imported by restore.py and slides_extra.py to
avoid duplication across slide-building scripts.
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

from constant_definitions.slides.layout import (
    POS_ONE, EMU_PER_INCH, GSLIDES_FILE_ID, FONT_NAME,
    ACCENT_R, ACCENT_G, ACCENT_B, DARK_R, DARK_G, DARK_B,
    GRID_R, GRID_G, GRID_B, LEGEND_R, LEGEND_G, LEGEND_B,
    WHITE_VAL,
)

ZERO_VAL = WHITE_VAL - WHITE_VAL


def _emu(inches):
    return int(inches * EMU_PER_INCH)


def _rgb(r, g, b):
    return {"red": r / WHITE_VAL, "green": g / WHITE_VAL, "blue": b / WHITE_VAL}


DARK_RGB = _rgb(DARK_R, DARK_G, DARK_B)
ACCENT_RGB = _rgb(ACCENT_R, ACCENT_G, ACCENT_B)
GRID_RGB = _rgb(GRID_R, GRID_G, GRID_B)
LEGEND_RGB = _rgb(LEGEND_R, LEGEND_G, LEGEND_B)
WHITE_RGB = _rgb(WHITE_VAL, WHITE_VAL, WHITE_VAL)


def tsty(obj_id, pt, rgb, bold=False, cell_loc=None):
    """Build an updateTextStyle request."""
    req = {"updateTextStyle": {"objectId": obj_id, "style": {
        "fontFamily": FONT_NAME,
        "fontSize": {"magnitude": pt, "unit": "PT"}, "bold": bold,
        "foregroundColor": {"opaqueColor": {"rgbColor": rgb}}},
        "textRange": {"type": "ALL"},
        "fields": "fontFamily,fontSize,bold,foregroundColor.opaqueColor.rgbColor"}}
    if cell_loc is not None:
        req["updateTextStyle"]["cellLocation"] = cell_loc
    return req


def tbox(oid, sid, x, y, w, h):
    """Build a createShape TEXT_BOX request."""
    return {"createShape": {"objectId": oid, "shapeType": "TEXT_BOX",
        "elementProperties": {"pageObjectId": sid,
            "size": {"width": {"magnitude": _emu(w), "unit": "EMU"},
                     "height": {"magnitude": _emu(h), "unit": "EMU"}},
            "transform": {"scaleX": POS_ONE, "scaleY": POS_ONE,
                "translateX": _emu(x), "translateY": _emu(y),
                "unit": "EMU"}}}}


def bg(sid):
    """Build an updatePageProperties request for dark background."""
    return {"updatePageProperties": {"objectId": sid,
        "pageProperties": {"pageBackgroundFill": {
            "solidFill": {"color": {"rgbColor": DARK_RGB}}}},
        "fields": "pageBackgroundFill.solidFill.color"}}


def del_slide(svc, marker_id):
    """Delete a slide containing an element with the given ID."""
    pres = svc.presentations().get(presentationId=GSLIDES_FILE_ID).execute()
    for s in pres.get("slides", []):
        ids = {e["objectId"] for e in s.get("pageElements", [])}
        if marker_id in ids:
            svc.presentations().batchUpdate(presentationId=GSLIDES_FILE_ID,
                body={"requests": [{"deleteObject": {
                    "objectId": s["objectId"]}}]}).execute()
            print("Deleted old slide: " + s["objectId"])
            return


def new_slide(svc):
    """Create a blank slide at the end and return its page object ID."""
    pres = svc.presentations().get(presentationId=GSLIDES_FILE_ID).execute()
    n = len(pres.get("slides", []))
    resp = svc.presentations().batchUpdate(presentationId=GSLIDES_FILE_ID,
        body={"requests": [{"createSlide": {"insertionIndex": n}}]}).execute()
    return resp["replies"][ZERO_VAL]["createSlide"]["objectId"]


def get_svc():
    """Return authenticated Slides API service."""
    from drive_client import get_credentials
    from googleapiclient.discovery import build as gbuild
    return gbuild("slides", "v1", credentials=get_credentials())
