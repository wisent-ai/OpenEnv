#!/usr/bin/env python3
"""Additional game-theory slides for the Kant Google Slides presentation.
Adds a game library overview slide showing domain breakdown across all
ten strategic domains, and a meta-gaming system slide describing rule
proposals, constitutional negotiation, gossip reputation, and the
six-rule governance catalog. Uses shared helpers from helpers.py for
consistent Wisent dark-theme styling across all presentation slides.
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

from common.games import GAMES
from constant_definitions.slides.layout import (
    POS_HALF, POS_ONE, POS_ONE_HALF, POS_TWO,
    POS_FOUR, POS_FIVE, POS_EIGHT, POS_NINE,
    GSLIDES_FILE_ID, PT_TITLE, PT_BODY, PT_SMALL, PT_LABEL, PT_STAT,
)
from helpers import (
    tsty, tbox, bg, del_slide, new_slide, get_svc,
    ACCENT_RGB, LEGEND_RGB, WHITE_RGB,
)

GL_TITLE_ID, GL_STAT_ID, GL_SUB_ID = "gl_title", "gl_stat", "gl_sub"
GL_LIST_ID = "gl_list"
GAME_COUNT = str(len(GAMES))
GAME_LIST = ", ".join(
    k.replace("_", " ").title() for k in sorted(GAMES.keys())
)

MG_TITLE_ID, MG_RULES_HDR_ID = "mg_title", "mg_rules_hdr"
MG_RULES_BODY_ID, MG_VARIANTS_HDR_ID = "mg_rules_body", "mg_variants_hdr"
MG_VARIANTS_BODY_ID, MG_FOOTER_ID = "mg_variants_body", "mg_footer"

MG_RULES_TEXT = (
    "\u2022  Equal Split \u2014 divide payoffs equally between players\n"
    "\u2022  Cooperation Bonus \u2014 reward for cooperative actions\n"
    "\u2022  Defection Penalty \u2014 punish defection\n"
    "\u2022  Minimum Guarantee \u2014 payoff floor for both players\n"
    "\u2022  Ban Defect \u2014 heavy penalty enforcing cooperation\n"
    "\u2022  None \u2014 base game payoffs unchanged"
)
MG_VARIANTS_TEXT = (
    "\u2022  Rule Proposal \u2014 simultaneous binding rule proposals;\n"
    "    agreement activates the chosen rule\u2019s payoff transform\n"
    "\u2022  Rule Signal \u2014 non-binding signals visible in history;\n"
    "    payoffs always come from the base game\n"
    "\u2022  Constitutional \u2014 multi-round negotiation; first\n"
    "    agreement locks in a rule for all subsequent rounds\n"
    "\u2022  Proposer\u2013Responder \u2014 asymmetric: one proposes,\n"
    "    other accepts or rejects the governance rule\n"
    "\u2022  Gossip \u2014 rate opponents as trustworthy, untrustworthy,\n"
    "    or neutral; reputation builds across episodes"
)
MG_FOOTER_TEXT = (
    "Pre-registered for PD, Stag Hunt, and Hawk-Dove  \u00b7  "
    "Dynamic composition for any base game"
)


def add_game_library_slide(svc):
    """Add a slide with big game count and full list of game names."""
    del_slide(svc, GL_TITLE_ID)
    sid = new_slide(svc)
    print("Adding game library slide (" + GAME_COUNT + " games)...")
    reqs = [
        bg(sid),
        tbox(GL_TITLE_ID, sid, POS_HALF, POS_HALF / POS_TWO, POS_NINE, POS_HALF),
        {"insertText": {"objectId": GL_TITLE_ID, "text": "Game Library"}},
        tsty(GL_TITLE_ID, PT_TITLE, ACCENT_RGB, bold=True),
        tbox(GL_STAT_ID, sid, POS_ONE, POS_ONE, POS_EIGHT, POS_ONE_HALF),
        {"insertText": {"objectId": GL_STAT_ID, "text": GAME_COUNT}},
        tsty(GL_STAT_ID, PT_STAT, ACCENT_RGB, bold=True),
        tbox(GL_SUB_ID, sid, POS_ONE, POS_TWO, POS_EIGHT, POS_HALF),
        {"insertText": {"objectId": GL_SUB_ID, "text": "unique games"}},
        tsty(GL_SUB_ID, PT_BODY, WHITE_RGB),
        tbox(GL_LIST_ID, sid, POS_HALF, POS_TWO + POS_ONE,
             POS_NINE, POS_TWO + POS_HALF),
        {"insertText": {"objectId": GL_LIST_ID, "text": GAME_LIST}},
        tsty(GL_LIST_ID, PT_SMALL, LEGEND_RGB),
    ]
    svc.presentations().batchUpdate(
        presentationId=GSLIDES_FILE_ID, body={"requests": reqs}).execute()
    print("Added game library slide.")


def add_meta_gaming_slide(svc):
    """Add a slide describing the meta-gaming governance system."""
    del_slide(svc, MG_TITLE_ID)
    sid = new_slide(svc)
    print("Adding meta-gaming slide...")
    reqs = [
        bg(sid),
        tbox(MG_TITLE_ID, sid, POS_HALF, POS_HALF / POS_TWO, POS_NINE, POS_ONE),
        {"insertText": {"objectId": MG_TITLE_ID,
            "text": "Meta-Gaming: Agents Change the Rules"}},
        tsty(MG_TITLE_ID, PT_TITLE, ACCENT_RGB, bold=True),
        tbox(MG_RULES_HDR_ID, sid, POS_HALF, POS_ONE_HALF, POS_FOUR, POS_HALF),
        {"insertText": {"objectId": MG_RULES_HDR_ID, "text": "Rule Catalog"}},
        tsty(MG_RULES_HDR_ID, PT_BODY, ACCENT_RGB, bold=True),
        tbox(MG_RULES_BODY_ID, sid, POS_HALF, POS_TWO, POS_FOUR, POS_TWO),
        {"insertText": {"objectId": MG_RULES_BODY_ID, "text": MG_RULES_TEXT}},
        tsty(MG_RULES_BODY_ID, PT_SMALL, WHITE_RGB),
        tbox(MG_VARIANTS_HDR_ID, sid, POS_FIVE, POS_ONE_HALF,
             POS_FOUR + POS_HALF, POS_HALF),
        {"insertText": {"objectId": MG_VARIANTS_HDR_ID,
            "text": "Interaction Paradigms"}},
        tsty(MG_VARIANTS_HDR_ID, PT_BODY, ACCENT_RGB, bold=True),
        tbox(MG_VARIANTS_BODY_ID, sid, POS_FIVE, POS_TWO,
             POS_FOUR + POS_HALF, POS_TWO + POS_HALF),
        {"insertText": {"objectId": MG_VARIANTS_BODY_ID,
            "text": MG_VARIANTS_TEXT}},
        tsty(MG_VARIANTS_BODY_ID, PT_SMALL, WHITE_RGB),
        tbox(MG_FOOTER_ID, sid, POS_HALF, POS_FOUR + POS_HALF,
             POS_NINE, POS_HALF),
        {"insertText": {"objectId": MG_FOOTER_ID, "text": MG_FOOTER_TEXT}},
        tsty(MG_FOOTER_ID, PT_SMALL, LEGEND_RGB),
    ]
    svc.presentations().batchUpdate(
        presentationId=GSLIDES_FILE_ID, body={"requests": reqs}).execute()
    print("Added meta-gaming slide.")


def main():
    svc = get_svc()
    add_game_library_slide(svc)
    add_meta_gaming_slide(svc)


if __name__ == "__main__":
    main()
