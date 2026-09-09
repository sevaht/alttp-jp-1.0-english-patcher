#!/usr/bin/env python3
"""Generate the whole English program: one :class:`Rom` loaded, edited, saved.

:func:`build` loads the pristine JP disassembly as a single whole-program
:class:`~snes_assembly_parser.Rom` and does everything on it: it
:meth:`~Rom.add`\\ s each relocated subsystem (built below as a
:class:`graft.Relocation` -- its placed, ``EN_``-namespaced pieces, each
knowing the ``org`` it lands at), :func:`hooks <apply_base_edits>` the base
banks to reach those copies *by name or ``#_`` address anchor* (never by which
file a routine lives in), and wires ``main.asm``. :meth:`~Rom.write` then emits
the entire fork: the base banks hooked in place, the graft grouped into
``bank_20`` .. ``bank_2E`` beside them (see :func:`graft.bank_header`), and
every untouched unit round-tripped byte-for-byte.

The relocated code is pulled *by name* from the US/JP disassembly; the binary
assets (font, graphics, palette) are US-ROM byte slices ``incbin``\\ 'd from
``bin/gfx/`` (alongside the base disassembly's own binaries there; regenerated
on the target by ``binextract-us.py``, see :mod:`us_assets`). Each subsystem
is self-contained -- its constants, helpers, and edits live in its own
function -- so there is nothing file-wide to trace through.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import jp_credits_font_asset, us_assets, us_credits_font_asset
from .graft import Relocation, bank_header, mirror, require_start, substitute
from .snes_assembly_parser import (
    DEFAULT_ROW_WIDTH,
    Assembly,
    Block,
    Edit,
    LandingPad,
    Line,
    Pool,
    Rom,
    datas,
    dbr_trampolines,
    free_space,
    incbin_line,
    instruction,
    instructions,
    note,
    notes,
)

USDASM = Path("../usdasm")
JPDASM = Path("../jpdasm")

#: Rom.write()'s default null_padbyte_threshold: a free-ROM gap over this many
#: bytes is filled with a single padbyte/pad jump instead of explicit db $FF
#: rows. 0 disables padbyte entirely (always explicit rows).
DEFAULT_NULL_PADBYTE_THRESHOLD = 128

#: nop_fill()'s default threshold: a JMP-past-a-gap's *interior* dead bytes
#: over this many bytes are filled with padbyte/pad instead of one NOP per
#: byte. Distinct from DEFAULT_NULL_PADBYTE_THRESHOLD (and much smaller) since
#: these gaps are a single edit site's byte-neutral filler, not a whole
#: free-ROM region -- most are tiny. 0 disables padbyte entirely (always
#: explicit NOPs).
DEFAULT_NOP_PADBYTE_THRESHOLD = 64


@dataclass(frozen=True)
class Sources:
    """The two input disassemblies the graft pulls from, bundled so each
    subsystem takes one argument instead of two in an easy-to-swap order."""

    us: Rom
    jp: Rom


# A landing-pad block's header comment: the real routines live a bank away, the
# freed JP entry-point names stay here and forward with a JSL/RTS bridge.
_REDIRECT_HEADER = (
    "; [ENG-REDIRECT] Landing pads: the real routines run in bank ${bank} "
    "(english/{file}).",
    "; They keep the JP entry-point names so unmodified same-bank JSR callers "
    "land here and",
    "; forward across the bank with a register-transparent JSL/RTS bridge (an "
    "argument in",
    "; A/X/Y passes straight through); the JP originals are preserved "
    "above as UNREACHABLE_*.",
)


def _redirect(bank: str, file: str) -> tuple[str, ...]:
    return tuple(
        line.format(bank=bank, file=file) for line in _REDIRECT_HEADER
    )


def _resource_lines(filename: str) -> list[str]:
    """A hand-written 65816 package resource (``resources/<filename>``), read
    verbatim as lines. Label namespacing / placement is the caller's job."""
    text = (
        resources.files("alttp_jp_english_patcher")
        .joinpath("resources", filename)
        .read_text(encoding="utf-8")
    )
    return text.splitlines()


# In-place save migrator (packed into the file-select bank, invoked at boot):
# US and vanilla-Japanese save slots are converted to our 6-word-name-at-$3D5
# format so they load and play.
def _save_migration_lines() -> list[str]:
    return _resource_lines("save_migration.asm")


# Module04_NameFile submodule 4: names an already-valid, blank-named save
# (an unconvertible JP import) without erasing it -- see edit_check_blank_name
# and StampNewFileTag.
def _name_fix_lines() -> list[str]:
    return _resource_lines("name_fix.asm")


def _row_fill(directive: str, value: str, count: int) -> list[Line]:
    """``count`` copies of ``value`` under ``directive`` (``db``/``dw``),
    wrapped at :data:`~.snes_assembly_parser.DEFAULT_ROW_WIDTH` per line --
    the same width :func:`~.snes_assembly_parser.free_space`'s own explicit
    rows use, so a hand-built filler row (one this codebase constructs
    itself, rather than one sized/anchored from a parsed disassembly) reads
    the same way."""
    values = [value] * count
    return datas(
        f"{directive} " + ", ".join(values[i : i + DEFAULT_ROW_WIDTH])
        for i in range(0, count, DEFAULT_ROW_WIDTH)
    )


def nop_fill(
    count: int,
    tag: str,
    comment: str,
    *,
    nop_padbyte_threshold: int = DEFAULT_NOP_PADBYTE_THRESHOLD,
) -> list[Line]:
    """Baseline stand-in for a full-only insertion: same byte count, so an
    edited routine's size -- and every anchor after it -- matches the full
    build exactly.

    A wall of one ``NOP`` per byte reads worse the bigger ``count`` gets, so
    only 1 byte stays a lone ``NOP``; anything bigger jumps past the gap, then
    fills it: a single ``db`` row of ``$EA`` at or under
    ``nop_padbyte_threshold``, else ``fillbyte``/``fill`` -- mirroring
    ``free_space``'s own small-gap-explicit / large-gap-padbyte split, just at
    a much smaller default cutoff, since these gaps are one edit site's
    filler, not a whole free-ROM region (``nop_padbyte_threshold <= 0``
    disables that, same as ``free_space``'s ``null_padbyte_threshold``).
    Either way the dead bytes read as harmless NOPs on the off chance they are
    ever reached.

    The "jump past the gap" is ``BRA +0``/``BRL +gap`` -- asar's *literal*
    relative-displacement syntax (``BRA +0`` assembles straight to
    ``$80 $00``, ``BRL +5`` to ``$82 $05 $00``) -- rather than the
    mnemonic with a label marking where the gap ends: this code hasn't been
    placed yet when ``nop_fill`` runs -- often not even in its final bank
    (:func:`text`, for one, mirrors its engine into bank ``$2E`` after this
    runs) -- so there's no real address for a label to carry, and a
    same-purpose invented label (with nothing else to call it) is a name to
    collide, a scope to get wrong, and one more thing to read. A branch's
    displacement is just "how far to the next byte after the gap", which is
    ``0``/``gap`` -- known upfront in Python -- so ``+gap`` needs neither a
    label nor a real address; asar resolves it as a literal offset, not an
    address expression. Likewise ``fill`` (unlike ``pad``) takes a byte
    *count*, not a target address, so the large-gap branch does not need one
    either (and sidesteps a separate asar bug: ``pad``'s target silently
    mis-evaluates to a bogus address when given a forward-referenced label).
    """
    header = notes([f"; [{tag}] {comment}"])
    if count <= 1:
        return header + (instructions(["NOP"]) if count else [])
    if count == 2:  # noqa: PLR2004 -- BRA +0 (2-byte NOP), nothing to fill
        return [*header, *instructions(["BRA +0"])]
    gap = count - 3
    lines = [*header, *instructions([f"BRL +{gap}"])]
    if gap == 1:
        lines += instructions(["NOP"])
    elif gap > 1:
        if nop_padbyte_threshold > 0 and gap > nop_padbyte_threshold:
            fill_line = note(f"fill {gap}")
            fill_line.size = gap
            lines += [note("fillbyte $EA"), fill_line]
        else:
            lines += _row_fill("db", "$EA", gap)
    return lines


# The VWF font blob's labels, shared by the text engine and the bank_00 upload;
# left un-namespaced so both reach the same ``TheFont``.
THEFONT_LABELS = frozenset({"TheFont", "TheFont_end"})


def flute_to_ocarina_dialogue_fixes(main: Assembly) -> None:
    """Renames "Flute"/"flute" to "Ocarina"/"ocarina" everywhere it occurs
    in ``main`` (9 of 10 occurrences in the dialogue/intro message data --
    the 10th, Message_0183, falls past ``Message_DataExtra`` and is
    :func:`flute_to_ocarina_overflow_fix`) -- the JP original's name for
    the item (matching every other Zelda game); the US translation renamed
    it "Flute". Every occurrence is spelled with literal characters only
    (no dictionary tokens participate), so each is a plain 5-byte ->
    7-byte splice, address-anchored per :func:`gba_dialogue_text_fixes`'s
    convention. None need re-wrapping: every line stays within the range
    other real lines in this same message data already reach.
    """
    main.splice(
        "#_1CA3BC:",  # Message_0067: "Here is the Flute!" -> "...Ocarina!"
        datas(
            [
                "db $59, $D8, $59, $0E, $1C, $1A, $2B, $22, $27, $1A"
                " ; the Ocarina"
            ]
        ),
        until="#_1CA3C4:",
    )
    main.splice(
        "#_1CB778:",  # Message_00A1: "...play the flute," -> "...ocarina,"
        datas(["db $28, $1C, $1A, $2B, $22, $27, $1A, $42 ; ocarina,"]),
        until="#_1CB77E:",
    )
    main.splice(
        "#_1CB7FB:",  # Message_00A2: "...son's flute...!" -> "...ocarina...!"
        datas(["db $28, $1C, $1A, $2B, $22, $27, $1A, $43, $3E ; ocarina…!"]),
        until="#_1CB802:",
    )
    main.splice(
        "#_1CB879:",  # Message_00A3: "keep the flute?" -> "...ocarina?"
        datas(
            [
                "db $24, $1E, $1E, $29, $59, $D8, $59, $28 ; keep the o",
                "db $1C, $1A, $2B, $22, $27, $1A, $3F ; carina?",
            ]
        ),
        until="#_1CB886:",
    )
    main.splice(
        "#_1CCB19:",  # Message_00E5 (1/2): "...playing the flute in"
        datas(
            [
                "db $BA, $32, $B3, $D8, $59, $28, $1C, $1A"
                " ; laying the oca",
                "db $2B, $22, $27, $1A, $59, $B4 ; rina in",
            ]
        ),
        until="#_1CCB25:",
    )
    main.splice(
        "#_1CCB83:",  # Message_00E5 (2/2): "buried my flute there"
        datas(
            [
                "db $32, $59, $28, $1C, $1A, $2B, $22, $27, $1A, $59"
                " ; y ocarina "
            ]
        ),
        until="#_1CCB8B:",
    )
    main.splice(
        "#_1CCC18:",  # Message_00E8: "find my flute?" -> "...ocarina?"
        datas(
            [
                "db $1D, $59, $26, $32, $59, $28, $1C, $1A ; d my oca",
                "db $2B, $22, $27, $1A, $3F ; rina?",
            ]
        ),
        until="#_1CCC23:",
    )
    main.splice(
        "#_1CCC4D:",  # Message_00E9 (1/4): move "looks" onto line 1 -- the
        # 2 extra chars from "flute"->"ocarina" (below) push "looks like
        # I can't play my ocarina" past line 2's budget; shifting "looks"
        # up onto line 1 ("...But it looks") keeps both lines safe.
        datas(["db $59, $25, $28, $28, $24, $2C, $75 ; <space>looks"]),
        until="#_1CCC4E:",
    )
    main.splice(
        "#_1CCC4E:",  # Message_00E9 (2/4): drop "looks<space>", now on line 1
        datas(["db $25, $22, $24 ; lik"]),
        until="#_1CCC56:",
    )
    main.splice(
        "#_1CCC66:",  # Message_00E9 (3/4): "...play my flute" (end of line)
        datas(["db $28, $1C, $1A, $2B, $22, $27, $1A ; ocarina"]),
        until="#_1CCC6B:",
    )
    main.splice(
        "#_1CCD08:",  # Message_00E9 (4/4): "the flute one last time…"
        datas(
            [
                "db $D8, $59, $28, $1C, $1A, $2B, $22, $27, $1A, $59"
                " ; the ocarina "
            ]
        ),
        until="#_1CCD10:",
    )


def flute_to_ocarina_overflow_fix(overflow: Assembly) -> None:
    """The one of the 10 dialogue occurrences (Message_0183) that falls
    past ``Message_DataExtra`` (bank $0E addressing) rather than in
    ``main`` -- see :func:`flute_to_ocarina_dialogue_fixes`.
    """
    overflow.splice(
        "#_0EEB8D:",  # Message_0183: "animals with his Flute." -> "...Ocarina."
        datas(
            [
                "db $B0, $2C, $59, $0E, $1C, $1A, $2B, $22, $27, $1A"
                " ; his Ocarina"
            ]
        ),
        until="#_0EEB95:",
    )


def gba_dialogue_text_fixes(main: Assembly) -> None:
    """GBA-era wording fixes applied to the dialogue/intro message data
    (``Message_Data``, US bank $1C): ``pegasus shoes`` -> ``pegasus boots``,
    ``faerie`` -> ``fairy`` (the plural, ``faeries`` -> ``fairies``, only
    occurs in the credits -- see ``credits_bank``'s ``edit_venus_faeries_
    to_fairies``), and ``wise men``/``wise man`` -> ``sages``/``sage`` (the
    last never occurs). Every message byte here is the dictionary/literal-
    character encoding ``CreateMessagePointers`` walks (see :func:`text`'s
    docstring); each edit below was decoded and re-encoded by hand against
    that scheme, not guessed.

    All twelve plain ``wise men`` -> ``sages`` occurrences share one
    byte-identical 5-byte encoding (``$E2, $D0, $59, $BE, $27`` -- the
    dictionary tokens for "wi"/"se"/" "/"me"/"n") and re-encode to an
    equal-length literal ``sages`` (``$2C, $1A, $20, $1E, $2C``), so one
    substring replace covers all of them; the boundary is safe because a
    handful of other same-shaped occurrences (an apostrophe, or the phrase
    split across a real line-break) never contain that literal 5-byte run in
    one line and so never match it -- each of those is edited individually
    below, first, so none of them can spuriously match the batch. Two
    messages (0038/0039) repeat the same split verbatim.
    """
    # The one occurrence with a following possessive: "men's" -> "'s line"
    # keeps growing past the batch's word boundary, so it is handled with
    # its own splice (+1 byte: 8 -> 9).
    main.splice(
        "#_1CCDA2:",
        datas(
            [
                "db $2C, $1A, $20, $1E, $2C, $51, $59, $25 ; sages' l",
                "db $B4 ; in",
            ]
        ),
        until="#_1CCDAA:",
    )
    # "seven wise men's" -> "seven sages'" (drop the extra "s" a plural
    # possessive doesn't take): -1 byte (14 -> 13).
    main.splice(
        "#_1C88CB:",
        datas(
            [
                "db $C3, $59, $D8, $59, $D0, $2F, $A0, $2C"
                " ; open the seven s",
                "db $1A, $20, $1E, $2C, $51 ; ages'",
            ]
        ),
        until="#_1C88D9:",
    )
    # "broken the wise[scroll]men's seal" -> "broken the sages'[scroll]seal"
    # -- the word itself can't straddle a line break, so the break moves to
    # right after it instead.
    main.splice(
        "#_1C9276:",
        datas(
            [
                "db $1B, $2B, $28, $24, $A0, $D8, $59, $2C ; broken the s",
                "db $1A, $20, $1E, $2C, $51 ; ages'",
                "db $73 ; scroll text",
                "db $D0, $1A, $25, $59, $8C ; seal and ",
            ]
        ),
        until="#_1C9288:",
    )
    # "of the wise[scroll]men.  I want" -> "of the sages.[scroll]I want",
    # same break-relocation idea; Message_0038 and Message_0039 repeat this
    # passage verbatim at different addresses. The scroll-text line itself
    # (between ``brk`` and ``after``) is left untouched by both splices.
    for start, brk, after, end in (
        ("#_1C9691:", "#_1C9695:", "#_1C9696:", "#_1C969E:"),
        ("#_1C9973:", "#_1C9977:", "#_1C9978:", "#_1C9980:"),
    ):
        main.splice(
            start,
            datas(["db $D8, $59, $2C, $1A, $20, $1E, $2C, $41 ; the sages."]),
            until=brk,
        )
        main.splice(after, datas(["db $08, $59, $DF, $27 ; I wan"]), until=end)
    # "of the wise[scroll]men..." -> "of the[scroll]sages..." (the message's
    # last line, so the shortened word just becomes its own line).
    main.splice(
        "#_1CB464:",
        datas(["db $C6, $59, $D8, $59 ; of the "]),
        until="#_1CB46A:",
    )
    main.splice(
        "#_1CB46B:",
        datas(["db $2C, $1A, $20, $1E, $2C, $43 ; sages…"]),
        until="#_1CB46E:",
    )
    # "of the wise[wait][scroll]men.  Only a" -> "of the sages.[wait]
    # [scroll]Only a".
    main.splice(
        "#_1CEBC3:",
        datas(
            [
                'db $59, $C6, $59, $D8, $59, $2C, $1A, $20 ; " of the sag"',
                'db $1E, $2C, $41 ; "es."',
            ]
        ),
        until="#_1CEBCA:",
    )
    main.splice(
        "#_1CEBCC:",
        datas(["db $0E, $27, $B9, $1A ; Only a"]),
        until="#_1CEBD4:",
    )
    # Seven occurrences split only by the disassembly's own cosmetic 8-byte
    # row chunking (no real line break involved), each re-encoding to the
    # same byte count so a straight per-row replace applies.
    main.replace_all(
        [
            (  # Message_0032: "of the seven wise men." -> "...sages."
                "#_1C9027: db $C6, $59, $D8, $59, $D0, $2F, $A0, $E2",
                "#_1C9027: db $C6, $59, $D8, $59, $D0, $2F, $A0, $2C",
                1,
            ),
            (
                "#_1C902F: db $D0, $59, $BE, $27, $41",
                "#_1C902F: db $1A, $20, $1E, $2C, $41",
                1,
            ),
            (  # Message_0038: "when the seven wise men created" -> "...sages"
                "#_1C95DC: db $59, $D0, $2F, $A0, $E2, $D0, $59, $BE",
                "#_1C95DC: db $59, $D0, $2F, $A0, $2C, $1A, $20, $1E",
                1,
            ),
            ("#_1C95E4: db $27", "#_1C95E4: db $2C", 1),
            (  # Message_0039: same passage, verbatim
                "#_1C98BE: db $59, $D0, $2F, $A0, $E2, $D0, $59, $BE",
                "#_1C98BE: db $59, $D0, $2F, $A0, $2C, $1A, $20, $1E",
                1,
            ),
            ("#_1C98C6: db $27", "#_1C98C6: db $2C", 1),
            (  # Message_003A: "of the wise men.   THE KING" -> "...sages..."
                "#_1C99B4: db $C6, $59, $D8, $59, $E2, $D0, $59, $BE",
                "#_1C99B4: db $C6, $59, $D8, $59, $2C, $1A, $20, $1E",
                1,
            ),
            (
                "#_1C99BC: db $27, $41, $89, $13, $07, $04, $59, $0A",
                "#_1C99BC: db $2C, $41, $89, $13, $07, $04, $59, $0A",
                1,
            ),
            (  # Message_0035: "the seven wise men, they" -> "...sages, they"
                "#_1C9341: db $D8, $59, $D0, $2F, $A0, $E2, $D0, $59",
                "#_1C9341: db $D8, $59, $D0, $2F, $A0, $2C, $1A, $20",
                1,
            ),
            (
                "#_1C9349: db $BE, $27, $42, $59, $D8, $32, $59, $AD",
                "#_1C9349: db $1E, $2C, $42, $59, $D8, $32, $59, $AD",
                1,
            ),
            (  # Message_0125: "the seven wise men sealed" -> "...sages..."
                "#_1CDF98: db $D8, $59, $D0, $2F, $A0, $E2, $D0, $59",
                "#_1CDF98: db $D8, $59, $D0, $2F, $A0, $2C, $1A, $20",
                1,
            ),
            (
                "#_1CDFA0: db $BE, $27, $59, $D0, $1A, $25, $1E, $1D",
                "#_1CDFA0: db $1E, $2C, $59, $D0, $1A, $25, $1E, $1D",
                1,
            ),
            (  # Message_0135: "As the wise men sealed" -> "As the sages..."
                "#_1CE994: db $00, $2C, $59, $D8, $59, $E2, $D0, $59",
                "#_1CE994: db $00, $2C, $59, $D8, $59, $2C, $1A, $20",
                1,
            ),
            (
                "#_1CE99C: db $BE, $27, $59, $D0, $1A, $25, $A4, $D8",
                "#_1CE99C: db $1E, $2C, $59, $D0, $1A, $25, $A4, $D8",
                1,
            ),
            (  # He gives you the Pegasus Shoes! -> ...Pegasus Boots!
                "#_1CA99E: db $12, $21, $28, $1E, $2C, $3E, $8A, $0D",
                "#_1CA99E: db $01, $28, $28, $2D, $2C, $3E, $8A, $0D",
                1,
            ),
            (  # You caught a faerie! -> ...a fairy!
                "#_1CC3D3: db $59, $1A, $59, $1F, $1A, $A6, $22, $1E",
                "#_1CC3D3: db $59, $1A, $59, $1F, $1A, $22, $2B, $32",
                1,
            ),
            (  # the figure of a faerie. -> ...a fairy.
                "#_1CF9E5: db $59, $1A, $59, $1F, $1A, $A6, $22, $1E",
                "#_1CF9E5: db $59, $1A, $59, $1F, $1A, $22, $2B, $32",
                1,
            ),
            (  # the 12 remaining plain "wise men" -> "sages" occurrences
                "$E2, $D0, $59, $BE, $27",
                "$2C, $1A, $20, $1E, $2C",
                12,
            ),
        ]
    )


def text(
    sources: Sources,
    *,
    changes: bool,
    gba_text_fixes: bool = True,
    flute_is_ocarina: bool = False,
    nop_padbyte_threshold: int = DEFAULT_NOP_PADBYTE_THRESHOLD,
) -> Relocation:
    """The US text subsystem: the VWF font (bank ``$20``), the message engine
    (mirror-placed to ``$2E``), and the message data (``$22``/``$23``).

    The engine is pulled whole by reference: the two endpoints plus
    ``TextCommandLengths`` (reached only by an out-of-bounds index) close over
    the entire live subsystem. Dead blocks drop out, their space held with an
    ``org`` so survivors keep their +$200000 mirror address.

    ``gba_text_fixes`` (only meaningful alongside ``changes``) applies
    :func:`gba_dialogue_text_fixes` to the message data; ``False`` leaves
    the US translation's original wording untouched. ``flute_is_ocarina``
    (also only meaningful alongside ``changes``) applies :func:`flute_to_
    ocarina_dialogue_fixes`; ``True`` renames "Flute" to "Ocarina" (the JP
    original's name) everywhere it occurs in dialogue.
    """
    us, jp = sources.us, sources.jp
    engine = us.extract(
        ("RenderText", "CreateMessagePointers", "TextCommandLengths"),
        recursive=True,
        external=THEFONT_LABELS,
        comments=True,
        gap_notes={
            "UNREACHABLE_0ED3CF": (
                "CreateMessagePointers over-reads "
                "RenderText_MoreInitialSettings,Y up to index $7F (past its "
                "20-byte end) across this pad into TextCommandLengths just "
                "below, so this offset is load-bearing."
            )
        },
    )
    main = us.blocks_until("Message_Data")
    overflow = us.blocks_until("Message_DataExtra")
    if changes and gba_text_fixes:
        gba_dialogue_text_fixes(main)
    if changes and flute_is_ocarina:
        flute_to_ocarina_dialogue_fixes(main)
        flute_to_ocarina_overflow_fix(overflow)
    # CreateMessagePointers walks byte-by-byte until it reads a literal $FF
    # (the table-end marker); in the US ROM this falls out for free since
    # Message_DataExtra runs to the natural, $FF-padded end of its bank. Our
    # relocated copy is pulled as just its own block (no trailing bank
    # padding), and the free ROM after it in bank $23 is $00-filled -- so
    # without an explicit terminator here, the scan runs past the real data
    # and never finds $FF (in practice: an effectively endless loop at boot,
    # since CreateMessagePointers runs once at startup). Needed
    # unconditionally -- both the baseline and the full build carry the
    # vanilla-preserved engine.
    overflow.append(
        datas(
            [
                "db $FF ; end of message table"
                " (CreateMessagePointers terminator)"
            ]
        )
    )
    decompress_hook = jp.address_of("DecompressFontGFX")
    masks_hook = jp.address_of("BuildSomeTextMasks")

    # [ENG-TEXT] vanilla US bug: one message line ("said, "Once I have
    # finished with...") draws past the VWF's 168px-per-half (`$0150`-byte)
    # row budget by a single pixel -- the font's quote-mark glyph (`"`)
    # declares an 8px advance width but its own ink
    # (decoded from us_font.2bpp) only occupies columns 0-4. Every other
    # glyph checked (period, comma, `O`) declares a width that fits its own
    # ink exactly, zero slack; the quote is the one glyph in the font that
    # doesn't follow that. Narrowing it to 6 -- its own ink width plus a
    # single pixel of trailing space -- clears the overflow (brings the line
    # to 167px) and leaves a small, deliberate gap between the open quote
    # and the next word (visually compared live against the close quote a
    # few lines later, "open."", which sits flush with zero gap against its
    # own neighbor -- same glyph, same tile, no ink discarded either way,
    # just one glyph carrying an extra pixel of its own declared width and
    # the other not). `"` appears in exactly two messages in the whole game
    # (grepped $4C across text.asm) -- this line, and one other (Agahnim's
    # "the prophecy says, "The Hero...") whose quotes each sit alone at a
    # scroll-segment boundary rather than flush against a line's own right
    # edge, so they aren't relying on that trailing padding either. Message
    # line breaks are all hardcoded by the data's own $75/$76/$73 ops, not
    # reflowed from character width, so this can't shift where any line
    # wraps -- it only changes this one line's own total pixel width.
    # Confirmed live (Mesen trace, same line): "with"'s own trailing
    # border, previously corrupted by the 1px overflow, renders correctly.
    engine.replace(
        "db   8,  7,  7,  7,  7,  4", "db   6,  7,  7,  7,  7,  4", 1
    )

    # (2) message-ID realignment, done in the ENGINE instead of the data. The
    # US has two messages JP lacks -- the Choose2High cursor prompts at IDs
    # $0B/$0C -- so every later message's US ID is JP's + 2. Rather than
    # delete them (which physically moves every byte of the data, exploding
    # the bank_22/bank_23 diff), teach CreateMessagePointers to hand those two
    # the out-of-range IDs $18C/$18D as it assigns IDs, so every real message
    # keeps its JP ID while the data stays byte-for-byte the US layout. Only
    # RenderText_Choose2HighOr3 references the two (repointed below,
    # full-only); without the redirect a 2-option "high" prompt (e.g. the
    # Great Fairy upgrade) would load whatever sits at ID $0B.
    #
    # The high slots must start at $18C, not the more obvious $18B: after the
    # loop's very last real message, it always does one extra iteration --
    # storing wherever $00/$02 land (the table-end $FF marker's own position)
    # into whatever slot X currently is -- before the *next* byte read
    # actually recognizes that $FF and exits. With 2 messages diverted, X
    # lands on exactly $04A1 (395 real messages * 3) at that point, which
    # used to be $18B's own slot -- silently overwriting its real pointer
    # with the $FF marker's position (bank $23, deep in padding) right after
    # this code correctly wrote it. Confirmed live (Mesen instruction trace
    # from a fresh boot): $18C wrote correctly and stayed correct (its own
    # slot, $04A4, isn't this collision point); $18B wrote correctly and was
    # then clobbered by that trailing store a few instructions later, which
    # is exactly what sent the Great Fairy bomb/arrow prompt reading garbage
    # and crashing. Shifting both slots by one dodges the fixed collision
    # point without changing anything else about the mechanism.
    #
    # Applied UNCONDITIONALLY, self-branching on `changes` at each site (real
    # redirect vs. same-size NOP filler in baseline), so CreateMessagePointers
    # is byte-neutral between baseline and full -- no cascade into RenderText
    # or the rest of the engine.
    if changes:
        engine.insert_after(
            "#_0ED3F9:",  # LDX #$0000 -- the table-index / ID-counter init
            instructions(
                [
                    "LDA.w #$0002 ; [ENG-TEXT] US-only msgs to give high IDs",
                    "STA.b $04    ; ($0B/$0C: the Choose2High cursor prompts)",
                    "LDA.w #$04A4 ; first high slot -> ID $18C ($18C * 3)",
                    "STA.b $06",
                ]
            ),
        )
    else:
        engine.insert_after(
            "#_0ED3F9:",
            nop_fill(
                10,
                "ENG-TEXT",
                "reserved: see CreateMessagePointers init",
                nop_padbyte_threshold=nop_padbyte_threshold,
            ),
        )
    if changes:
        engine.insert_before(
            "#_0ED3FC:",  # the per-message store, right past .next_message
            [
                *instructions(
                    [
                        "CPX.w #$0021 ; [ENG-TEXT] slot for ID $0B, where the",
                        "BNE .store   ; two US-only prompts fall...",
                        "LDA.b $04    ; ...divert them to $18C/$18D, then let",
                        "BEQ .store   ; the real msg $0B take this slot",
                        "PHX",
                        "LDX.b $06        ; the $18C/$18D table slot",
                        "LDA.b $00",
                        "STA.l $7F71C0,X",
                        "LDA.b $01",
                        "STA.l $7F71C1,X",
                        "PLX              ; keep the real ID index put",
                        "LDA.b $06",
                        "CLC",
                        "ADC.w #$0003     ; next high slot",
                        "STA.b $06",
                        "DEC.b $04",
                        "BRA .next_byte   ; walk msg; don't consume an ID",
                    ]
                ),
                Line.from_line(".store"),
            ],
        )
    else:
        engine.insert_before(
            "#_0ED3FC:",
            nop_fill(
                37,
                "ENG-TEXT",
                "reserved: see CreateMessagePointers check",
                nop_padbyte_threshold=nop_padbyte_threshold,
            ),
        )

    if changes:
        # (1) [NAME] field: the name is a 6-word field at $3D5 (widened -- see
        # file_select), so the stock US 6-char handler works; just repoint its
        # read.
        name_field: list[Edit] = [("LDA.l $7003D9,X", "LDA.l $7003D5,X", 1)]
        # (3) repoint RenderText_Choose2HighOr3's two cursor prompts to the
        # high IDs the engine now assigns them.
        engine_edits: dict[str, list[Edit]] = {
            "ParseText_WritePlayerName": name_field,
            "RenderText_Choose2HighOr3": [
                (
                    "dw $000B",
                    "dw $018C    ; [ENG-TEXT] cursor prompt 1"
                    " (see CreateMessagePointers)",
                    1,
                ),
                ("dw $000C", "dw $018D    ; [ENG-TEXT] cursor prompt 2", 1),
            ],
        }
        engine.apply_edit_table(engine_edits)
    engine.ensure_anchors()  # anchor the hand-written lines (either build)

    engine_org = mirror(require_start(engine))  # $0EC440 -> $2EC440
    engine_text = engine.render(engine_org)
    overflow_note = "MESSAGE overflow (US bank_0E)."

    # Every JP name this subsystem intercepts (freed in the base): the first
    # three claim a bare alias in the relocated engine; the last two are the
    # override stubs below (verbatim, already carrying a bare alias). None have
    # a same-bank caller, so all resolve to aliases (no pad).
    relocation = Relocation(
        hooked=(
            "RenderText",
            "Module0E_02_RenderText",
            "CreateMessagePointers",
            "DecompressFontGFX",
            "BuildSomeTextMasks",
        ),
        shared=THEFONT_LABELS,
        changes=changes,
    )
    # Our VWF text font (us_font.2bpp); a raw blob (TheFont/TheFont_end stay
    # bare, shared), read by the engine and the bank_00 upload.
    relocation.place(
        Assembly(
            [
                note("TheFont:"),
                incbin_line(
                    "bin/gfx/us_font.2bpp",
                    us_assets.asset("us_font.2bpp").size,
                ),
                note("TheFont_end:"),
            ]
        ),
        0x208000,
        "Our VWF text font (font.2bpp); read by the bank_00 upload.",
        namespace=False,
    )
    relocation.place(
        main, 0x228000, "MESSAGE data (US text.asm bank $1C); free bank."
    )
    relocation.place(overflow, 0x238000, overflow_note)
    if changes:
        # Re-caption the name filter (a multi-line comment swap on the rendered
        # engine text), then the two override stubs -- hand-written in final
        # form (own bare hook alias + EN_ name), emitted verbatim.
        engine_text = substitute(
            engine_text,
            "; I hate this thing...",
            "\n".join(  # noqa: FLY002
                [
                    "; [ENG-NAME] Names are entered on the US file-select"
                    " screen, so they",
                    "; store native US character codes. This is the stock US",
                    "; RenderText_FilterName (maps name-entry codes to VWF"
                    " glyph codes,",
                    "; I/i and '!' special cases + the lowercase-encode"
                    " branch).",
                ]
            ),
        )
        relocation.place(
            "; [ENG-FSFONT] JP's DecompressFontGFX VWF-rendered the JP font"
            " into $7E2000; the US ROM has no\n"
            "; such routine -- its text font is the plain 2bpp TheFont,"
            " uploaded to VRAM $E000 by\n"
            "; TransferFontToVRAM. No decompression needed.\n"
            "DecompressFontGFX:\n"
            "EN_DecompressFontGFX:\n"
            f"#_{mirror(decompress_hook):06X}: RTL ; US isn't compressed",
            mirror(decompress_hook),
            f"Hook: DecompressFontGFX (mirror of JP ${decompress_hook:06X}).",
            namespace=False,
        )
        relocation.place(
            "; [ENG-TEXT] BuildSomeTextMasks is a no-op here (masks come from"
            " the PerformVWFing tables);\n"
            "; keep the hook so its JP callers land somewhere valid.\n"
            "BuildSomeTextMasks:\n"
            "EN_BuildSomeTextMasks:\n"
            f"#_{mirror(masks_hook):06X}: RTL",
            mirror(masks_hook),
            f"Hook: BuildSomeTextMasks (mirror of JP ${masks_hook:06X}).",
            namespace=False,
        )
    relocation.place(
        engine_text,
        engine_org,
        "Text ENGINE, mirror-placed from US bank_0E $0EC440.",
    )
    return relocation


def font_upload(sources: Sources, *, changes: bool) -> Relocation:
    """Bank ``$20``: JP ``TransferFontToVRAM``, mirror-placed and repointed to
    upload our plain-2bpp ``TheFont`` to VRAM $E000 (was a $7E2000 VWF buffer).
    """
    jp = sources.jp
    root = "TransferFontToVRAM"
    routine = jp.extract([root], recursive=True, external=THEFONT_LABELS)
    if changes:
        # The three font-source operands, same byte width so length is
        # unchanged.
        routine.replace("LDA.b #$7E", "LDA.b #TheFont>>16", count=1)
        routine.replace("LDA.w #$7E2000", "LDA.w #TheFont", count=1)
        routine.replace(
            "LDX.w #$0FFF", "LDX.w #(TheFont_end-TheFont)/2-1", count=1
        )
        routine.lines.insert(
            0,
            note(
                "; [ENG-GFX] Uploads TheFont (font.2bpp) to VRAM $E000 -- the"
                " US form (JP uploaded a $7E2000 VWF buffer)."
            ),
        )
    relocation = Relocation(
        hooked=(root,), shared=THEFONT_LABELS, changes=changes
    )
    relocation.place_mirror(
        routine, "bank-$00 TransferFontToVRAM (mirror of JP $00E596)."
    )
    return relocation


def hud_max_capacity_color(
    *, changes: bool, yellow_counts_at_absolute_max: bool = False
) -> Relocation:
    """Bank ``$29`` (free space): recolors the HUD bomb/arrow/rupee digit
    counters to gold when the player is at maximum capacity (50 bombs, 70
    arrows, 999 rupees), matching the GBA re-release.

    Each hooks in right after ``UpdateHUDBuffer``'s own digit-drawing
    (bank_0D) for that counter finishes -- so its digit tiles are already
    written to the HUD buffer, each with palette 1 ($24xx) baked into its
    high byte -- and conditionally EORs them with $1000 (the bit difference
    between palette 1 and palette 5's attribute byte, $24 xor $34; see
    ``apply_base_edits``'s ``PaletteData`` edit for why palette 5 -- an
    otherwise-idle slot repointed to a matching red-outline/gold-fill
    palette) when at max.

    By default bombs/arrows compare the live count against ``CapacityUpgrades``
    indexed by the player's own upgrade count ($7EF370/$7EF371) -- the same
    lookup the bomb/arrow-refill code (also bank_0D) uses to know when to
    stop refilling -- so it reads as "matches whatever cap you currently
    have", turning gold at e.g. 40/40 arrows on an early, not-fully-upgraded
    save rather than only the true 70; confirmed against the GBA
    re-release's own behavior (a common online claim that the GBA version
    uses the absolute cap instead turned out not to hold up under actual
    testing). ``yellow_counts_at_absolute_max=True``
    (``--yellow-counts-at-absolute-max``) instead compares the live count
    directly against a fixed constant ($32/50, $46/70) -- the absolute,
    fully-upgraded cap, ignoring the player's current tier. Rupees are
    unaffected either way: they have no tiered capacity, so 999 is always a
    fixed-constant compare against the same hardcoded cap the rupee-fill
    code (bank_0D) already clamps both the real and displayed rupee count
    to. Doing the recolor post-hoc, on the already-drawn tiles,
    rather than diverting the draw itself needs no call back into bank_0D's
    ``HexToDec`` -- a same-bank-only routine (it returns via RTS, not RTL)
    that would otherwise force this code to live in bank_0D itself, where
    there's no free space this size.

    An earlier version of this reused palette 2 (the existing magic-gauge
    gold, already loaded every frame) instead of touching any palette data
    -- zero risk, but visually too subtle to notice in play: palette 2
    pairs its gold with the *same white fill* every other digit already
    uses, so only the thin outline changed color. Confirmed working
    correctly the whole time via live pixel sampling (WRAM/VRAM/CGRAM all
    correct) before that was diagnosed as a design mismatch, not a bug --
    the fill itself needs to turn gold, which means owning a palette slot.
    """
    if yellow_counts_at_absolute_max:
        bombs_check = [
            "LDA.l $7EF343    ; current bombs",
            "CMP.b #$32       ; 50 (absolute max, all upgrades)",
        ]
        arrows_check = [
            "LDA.l $7EF377    ; current arrows",
            "CMP.b #$46       ; 70 (absolute max, all upgrades)",
        ]
    else:
        bombs_check = [
            "LDA.l $7EF370    ; bomb capacity-upgrade index",
            "TAY",
            "LDA.l $7EF343    ; current bombs",
            "CMP.w CapacityUpgrades_bombs_hex,Y ; this tier's own cap",
        ]
        arrows_check = [
            "LDA.l $7EF371    ; arrow capacity-upgrade index",
            "TAY",
            "LDA.l $7EF377    ; current arrows",
            "CMP.w CapacityUpgrades_arrows_hex,Y ; this tier's own cap",
        ]
    relocation = Relocation(changes=changes)
    relocation.place(
        Assembly.from_content(
            [
                "; [ENG-HUD] UpdateHUDBuffer (bank_0D) always draws the",
                "; bomb/arrow HUD digits with palette 1 ($24xx, white fill)",
                "; -- this recolors them to palette 5 ($34xx, gold fill,",
                "; same red outline) when at max capacity (see",
                "; hud_max_capacity_color for the absolute-vs-current-tier",
                "; choice), matching the GBA re-release's yellow-at-max",
                "; counters (see PaletteData's edit in apply_base_edits for",
                "; the palette itself). Hooked in at UpdateHUDBuffer's very",
                "; last Arrows-digit store (bank_0D), replayed here, by",
                "; which point all 4 digit tiles (bomb tens/ones, arrow",
                "; tens/ones) are already sitting in the HUD buffer.",
                "HUD_RecolorBombsArrowsAtMax:",
                "STA.l $7EC760    ; replay the displaced store (arrow ones)",
                "SEP #$30",
                *bombs_check,
                "BNE .bombs_not_max",
                "REP #$30",
                "LDA.l $7EC758",
                "EOR.w #$1000",
                "STA.l $7EC758",
                "LDA.l $7EC75A",
                "EOR.w #$1000",
                "STA.l $7EC75A",
                "SEP #$30",
                ".bombs_not_max",
                *arrows_check,
                "BNE .arrows_not_max",
                "REP #$30",
                "LDA.l $7EC75E",
                "EOR.w #$1000",
                "STA.l $7EC75E",
                "LDA.l $7EC760",
                "EOR.w #$1000",
                "STA.l $7EC760",
                "SEP #$30",
                ".arrows_not_max",
                "REP #$30         ; Keys (rejoin point) expects 16-bit A",
                "JML $0DFCEC",
            ]
        ).ensure_anchors(),
        0x29D000,
        "Max-capacity bomb/arrow HUD digit color.",
    )
    relocation.place(
        Assembly.from_content(
            [
                "; [ENG-HUD] Same idea as HUD_RecolorBombsArrowsAtMax, for",
                "; the 3-digit rupee counter. Hooked in at UpdateHUDBuffer's",
                "; very last Rupees-digit store (bank_0D), replayed here, by",
                "; which point all 3 digit tiles (hundreds/tens/ones) are",
                "; already in the HUD buffer. $7EF362 (the displayed rupee",
                "; count, ticking toward the real $7EF360 -- see bank_0D's",
                "; own rupee-fill code) is what UpdateHUDBuffer itself draws",
                "; from, so comparing it directly against 999 matches",
                "; exactly what's on screen, including mid-tick-animation.",
                "HUD_RecolorRupeesAtMax:",
                "STA.l $7EC754    ; replay the displaced store (ones digit)",
                "LDA.l $7EF362    ; current displayed rupees",
                "CMP.w #$03E7     ; 999",
                "BNE .rupees_not_max",
                "LDA.l $7EC750",
                "EOR.w #$1000",
                "STA.l $7EC750",
                "LDA.l $7EC752",
                "EOR.w #$1000",
                "STA.l $7EC752",
                "LDA.l $7EC754",
                "EOR.w #$1000",
                "STA.l $7EC754",
                ".rupees_not_max",
                "JML $0DFCA4      ; Bombs (rejoin point)",
            ]
        ).ensure_anchors(),
        0x29D100,
        "Max-capacity rupee HUD digit color.",
    )
    return relocation


def credits_font_upload(
    sources: Sources, *, changes: bool, credits_font: str = "jp"
) -> Relocation:
    """Bank ``$20`` (free space): a second, independent ``TransferFontToVRAM``
    copy, repointed to upload the credits' font from a precomputed flat
    resource instead of JP's runtime-decompressed ``$7E2000`` VWF buffer.

    ``credits_font`` picks which resource: ``"jp"`` (default) is
    ``jp_credits_font.2bpp``, JP 1.0's own bold font, decompressed *offline*
    (``binextract-jp-credits-font.py``, from the raw JP ROM bytes, no
    emulator) into a plain flat 2bpp sheet. ``"us"`` is
    ``us_credits_font.2bpp``, the same
    tile-slot layout with each character's pixel data instead pulled from
    the US dialogue font (``binextract-us-credits-font.py``, see
    :mod:`us_credits_font_asset`) -- letting credits look like the rest of
    the (US-fonted) game instead of standing out. Either way this is the
    same flat-upload shape :func:`font_upload` already uses for the English
    dialogue font, just a different source and a different pair of callers
    (credits' own font-load sites, wired in ``apply_base_edits``). This is
    why credits needs none of
    ``BuildSomeTextMasks``/``DecompressFontGFX``/``TransferFontToVRAM``'s JP
    originals at runtime at all: the decompression/lookup already happened
    once, ahead of time, and its *output* is what gets `incbin`'d here.
    """
    filename, size, style_note = (
        (
            "jp_credits_font.2bpp",
            jp_credits_font_asset.SIZE,
            "JP 1.0's own bold font",
        )
        if credits_font == "jp"
        else (
            "us_credits_font.2bpp",
            us_credits_font_asset.SIZE,
            "the US dialogue font",
        )
    )
    jp = sources.jp
    # Credits_InitializeTheActualCredits is never relocated (only its own
    # readers/tables are, in credits_bank), so this address is stable and
    # identical whether read from the pristine source or the mutated build.
    # Resume point is right after its own JSL Credits_InitializePolyhedral
    # (exactly 4 bytes later -- the whole instruction) -- see the long
    # comment on EN_CreditsFont_ClearAttributionTilemap below for why the
    # clear has to land *before* that call, not after it.
    credits_init_polyhedral_call = (
        jp.block("Credits_InitializeTheActualCredits")
        .line("JSL Credits_InitializePolyhedral")
        .address
    )
    if credits_init_polyhedral_call is None:
        msg = (
            "Credits_InitializeTheActualCredits: JSL"
            " Credits_InitializePolyhedral is not a live-anchored"
            " instruction"
        )
        raise ValueError(msg)
    credits_init_resume = credits_init_polyhedral_call + 4
    # Only needed for --credits-font us's own palette patch (see the footer
    # below) -- the resume point there lands after this call instead of
    # after Credits_InitializePolyhedral.
    credits_palette_load_call = (
        jp.block("Credits_InitializeTheActualCredits")
        .line("JSL PaletteLoad_HUD")
        .address
    )
    if credits_palette_load_call is None:
        msg = (
            "Credits_InitializeTheActualCredits: JSL PaletteLoad_HUD is"
            " not a live-anchored instruction"
        )
        raise ValueError(msg)
    # Credits_InitializePolyhedral's own premature STA.b $13 -- see
    # EN_CreditsInitPolyhedral_SkipPrematureUnblank below for why this is
    # skipped rather than raced against (again).
    polyhedral_premature_write = (
        jp.block("Credits_InitializePolyhedral").line("STA.b $13").address
    )
    if polyhedral_premature_write is None:
        msg = (
            "Credits_InitializePolyhedral: STA.b $13 is not a"
            " live-anchored instruction"
        )
        raise ValueError(msg)
    # EN_CreditsFont_ClearAttributionTilemap's own DMA fill body, pulled
    # (not hand-written) from EraseTilemaps_bg3's own BG3-fill pass
    # (bank_00) -- see the long comment where it's placed below.
    clear_attribution_tilemap_dma = jp.block("EraseTilemaps", comments=False)
    clear_attribution_tilemap_dma.replace(
        "LDA.w #$6000 ; VRAM $C000", "LDA.w #$6800 ; VRAM $D000", count=2
    )
    clear_attribution_tilemap_dma.splice(
        "LDA.b $02", [instruction("LDA.l EN_CreditsBlankFillTile")]
    )
    clear_attribution_tilemap_start = clear_attribution_tilemap_dma.find(
        "LDA.l EN_CreditsBlankFillTile"
    )
    clear_attribution_tilemap_stop = clear_attribution_tilemap_dma.find(
        "SEP #$20", clear_attribution_tilemap_start
    )
    clear_attribution_tilemap_lines = clear_attribution_tilemap_dma.lines[
        clear_attribution_tilemap_start:clear_attribution_tilemap_stop
    ]
    # EraseTilemaps' BG3 pass (this slice) relies on ambient state it never
    # sets itself: LDY.b #$02 (MDMAEN's trigger value), LDX.b #$80 (the
    # high-byte-pass VMAIN mode), and DMA1ADDRB=0 (the DMA source bank) are
    # all set once, earlier, by the BG1/2 pass this slice starts after, and
    # stay live by fall-through within that one routine. Standalone, this
    # slice can't rely on any of that: X/Y/DMA1ADDRB hold whatever
    # unrelated code left them at, so MDMAEN could trigger the wrong DMA
    # channel (or none), the high-byte pass could use the wrong VMAIN mode,
    # and the DMA source could read from the wrong bank entirely -- caught
    # by a direct byte comparison against the last confirmed-working build,
    # not by inspection. `EraseTilemaps` also never resets VMAIN back to
    # $00 at its own end (it just SEP/RTLs) -- fine there, but this routine
    # falls through into JSL Credits_InitializePolyhedral and whatever
    # runs after it, so it restores VMAIN's default mode explicitly rather
    # than leaving the high-byte-pass mode behind. Re-adding these four
    # (`.insert()`/`.append()` on the already-sliced list -- Assembly.
    # insert_before would find the *first* of several matching occurrences
    # in the full routine, i.e. the BG1/2 pass, not this slice's own) is
    # the one place this routine isn't a pure pull -- everything else
    # still is.
    clear_attribution_tilemap_lines.insert(
        next(
            i
            for i, line in enumerate(clear_attribution_tilemap_lines)
            if "STA.w DMA1ADDRL" in str(line)
        )
        - 1,
        instruction("STZ.w DMA1ADDRB"),
    )
    clear_attribution_tilemap_lines.insert(
        next(
            i
            for i, line in enumerate(clear_attribution_tilemap_lines)
            if "STY.w MDMAEN" in str(line)
        ),
        instruction("LDY.b #$02"),
    )
    clear_attribution_tilemap_lines.insert(
        next(
            i
            for i, line in enumerate(clear_attribution_tilemap_lines)
            if "STX.w VMAIN" in str(line)
        ),
        instruction("LDX.b #$80"),
    )
    clear_attribution_tilemap_lines.append(instruction("STZ.w VMAIN"))
    routine = jp.extract(["TransferFontToVRAM"], recursive=True)
    routine.suffix(["TransferFontToVRAM"], "_Credits")
    if changes:
        # Same 2 byte-width-preserving operand swaps as font_upload()'s
        # copy; the original word-count load already matches this
        # resource's own 8192-byte size exactly (both are a flat
        # 4096-word/8192-byte transfer), so it needs no edit.
        routine.replace("LDA.b #$7E", "LDA.b #CreditsFont>>16", count=1)
        routine.replace("LDA.w #$7E2000", "LDA.w #CreditsFont", count=1)
        routine.lines.insert(
            0,
            note(
                f"; [ENG-CREDITS-FONT] Uploads the credits' font"
                f" ({filename}, {style_note}) to VRAM $E000 -- decompressed"
                " offline, so no runtime decompression is needed."
            ),
        )
    relocation = Relocation(changes=changes)
    relocation.place(
        Assembly(
            [
                note("CreditsFont:"),
                incbin_line(f"bin/gfx/{filename}", size),
                note("CreditsFont_end:"),
            ]
        ),
        0x209000,
        f"The credits' font ({filename}, {style_note}), computed offline.",
        namespace=False,
    )
    relocation.place(
        routine,
        0x20B000,
        "Credits' own TransferFontToVRAM copy (mirror-independent -- a"
        " second, separately-sourced upload routine, not JP $00E596's own"
        " mirror slot).",
    )
    relocation.place(
        Assembly.from_content(
            [
                "; [ENG-CREDITS-FONT] Credits_InitializeTheActualCredits does"
                " its own JSL TransferFontToVRAM *before* JSL Credits_"
                "LoadCoolBackground -- the unsafe order: InitializeTilesets",
                "; (Credits_LoadCoolBackground's own call, a CPU-driven, not"
                " DMA, VRAM copy loop -- interruptible by NMI mid-transfer,"
                " unlike a DMA) then corrupts the just-uploaded font at VRAM",
                "; $E000, confirmed live via a long Lua VRAM trace (byte-"
                "correct for ~40 frames, then wrong for the rest of the"
                " credits; a same-length vanilla JP 1.0 trace showed zero",
                "; corruption the entire run, so this really is build-"
                "specific timing, not a pre-existing issue). A fixed-cycle"
                " busy-wait here (tried first) papered over it by luck of",
                "; how much real time it happened to add, the same way"
                " DecompressFontGFX's real (removed) decompression cost"
                " always happened to in vanilla -- both are just timing",
                "; padding for an ordering that's wrong regardless of how"
                " much padding it gets. Credits_LoadOverworldScene_PrepGFX"
                " (bank_02, credits' other font-load site) already does",
                "; this the safe way -- its own InitializeTilesets call"
                " runs *before* its font upload, not after, so the upload"
                " is the last VRAM-graphics write in that routine, not the",
                "; first -- no padding of any kind needed there. This"
                " matches that same safe order instead of padding around"
                " the unsafe one: skip Credits_InitializeTheActualCredits's",
                "; own JSL TransferFontToVRAM call site entirely (nothing to"
                " reorder around here -- it's simply dead now) and reproduce"
                " the very next instruction, its own JSL Credits_",
                "; LoadCoolBackground (untouched, still reachable in place,"
                " but only via this copy now, not the original flow) --"
                " EN_CreditsFont_ClearAttributionTilemap (below) does the",
                "; real upload itself, now running after Credits_"
                "LoadCoolBackground instead of before it.",
                "CreditsFont_SkipOldUploadSite:",
                "JSL Credits_LoadCoolBackground",
                f"JML ${credits_init_polyhedral_call:06X}",
            ]
        ).ensure_anchors(),
        0x20B030,
        "Credits_InitializeTheActualCredits: skip the old (unsafe-order)"
        " font upload call site.",
    )
    relocation.place(
        Assembly.from_content(
            [
                "; [ENG-CREDITS-FONT] Credits_InitializePolyhedral's own"
                " STA.b $13 (LDA.b #$0F/STA.b $13, jpdasm bank_0C) sets"
                " full brightness/force-blank-off immediately, at its own",
                "; tail -- but in the normal flow it's always overwritten"
                " by Credits_InitializeTheActualCredits's own later STZ.b"
                " $13 before it ever matters (Credits_BrightenTriangles,",
                "; the very next module state, ramps brightness up from 0"
                " on its own regardless of what Polyhedral wrote here)."
                " Reordering the font upload (EN_CreditsFont_",
                "; ClearAttributionTilemap) and adding the VRAM clear both"
                " made sure that *if* an NMI catches this premature write"
                " before the later STZ.b $13, nothing garbage-looking is",
                "; exposed -- but confirmed live, a real (not garbage,"
                " just visibly wrong) 1-frame flash to full brightness"
                " still happens, because the write itself still reaches",
                "; hardware. Since it's provably never needed (always"
                " clobbered before it would otherwise matter), skip it"
                " outright instead of racing it a third time -- $13 simply",
                "; stays whatever Credits_InitializeTheActualCredits's own"
                " EnableForceBlank already set it to ($80, force-blank on)"
                " for the rest of Credits_InitializePolyhedral's own",
                "; execution, until that later STZ.b $13 sets it correctly."
                " Reproduces the displaced INC.b $11, then rejoins right"
                " after it (RTL) -- see the relocate_block call in",
                "; apply_base_edits.",
                "CreditsInitPolyhedral_SkipPrematureUnblank:",
                "INC.b $11",
                f"JML ${polyhedral_premature_write + 4:06X}",
            ]
        ).ensure_anchors(),
        0x20B040,
        "Credits_InitializePolyhedral: skip the premature INIDISP write.",
    )
    clear_attribution_tilemap_header = Assembly.from_content(
        [
            "; [ENG-CREDITS-FONT] Credits_InitializeTheActualCredits sets"
            " the attribution overlay's own VRAM destination to word"
            " $6800 (LDA.w #$6800/STA.b $C8, jpdasm bank_0E), but its own"
            " EraseTilemaps_bg3 call -- right at the top of the same",
            "; routine -- only clears $6000-$67FF (2048 words, traced"
            " byte-for-byte through its two DMA fill passes in bank_00)."
            " $6800 onward is never synchronously cleared; the only",
            "; thing that ever writes there is"
            " Credits_AddNextAttribution's own queued write (staged in"
            " WRAM, landed by the next NMI's DoNMIUpdates), landing"
            " whenever the next NMI happens to flush it, not synchronously.",
            "; This clears it up front instead of leaving that gap:"
            " EN_CreditsBlankFillTile's own tilemap word (the same value"
            " Credits_AddNextAttribution itself already writes to blank a",
            "; line, so a pre-cleared line and a genuinely-blank one are"
            " pixel-for-pixel identical) across the whole 2048-word span"
            " (BG2SC is set to $12 a few instructions later, whose size",
            "; field selects a 32x64 map). Hooked in here, right before the"
            " JSL Credits_InitializePolyhedral it replaces, mainly so the"
            " font upload (below) and this clear both land well before",
            "; anything could possibly display -- defense in depth,"
            " alongside EN_CreditsInitPolyhedral_SkipPrematureUnblank"
            " (credits_font_upload, bank $20) actually preventing",
            "; Credits_InitializePolyhedral's own STA.b $13 from ever"
            " reaching hardware early to begin with (that write, not this"
            " region staying stale, turned out to be the real cause of the",
            "; transition-into-credits garbage flash -- a direct synchronous"
            " DoNMIUpdates call, tried first and reverted, was actively"
            " worse: too broad a routine, gated on live-gameplay WRAM state",
            "; not valid yet mid-setup, it corrupted the background"
            " instead). Reproduces the displaced JSL Credits_"
            "InitializePolyhedral itself, then rejoins right after it --",
            "; see the relocate_block call in apply_base_edits.",
            "; The DMA fill body below (STA.b $00 through the second"
            " STY.w MDMAEN) is not hand-written -- it's"
            " EraseTilemaps_bg3's own BG3-fill pass (bank_00), pulled",
            "; verbatim and tweaked: VMADDR $6000->$6800 (both"
            " occurrences, byte-neutral, .replace()), and the fill-tile"
            " source LDA.b $02 (that pass's own BG3 floor tile, loaded",
            "; by its header, not pulled here) swapped for a direct"
            " LDA.l EN_CreditsBlankFillTile (a real size change, 2->4"
            " bytes, so .splice()+instruction() instead of .replace()). Same"
            " low+high-byte DMA idiom, same register-width discipline (A",
            "; kept 16-bit for the whole pass -- several stores are paired"
            " 16-bit writes landing on two adjacent DMA registers at once,"
            " e.g. STA.w DMA1ADDRL also writing DMA1ADDRL+1 -- X/Y stay",
            "; 8-bit throughout, untouched) -- all of it comes from bytes"
            " that already assemble and already run correctly in the base"
            " game, not re-derived by hand.",
            "; The font upload (JSL EN_TransferFontToVRAM_Credits) happens"
            " here too now, first thing -- see EN_CreditsFont_"
            "SkipOldUploadSite above for why: Credits_InitializeTheActual",
            "; Credits's own font-load site used to run this before"
            " Credits_LoadCoolBackground, an unsafe order; this reorders it"
            " to run after, matching Credits_LoadOverworldScene_PrepGFX's",
            "; own already-safe order. TransferFontToVRAM manages its own"
            " register widths internally (ends SEP #$30) regardless of what"
            " it's called with, so its position relative to the PHP/REP",
            "; #$20 pair below doesn't matter -- placed first simply"
            ' because "load the font" reads better before "clear the text'
            ' area" than after.',
            "CreditsFont_ClearAttributionTilemap:",
            "PHP",
            "JSL EN_TransferFontToVRAM_Credits",
            "REP #$20",
        ]
    ).ensure_anchors()
    if credits_font == "us":
        # [ENG-CREDITS-FONT] TheFont (the US dialogue font, used here for
        # the TOP/BOTTOM/white location-caption role only -- see
        # us_credits_font_asset's module docstring for why the SMALL/yellow
        # role is never touched) shades curved/gapped letters via color
        # CONTRAST between pixel values 1 and 2/3, not true zero-value
        # gaps. Credits' own font palette 7 (TOP/BOTTOM role) is otherwise
        # degenerate by design for JP's own bold glyphs -- only color index
        # 2 is a real color; indices 1 and 3 are both black, same as index
        # 0 (background) -- fine for JP's own glyphs, but it silently
        # drops TheFont's own outline/accent shading, collapsing every
        # curved letter into a solid block. Patched below (CGRAM 29/31 =
        # palette 7's own indices 1/3) with real colors, not invented ones
        # -- pulled from the US ROM's own bank_1B HUD-palette data (PC
        # $0DD660, PaletteData_hud_00 in the underlying disassembly; this
        # exact [black, black outline, white fill, red accent] quad
        # recurs at multiple rows in that same table, e.g. PC $0DD668:
        # $0000, $0018, $7FFF, $0000 -- the standard "real text" palette
        # shape this ROM uses elsewhere, not a one-off guess). This is
        # additive: color indices
        # 28/30 (background and the existing bright fill) are untouched,
        # and this doesn't modify Palettes_HUD (jpdasm bank_1B -- a shared
        # sprite/HUD accent table, also used for sword/shield glow effects
        # and other HUD elements, confirmed via a static ROM byte-pattern
        # search -- not safe to edit directly), so jp mode and every other
        # CGRAM 7 consumer are unaffected. Palette 3 (SMALL role) is never
        # touched at all now -- that role always renders JP's own font
        # (see us_credits_font_asset), whose glyphs were designed around
        # this exact degenerate palette; patching it would only break them.
        #
        # A one-off write placed right here (before Credits_
        # InitializePolyhedral even runs) doesn't stick, confirmed live
        # (byte dump at the actual caption frame): Credits_
        # InitializeTheActualCredits's own JSL PaletteLoad_HUD -- which
        # runs later in its normal flow, right after Credits_
        # InitializePolyhedral -- reloads all of PAL_HUD wholesale from
        # ROM's Palettes_HUD table, clobbering anything written before it.
        # Placed in the footer instead (below, after the reproduced JSL
        # PaletteLoad_HUD), it patches PAL_HUD (symbols_wram.asm: PAL_HUD7
        # = $7EC538, index 1 at +2, index 3 at +6 -- "Main palette block
        # written to CGRAM", the source bank_00's own DOCGRAM-gated DMA,
        # #_008B87's .skip_CGRAM block, copies wholesale into real CGRAM
        # every time DOCGRAM/$7E0015 is set) *after* PaletteLoad_HUD's own
        # reload -- confirmed live via an exec/write trace this still
        # wasn't enough: Credits_LoadOverworldScene_PrepGFX's own
        # Overworld_CopyPalettesToCache call (bank_02, #_02C4CD -- runs on
        # every credits location, right after its font upload) re-syncs
        # PAL_HUD wholesale from PALB_HUD ($7EC300 -- "Palette buffer 2",
        # a *second*, more foundational WRAM copy that PAL_HUD itself gets
        # refreshed from) a few frames later, clobbering PAL_HUD again
        # with PALB_HUD's still-original values. Patching PALB_HUD too
        # (same indices/offsets, $7EC300 base instead of $7EC500) closes
        # that gap: whichever buffer a given reload actually reads from,
        # it now already has the new colors.
        credits_font_palette_patch = instructions(
            [
                "REP #$20",
                "LDA.w #$0000",
                "STA.l $7EC33A ; PALB_HUD7 index 1 (staging copy)",
                "STA.l $7EC53A ; PAL_HUD7 index 1 (CGRAM color 29)",
                "LDA.w #$0018",
                "STA.l $7EC33E ; PALB_HUD7 index 3 (staging copy)",
                "STA.l $7EC53E ; PAL_HUD7 index 3 (CGRAM color 31)",
            ]
        )
        # Reproduces Credits_InitializeTheActualCredits's own instructions
        # from JSL Credits_InitializePolyhedral through JSL PaletteLoad_HUD
        # (inclusive) verbatim -- not hand-transcribed -- so the palette
        # patch above can run right after the real PaletteLoad_HUD call
        # instead of before it.
        init_block = jp.block("Credits_InitializeTheActualCredits")
        poly_index = init_block.find("JSL Credits_InitializePolyhedral")
        palette_load_index = init_block.find("JSL PaletteLoad_HUD")
        reproduced_through_palette_load = init_block.lines[
            poly_index : palette_load_index + 1
        ]
        footer_resume = credits_palette_load_call + 4
        clear_attribution_tilemap_footer = Assembly(
            [
                instruction("PLP"),
                *reproduced_through_palette_load,
                *credits_font_palette_patch,
                instruction(f"JML ${footer_resume:06X}"),
            ]
        ).ensure_anchors()
    else:
        clear_attribution_tilemap_footer = Assembly.from_content(
            [
                "PLP",
                "JSL Credits_InitializePolyhedral",
                f"JML ${credits_init_resume:06X}",
            ]
        ).ensure_anchors()
    relocation.place(
        Assembly(
            clear_attribution_tilemap_header.lines
            + clear_attribution_tilemap_lines
            + clear_attribution_tilemap_footer.lines
        ),
        0x20B060,
        "Synchronous VRAM $6800-$6FFF clear before credits unblank -- fixes"
        " the transition-into-credits garbage flash.",
    )
    if credits_font == "us":
        # Credits_InitializeTheActualCredits's own JSL PaletteLoad_HUD only
        # runs once, at the very start of the whole credits sequence -- but
        # each individual location shown during the sequence gets its own
        # fresh PAL_HUD reload too, via Credits_LoadOverworldScene_PrepGFX's
        # (bank_02) own unconditional JSL PaletteLoad_HUD (confirmed live:
        # the very first caption still showed degenerate colors even with
        # the patch above in place, byte dump at the actual caption frame).
        # Same fix, same hook shape, at this second call site: reproduce
        # the displaced JSL PaletteLoad_HUD, patch right after it, resume.
        overworld_palette_load_call = (
            jp.block("Credits_LoadOverworldScene_PrepGFX")
            .line("JSL PaletteLoad_HUD")
            .address
        )
        if overworld_palette_load_call is None:
            msg = (
                "Credits_LoadOverworldScene_PrepGFX: JSL PaletteLoad_HUD is"
                " not a live-anchored instruction"
            )
            raise ValueError(msg)
        relocation.place(
            Assembly(
                [
                    note(
                        "; [ENG-CREDITS-FONT] Same palette patch as EN_"
                        "CreditsFont_ClearAttributionTilemap's own footer"
                        " (see credits_font_upload above for the full"
                        " explanation) -- Credits_LoadOverworldScene_"
                        "PrepGFX reloads PAL_HUD fresh on every credits"
                        " location, not just once at startup, so this"
                        " needs the same patch reapplied here too."
                    ),
                    note("CreditsOverworldPaletteLoad_Patch:"),
                    instruction("JSL PaletteLoad_HUD"),
                    *credits_font_palette_patch,
                    instruction("SEP #$20"),
                    instruction(f"JML ${overworld_palette_load_call + 4:06X}"),
                ]
            ).ensure_anchors(),
            0x20B120,
            "Credits_LoadOverworldScene_PrepGFX: patch the palette again"
            " after its own PaletteLoad_HUD reload.",
        )
    return relocation


def credits_bank(
    sources: Sources,
    *,
    changes: bool,
    keep_jp_credits: bool = False,
    gba_text_fixes: bool = True,
    flute_is_ocarina: bool = False,
) -> Relocation:
    """Bank ``$2E``: the JP credits reader + tables, mirror-placed, kept on
    JP's own font and text -- the JP credits are already English, and its
    own (bolder) font reads better than a US-font swap, so the credits look
    different from the rest of the (US-fonted) game by design.

    Unless ``keep_jp_credits``, a handful of JP 1.0 mistakes fixed in the US
    release are applied on top: THE LOYAL PRIEST -> SAGE, FINGER WEBS FOR
    SALE -> FLIPPERS FOR SALE (re-centered -- the US left it off-center),
    OCARINA BOY -> FLUTE BOY, GANNON'S TOWER -> GANON'S TOWER, and a new
    ENGLISH SCRIPT WRITERS attribution section the US added and JP never
    had. JP and US share one underlying character-code space for both
    ``CreditsTextLine`` and the ``Credits_AddEndingSequenceText`` "SMALL"
    captions (only the glyph-tile *pointer* tables ever differed -- now
    unused), so every US byte reused below is verbatim, not re-encoded.

    Every ``Credits_AddEndingSequenceText`` "SMALL"/location caption follows
    one consistent centering rule: even length -> perfectly centered, odd
    length -> off by exactly one column (it can't split evenly). The three
    replaced/moved-length captions above stay on that same convention
    rather than whatever their new word length happens to land on; one
    further caption whose unmodified JP text broke the rule (SAHASRALAH'S
    HOMECOMING) is nudged back in line too, still gated on
    ``keep_jp_credits`` since it's a deviation from pure JP 1.0.

    ``gba_text_fixes`` (independent of ``keep_jp_credits``: GBA-era wording
    fixes, not JP-1.0-mistake fixes) recenters and relabels the SMITHERY
    location caption to SMITHY, and changes the VENUS. QUEEN OF FAERIES
    caption to FAIRIES, matching the same wording changes
    :func:`gba_dialogue_text_fixes` makes to the dialogue. ``flute_is_
    ocarina`` (also independent of ``keep_jp_credits``) skips
    ``edit_flute_boy`` entirely, leaving JP's own pristine "OCARINA BOY
    PLAYS AGAIN" caption in place (already correctly centered for its own
    length) instead of converting it to "FLUTE BOY PLAYS AGAIN".

    Placed in three contiguous groups (JP interleaves them with credits code
    we do not relocate). The readers return long -- they are reached across
    banks via bank_0E landing pads -- and keep only their EN_ names (the bare
    aliases live under the pads in bank_0E), so no hooks/shared here.
    """
    jp = sources.jp

    groups: tuple[tuple[Block | Pool, ...], ...] = (
        (
            Block("Credits_CharacterToTile"),
            Block("CreditsBlankFillTile"),
            Pool("CreditsTextLine"),
        ),
        (
            Pool("Credits_AddNextAttribution"),
            Block("Credits_AddNextAttribution"),
        ),
        (
            Pool("Credits_AddEndingSequenceText"),
            Block("Credits_AddEndingSequenceText"),
        ),
    )
    readers = ("Credits_AddNextAttribution", "Credits_AddEndingSequenceText")

    def edit_priest_to_sage(block: Assembly) -> None:
        # 14 chars, even -- centered means $C4D2, not JP's $C4D0 (which
        # centered 16-char PRIEST).
        block.splice(
            "; SMALL: THE LOYAL PRIEST",
            notes(["; SMALL: THE LOYAL SAGE"])
            + datas(
                [
                    "dw $6962, $1B00 ; VRAM $C4D2 | 28 bytes",
                    "db $2D, $21, $1E, $9F, $25, $28, $32, $1A",
                    "db $25, $9F, $2C, $1A, $20, $1E",
                ]
            )
            + notes([""]),
            until="; TOP: SANCTUARY",
        )

    def edit_flippers_for_sale(block: Assembly) -> None:
        # 17 chars, odd -- US kept JP's $C4CC (centered for 20-char FINGER
        # WEBS), landing off-center; $C4CE is the best achievable (-1).
        block.splice(
            "; SMALL: FINGER WEBS FOR SALE",
            notes(["; SMALL: FLIPPERS FOR SALE"])
            + datas(
                [
                    "dw $6762, $2100 ; VRAM $C4CE | 34 bytes",
                    "db $1F, $25, $22, $29, $29, $1E, $2B, $2C",
                    "db $9F, $1F, $28, $2B, $9F, $2C, $1A, $25",
                    "db $1E",
                ]
            )
            + notes([""]),
            until="; TOP: ZORA'S WATERFALL",
        )

    def edit_flute_boy(block: Assembly) -> None:
        # 21 chars, odd -- JP's $C4C8 centered 23-char OCARINA BOY (also
        # odd) at -1; keeping it for the 2-shorter FLUTE BOY drifts to -3.
        # $C4CA is the best achievable (-1), same idea as FLIPPERS above.
        block.splice(
            "; SMALL: OCARINA BOY PLAYS AGAIN",
            notes(["; SMALL: FLUTE BOY PLAYS AGAIN"])
            + datas(
                [
                    "dw $6562, $2900 ; VRAM $C4CA | 42 bytes",
                    "db $1F, $25, $2E, $2D, $1E, $9F, $1B, $28",
                    "db $32, $9F, $29, $25, $1A, $32, $2C, $9F",
                    "db $1A, $20, $1A, $22, $27",
                ]
            )
            + notes([""]),
            until="; TOP: HAUNTED GROVE",
        )

    def edit_recenter_sahasralahs_homecoming(block: Assembly) -> None:
        # 23 chars, odd -- the one unmodified-text caption at +1 instead of
        # the -1 every other JP caption of its length uses; $C4C8 is -1.
        block.replace(
            "dw $6562, $2D00 ; VRAM $C4CA | 46 bytes",
            "dw $6462, $2D00 ; VRAM $C4C8 | 46 bytes",
            1,
        )
        # The apostrophe's top-stroke is a separate 1-character .chargfx
        # entry, drawn on the row above to sit over the apostrophe in the
        # line just moved -- it has to shift the same -1 column with it.
        block.replace(
            "dw $4F62, $0100 ; VRAM $C49E | 2 bytes",
            "dw $4E62, $0100 ; VRAM $C49C | 2 bytes",
            1,
        )

    def edit_venus_faeries_to_fairies(block: Assembly) -> None:
        # FAERIES -> FAIRIES, same length (7 chars) -- no recentering needed.
        block.replace(
            "; SMALL: VENUS. QUEEN OF FAERIES",
            "; SMALL: VENUS. QUEEN OF FAIRIES",
            1,
        )
        block.replace(
            "db $1F, $1A, $1E, $2B, $22, $1E, $2C",
            "db $1F, $1A, $22, $2B, $22, $1E, $2C",
            1,
        )

    def edit_smithery_to_smithy(block: Assembly) -> None:
        # 6 chars, even -- centered means +1 column from JP's 8-char
        # SMITHERY on both the TOP and BOTTOM copies (+$0100 raw, the same
        # per-column step edit_recenter_sahasralahs_homecoming's apostrophe
        # shift above uses).
        block.splice(
            "; TOP: SMITHERY",
            notes(["; TOP: SMITHY"])
            + datas(
                [
                    "dw $ED62, $0B00 ; VRAM $C5DA | 12 bytes",
                    "db $6F, $69, $65, $70, $64, $75",
                ]
            )
            + notes([""]),
            until="; BOTTOM: SMITHERY",
        )
        block.splice(
            "; BOTTOM: SMITHERY",
            notes(["; BOTTOM: SMITHY"])
            + datas(
                [
                    "dw $0D63, $0B00 ; VRAM $C61A | 12 bytes",
                    "db $95, $8F, $8B, $96, $8A, $9B",
                ]
            )
            + notes([""]),
            until=".chargfx_kak2",
        )

    def edit_ganons_tower(block: Assembly) -> None:
        # Both forms drop the second "N" (GANNON'S -> GANON'S); the
        # centering offset is unaffected either way (compact form still
        # solves 2*offset+len=32 one letter shorter; the LEVEL# form uses a
        # fixed offset like every other LEVEL# line, not per-line centering).
        block.splice(
            "; 8 GANNON'S TOWER",
            notes(["; 8 GANON'S TOWER", ".line6B"])
            + datas(
                [
                    "db $08, $1D ; spacing, 0x1E bytes",
                    "db $5B, $9F, $63, $5D, $6A, $6B, $6A, $77",
                    "db $6F, $9F, $70, $6B, $73, $61, $6E",
                ]
            )
            + notes([""]),
            until="; LEVEL8 GANNON'S TOWER",
        )
        block.splice(
            "; LEVEL8 GANNON'S TOWER",
            notes(["; LEVEL8 GANON'S TOWER", ".line6C"])
            + datas(
                [
                    "db $03, $27 ; spacing, 0x28 bytes",
                    "db $0B, $04, $15, $04, $0B, $81, $9F, $89",
                    "db $83, $90, $91, $90, $9D, $95, $9F, $96",
                    "db $91, $99, $87, $94",
                ]
            )
            + notes([""]),
            until="; TOTAL GAMES PLAYED",
        )

    def edit_add_english_script_writers(block: Assembly) -> None:
        # US-only attribution JP never had. Bytes are US's own, verbatim
        # (same character-code space as JP -- see the docstring); the two
        # header bytes are a centering offset + encoded length, also
        # independent of JP/US, so no re-derivation is needed either.
        # New lines get descriptive labels (not JP's or US's own .lineXX
        # numbering, which already means different things in each ROM) and
        # land right before "SPECIAL THANKS TO", mirroring where the US
        # inserts them relative to its own "TOMOAKI KUROUME".
        block.insert_before(
            "; SPECIAL THANKS TO",
            notes(
                [
                    "; [ENG-CREDITS] ENGLISH SCRIPT WRITERS (not in JP 1.0;",
                    "; added to match the US release's attribution).",
                    ".lineEnglishScriptWriters",
                ]
            )
            + datas(
                [
                    "db $05, $2B ; spacing, 0x2C bytes",
                    "db $1E, $27, $20, $25, $22, $2C, $21, $9F",
                    "db $2C, $1C, $2B, $22, $29, $2D, $9F, $30",
                    "db $2B, $22, $2D, $1E, $2B, $2C",
                ]
            )
            + notes(["", "; [ENG-CREDITS] DANIEL OWSEN", ".lineDanielOwsenA"])
            + datas(
                [
                    "db $0A, $17 ; spacing, 0x18 bytes",
                    "db $60, $5D, $6A, $65, $61, $68, $9F, $6B",
                    "db $73, $6F, $61, $6A",
                ]
            )
            + notes(["", "; [ENG-CREDITS] DANIEL OWSEN", ".lineDanielOwsenB"])
            + datas(
                [
                    "db $0A, $17 ; spacing, 0x18 bytes",
                    "db $86, $83, $90, $8B, $87, $8E, $9F, $91",
                    "db $99, $95, $87, $90",
                ]
            )
            + notes(
                ["", "; [ENG-CREDITS] HIROYUKI YAMADA", ".lineHiroyukiYamadaA"]
            )
            + datas(
                [
                    "db $08, $1D ; spacing, 0x1E bytes",
                    "db $64, $65, $6E, $6B, $75, $71, $67, $65",
                    "db $9F, $75, $5D, $69, $5D, $60, $5D",
                ]
            )
            + notes(
                ["", "; [ENG-CREDITS] HIROYUKI YAMADA", ".lineHiroyukiYamadaB"]
            )
            + datas(
                [
                    "db $08, $1D ; spacing, 0x1E bytes",
                    "db $8A, $8B, $94, $91, $9B, $97, $8D, $8B",
                    "db $9F, $9B, $83, $8F, $83, $86, $83",
                ]
            )
            + notes([""]),
        )
        # Grow the .pointers run between JP's own TOMOAKI KUROUME
        # (.line42) and SPECIAL THANKS TO (.line43) from JP's existing 10
        # blank slots to the US's own 26-slot pacing around this section
        # (8 blank / title / 3 blank / name / name / 2 blank / name / name
        # / 8 blank) -- a net +16 entries.
        after = block.find("dw .line42-.data") + 1
        before = block.find("dw .line43-.data", after)
        if before - after != 10:  # noqa: PLR2004
            msg = (
                "credits_bank: expected 10 blank .pointers entries between "
                f".line42/.line43, found {before - after}"
            )
            raise ValueError(msg)
        blank = "dw .line01-.data"
        new_pointers = (
            [blank] * 8
            + ["dw .lineEnglishScriptWriters-.data"]
            + [blank] * 3
            + ["dw .lineDanielOwsenA-.data", "dw .lineDanielOwsenB-.data"]
            + [blank] * 2
            + [
                "dw .lineHiroyukiYamadaA-.data",
                "dw .lineHiroyukiYamadaB-.data",
            ]
            + [blank] * 8
        )
        block.lines[after:before] = datas(new_pointers)
        # .stats_lines (same pool) holds 14 absolute $CA*2 position markers
        # -- compared against the scroll position to know when to draw each
        # quest-history number (Credits_AddNextAttribution) -- and all 14
        # sit after the insertion point above (CA 326-390, vs. the
        # insertion's CA 269), so each needs +$20 (16 lines * 2) to keep
        # landing on the same logical quest-history rows, not 16 lines early.
        block.splice(
            "dw $028C",
            datas(
                [
                    "dw $02AC",
                    "dw $02B4",
                    "dw $02BC",
                    "dw $02C4",
                    "dw $02CC",
                    "dw $02D6",
                    "dw $02DE",
                    "dw $02E6",
                    "dw $02EE",
                    "dw $02F6",
                    "dw $02FE",
                    "dw $0306",
                    "dw $030E",
                    "dw $032C",
                ]
            ),
            until="pool off",
        )

    def edit_grow_pointer_bound(block: Assembly) -> None:
        # +32 bytes (16 more .pointers entries, 2 bytes each) for
        # edit_add_english_script_writers's timeline growth above -- this is
        # how far the reader is willing to walk .pointers before stopping.
        block.replace("CPY.w #$0310", "CPY.w #$0330", 1)

    # The readers are the hooks: a same-bank caller in bank_0E's credits driver
    # (not relocated) reaches them, so caller-analysis gives each a landing pad
    # in bank_0E's freed ROM.
    relocation = Relocation(
        hooked=readers,
        carried=frozenset(
            member.name for members in groups for member in members
        ),
        pad_region="NULL_0EEDFB",
        pad_header=_redirect("2E", "en_credits.asm"),
        changes=changes,
    )
    for members in groups:
        group = jp.concat(list(members))
        names = {member.name for member in members}
        if changes:
            if names & set(readers):
                group.return_long()
            if not keep_jp_credits:
                if "CreditsTextLine" in names:
                    edit_ganons_tower(group)
                    edit_add_english_script_writers(group)
                if "Credits_AddNextAttribution" in names:
                    edit_grow_pointer_bound(group)
                if "Credits_AddEndingSequenceText" in names:
                    edit_priest_to_sage(group)
                    edit_flippers_for_sale(group)
                    if not flute_is_ocarina:
                        edit_flute_boy(group)
                    edit_recenter_sahasralahs_homecoming(group)
            if gba_text_fixes and "Credits_AddEndingSequenceText" in names:
                edit_venus_faeries_to_fairies(group)
                edit_smithery_to_smithy(group)
        start = require_start(group)
        relocation.place(
            group, mirror(start), f"credits region, mirror of JP ${start:06X}."
        )
    return relocation


def item_menu(
    sources: Sources,
    *,
    changes: bool,
    gba_text_fixes: bool = True,
    flute_is_ocarina: bool = False,
) -> Relocation:
    """Bank ``$2D``: the US item menu, mirror-placed. Four entry routines get
    a DBR-setting trampoline (their bodies become ``<name>_body`` and return
    long); the name-text tables get the US content, with the 2-row US
    ``Mirror`` table filling JP's 4-row slot -- reserved in BOTH builds
    (real copies in full, same-size labelled padding in the baseline) so the
    tables after it -- and ``MenuCursorPositions`` (always ``+$20``) -- keep a
    stable address and don't cascade in the baseline<->full review diff.

    ``gba_text_fixes`` (only meaningful alongside ``changes``) edits two
    entries in ``ItemMenuNameText_Bottles`` -- shared by the bottle-content
    cursor menu and the equipped-Y-item name display: the bottled "FAERIE"
    entry becomes "FAIRY" (single row, left-aligned, unchanged layout), and
    "GOOD BEE" becomes "GOLDEN"/"BEE" (two rows, top-left/bottom-right,
    matching ``ItemMenuNameText_Mirror``'s "MAGIC"/"MIRROR" layout) -- the
    names the GBA re-release and every later game use. ``flute_is_ocarina``
    (also only meaningful alongside ``changes``, independent of
    ``gba_text_fixes``) renames ``ItemMenuNameText_Flute``'s "FLUTE" entry
    (both the inactive and active icon-state copies) to "OCARINA" -- the
    JP original's name for the item, still single-row/left-aligned since
    "OCARINA" is exactly 7 letters.
    """
    us, jp = sources.us, sources.jp
    entries = (
        "UpdateBottleMenu",
        "DrawAbilityText",
        "SetLiftText",
        "DrawEquippedYItem",
    )
    # (US block, EN name, return-long). Entry routines -> <name>_body.
    bodies = (
        ("BottleMenuCursorPosition", "BottleMenuCursorPosition", False),
        ("UpdateBottleMenu", "UpdateBottleMenu_body", True),
        ("DrawMenuIcon", "DrawMenuIcon", False),
        ("DrawAbilityText", "DrawAbilityText_body", True),
        ("SetLiftText", "SetLiftText_body", True),
        ("DrawEquippedYItem", "DrawEquippedYItem_body", True),
    )
    # One contiguous US region so the Mirror expansion cascades the rest
    # exactly.
    name_text = (
        "ItemMenuNameText_YItems",
        "ItemMenuNameText_Bottles",
        "ItemMenuNameText_Powder",
        "ItemMenuNameText_Flute",
        "ItemMenuNameText_Mirror",
        "ItemMenuNameText_Bow",
        "ItemIcons",
        "AbilityText",
    )

    # The four entries are the hooks. Their same-bank JSR callers (the item-
    # menu dispatch) stay in bank_0D, so caller-analysis gives each a landing
    # pad in bank_0D's freed ROM (the bare name -> a DBR trampoline here). The
    # relocated blocks (bodies + name-text + cursor) are what moves along.
    relocation = Relocation(
        hooked=entries,
        carried=frozenset(
            [us_name for us_name, _, _ in bodies]
            + [*name_text, "MenuCursorPositions"]
        ),
        pad_region="NULL_0DAFDD",
        pad_header=_redirect("2D", "en_item_menu.asm"),
        changes=changes,
    )

    def place(body: Assembly, org: int) -> None:
        relocation.place(body, org, f"item-menu @ ${org:06X}.")

    if changes:
        place(dbr_trampolines(entries), 0x2DE100)
    for us_name, en_name, long_return in bodies:
        body = us.block(us_name, comments=False)
        if changes and long_return:
            body.return_long(restore_bank=True)  # PLB before RTL (DBR restore)
        if en_name != us_name:
            body.lines[0].label = en_name  # rename the routine label
        place(body, mirror(require_start(body)))

    def expand_mirror_slot(region: Assembly) -> None:
        # US ItemMenuNameText_Mirror is 2 dw rows; JP's slot wants 4. Fill the
        # extra 2-row ($20) span in BOTH builds -- the real US rows duplicated
        # (full), or same-size labelled padding (baseline) -- so the tables
        # after Mirror (and MenuCursorPositions) keep their +$20 slot in either
        # build and never cascade in the review diff.
        start = next(
            index
            for index, line in enumerate(region.lines)
            if line.label and line.label.endswith("ItemMenuNameText_Mirror")
        )
        end = next(
            index
            for index in range(start + 1, len(region.lines))
            if region.lines[index].is_top_level_label
        )
        rows = [
            line for line in region.lines[start:end] if line.opcode == "dw"
        ]
        if len(rows) != 2:  # noqa: PLR2004
            msg = f"Mirror: expected 2 US rows, found {len(rows)}"
            raise ValueError(msg)
        if changes:
            fill = [Line.from_line(str(row)) for row in rows]
            for copy, src in zip(fill, rows, strict=True):
                copy.size = src.size
        else:
            fill = notes(
                [
                    "; [ENG-ITEM] reserved: JP's Mirror slot is 4 rows (US",
                    "; has 2); the full build fills these with the copies.",
                ]
            )
            for row in rows:
                fill += _row_fill("dw", "$FFFF", row.size // 2)
        region.lines[end:end] = fill

    def restore_ability_jp_rows(region: Assembly) -> None:
        # Rows 10-11 of AbilityText keep the JP tile values (not US).
        jp_rows = [
            line.arguments
            for line in jp.block("AbilityText", comments=False).lines
            if line.opcode == "dw"
        ]
        row = 0
        inside = False
        for line in region.lines:
            if line.label and line.label.endswith("AbilityText"):
                inside = True
                continue
            if inside and line.is_top_level_label:
                break
            if inside and line.opcode == "dw":
                if row in (10, 11):
                    line.arguments = jp_rows[row]
                row += 1

    def fix_good_bee_to_golden_bee(region: Assembly) -> None:
        # US calls this bottled bee the "Good Bee"; the GBA re-release (and
        # every later game) calls it the "Golden Bee". Two rows, matching
        # ItemMenuNameText_Mirror's MAGIC/MIRROR layout: GOLDEN top-left,
        # BEE bottom-right. Menu-font tiles are alphabetical from $2550 (A).
        index = region.find(
            "dw $2556, $255E, $255E, $2553, $24F5, $2551, $2554, $2554"
        )
        top, bottom = region.lines[index - 1], region.lines[index]
        top.arguments = [
            "$2556",
            "$255E",
            "$255B",
            "$2553",
            "$2554",
            "$255D",
            "$24F5",
            "$24F5",
        ]  # GOLDEN
        bottom.arguments = [
            "$24F5",
            "$24F5",
            "$24F5",
            "$24F5",
            "$24F5",
            "$2551",
            "$2554",
            "$2554",
        ]  # BEE
        bottom.comment = " golden bee"

    def fix_faerie_to_fairy(region: Assembly) -> None:
        # Single row, left-aligned (matches BEE's own single-row layout
        # just below it).
        row = region.line(
            "dw $2555, $2550, $2554, $2561, $2558, $2554, $24F5, $24F5"
        )
        row.arguments = [
            "$2555",
            "$2550",
            "$2558",
            "$2561",
            "$2568",
            "$24F5",
            "$24F5",
            "$24F5",
        ]  # FAIRY
        row.comment = " fairy"

    def fix_flute_to_ocarina(region: Assembly) -> None:
        # JP calls this the Ocarina (matching every other Zelda game); the
        # US translation renamed it "Flute". Single row, left-aligned, same
        # layout as FLUTE -- OCARINA is exactly 7 letters, so it still fits
        # the 8-tile row with one blank left over. Both bottle-menu entries
        # (inactive/active flute icon state) are byte-identical.
        ocarina = [
            "$255E",
            "$2552",
            "$2550",
            "$2561",
            "$2558",
            "$255D",
            "$2550",
            "$24F5",
        ]
        needle = "dw $2555, $255B, $2564, $2563, $2554, $24F5, $24F5, $24F5"
        first = region.find(needle)
        region.lines[first].arguments = list(ocarina)
        region.lines[first].comment = " ocarina"
        second = region.find(needle, first + 1)
        region.lines[second].arguments = list(ocarina)
        region.lines[second].comment = " ocarina"

    region = us.concat([*name_text])
    expand_mirror_slot(region)  # both builds: real copies (full) or padding
    if changes:
        edit_table: dict[str, list[Edit]] = {
            "AbilityText": [restore_ability_jp_rows]
        }
        if gba_text_fixes:
            edit_table["ItemMenuNameText_Bottles"] = [
                fix_faerie_to_fairy,
                fix_good_bee_to_golden_bee,
            ]
        if flute_is_ocarina:
            edit_table["ItemMenuNameText_Flute"] = [fix_flute_to_ocarina]
        region.apply_edit_table(edit_table)
    place(region, mirror(require_start(region)))

    cursor = us.block("MenuCursorPositions", comments=False)
    # Cursor positions follow the Mirror table, whose 4-row slot is reserved in
    # both builds, so the cursor always sits $20 past its US mirror.
    place(cursor, mirror(require_start(cursor)) + 0x20)
    return relocation


def file_select(
    sources: Sources,
    *,
    changes: bool,
    null_padbyte_threshold: int,
    nop_padbyte_threshold: int = DEFAULT_NOP_PADBYTE_THRESHOLD,
    us_title_screen: bool = True,
) -> Relocation:
    """Bank ``$2C``: the US file-select / copy / erase / name-entry. Pulled
    whole by recursion from the entry points; a few come from JP for the
    dual-save backup the US dropped. US-sourced routines are mirror-placed at
    their US address + $200000 (stable, recognisable addresses); the JP
    restorations -- too big for their US slots -- plus the IRQ handler and save
    migrator pack into the free bottom of the bank, all gaps labelled NULL_.

    ``IntroLogoTilemap`` is one of the entry points despite belonging to the
    title screen, not file-select: it is module 0's slot in the per-module
    tilemap-rebuild table this bank's other tilemap entries (``FileSelect
    Tilemap`` and siblings) belong to, so something has to fill it regardless.
    US by default (matching the US logo/sword art :func:`title_screen_
    graphics` repoints ``GFX_40``/``GFX_41`` at); with ``us_title_screen``
    off (``--title-screen jp``), the JP-native version instead.
    """
    us, jp = sources.us, sources.jp
    # The entry points -- every symbol un-relocated code reaches this subsystem
    # by. They are both the recursion ROOTS (closure pulls the transitive
    # helpers/data) and the HOOKS kept bare so unmodified callers resolve.
    hooks = frozenset(
        {
            "Module01_FileSelect",
            "Module02_CopyFile",
            "Module03_KILLFile",
            "Module04_NameFile",
            "CopySaveToWRAM",
            "IntroLogoTilemap",
            "FileSelectTilemap",
            "FileSelectKILLFileTilemap",
            "FileSelectCopyFileTilemap",
            "NamePlayerTilemap",
        }
    )
    # Pulled from JP (restore the dual-save backup the US ROM removed).
    jp_blocks = frozenset(
        {
            "FileSelect_InitializeGFX",
            "KILLFile_FindFileIndices",
            "NameFile_EraseSave",
            "InitializeSaveFile",
        }
        | (set() if us_title_screen else {"IntroLogoTilemap"})
    )

    def build_irq_handler() -> Assembly:
        # The name-entry V-IRQ raster split, derived from JP
        # NoIRQThread's .not_mode7 sublabel: bank_00 keeps the IRQ-active check
        # (LDA $0128 / BEQ .IRQ_inactive) and JMLs here for the split (see
        # apply_base_edits); we JML back to .IRQ_inactive ($00821B). JP already
        # loads the #$38 split; we only add the name-entry ($0128 == $01) case
        # that loads #$74 instead, reusing JP's #$38 as the default.
        irq = jp.block("NoIRQThread").subblock(".not_mode7")
        # bank_00 keeps the active-check; replace it with our entry label.
        irq.splice(
            ".not_mode7",
            [
                "; [ENG-FS] Name-entry V-IRQ raster split (JP",
                "; NoIRQThread.not_mode7 + the $0128 name-entry case).",
                "IRQActiveHandler:",
            ],
            until="LDA.w TIMEUP",
        )
        irq.insert_before(
            "LDA.b #$38",
            [
                *instructions(
                    [
                        "LDA.w $0128",
                        "CMP.b #$01",
                        "BNE .default_split",
                        "LDA.b #$74",
                        "BRA .store_split",
                    ]
                ),
                ".default_split",
            ],
        )
        irq.insert_after("LDA.b #$38", [".store_split"])
        irq.append(instructions(["JML $00821B"]))
        return irq

    def edit_initialize_gfx(block: Assembly) -> None:
        # JP FileSelect_InitializeGFX -> English (name-banner + BG3 setup).
        # Each site is made byte-neutral between baseline/full (real edit vs.
        # same-size NOP filler in baseline) so this routine's size -- and thus
        # every later routine's address -- never depends on `changes`.
        if changes:
            block.delete("STZ.w $0AB6", until="JSL PaletteLoad_UnderworldSet")
            block.insert_after(
                "STA.w $0AA9",
                instructions(["LDA.b #$06", "STA.w $0AB6", "STA.w $0710"]),
            )
        else:
            # unedited site is 3 bytes; the full build's replacement is 8.
            block.insert_after(
                "STZ.w $0AB6",
                nop_fill(
                    5,
                    "ENG-FS",
                    "reserved: see FileSelect_InitializeGFX>0AB6",
                    nop_padbyte_threshold=nop_padbyte_threshold,
                ),
            )
        if changes:
            block.delete("LDA.b #$01", until="STA.w $0AB2")
            block.insert_after(
                "JSL PaletteLoad_OWBG3", instructions(["LDA.b #$00"])
            )
        # (already byte-neutral: LDA.b #$01 (2) -> LDA.b #$00 (2); no baseline
        # padding needed.)
        if changes:
            block.insert_after(
                "STA.w $0AA1", instructions(["LDA.b #$51", "STA.w $0AA2"])
            )
        else:
            block.insert_after(
                "STA.w $0AA1",
                nop_fill(
                    5,
                    "ENG-FS",
                    "reserved: see FileSelect_InitializeGFX>0AA1",
                    nop_padbyte_threshold=nop_padbyte_threshold,
                ),
            )

    def edit_erase(block: Assembly) -> None:
        # JP NameFile_EraseSave -> English blank fill + flags. Same
        # byte-neutral-between-builds treatment as edit_initialize_gfx.
        if changes:
            block.delete("STZ.w $0B10", until="STZ.w $0B15")
            block.insert_after("STA.w $0128", instructions(["STZ.w $0B10"]))
            block.replace("LDA.b #$3E", "LDA.b #$83", count=1)
            block.replace("LDA.w #$019C", "LDA.w #$01F0", count=1)
            block.replace("LDA.w #$018C", "LDA.w #$00A9", count=1)
        # (the STZ.w $0B10 move and the 3 replaces are already byte-neutral;
        # no baseline padding needed for any of them.)
        if changes:
            # widen: prepend two blank-writes so all 6 words ($3D5-$3DF) clear.
            block.insert_before(
                "STA.l $7003D9,X",
                instructions(["STA.l $7003D5,X", "STA.l $7003D7,X"]),
            )
        else:
            block.insert_before(
                "STA.l $7003D9,X",
                nop_fill(
                    8,
                    "ENG-FS",
                    "reserved: see NameFile_EraseSave>7003D9",
                    nop_padbyte_threshold=nop_padbyte_threshold,
                ),
            )

    def edit_widen_idle_exit(block: Assembly) -> None:
        # FileSelect_HandleInput's very first "no relevant input this frame"
        # check has to reach all the way to the routine's own .exit -- it was
        # already close to BEQ's +-127 limit in the unmodified US source, and
        # this routine's own growth here (edit_check_blank_name's real JSR or
        # its same-size baseline filler -- either way, both builds grow it
        # identically) tips it 1 byte over. Widen to an unconditional long
        # jump; applied unconditionally (not gated on `changes`) since both
        # builds need it equally.
        block.splice(
            "BEQ .exit",
            [*instructions(["BNE .not_idle", "JMP .exit"]), ".not_idle"],
        )

    def edit_check_blank_name(block: Assembly) -> None:
        # US FileSelect_HandleInput: a valid save ($BF,X != 0, about to fall
        # into the normal load-and-play path) whose name is all-blank (see
        # ConvertJP's blank fallback for an unconvertible JP import) needs a
        # name before it can be played. The actual check/redirect lives
        # out-of-line (FileSelect_NameIsBlankRedirect, name_fix.asm) so this
        # inline footprint is just a 3-byte JSR -- this routine's own
        # pre-existing short branches (BEQ/BRA to its ".exit") sit mid-routine,
        # before this insertion point, with little headroom to spare. Packed
        # low (see low_names) since it grows past its original
        # US slot -- byte-neutral between builds like FileSelect_InitializeGFX/
        # NameFile_EraseSave's edits, so a same-size filler stands in for the
        # baseline.
        if changes:
            block.insert_after(
                "BEQ .no_file_there",
                instructions(["JSR FileSelect_NameIsBlankRedirect"]),
            )
        else:
            block.insert_after(
                "BEQ .no_file_there",
                nop_fill(
                    3,
                    "ENG-FS",
                    "reserved: see FileSelect_HandleInput>blank-name check",
                    nop_padbyte_threshold=nop_padbyte_threshold,
                ),
            )

    def edit_add_rename_submodule(block: Assembly) -> None:
        # Append submodule 4 (NameFile_SetupRename, see name_fix.asm /
        # edit_check_blank_name) to the dispatch table. Module04_NameFile is
        # packed low (see low_names), so -- like edit_initialize_gfx/
        # edit_erase -- it needs the same size in both builds; a `dl` entry
        # is data, never executed as code, so the baseline stand-in is a
        # plain 3-byte placeholder rather than nop_fill (which is for
        # instruction-flow filler).
        if changes:
            block.insert_after(
                "dl NameFile_DoTheNaming", datas(["dl NameFile_SetupRename"])
            )
        else:
            block.insert_after(
                "dl NameFile_DoTheNaming",
                notes(["; [ENG-FS] reserved: unused submodule 4 slot"])
                + datas(["dl $000000"]),
            )

    def edit_handle_input_marker(block: Assembly) -> None:
        # US FileSelect_HandleInput reads the $3E5 marker natively; JP's is
        # $3E1. Same length either way, but gated (not a plain tuple) now
        # that this routine is packed low (see low_names) and so always
        # edited -- baseline should still read the untouched US marker,
        # matching edit_initialize_gfx/edit_erase's own already-byte-neutral
        # replaces, which are gated the same way.
        if changes:
            block.replace("LDA.l $7003E5,X", "LDA.l $7003E1,X", count=1)

    # Player name: widen the SRAM name to a contiguous 6-word field at $3D5
    # (ending just before the JP checksum marker at $3E1), so the US
    # 6-char-native routines just need their base repointed -4 ($7003D9 ->
    # $7003D5). FileSelect_HandleInput/DrawDeaths touch only post-name data
    # (SCHKSM $3E1, deaths $401), so they are unchanged.
    shift = ("LDA.l $7003D9,X", "LDA.l $7003D5,X", 1)
    name_edits: dict[str, list[Edit]] = {
        "CopyFile_SelectionAndBlinker": [shift],
        "CopyFile_TargetSelectionAndBlink": [shift],
        "FileSelect_CopyNameToStripes": [shift],
        "NameFile_DrawSelectedCharacter": [shift],
        # the char write + the terminator read -- both $7003D9,X.
        "NameFile_DoTheNaming": [("$7003D9", "$7003D5", 2)],
        # the special-name cheat check (name-word 1 -> mushroom + items).
        "InitializeSaveFile": [("LDA.l $7003D9", "LDA.l $7003D5", 1)],
    }
    edits: dict[str, list[Edit]] = {
        "FileSelect_HandleInput": [
            edit_handle_input_marker,
            edit_check_blank_name,
            edit_widen_idle_exit,
        ],
        "FileSelect_DrawDeaths": [("LDA.l $700405,X", "LDA.l $700401,X", 1)],
        "ReinitializeFileSelectGraphics": [
            (
                "JSL PaletteLoadForFileSelect",
                "JSL USFS_PaletteLoadForFileSelect",
                1,
            )
        ],
        "FileSelect_InitializeGFX": [edit_initialize_gfx],
        "NameFile_EraseSave": [edit_erase],
        "Module04_NameFile": [edit_add_rename_submodule],
        **name_edits,
    }

    def edit_mark_checksum_only(block: Assembly) -> None:
        # A zero-byte, scope-transparent label marking InitializeSaveFile's
        # checksum-recompute tail (right before the `LDX.b $00` that reloads
        # the slot offset), so StampNewFileTag's rename-only branch below can
        # jump straight there. Anchored on the copy-loop's own exit branch
        # since `LDX.b $00` alone isn't unique in this routine (it also
        # appears right after the checksum loop).
        block.insert_after("BPL .copy_next", ["#checksum_only:"])

    # Save compatibility: migrate foreign slots to our format (skippable with
    # save_compat=False). The migrator (save_migration.asm resource) lands in
    # this bank; it is *invoked* at boot from bank_00 (see apply_base_edits) so
    # it runs before InitializeMemoryAndSRAM's $3E1 sanity zeroing.
    if changes:
        # Tag freshly-created files as ours -- byte-neutrally, so
        # InitializeSaveFile keeps its size and its US mirror slot. The $55AA
        # marker's own `STA.l $7003E1,X` (4 bytes) is swapped for a `JSL
        # StampNewFileTag` (also 4 bytes); the stub (built in the tail below)
        # writes the marker AND our $410 tag, then RTLs -- or, for a
        # blank-name rename (see edit_check_blank_name), skips straight to
        # the checksum recompute instead (edit_mark_checksum_only's label).
        edits["InitializeSaveFile"].append(
            ("STA.l $7003E1,X", "JSL StampNewFileTag", 1)
        )
        edits["InitializeSaveFile"].append(edit_mark_checksum_only)

    def stamp_new_file_stub() -> list[Line]:
        return (
            notes(
                [
                    ";" + "=" * 99,
                    "; [ENG-FS] New-file format tag, out-of-line so",
                    "; InitializeSaveFile stays byte-neutral (a JSL replaces",
                    "; its $55AA-marker STA -- same 4 bytes). A holds $55AA",
                    "; on entry. If the marker's already set, this save was",
                    "; never erased -- a blank-name rename, not a new file",
                    "; -- so skip the marker/bomb-wall/deaths/starting-items",
                    "; reset and jump straight to the checksum recompute.",
                    "StampNewFileTag:",
                ]
            )
            + instructions(
                [
                    "CMP.l $7003E1,X    ; already $55AA? (A is still $55AA)",
                    "BEQ .rename_only",
                    "STA.l $7003E1,X    ; the $55AA marker",
                    "LDA.w #$0006",
                    "STA.l $700410,X    ; our-format tag",
                    "RTL",
                ]
            )
            + notes([".rename_only"])
            + instructions(
                [
                    "TSC                 ; discard the JSL return address",
                    "CLC                 ; (3 bytes) so RTL doesn't land back",
                    "ADC.w #$0003        ; after this JSL -- we're jumping",
                    "TCS                 ; past it, to the checksum tail",
                    "JML checksum_only",
                ]
            )
        )

    # ---- placement: mirror the US routines, pack the JP restorations low ----
    # Bank $2C is the +$200000 mirror of US bank $0C, where every routine we
    # pull originates. US-sourced routines are MIRROR-placed at their US
    # address + $200000, so each keeps a stable, source-recognisable address
    # and a byte-neutral edit never renumbers its neighbours (a clean
    # baseline<->full diff). The JP restorations are BIGGER than the US slots
    # they'd occupy (that is why they were pulled), so they cannot sit in the
    # US layout; they -- plus the CopyFile_FindFileIndices stub that falls into
    # KILLFile_FindFileIndices, and the IRQ handler + save migrator -- pack
    # into the free bottom of the bank ($2C8000..). Every gap is a labelled
    # NULL_ free-ROM block (the disassembly convention we owe the expanded
    # banks).
    reachable = us.closure(sorted(hooks), recursive=True)
    carried = frozenset(entry.name for entry in reachable)
    packed_blocks = list(
        dict.fromkeys(
            entry.name for entry in reachable if isinstance(entry, Block)
        )
    )
    # Intro_SetStripesAndAdvance is a US #-label the closure cannot reach; the
    # relocated code JSRs it in every build (baseline included), so pull it
    # explicitly or the baseline has a dangling reference and won't assemble.
    packed_blocks += ["Intro_SetStripesAndAdvance"]

    # The routines that CANNOT mirror-place (JP restorations bigger than their
    # US slot). This is also the pack order; the CopyFile_FindFileIndices /
    # KILLFile_FindFileIndices pair must stay adjacent (the former sets A=$07
    # and falls straight into the latter).
    # InitializeSaveFile is NOT here: a short BNE in NameFile_DoTheNaming pins
    # it to the US layout, so it stays mirror-placed (it fits the US slot
    # exactly, and its tag edit is byte-neutral -- see StampNewFileTag).
    # FileSelect_HandleInput and Module04_NameFile are ALSO here now: neither
    # is branch-distance-pinned (FileSelect_HandleInput is only JSL'd;
    # Module04_NameFile is only reached via bank_00's RunModule data table),
    # and mirror-placing left zero slack before the next mirrored routine in
    # each case (CopySaveToWRAM sits byte-adjacent to FileSelect_HandleInput
    # in the source US ROM), so growing either one here (edit_check_blank_name/
    # edit_add_rename_submodule) would overlap the next mirrored routine.
    # Packing them low needs the same byte-neutral treatment as the other
    # low_names members.
    low_names = [
        "FileSelect_InitializeGFX",
        "CopyFile_FindFileIndices",
        "KILLFile_FindFileIndices",
        "NameFile_EraseSave",
        "FileSelect_HandleInput",
        "Module04_NameFile",
        "IntroLogoTilemap",
    ]
    low_set = frozenset(low_names)

    def pull(name: str, *, edited: bool) -> Assembly:
        # An asar pool is emitted inline just before its routine, so a pulled
        # segment is (pool + block) and its first byte is the pool's, not the
        # routine label's -- callers that need the placement address use
        # require_start(), not address_of().
        origin = jp if name in jp_blocks else us
        seg: list[Line] = []
        if name in origin.pool_names:
            seg += origin.pool(name, comments=False).lines
        seg += origin.block(name, comments=False).lines
        asm = Assembly(seg)
        if edited:
            asm.apply_edits(edits.get(name, []))
        return asm

    def width(asm: Assembly) -> int:
        return sum(line.size for line in asm.lines)

    # Packed low region, from the bank base.
    low_org = 0x2C8000
    low: list[Line] = []
    pc = low_org
    for name in low_names:
        # edited=True unconditionally: edit_initialize_gfx/edit_erase (the
        # only low_names entries with edits) branch on `changes` themselves
        # and pad the baseline to match the full build's size (see above), so
        # every build is already byte-neutral here -- no growth to absorb.
        body = pull(name, edited=True)
        low += body.lines
        pc += width(body)
    # NB: the IRQ handler + save migrator are appended AFTER the US region
    # (see `tail` below), not here -- keeping the low region (and thus the big
    # bridge NULL_) byte-identical between baseline and full.
    low_end = low_org + sum(line.size for line in low)

    # Routines mirror-placed at their US address + $200000, ascending order,
    # with a labelled NULL_ gap bridging the unpulled space between them. A
    # US-sourced segment keys on its own first byte (an inline pool precedes
    # the routine); a JP restoration kept here (InitializeSaveFile) takes its
    # US-equivalent slot instead of its own JP mirror.
    def us_place(name: str, seg: Assembly) -> int:
        if name in jp_blocks:
            return mirror(us.address_of(name))
        return mirror(require_start(seg))

    us_segs = [
        (us_place(name, seg), name, seg)
        for name in packed_blocks
        if name not in low_set
        for seg in [pull(name, edited=changes)]
    ]
    us_segs.sort(key=lambda item: item[0])
    us_region: list[Line] = []
    first_mirror = us_segs[0][0]
    pc = first_mirror
    for target, name, seg in us_segs:
        if target < pc:
            msg = f"file_select: US mirror overlap at {name}"
            raise ValueError(msg)
        if target > pc:
            us_region += free_space(
                pc, target - pc, null_padbyte_threshold=null_padbyte_threshold
            )
            pc = target
        us_region += seg.lines
        pc += width(seg)

    # New code JP has no home for: the name-entry IRQ handler and the save
    # migrator (both reached by label from bank_00). Appended after the US
    # region so their full-only presence never shifts the low region or the
    # mirrored routines -- it lands in the free tail instead.
    tail: list[Line] = []
    if changes:  # IRQ handler built from a sublabel, not a top-level block
        tail += build_irq_handler().lines
        tail += stamp_new_file_stub()  # InitializeSaveFile's byte-neutral tag
        # ensure_anchors gives the hand-written asm the #_<hex> per-line labels
        # the disassembly convention wants; render re-stamps them at emit time.
        tail += (
            Assembly.from_content(_save_migration_lines())
            .ensure_anchors()
            .lines
        )
        tail += Assembly.from_content(_name_fix_lines()).ensure_anchors().lines

    if first_mirror < low_end:
        msg = "file_select: low region overruns the US mirror start"
        raise ValueError(msg)
    bridge = free_space(
        low_end,
        first_mirror - low_end,
        null_padbyte_threshold=null_padbyte_threshold,
    )
    bank = low + bridge + us_region + tail

    relocation = Relocation(
        hooked=tuple(sorted(hooks)), carried=carried, changes=changes
    )
    relocation.place(
        Assembly(bank),
        low_org,
        "file-select: US routines mirror-placed; JP restorations packed low, "
        "IRQ handler + save migrator in the tail.",
    )
    return relocation


def graphics(*, changes: bool) -> Relocation:
    """Bank ``$26``: the US menu/HUD + file-select graphics sheets (the
    kana/Latin font sheets ``GFX_DC``/``GFX_DD`` and the file-select "linoleum"
    background ``GFX_39``). Binary US-ROM slices, ``incbin``\\ 'd; each claims
    its freed JP name so the game's tables reach the US art.
    """
    # (JP sheet name, its US-ROM slice under bin/gfx/). Packed from $268000;
    # referenced by symbol, so the exact address is asar's job.
    sheets = (
        ("GFX_DC", "us_gfx_dc.2bppc"),  # US menu/file-select font ($69)
        ("GFX_DD", "us_gfx_dd.2bppc"),  # US menu/file-select font ($6A)
        ("GFX_39", "us_gfx_39.3bppc"),  # US file-select "linoleum" bg ($39)
    )
    # Why each freed JP sheet is dead (annotates the base definition).
    dead_notes = {
        "GFX_39": (
            "; [ENG-GFX] JP menu-bg sheet $39 repointed at the US linoleum "
            "(GFX_39, usgfx.asm);",
            "; this JP data is no longer referenced.",
        ),
        "GFX_DC": (
            "; [ENG-GFX] JP menu-font sheet $69 repointed at the US font "
            "(GFX_DC, usgfx.asm).",
        ),
        "GFX_DD": (
            "; [ENG-GFX] JP $6A repointed at the US font (GFX_DD, usgfx.asm); "
            "data stays live via GFX_71.",
        ),
    }
    lines: list[Line] = []
    for name, filename in sheets:
        lines.append(note(f"{name}:"))
        lines.append(
            incbin_line(f"bin/gfx/{filename}", us_assets.asset(filename).size)
        )
        lines.append(note(""))
    relocation = Relocation(
        hooked=tuple(name for name, _ in sheets),
        hook_notes=dead_notes,
        changes=changes,
    )
    relocation.place(
        Assembly(lines[:-1]),
        0x268000,
        "US graphics sheets (menu/HUD + file-select).",
    )
    return relocation


def file_select_palette(*, changes: bool) -> Relocation:
    """Bank ``$27``: the file-select US palette overlay. Four CGRAM rows differ
    JP<->US; ``USFS_PaletteLoadForFileSelect`` (our own routine, hand-written
    -- not sourced from either disassembly, so it is full-only; see
    ``resources/usfs_palette_load.asm``) runs the stock US load and then
    overlays those four US palettes from ``USFS_Palette`` (a plain US-ROM
    ``PaletteData`` byte slice, ``incbin``\\ 'd from ``bin/gfx/us_palette.bin``
    -- present in both builds, like the graft's other raw asset extractions).
    ``file_select`` repoints the one ``JSL`` at it. Keeping all four palettes
    here (not filling JP's empty ``owanim_00`` in bank_1B) keeps every US
    palette in the graft and the base edit a pure repoint -- the shared
    ``PaletteLoadSingle`` reads bank $1B only, so a 2nd-MB copy is unreachable
    through the stock load anyway.
    """
    relocation = Relocation(changes=changes)
    lines: list[Line] = []
    if changes:
        lines += (
            Assembly.from_content(_resource_lines("usfs_palette_load.asm"))
            .ensure_anchors()
            .lines
        )
        lines.append(note(""))
    lines += [
        note(
            "; Four US file-select palettes (colors 1-7 each), CGRAM-row"
            " order 5, 7, 9, 11."
        ),
        note("USFS_Palette:"),
        incbin_line(
            "bin/gfx/us_palette.bin", us_assets.asset("us_palette.bin").size
        ),
    ]
    relocation.place(
        Assembly(lines),
        0x278000,
        "file-select US palette overlay + data.",
        namespace=False,
    )
    return relocation


def title_screen(
    sources: Sources, *, changes: bool, us_title_screen: bool
) -> Relocation:
    """Bank ``$28``: the US title-screen sword animation -- on by default
    (``--title-screen jp`` opts out). Same present-but-inert shape as
    :func:`graphics`: this bank's content is always built and placed, in
    every build (including ``--baseline``) -- only ``hooked`` (and so the
    base-bank landing pad and every alias) is conditional on
    ``us_title_screen``, so a build with the flag off leaves the base
    pointing at JP's own title screen untouched, same as
    :func:`title_screen_graphics` already does for its sheets.

    ``Intro_SwordStab`` (the US-only dispatch state) and everything it reaches
    -- ``Intro_InitLogoSword``/``Intro_HandleLogoSword`` and the three
    ``LogoSword_*`` sub-states, with their pools -- are pulled whole (by name,
    not recursion: they are scattered non-contiguously across US bank $0C
    among plenty of unrelated code, so a closure/concat pull would have to
    declare every gap between them; the small, fixed list below is simpler)
    from the US disassembly, unedited. Also pulled: bank $00's
    ``IntroLogoPaletteFadeIn``/``IntroTitleCardPaletteFadeIn`` (US-only
    wrappers around the shared ``PaletteFilter_RestoreAdditive``, which both
    ROMs have) -- these two, not the shared routine itself, since they end in
    ``RTL`` and it ends in ``RTS``: a same-bank ``JSR`` (as their one caller,
    below, used in the original US ROM) is exactly what an ``RTS`` needs, but
    that caller is a cross-bank ``JSL`` here, which needs (and gets, since
    these two get relocated alongside it) an ``RTL`` partner -- calling the
    bare ``RTS`` routine directly across banks silently leaves its pushed bank
    byte on the stack, corrupting every return after it.

    Also pulled (end of ``names``, see the comment there): ``Intro_
    FadeLogoIn``/``Intro_PopSubtitleCard``/``Intro_TrianglesBeforeAttract``,
    the three Module00_Intro dispatch states the US ROM restructured around
    the sword -- unedited except ``Intro_PopSubtitleCard``'s own 2 small
    edits (also below). Everything genuinely novel -- with no ROM original to
    pull instead -- is hand-written in ``resources/title_screen.asm``:
    ``TitleScreenUS_DrawTriangle``'s subtype-dispatch logic and
    ``Module00_Intro_Dispatch``'s own hybrid table. Every source lands in
    this one relocation so cross-references between pulled and hand-written
    pieces (by their bare US names) get ``EN_``-namespaced together in one
    pass.
    """
    us = sources.us
    # Source order (pool immediately precedes its same-named routine); each
    # entry pulls its own pool automatically if one exists under that name.
    names = (
        "Intro_SwordStab",
        "Intro_InitLogoSword",
        "Intro_HandleLogoSword",
        "LogoSword_IdleState",
        "LogoSword_EyeTwinkle",
        "LogoSword_BladeShimmer",
        # PaletteFilter_RestoreAdditive: IntroLogoPaletteFadeIn/
        # IntroTitleCardPaletteFadeIn JSR it internally (same-bank in the
        # original US ROM); it has to move here too, not just get JSL'd from
        # its original bank_00 spot, since it returns with a plain RTS (a
        # cross-bank JSL to an RTS routine leaves the pushed bank byte on the
        # stack, corrupting every return after it).
        "PaletteFilter_RestoreAdditive",
        # Not IntroTitleCardPaletteFadeIn separately: it is an
        # address-transparent (#-prefixed) sublabel sharing this block's
        # tail (.finish), not a boundary block() stops at, so pulling
        # IntroLogoPaletteFadeIn already carries it along.
        "IntroLogoPaletteFadeIn",
        # Intro_LoadAllPalettes (bank $02): despite the shared name and the
        # shared caller (Intro_LoadAllPalettes_long, itself reached from
        # both JP's Intro_CreateTextPointers and US's Intro_LoadTextAnd
        # Palettes -- see title_screen()'s own module docstring on why that
        # caller stays JP-sourced), *this* routine's own body is not shared:
        # JP sets $0AB3=$04 (US: $05, a different PaletteLoad_OWBGMain
        # "area"), skips $0AAC/$0ABD entirely, and never calls
        # PaletteLoad_SpritePal0Left. Confirmed by diffing this routine's
        # text between the disassemblies directly, after a live-CGRAM diff
        # against the real US ROM (byte-identical VRAM, still visibly wrong
        # colors) pointed at a palette-*load*, not palette-*data*, mismatch.
        # Only one caller in the entire game (grepped), so wholesale
        # replacement -- not a byte-neutral in-place edit -- is safe; see
        # apply_base_edits's relocate_block for the JP-side redirect (a
        # same-size swap: Intro_LoadAllPalettes_long is coincidentally
        # exactly 4 bytes, matching a JML's own footprint, so no growth).
        "Intro_LoadAllPalettes",
        # AnimateSceneSprite_DrawCopyright (bank $0C): draws the "(c)1991
        # Nintendo" copyright line as 10 fixed OAM tiles from a `.groups`
        # pool. US's own copy is a *longer* pool (13 groups, count $0D vs
        # JP's $0A) reading "(c)1991,1992 Nintendo" -- an extra sheet tile
        # ($68, a comma) plus 2 reused digit tiles. Its caller
        # (AnimateSceneSprite_Copyright) is reached through a tail-jump
        # chain (JumpTableLocal computes same-bank-only targets by
        # construction -- it pops and reuses its own caller's return
        # address, so the bank byte can never change), so it cannot move
        # to this bank itself; only the callee moves, hooked with a
        # landing pad back in bank $0C's own free ROM (see
        # apply_base_edits/_wire_hooks). Pulled here (not the caller) so
        # apply_base_edits can hook it the normal way.
        "AnimateSceneSprite_DrawCopyright",
        # AnimateSceneSprite_AddObjectsToOAMBuffer: DrawCopyright's own JSR
        # target (same-bank in the original, self-contained -- no JSR/JSL
        # of its own, confirmed by reading its full body). Moving
        # DrawCopyright to this bank without moving this one too would
        # leave DrawCopyright's internal JSR unable to reach it (still
        # bank $0C). Its other 3 same-bank callers (AnimateSceneSprite_
        # Triangle/TitleCard/Sparkle, none touched by this feature) also
        # get a landing pad automatically -- one pad serves all of them,
        # since they all JSR the same address. Pulling it here (alongside
        # its one in-relocation caller) means DrawCopyright's own internal
        # reference to it gets EN_-namespaced too, resolving straight to
        # the bank-$28 copy instead of round-tripping through the bank-$0C
        # landing pad meant for the *other* 3 (still-external) callers.
        "AnimateSceneSprite_AddObjectsToOAMBuffer",
        # The three Module00_Intro dispatch states the US ROM restructured
        # around the sword (Module00_Intro_Dispatch, in title_screen.asm,
        # points here instead of at JP's own copies). Intro_FadeLogoIn and
        # Intro_TrianglesBeforeAttract are pulled unedited -- confirmed
        # byte-for-byte identical to JP's own versions plus the sword calls
        # already spliced in by the US ROM itself, nothing left to hand-edit.
        # Intro_PopSubtitleCard gets 2 edits below. Deliberately last in this
        # tuple: title_screen.asm's own hand-written content (LoadAllPalettes
        # onward) used to start exactly here, so appending instead of
        # inserting keeps this bank's layout, and the built ROM's bytes,
        # unchanged.
        "Intro_FadeLogoIn",
        "Intro_PopSubtitleCard",
        "Intro_TrianglesBeforeAttract",
    )
    # Intro_InitLogoSword has no RTS/RTL of its own: it falls straight through
    # into Intro_HandleLogoSword (both share the "pool Intro_HandleLogoSword"
    # .char/.position_x/.position_y tables, defined once, ahead of
    # InitLogoSword, in the source). Pulling per-name in `names` order would
    # place that pool between the two instead -- its data bytes would run
    # right into InitLogoSword's fallthrough as bogus instructions -- so its
    # one pool pull is hoisted ahead of the whole loop instead.
    sword_lines: list[Line] = [
        *us.pool("Intro_HandleLogoSword", comments=False).lines
    ]
    for name in names:
        if name == "AnimateSceneSprite_DrawCopyright":
            # Combined (not appended straight to sword_lines like the
            # other names below): its pool and body both need renaming
            # together before namespacing, see the comment block below.
            piece = Assembly(
                [
                    *us.pool(name, comments=False).lines,
                    *us.block(name, comments=False).lines,
                ]
            )
            # Its own internal call also needs to become a JSL: once
            # AddObjectsToOAMBuffer moves to this bank too, the reference
            # resolves to the (now same-bank) EN_ copy, which ends in RTL
            # (see below) to satisfy its *other*, still-external, landing-
            # pad-mediated callers -- a same-bank JSR/RTS pair would pop
            # one byte short of what that RTL pushes. Growing JSR (3
            # bytes) to JSL (4) is safe here (unlike a base-bank edit):
            # this whole piece gets freshly re-rendered at its own org, so
            # every line after it just shifts to match its new size.
            index = piece.find("JSR AnimateSceneSprite_AddObjectsToOAMBuffer")
            piece.lines[index] = instruction(
                "JSL AnimateSceneSprite_AddObjectsToOAMBuffer"
            )
            # DBR fix: this routine reads its .groups pool through direct-
            # page-indirect addressing (LDA.b ($08),Y), which resolves the
            # pool's bank from the *data* bank register (DBR), not the
            # code's own (program) bank -- moving code with JML/JSL never
            # touches DBR, so it stays whatever the pre-existing caller
            # chain set it to (bank $0C -- coincidentally correct for the
            # original, unmoved routine, since it lived there too). Found
            # by direct emulator inspection: cpu.dbr read $0C at the
            # indirect-read instruction, while $08/$09 correctly pointed
            # at .groups' real, bank-$28 address -- the pool pointer was
            # right, the bank DBR supplied for it was not, so every read
            # landed on unrelated bank-$0C bytes instead. A
            # dbr_trampolines() entry stub (PHB/PHK/PLB, i.e. save the
            # caller's DBR then set DBR = this bank) fixes it for this
            # whole call -- DBR then stays $28 through the subsequent JSL
            # to AddObjectsToOAMBuffer too, since JSL doesn't touch it
            # either. AddObjectsToOAMBuffer's own generic landing pad
            # deliberately does NOT get this treatment: its other 3
            # external callers (Triangle/TitleCard/Sparkle) have their
            # *own* pools still in bank $0C, so forcing DBR=$28 there
            # would break *their* reads instead. The pool (not just the
            # routine) has to be renamed alongside it -- dbr_trampolines'
            # own stub claims the bare "AnimateSceneSprite_DrawCopyright"
            # name, which would otherwise collide with the pool directive
            # still using it too.
            piece.suffix(["AnimateSceneSprite_DrawCopyright"], "_body")
            piece.return_long(restore_bank=True)
            sword_lines += dbr_trampolines(
                ["AnimateSceneSprite_DrawCopyright"]
            ).lines
            sword_lines += piece.lines
            continue
        if name in us.pool_names and name != "Intro_HandleLogoSword":
            sword_lines += us.pool(name, comments=False).lines
        block = us.block(name, comments=False)
        if name == "AnimateSceneSprite_AddObjectsToOAMBuffer":
            # Hooked with a landing pad (its same-bank-$0C callers stay
            # behind) -- a cross-bank JSL needs an RTL partner, not the
            # JSR-only RTS this routine ends in.
            block.return_long()
        if name == "Intro_PopSubtitleCard":
            # FadeMusicAndResetSRAMMirror stays behind in JP's own bank $0C
            # (a shared routine, not pulled here), so US's same-bank JMP.w
            # has to become a cross-bank JML now that this routine lives in
            # bank $28. Growing 3 -> 4 bytes is safe: the whole relocation is
            # freshly re-rendered at its own org, so every later line just
            # shifts to match -- unlike a base-bank edit, nothing here has a
            # fixed offset to preserve. instruction(), not a plain string,
            # so the new line's size is actually recomputed (splice() treats
            # a raw string as an unsized comment, not code).
            block.splice(
                "JMP.w FadeMusicAndResetSRAMMirror",
                [instruction("JML FadeMusicAndResetSRAMMirror")],
            )
            # US's plain `INC.b $11` would advance to dispatch slot 8, which
            # has to stay vanilla here (Attract_LoadNewScene and Save & Quit
            # both enter with $11=8 -- see Module00_Intro_Dispatch's own
            # comment) -- jump straight to slot 10 instead.
            block.splice(
                "INC.b $11", instructions(["LDA.b #$0A", "STA.b $11"])
            )
        sword_lines += block.lines
    hand = Assembly.from_content(
        _resource_lines("title_screen.asm")
    ).ensure_anchors()

    def triangle_tables() -> list[Line]:
        """The 4 OAM object tables TitleScreenUS_DrawTriangle picks between,
        pulled from the two US routines it merges instead of transcribed.
        Emitted as plain sublabels of that hand-written routine (its own
        enclosing scope), not as their own asar pool -- each pulled pool's
        own ``pool``/``pool off`` directive lines are dropped so the tables
        belong to the routine above them, exactly like the hand-transcribed
        copies they replace did.
        """
        title = us.pool("AnimateSceneSprite_DrawTriangle", comments=False)
        room = us.pool(
            "AnimateSceneSprite_DrawTriforceRoomTriangle", comments=False
        )
        lines: list[Line] = []
        for index, (source, sublabel, renamed) in enumerate(
            (
                (title, ".rightside_objects", None),
                (title, ".leftside_objects", None),
                # Renamed: the triforce-room routine names its own two
                # tables identically, and both pairs share one scope here.
                (room, ".rightside_objects", ".tf_rightside_objects"),
                (room, ".leftside_objects", ".tf_leftside_objects"),
            )
        ):
            piece = source.subblock(sublabel, comments=False)
            if renamed is not None:
                piece.rename_label(sublabel, renamed)
            if index:
                lines.append(note(""))
            lines += [line for line in piece.lines if line.opcode != "pool"]
        return lines

    hand.splice(
        "[PULLED] .rightside_objects",
        triangle_tables(),
        until="; Replaces Attract_Initialize",
    )
    # TitleScreenUS_AttractInitializePalettes's own body: not the whole US
    # Attract_Initialize (its own comment above explains why -- the routine
    # diverges again a little further in, past PaletteLoad_LinkArmorAnd
    # Gloves, in a way this build deliberately keeps JP-sourced: JP loads
    # attract-plaque MESSAGE 0110, US loads MESSAGE 0112, and this build
    # keeps JP message IDs everywhere -- see text()'s CreateMessagePointers
    # realignment), just this palette-loading prefix, sliced out of US's own
    # routine instead of transcribed.
    attract_initialize = us.block("Attract_Initialize", comments=False)
    prefix_start = attract_initialize.find("JSL TransferAttractPlaques")
    prefix_end = attract_initialize.find("JSL PaletteLoad_LinkArmorAndGloves")
    hand.splice(
        "[PULLED] US Attract_Initialize's own palette-loading prefix",
        attract_initialize.lines[prefix_start:prefix_end],
    )
    relocation = Relocation(
        hooked=(
            (
                "AnimateSceneSprite_DrawCopyright",
                "AnimateSceneSprite_AddObjectsToOAMBuffer",
            )
            if us_title_screen
            else ()
        ),
        # AnimateSceneSprite_DrawCopyright is one of AddObjectsToOAMBuffer's
        # own callers, and it moved along in this same relocation -- telling
        # needs_landing_pad that avoids a spurious pad-vs-alias mismatch for
        # that one internal reference (still-external callers force a pad
        # regardless, since they're the only ones left outside `carried`).
        carried=frozenset({"AnimateSceneSprite_DrawCopyright"}),
        pad_region="NULL_0CFFF6",
        pad_header=(
            "; [ENG-TITLE] AnimateSceneSprite_DrawCopyright/"
            "AddObjectsToOAMBuffer moved to bank $28 for the longer US"
            " copyright text (US title screen, the default).",
        ),
        changes=changes,
    )
    relocation.place(
        Assembly([*sword_lines, *hand.lines]),
        0x288000,
        "US title-screen sword animation (the default; --title-screen jp"
        " opts out).",
    )
    return relocation


def _local_label_address(block: Assembly, label: str) -> int:
    """The live address ``label`` (a dotted sublabel) points at within
    ``block`` -- a label-only line carries no anchor of its own, so this is
    the address of the first live-anchored line at/after its declaration."""
    for index, line in enumerate(block.lines):
        if line.label == label:
            for later in block.lines[index:]:
                if later.address is not None:
                    return later.address
            break
    msg = f"{label!r} is not a resolvable local label in this block"
    raise ValueError(msg)


def attract_text_timing_fix(sources: Sources, *, changes: bool) -> Relocation:
    """Bank ``$20`` (free space): widens the attract mode's fixed on-screen
    durations for its four auto-scrolling story captions -- the "Long ago,
    in the beautiful kingdom of Hyrule..." intro and the throne-room/prison/
    altar dungeon captions -- to the US ROM's own values, not JP 1.0's.

    Each caption's screen time is a plain WRAM countdown that fades the
    scene out on its own schedule, unrelated to the message's own pacing
    bytes (``set scroll speed``/``wait`` op-codes, ``text.asm``) -- this
    build's ``CreateMessagePointers`` realignment (:func:`text`) already
    carries those over untouched, keeping JP's message *IDs* while grafting
    in the English *content* behind them (offset by the 2 US-only Choose2
    High prompts JP lacks -- JP ID N holds the content at US ID N+2, e.g.
    JP $0111/``AttractScene_ThroneRoom`` holds US $0113, "...Agahnim came to
    Hyrule to release the seal. He eliminated the good King of Hyrule..."
    -- confirmed against the JP 1.0 disassembly's own gloss for Message_0111,
    which names him by his Japanese transliteration, アグニム). JP 1.0 sized
    each countdown for its own short, kanji-dense text; with the much longer
    English translation now living behind those same IDs, the countdown --
    and so the scene -- ends before the text finishes scrolling in,
    confirmed live. The US ROM hits the identical problem in reverse (its
    own longer English text against the same JP-authored code shape) and
    fixes it the same way: bigger duration values, pulled here verbatim
    from its own ROM, not re-derived.

    The intro caption's own counter (``$0200``, ``AttractDramatize_
    PolkaDots``) is already 16-bit in JP 1.0, so only its value needs
    raising (a plain, byte-neutral ``set_operand`` in ``apply_base_edits``,
    no relocation needed). The three dungeon captions share one 8-bit
    counter (``$64``) sized for JP's largest value (208, fits a byte); the
    US ROM's largest (576) does not, so US widens every read/write of it to
    16-bit too -- ``AttractText_DungeonSceneCountdown`` (below) replaces the
    shared decrement in ``Attract_DoTextInDungeonScene``, and each dungeon
    scene gets its own widened init (``AttractText_*Duration``) and widened
    consumer check (``AttractText_*Check``) -- seven small relocations
    total, each an exact-fit ``JML`` (no orphan bytes) with the original
    JP 1.0 instructions confirmed byte-for-byte identical to the US ROM's
    once REP/SEP-widened, aside from the duration values themselves.
    """
    jp = sources.jp

    def block_address(name: str, needle: str) -> int:
        address = jp.block(name).line(needle).address
        if address is None:
            msg = f"{name}: {needle!r} is not a live-anchored instruction"
            raise ValueError(msg)
        return address

    def duration_stub(label: str, block: str, new_value: str) -> Assembly:
        sta = block_address(block, "STA.b $64")
        resume = sta + 2
        return Assembly.from_content(
            [
                f"{label}:",
                "REP #$20",
                f"LDA.w #{new_value}",
                "STA.b $64",
                "SEP #$20",
                f"JML ${resume:06X}",
            ]
        ).ensure_anchors()

    def check_stub(
        label: str, block: str, branch: str, target_label: str
    ) -> Assembly:
        lda = block_address(block, "LDA.b $64")
        resume = lda + 4
        taken = _local_label_address(jp.block(block), target_label)
        return Assembly.from_content(
            [
                f"{label}:",
                "REP #$20",
                "LDA.b $64",
                "SEP #$20",
                f"{branch} .taken",
                f"JML ${resume:06X}",
                ".taken",
                f"JML ${taken:06X}",
            ]
        ).ensure_anchors()

    relocation = Relocation(changes=changes)
    relocation.place(
        duration_stub(
            "ThroneRoomDuration", "AttractScene_ThroneRoom", "$0210"
        ),
        0x20C000,
        "AttractScene_ThroneRoom: widen the on-screen-duration counter to"
        " the US ROM's own value (528 frames, was 192).",
    )
    relocation.place(
        duration_stub("PrisonDuration", "AttractScene_Prison", "$0240"),
        0x20C030,
        "AttractScene_Prison: widen the on-screen-duration counter to the"
        " US ROM's own value (576 frames, was 208).",
    )
    relocation.place(
        duration_stub("AltarDuration", "AttractScene_AgahnimAltar", "$00C0"),
        0x20C060,
        "AttractScene_AgahnimAltar: widen the on-screen-duration counter's"
        " storage to 16-bit (value unchanged, 192 frames) -- matches the"
        " US ROM, and keeps it consistent with the shared 16-bit"
        " countdown/checks below.",
    )
    dec_address = block_address("Attract_DoTextInDungeonScene", "DEC.b $64")
    relocation.place(
        Assembly.from_content(
            [
                "; [ENG-ATTRACT] Attract_DoTextInDungeonScene's own"
                " shared decrement, widened to match the three dungeon"
                " scenes' own now-16-bit $64 (see the *Duration stubs",
                "; above) -- matches the US ROM. DEC.b $64 is done here,"
                " under our own REP #$20, not by jumping back into JP"
                " 1.0's original DEC.b $64: DEC (like LDA/STA) is",
                "; width-sensitive, and the original instruction runs at"
                " its own native 8-bit width regardless of what called"
                " it, so jumping back into it would only decrement the",
                "; low byte -- the high byte would never change, and the"
                " 16-bit value would never actually reach 0 (confirmed"
                " live: the scene hung indefinitely once the caption",
                "; finished rendering, since RenderText's own pacing is"
                " independent of this counter). The original instruction"
                " is orphaned (dead, no longer jumped into) rather than",
                "; reused. BEQ .skip skips the decrement once the"
                " counter's already at 0 (avoiding underflow), same as"
                " JP 1.0's original 8-bit check.",
                "DungeonSceneCountdown:",
                "REP #$20",
                "LDA.b $64",
                "BEQ .skip",
                "DEC.b $64",
                ".skip",
                "SEP #$20",
                f"JML ${dec_address + 2:06X}",
            ]
        ).ensure_anchors(),
        0x20C090,
        "Attract_DoTextInDungeonScene: widen the shared $64 decrement to"
        " 16-bit.",
    )
    relocation.place(
        check_stub(
            "ThroneRoomCheck",
            "AttractDramatize_ThroneRoom",
            "BNE",
            ".continue_dramatization",
        ),
        0x20C0C0,
        "AttractDramatize_ThroneRoom: widen the $64 consumer check to"
        " 16-bit.",
    )
    relocation.place(
        check_stub(
            "PrisonCheck", "Dramaghanim_MoveAndSpin", "BEQ", ".continue"
        ),
        0x20C0F0,
        "Dramaghanim_MoveAndSpin (Prison): widen the $64 consumer check to"
        " 16-bit.",
    )
    relocation.place(
        check_stub("AltarCheck", "Dramagahnim_IdleGuiltily", "BNE", ".exit"),
        0x20C120,
        "Dramagahnim_IdleGuiltily (Agahnim's Altar): widen the $64 consumer"
        " check to 16-bit.",
    )
    return relocation


def title_screen_graphics(
    *, changes: bool, us_title_screen: bool
) -> Relocation:
    """Bank ``$29``: the US title-logo BG art (``GFX_40``/``GFX_41``) and the
    triforce+sword OBJ sheet (``GFX_7B``) -- on by default (``--title-screen
    jp`` opts out). Same incbin-and-repoint shape as :func:`graphics`,
    including its present-but-inert baseline behavior: this bank's sheets are
    always built and placed, every build -- unlike ``graphics``'s own sheets,
    which are unconditionally hooked (``_wire_hooks`` frees a hooked name
    from the base regardless of ``Relocation.changes``, which only controls
    the alias, not the free), the ``hooked`` tuple here is itself conditional
    on ``us_title_screen``, so a build with the flag off never frees these JP
    sheet names at all and the game keeps resolving them to JP's own art.

    Intro_InitializeDefaultGFX (bank $0C, shared/unedited) loads BG1's
    character memory once, at boot, by calling ``InitializeTilesets`` with
    ``$0AA1``=$23/``$0AA2``=$51 -- indices into ``SheetsTable_AA1`` (an
    8-sheet row per tileset ID) and ``SheetsTable_AA2`` (a 4-sheet row that
    overrides 4 of AA1's 8 slots when its own entry is nonzero), both bank
    $00. Both tables' *code* is shared/byte-identical between the
    disassemblies, but these two rows' *data* is not: AA1 row $23 reads
    ``$00/$39/$39/$72/$40/$41/$39/$0F`` in JP, ``$16/$39/$1D/$17/$40/$41/$39/
    $1E`` in US; AA2 row $51 (which overrides AA1 row $23's 4th slot, i.e.
    the $72/$17 one) reads ``$72/$40/$41/$39`` in JP, ``$17/$40/$41/$39`` in
    US (all confirmed by reading both disassemblies' own copies of the
    tables directly -- not inferred). Since this whole patcher's base is
    JP's bank $00, unmodified, the game loads *JP's* sheet list regardless
    of ``us_title_screen`` -- :func:`apply_base_edits`'s two row rewrites are
    what make it load these sheets instead (four here, plus $39 -- already
    repointed unconditionally by :func:`graphics`, file-select's linoleum --
    and $40/$41, the logo text). Confirmed empirically: swapping just these
    sheets, plus both row rewrites, reproduces the real US ROM's title
    screen VRAM content exactly (a diff of live VRAM dumps -- tilemap *and*
    character data -- came back byte-for-byte identical). $5C/$5D (also
    visible in a live US-ROM VRAM dump, but from
    ``DecompressAnimatedUnderworldTiles``'s separate, hardcoded-Y intro
    call, not this table) turned out to be unrelated to the title screen's
    own appearance -- repointing them made no difference to any pixel, so
    they are deliberately left JP-sourced (repointing them would have swapped
    dungeon water/lava tile animation game-wide for no benefit).

    GFX_A5 is a different case: it's the sword-blade/hilt *OBJ* sheet (the
    others above are all BG). It's reached through SheetsTable_AA3 row $7D
    (read by InitializeTilesets via $0AA3, set to $7D by the same
    Intro_InitializeDefaultGFX) -- confirmed by live-tracing VRAM writes in
    the real US ROM down to the exact tiles the sword sprite pool
    (EN_Intro_HandleLogoSword's .char/.position_x/.position_y, pulled in
    title_screen()) references. JP's row $7D leaves that slot's sheet ID
    unchanged (a no-op, since JP's title screen has no sword); US's sets it
    to sheet $32, i.e. GFX_A5 -- so apply_base_edits's row rewrite alone
    only gets as far as loading *JP's own* GFX_A5 into VRAM, which isn't the
    sword at all (confirmed: rendering those VRAM tiles showed an unrelated
    diagonal lattice pattern, not a blade). Repointing GFX_A5 itself here is
    the fix, and it is *not* fully scoped to the title screen the way the
    sheets above are: SheetsTable_AA3 row $42 (slot 2) also references sheet
    $32 by JP's own numbering, and $0AA3 is written from many real-gameplay
    sites (room/area tileset loads) -- so some other, not yet identified JP
    room whose tileset is row $42 would show the US sword-sheet content
    instead of its own real graphic with the (default) US title screen on.
    Accepted knowingly: contained to a flag (``--title-screen jp`` opts
    out), and unswapped left the intro with no sword sprite at all (the
    original, worse problem).
    """
    sheets = (
        ("GFX_16", "us_gfx_16.3bppc"),  # US title-screen intro-tileset 1/4
        ("GFX_17", "us_gfx_17.3bppc"),  # US title-screen intro-tileset 2/4
        ("GFX_1D", "us_gfx_1d.3bppc"),  # US title-screen intro-tileset 3/4
        ("GFX_1E", "us_gfx_1e.3bppc"),  # US title-screen intro-tileset 4/4
        ("GFX_40", "us_gfx_40.3bppc"),  # US title-logo BG art, sheet 1/2
        ("GFX_41", "us_gfx_41.3bppc"),  # US title-logo BG art, sheet 2/2
        ("GFX_7B", "us_gfx_7b.3bpp"),  # US triforce+sword OBJ sheet
        ("GFX_A5", "us_gfx_a5.3bppc"),  # US sword-blade/hilt OBJ sheet
    )
    dead_notes: dict[str, tuple[str, ...]] = {
        name: (
            f"; [ENG-TITLE] JP intro-tileset sheet "
            f"${name.removeprefix('GFX_')} repointed at the US sheet",
            f"; ({name}, usgfx.asm); only live with the (default) US"
            " title screen.",
        )
        for name, _ in sheets
    }
    lines: list[Line] = []
    for name, filename in sheets:
        lines.append(note(f"{name}:"))
        lines.append(
            incbin_line(f"bin/gfx/{filename}", us_assets.asset(filename).size)
        )
        lines.append(note(""))
    relocation = Relocation(
        hooked=(tuple(name for name, _ in sheets) if us_title_screen else ()),
        hook_notes=dead_notes,
        changes=changes,
    )
    relocation.place(
        Assembly(lines[:-1]),
        0x298000,
        "US title-screen graphics (the default; --title-screen jp opts"
        " out).",
    )
    return relocation


# ---------------------------------------------------------------------------
# base-disassembly edits
# ---------------------------------------------------------------------------
# Reaching the relocated graft from the unmodified JP banks has two parts, both
# keyed *by name or #_ address anchor* through the whole-program Rom (never by
# which file a routine lives in, and failing loud on upstream drift):
#   * _wire_hooks -- the base half of every hook, DERIVED from the relocations
#     and the program's callers: free each hooked JP name, and decide per name
#     whether the relocated copy's bare alias suffices or a landing pad is
#     needed (a same-bank caller stays behind). Nothing to re-list here.
#   * apply_base_edits -- the few edits that are NOT a plain hook.
def _en_pad(name: str) -> LandingPad:
    """A landing pad forwarding the freed JP ``name`` to its ``EN_`` copy."""
    return LandingPad(name, f"EN_{name}")


def _wire_hooks(english: Rom, jp: Rom, relocations: list[Relocation]) -> None:
    """Wire the base half of every hook, deciding alias-vs-pad from callers.

    For each relocation: split its :attr:`~graft.Relocation.hooked` names into
    the ones a bare alias reaches (recorded back on the relocation as
    ``aliased``, for ``EN_`` namespacing) and the ones a same-bank caller
    strands (a landing pad in the relocation's ``pad_region``). Then free every
    hooked name in the base. Run *before* the relocations are added, so the
    alias set it records is the one their pieces emit.
    """
    for relocation in relocations:
        pad_names = tuple(
            name
            for name in relocation.hooked
            if jp.needs_landing_pad(name, relocated=relocation.carried)
        )
        relocation.aliased = frozenset(relocation.hooked) - frozenset(
            pad_names
        )
        for name in relocation.hooked:
            english.hook(name, comment=relocation.hook_notes.get(name, ()))
        if pad_names:
            if relocation.pad_region is None:
                msg = f"hooks {pad_names} need a pad but no pad_region is set"
                raise ValueError(msg)
            english.landing_pads(
                relocation.pad_region,
                [_en_pad(name) for name in pad_names],
                header=relocation.pad_header,
            )


def _address_of_line(english: Rom, name: str, needle: str) -> int:
    """The live address of the ``needle``-matching line in block ``name``."""
    address = english.block(name).line(needle).address
    if address is None:
        msg = f"{name}: {needle!r} is not a live-anchored instruction"
        raise ValueError(msg)
    return address


def _address_of_pool_line(english: Rom, name: str, needle: str) -> int:
    """The live address of the ``needle``-matching line in pool ``name``."""
    address = english.pool(name).line(needle).address
    if address is None:
        msg = f"{name}: {needle!r} is not a live-anchored pool line"
        raise ValueError(msg)
    return address


def mothula_damage_bugfix(
    *, changes: bool, fix_mothula_bugs: bool = False
) -> Relocation:
    """Bank ``$29`` (free space): fixes Mothula (sprite ``$88``, Skull
    Woods) being immune to golden-sword swings/dashes, tempered-sword spin
    attacks, and golden-sword spin attacks -- with full GBA parity (the
    spike-immunity side is *not* disturbed either).

    Every hit Link lands resolves to a "damage class" (1-5: fighter/
    master/tempered/golden sword swing = 1/2/3/4, spin attack = one class
    higher than the same-tier swing (2/3/4/5), poke roughly one lower --
    ``.damage_class``, bank_06, inside ``CheckSwordDamageToSprite``) which
    indexes the target sprite's own row in a per-(sprite, class) "defense
    subclass" chart (``EnemyDamageCompressed``, decompressed once at boot
    by ``DecompressEnemyDamageSubclasses``, bank_02, into WRAM
    ``$7F6000``); that subclass then indexes ``DamageSubclassValue``
    (bank_0D) for the actual damage dealt. Mothula's own row sets defense
    subclass 0 (-> always 0 damage, i.e. "immune") for classes 4 and 5 --
    decompressed and decoded by hand (matches a longtime community
    finding, not guessed): this blocks golden-sword regular swings,
    golden-sword dashes/charges, and tempered-sword spin attacks (all
    class 4), and golden-sword spin attacks (class 5). This isn't purely
    an oversight: Mothula's own room is filled with the roaming spike
    hazards it periodically summons, which -- per the community damage-
    class legend (every "Spike Trap" icon sits in class 4 alongside those
    same sword hits) -- deal contact damage through this same class-4/5
    slot, so the original devs zeroed it out to stop Mothula chipping
    itself to death on its own spikes between player hits.

    Rather than un-zeroing classes 4/5 directly (which would also
    un-zero the spike-contact damage sharing that slot), this instead
    matches the community's known-working fix (``mothula_l4.ips``, prior
    art for the same bug, independently confirmed via its approach
    rather than copied byte-for-byte): redirect Mothula's own class-4/5
    *sword* hits to damage class 6 before they ever reach the table
    lookup, and give class 6 -- confirmed via the community damage-class
    legend to be used by nothing but the Bow, not by any sword tier or
    the Spike Trap -- a real (nonzero) defense subclass in Mothula's row.
    Classes 4 and 5 themselves are left at their original subclass 0, so
    spike contact damage (whatever code path deals it) is completely
    unaffected -- this is the difference from an earlier version of this
    fix, which edited classes 4/5 directly and would likely have
    reintroduced Mothula self-damaging on its own spikes.

    Three hooks:
    1. Right after ``DecompressEnemyDamageSubclasses``'s nibble-unpacking
       loop finishes (before it returns, bank_02) -- give Mothula's row a
       nonzero (1, matching classes 1-3's 2/4/8 damage scaling) defense
       subclass for class 6 only. Simplest done once, right after the
       whole table exists in WRAM, rather than editing the still-
       compressed ROM data.
    2. Inside ``CheckSwordDamageToSprite`` itself (bank_06), right after
       it restores the sprite-slot ``X`` (the ``PLX`` following the
       class-assignment/hammer-override logic) -- if the current sprite
       is Mothula and the class just assigned is 4 or 5, rewrite it to 6.
       Placed after ``PLX`` specifically because ``X`` is repurposed as a
       scratch table index for the class lookup itself and only holds the
       sprite slot again once restored.
    3. A second, genuinely distinct bug, found by tracing why a sword hit
       can still be "eaten" if Mothula happens to touch a spike in the
       same frame: ``ApplyDamageToSprite`` doesn't apply damage
       immediately. It queues the larger of this frame's damage and
       whatever's already queued into per-sprite WRAM ``$0CE2,X`` (so two
       hits in one frame don't lose the bigger one) -- *except* that a
       hit resolving to exactly 0 damage takes a different path
       (``.new_damage_less``/``.powder_damage``) that unconditionally
       ``JMP``s to ``.clear_queued_damage``, zeroing ``$0CE2,X``
       regardless of what was already queued. Since a spike touch always
       resolves to 0 damage against Mothula (defense subclass 0 for
       classes 4/5, untouched by hook 2 above), a spike touch landing in
       the same frame as a real sword hit wipes the sword's queued
       damage -- independent of which damage class the sword hit used,
       so hook 2 alone can't fix it (and neither does the community's
       ``mothula_l4.ips``, confirmed by the user's own testing). The
       actual spike-contact attempt happens in ``CheckSpriteTileProperty``
       (bank_06): any sprite standing on tile type ``$44`` (spike floor)
       unconditionally tries a class-4 hit via
       ``CheckAncillaDamageToSprite_preset``. Hooked right before that
       call to skip it entirely when the sprite is Mothula -- the gating
       checks above it (touch-eligibility, frozen-sprite bail-out) still
       run normally for every sprite including Mothula, only the damage
       attempt (and its reaction/knockback side effect, itself harmless
       since it would've been a 0-damage hit anyway) is skipped, so a
       spike touch on Mothula can no longer clear a real queued hit.
    """
    relocation = Relocation(changes=changes)
    if changes and fix_mothula_bugs:
        relocation.place(
            Assembly.from_content(
                [
                    "; [ENG-MOTHULA] Hooked from DecompressEnemyDamage"
                    "Subclasses's loop-exit check (bank_02) -- see"
                    " mothula_damage_bugfix's docstring. Mothula (sprite"
                    " $88) is DMGTABLE row $880-$88F; class 6 (Bow --"
                    " unused by any sword tier or the Spike Trap) sits",
                    "; at $886. Loop still running (X<$1000) falls"
                    " through to .continue_decompress, an unmodified"
                    " replay of the original loop-continue branch.",
                    "MothulaDamageFix:",
                    "CPX.w #$1000",
                    "BCC .continue_decompress",
                    "LDA.b #$01",
                    "STA.l $7F6886 ; Mothula: damage class 6 (was 0)",
                    "SEP #$30",
                    "RTL",
                    "",
                    ".continue_decompress",
                    "JML $02FC02 ; DecompressEnemyDamageSubclasses .next",
                ]
            ).ensure_anchors(),
            0x29D200,
            "Mothula damage-table bugfix (see apply_base_edits).",
        )
        relocation.place(
            Assembly.from_content(
                [
                    "; [ENG-MOTHULA] Hooked from CheckSwordDamageToSprite"
                    " (bank_06) right after it restores the sprite-slot X"
                    " -- see mothula_damage_bugfix's docstring. Mothula's",
                    "; own class 4/5 sword hits get redirected to class"
                    " 6 (spike-safe); every other sprite/class is"
                    " untouched. Replays the 2 displaced instructions"
                    " (LDA.b #$10 / STA.b $47) before rejoining.",
                    "MothulaClassRedirect:",
                    "LDA.w $0E20,X ; current sprite type",
                    "CMP.b #$88 ; Mothula",
                    "BNE .not_mothula",
                    "LDA.w $0CF2 ; damage class just assigned",
                    "CMP.b #$04",
                    "BEQ .redirect",
                    "CMP.b #$05",
                    "BNE .not_mothula",
                    ".redirect",
                    "LDA.b #$06",
                    "STA.w $0CF2",
                    ".not_mothula",
                    "LDA.b #$10",
                    "STA.b $47",
                    "JML $06ED8D ; CheckSwordDamageToSprite fallthrough"
                    " into ApplyDamageToSprite setup",
                ]
            ).ensure_anchors(),
            0x29D300,
            "Mothula damage-class redirect (see apply_base_edits).",
        )
        relocation.place(
            Assembly.from_content(
                [
                    "; [ENG-MOTHULA] Hooked from CheckSpriteTileProperty"
                    " (bank_06), right before the class-4 spike-floor"
                    " damage attempt -- see mothula_damage_bugfix's"
                    " docstring point 3. A 0-damage hit (guaranteed for",
                    "; Mothula, class 4/5 defense subclass is still 0)"
                    " would otherwise clear any already-queued real"
                    " sword damage this frame. Skips the attempt (and"
                    " its harmless-anyway reaction side effect)",
                    "; entirely for Mothula; every other sprite is"
                    " unaffected. Replays LDA.b #$04 / JSL before"
                    " rejoining for non-Mothula sprites.",
                    "MothulaSpikeImmunity:",
                    "LDA.w $0E20,X ; current sprite type",
                    "CMP.b #$88 ; Mothula",
                    "BEQ .skip_for_mothula",
                    "LDA.b #$04",
                    "JSL CheckAncillaDamageToSprite_preset",
                    "JML $06E849 ; CheckSpriteTileProperty continuation",
                    "",
                    ".skip_for_mothula",
                    "JML $06E856 ; ...skip_damage_or_recoil"
                    " (BRA .succeed_and_exit)",
                ]
            ).ensure_anchors(),
            0x29D400,
            "Mothula spike-contact damage-queue fix (see apply_base_edits).",
        )
    return relocation


def _wire_credits_overworld_palette_patch(
    english: Rom, credits_font: str
) -> None:
    """See EN_CreditsOverworldPaletteLoad_Patch (credits_font_upload, bank
    $20): Credits_LoadOverworldScene_PrepGFX reloads PAL_HUD fresh on every
    credits location (not just once at startup like Credits_
    InitializeTheActualCredits), so the font's palette patch needs
    reapplying here too.
    """
    if credits_font != "us":
        return
    overworld_palette_load_call = _address_of_line(
        english, "Credits_LoadOverworldScene_PrepGFX", "JSL PaletteLoad_HUD"
    )
    english.relocate_block(
        overworld_palette_load_call,
        "EN_CreditsOverworldPaletteLoad_Patch",
        resume=overworld_palette_load_call + 4,
    )


def apply_base_edits(
    english: Rom,
    *,
    keep_jp_credits: bool = False,
    weathercock_fix: bool = True,
    keep_religious_imagery: bool = False,
    epilepsy_fix: bool = True,
    us_title_screen: bool = True,
    low_health_beep: bool = True,
    fix_mothula_bugs: bool = False,
    credits_font: str = "jp",
) -> None:
    """Apply the base edits that are not plain hooks (see _wire_hooks)."""
    # Save compatibility: invoke the migrator (in bank $2C) from bank_00's
    # InitializeMemoryAndSRAM, BEFORE it zeroes any main slot's $3E1 word whose
    # value isn't the $55AA marker. A US save's marker lives at $3E5, so its
    # $3E1 would be cleared (breaking the checksum) before the file-select
    # runs. The relocated stub migrates, replays the displaced LDA, JMLs back.
    english.relocate_block(
        0x0087EF,
        "EN_MigrateAtBoot",
        resume=0x0087F3,
        comment=(
            "; [ENG-FS] migrate foreign save slots before the $3E1 sanity "
            "check (us_menu.asm).",
        ),
    )
    # V-IRQ active block -> the relocated name-entry raster-split handler.
    english.relocate_block(
        0x008205,
        "EN_IRQActiveHandler",
        resume=0x00820A,
        orphan=(0x38,),
        comment=(
            "; [ENG-FS] V-IRQ active block -> relocated handler in bank $2C "
            "(us_menu.asm).",
        ),
    )
    # Byte-neutral operand swaps: US BG3 blank tile, the US 126-tile text box,
    # and the three font-upload operands (-> our TheFont, uploaded to $E000).
    english.set_operand(
        0x008335,
        "LDA.w #$00A9",
        comment="[ENG-FS] US BG3 blank tile (was $0188 hex-pattern glyph)",
    )
    english.set_operand(
        0x008D02,
        "LDX.w #$07E0",
        comment="[ENG-TEXT] US 126-tile text box (was $0780 / 120)",
    )
    english.set_operand(0x00E557, "LDA.b #TheFont>>16")
    english.set_operand(0x00E563, "LDA.w #TheFont")
    english.set_operand(0x00E568, "LDX.w #(TheFont_end-TheFont)/2-1")
    if weathercock_fix:
        # The animated "weathercock" (windmill vane) VRAM tile ($1DE) cycles
        # between 3 pre-decompressed variants (a periodic DMA-source swap,
        # not a re-decompression). JP 1.0 and the US ROM leave the vane's
        # right end open -- missing the bordering pixel that closes off its
        # tip -- in all 3; the EU release adds it (bit 0 of the tile's
        # row-6-bitplane-0 and row-7-bitplane-1 bytes, cleared). Each pair
        # sits in a literal ("raw copy") run -- the tile's own pixel data
        # verbatim, not a compressed/back-referenced encoding -- confirmed
        # by decompressing GFX_5A/GFX_5B (bank $14) in Python and diffing
        # against the same sheets decompressed from an EU ROM.
        english.set_operand(
            _address_of_line(
                english, "GFX_5A", "db $3C, $10, $14, $00, $67, $00, $A7, $03"
            ),
            "db $3C, $10, $14, $00, $66, $00, $A7, $02",
            comment="[ENG-GFX] EU-matching weathercock fix",
        )
        english.set_operand(
            _address_of_line(
                english, "GFX_5A", "db $00, $34, $00, $2C, $08, $7F, $18, $BF"
            ),
            "db $00, $34, $00, $2C, $08, $7E, $18, $BF",
            comment="[ENG-GFX] EU-matching weathercock fix",
        )
        english.set_operand(
            _address_of_line(english, "GFX_5A", "db $0B, $00, $00"),
            "db $0A, $00, $00",
            comment="[ENG-GFX] EU-matching weathercock fix",
        )
        english.set_operand(
            _address_of_line(
                english, "GFX_5B", "db $1C, $08, $1C, $00, $77, $00, $A7, $03"
            ),
            "db $1C, $08, $1C, $00, $76, $00, $A7, $02",
            comment="[ENG-GFX] EU-matching weathercock fix",
        )
    if not keep_religious_imagery:
        # Eastern Palace's floor tile ($1A8, a 2x2 tilemap block) is a Star
        # of David in JP 1.0; the US ROM swaps it for a generic tile. Two
        # copies (GFX_19, GFX_1A -- both loaded together for this room) hold
        # the same 23-byte literal run (3bpp source, bytes 0-22 of the
        # 24-byte tile; byte 23 comes from a separate, shared "repeat word"
        # command whose value already matches between JP and US, so it's
        # left untouched). Confirmed by diffing this tile's live VRAM
        # between an unmodified US ROM and this JP-English build, standing
        # in the same spot in each.
        for sheet in ("GFX_19", "GFX_1A"):
            english.set_operand(
                _address_of_line(
                    english, sheet, "db $07, $FF, $0D, $FF, $11, $FF, $22, $FF"
                ),
                "db $07, $FF, $08, $FF, $11, $FF, $23, $FF",
                comment="[ENG-GFX] US-matching Eastern Palace floor tile",
            )
            english.set_operand(
                _address_of_line(
                    english, sheet, "db $7F, $FF, $C4, $FF, $A8, $FF, $90, $FF"
                ),
                "db $41, $FF, $83, $FF, $96, $FF, $BC, $FF",
                comment="[ENG-GFX] US-matching Eastern Palace floor tile",
            )
            english.set_operand(
                _address_of_line(
                    english, sheet, "db $03, $0F, $1F, $3F, $7F, $7F, $FF"
                ),
                "db $07, $0F, $1F, $3F, $7F, $FF, $FF",
                comment="[ENG-GFX] US-matching Eastern Palace floor tile",
            )

    # HUD bomb/arrow/rupee digit counters turn gold at max capacity
    # (GBA-matching feature; see hud_max_capacity_color). Repurposes the
    # HUD's palette 5 (.hud_05 / #Palettes_HUD_RedYellow, PaletteData in
    # bank_1B) -- used only by the file-select/equipment-preview screen's
    # "Fire Shield" icon, never shown alongside the live gameplay HUD --
    # into a proper "same red outline as every other digit, gold ($F8F878)
    # fill instead of white" palette. Byte-neutral: same 4-color/8-byte
    # table shape.
    english.set_operand(
        _address_of_line(
            english, "PaletteData", "dw  $0000,  $00B8,  $433D,  $0000"
        ),
        "dw  $0000,  $0018,  $3FFF,  $0000",
        comment="[ENG-HUD] palette 5: red outline (unchanged), $F8F878 fill",
    )
    # UpdateHUDBuffer's very last Arrows-digit store (bank_0D) is
    # STA.l $7EC760, itself an exact JML-sized 4 bytes -- no orphan needed.
    # By this point all 4 digit tiles (bomb + arrow, tens + ones) are
    # already in the HUD buffer.
    english.relocate_block(
        0x0DFCE8,
        "EN_HUD_RecolorBombsArrowsAtMax",
        resume=0x0DFCEC,
        comment=(
            "; [ENG-HUD] Last Arrows-digit store -> relocated copy that"
            " also recolors at max capacity (hud_max_capacity_color).",
        ),
    )
    # Same for the rupee counter. UpdateHUDBuffer's very last Rupees-digit
    # store (bank_0D) is STA.l $7EC754, also an exact JML-sized 4 bytes.
    english.relocate_block(
        0x0DFCA0,
        "EN_HUD_RecolorRupeesAtMax",
        resume=0x0DFCA4,
        comment=(
            "; [ENG-HUD] Last Rupees-digit store -> relocated copy that"
            " also recolors at max capacity (hud_max_capacity_color).",
        ),
    )
    if epilepsy_fix:
        # OversaturateColor (bank $02) is the "bright" half of the shared
        # full-screen flash effect: HandleScreenFlash alternates every frame
        # between this and RestorePalettesAfterFlash while a WRAM counter
        # ($0FF9) is nonzero, driving every screen-flash in the game
        # (Agahnim's altar cutscene and boss lightning, Vitreous's and the
        # Ether Medallion's lightning -- both spawned through the same
        # Sprite_SpawnLightning -- and the Magic Bat's power-up flash).
        # JP 1.0 boosts each of a color's 5-bit R/G/B channels by +14 (of a
        # max 31) per flash frame, clamped; a later Japanese revision (JP
        # header version byte $02, matching a "Virtual Console" ROM dump)
        # tones this down to +2 -- Nintendo's photosensitive-epilepsy-safety
        # pass on this game's flash effects. Confirmed the *only* real
        # difference (not a relocation artifact -- most of the ROM's raw
        # bytes differ between these two builds purely because later code
        # shifted addresses around; HandleScreenFlash itself, and every
        # $0FF9-duration constant checked, are byte-identical once that
        # shift is accounted for) is these 3 add-immediate operands.
        english.set_operand(
            _address_of_line(english, "OversaturateColor", "ADC.w #$000E"),
            "ADC.w #$0002",
        )
        english.set_operand(
            _address_of_line(english, "OversaturateColor", "ADC.w #$01C0"),
            "ADC.w #$0040",
        )
        english.set_operand(
            _address_of_line(english, "OversaturateColor", "ADC.w #$3800"),
            "ADC.w #$0800",
            comment="[ENG-GFX] toned-down flash brightness (later JP rev)",
        )
    if not low_health_beep:
        # RefillLogic (bank_0D) queues SFX2.2B (the repeating low-health
        # warning beep) once every 32 frames whenever current health is
        # below HeartBeepThresholds[current max hearts] -- gated first on
        # "not mid-cutscene" / "not mid-heal", both plain BNE-past checks
        # ahead of the threshold compare and the SFX queue/timer-tick
        # itself. Turning the first of those into an unconditional branch
        # (same 2-byte size) skips the whole block -- threshold compare,
        # SFX queue, and the beep-timer WRAM word ($04CA, read/written
        # nowhere else in the ROM) -- without touching health/heart state
        # or any other HUD drawing.
        english.set_operand(
            _address_of_line(english, "RefillLogic", "BNE .done_beeping"),
            "BRA .done_beeping",
            comment="[ENG-HUD] skip the low-health warning beep entirely",
        )
    if fix_mothula_bugs:
        # See mothula_damage_bugfix's docstring. Redirect right after the
        # nibble-unpacking loop's own CPX/BCC exit check (5 bytes: CPX.w
        # #$1000 is 3, BCC .next is 2) into our replacement, which
        # replays that same check before doing anything else -- the 1
        # orphaned byte is the tail of the displaced BCC, now unreachable.
        cpx_address = _address_of_line(
            english, "DecompressEnemyDamageSubclasses", "CPX.w #$1000"
        )
        english.relocate_block(
            cpx_address,
            "EN_MothulaDamageFix",
            resume=cpx_address + 5,
            orphan=(0x00,),
        )
        # Redirect right after CheckSwordDamageToSprite (bank_06) restores
        # the sprite-slot X (PLX), replacing the 2 instructions it sets up
        # next (LDA.b #$10 / STA.b $47 -- 4 bytes total, matching JML
        # exactly, no orphan) with our class-4/5-to-6 Mothula check.
        redirect_address = _address_of_line(
            english, "CheckSwordDamageToSprite", "LDA.b #$10"
        )
        resume_address = _address_of_line(
            english, "CheckSwordDamageToSprite", "LDA.b #$9D"
        )
        english.relocate_block(
            redirect_address, "EN_MothulaClassRedirect", resume=resume_address
        )
        # Redirect right before CheckSpriteTileProperty's (bank_06) class-4
        # spike-floor damage attempt (LDA.b #$04 / JSL, 6 bytes total --
        # a 4-byte JML plus 2 orphan bytes) with our Mothula skip-check.
        spike_address = _address_of_line(
            english, "CheckSpriteTileProperty", "LDA.b #$04"
        )
        spike_resume_address = _address_of_line(
            english, "CheckSpriteTileProperty", "LDA.w $0EF0,X"
        )
        english.relocate_block(
            spike_address,
            "EN_MothulaSpikeImmunity",
            resume=spike_resume_address,
            orphan=(0x00, 0x00),
        )
    # Credits keeps its own JP-native (bolder) font, but no longer via JP's
    # own runtime decompression pipeline: jp_credits_font.2bpp
    # (credits_font_upload, bank $20) is that same font pre-decompressed
    # *offline* (binextract-jp-credits-font.py, straight from the raw JP
    # ROM bytes -- no emulator), so credits' scene-load and staff-scroll
    # init only need to point their TransferFontToVRAM call at that flat
    # resource instead of JP's own $7E2000-sourced copy -- no
    # BuildSomeTextMasks, no TEXTDECOMP WRAM table, at all, ever. Every other
    # DecompressFontGFX/TransferFontToVRAM caller (the game's normal font
    # loads) is unaffected and keeps the usual English hooks.
    #
    # DecompressFontGFX itself is a no-op everywhere in this build (real
    # decompression is never needed again) -- both credits font-load sites
    # just fall through to that same bare no-op like every other caller,
    # same as everywhere else. Credits_LoadOverworldScene_PrepGFX (bank_02)
    # already calls InitializeTilesets *before* its own font upload, the
    # safe order (see EN_CreditsFont_ClearAttributionTilemap, credits_font_
    # upload, for why order matters here at all) -- only its
    # TransferFontToVRAM call needs redirecting to our own asset-backed
    # copy.
    english.rewrite_reference(
        _address_of_line(
            english,
            "Credits_LoadOverworldScene_PrepGFX",
            "JSL TransferFontToVRAM",
        ),
        "TransferFontToVRAM",
        "EN_TransferFontToVRAM_Credits",
    )
    # Credits_InitializeTheActualCredits does it the *unsafe* way (font
    # upload, then Credits_LoadCoolBackground's own InitializeTilesets) --
    # see EN_CreditsFont_SkipOldUploadSite (credits_font_upload, bank $20)
    # for the fix: skip this call site entirely rather than pad around the
    # bad order with a wait. The real upload now happens inside
    # EN_CreditsFont_ClearAttributionTilemap instead, which already runs
    # after Credits_LoadCoolBackground.
    old_font_upload_call = _address_of_line(
        english, "Credits_InitializeTheActualCredits", "JSL TransferFontToVRAM"
    )
    english.relocate_block(
        old_font_upload_call,
        "EN_CreditsFont_SkipOldUploadSite",
        resume=old_font_upload_call + 4,
    )
    # See EN_CreditsFont_ClearAttributionTilemap (credits_font_upload, bank
    # $20) for why this is needed, and why it hooks in *before* JSL
    # Credits_InitializePolyhedral specifically (not later, at STZ.b $13 --
    # confirmed live via a Lua INIDISP write-trace that hooking there was
    # too late: Credits_InitializePolyhedral's own premature $13=$0F had
    # already reached hardware for several frames by that point).
    credits_init_polyhedral_call = _address_of_line(
        english,
        "Credits_InitializeTheActualCredits",
        "JSL Credits_InitializePolyhedral",
    )
    english.relocate_block(
        credits_init_polyhedral_call,
        "EN_CreditsFont_ClearAttributionTilemap",
        resume=credits_init_polyhedral_call + 4,
        comment=(
            "; [ENG-CREDITS-FONT] Credits_InitializeTheActualCredits -> "
            "upload the font, then clear VRAM $6800-$6FFF, before "
            "Credits_InitializePolyhedral's own premature unblank (fixes "
            "the transition-into-credits garbage flash).",
        ),
    )
    # See EN_CreditsInitPolyhedral_SkipPrematureUnblank (credits_font_
    # upload, bank $20): the VRAM clear above only makes sure nothing
    # garbage-looking is exposed *if* Credits_InitializePolyhedral's own
    # premature $13 write reaches hardware -- confirmed live it still can,
    # a real (visibly wrong, if no longer garbage) 1-frame flash to full
    # brightness. Skips that write outright instead.
    polyhedral_premature_write = _address_of_line(
        english, "Credits_InitializePolyhedral", "STA.b $13"
    )
    english.relocate_block(
        polyhedral_premature_write,
        "EN_CreditsInitPolyhedral_SkipPrematureUnblank",
        resume=polyhedral_premature_write + 4,
    )
    _wire_credits_overworld_palette_patch(english, credits_font)
    # Real US's own Intro_InitializeMemory dispatch (usdasm bank_0C) has
    # eleven steps; JP's (this bank) has twelve, because JP splits US's
    # single Intro_LoadTextAndPalettes step into two: this one and a
    # separate DecompressFontGFX dispatch slot -- which, like every other
    # DecompressFontGFX caller, is a no-op in the English build. That
    # dispatch slot still costs a full frame just to be reached and
    # immediately RTL. The now-dead BuildSomeTextMasks call site is
    # repurposed to bump $B0 an extra time, so next frame's dispatch skips
    # straight over the dead slot -- matching the real US ROM's frame
    # count exactly without touching the shared dispatch table itself.
    intro_dead_call = _address_of_line(
        english, "Intro_CreateTextPointers", "JSL BuildSomeTextMasks"
    )
    skip_dead_slot = Relocation()
    skip_dead_slot.place(
        Assembly.from_content(
            [
                "Intro_CreateTextPointers_SkipDeadSlot:",
                "INC.b $B0",
                f"JML ${intro_dead_call + 4:06X}",
            ]
        ).ensure_anchors(),
        0x2EFD30,
        "[ENG-TEXT] Intro_CreateTextPointers: skip Module00_Intro's next"
        " dispatch slot (DecompressFontGFX, a permanent no-op here) -- see"
        " above.",
    )
    english.add(skip_dead_slot)
    english.relocate_block(
        intro_dead_call,
        "EN_Intro_CreateTextPointers_SkipDeadSlot",
        resume=intro_dead_call + 4,
    )
    if not keep_jp_credits:
        # Credits_FadeColorAndBeginAnimating ends the staff-credits scroll
        # once a frame counter ($EA, advancing the scroll line $CA by 1
        # every 8 ticks) reaches a hardcoded terminal value; +128 ($80)
        # accounts for edit_add_english_script_writers's 16 extra lines
        # (16 lines * 8 ticks/line), preserving the original pause length
        # between the last line and the cutoff.
        english.set_operand(
            _address_of_line(
                english, "Credits_FadeColorAndBeginAnimating", "CMP.w #$0CD8"
            ),
            "CMP.w #$0D58",
        )
    # File-select: keep the one in-bank CopySaveToWRAM reference pointing at
    # the preserved JP original (which the graft leaves in place, unmoved -- so
    # no org re-pin is needed).
    english.rewrite_reference(
        0x0CCE8B, "CopySaveToWRAM", "UNREACHABLE_CopySaveToWRAM"
    )
    if us_title_screen:
        # Module00_Intro's dispatch table repointed at title_screen()'s bank
        # $28 (four set_operand swaps, so the table's byte width -- and every
        # anchor after it -- never changes): `LDA.b $11 / JSL JumpTableLong`
        # at Module00_Intro's `.run_submodule` reads its own dispatch table
        # from JumpTableLong's return address (see title_screen.asm's
        # Module00_Intro_Dispatch), so relocating just the `JSL
        # JumpTableLong` call brings a grown (10 -> 11 entry) copy of the
        # table to bank $28 with it -- no base-bank table entries touched
        # in place, and slot 8 (which Save & Quit and Attract_LoadNewScene
        # both jump to directly) stays exactly vanilla. `JSL JumpTableLong`
        # is exactly 4 bytes, matching a JML's own footprint: no orphan
        # bytes, and the original (now unreachable) 10-entry table right
        # after it is left as-is.
        english.relocate_block(
            0x0CC115,
            "EN_Module00_Intro_Dispatch",
            resume=0x0CC119,
            comment=(
                "; [ENG-TITLE] Module00_Intro's dispatch -> a grown copy "
                "(bank $28, title_screen.asm) with the US-ified states",
                "; spliced in and slot 8 left vanilla. "
                "US title screen only (default; --title-screen jp opts"
                " out).",
            ),
        )
        # SheetsTable_AA1 row $23 (bank $00, read by InitializeTilesets,
        # called from Intro_InitializeDefaultGFX with $0AA1=$23): the code
        # reading this table is byte-identical between the disassemblies,
        # but this row's own data is not -- JP's reads $00/$39/$39/$72/$40/
        # $41/$39/$0F, US's reads $16/$39/$1D/$17/$40/$41/$39/$1E (checked
        # directly against both disassemblies' own copies of the table).
        # Since this whole patcher's base is JP's bank $00, the game loads
        # JP's sheet list into the title screen's BG1 tiles regardless of
        # this flag unless the row itself is rewritten -- title_screen_
        # graphics() repointing GFX_16/17/1D/1E to US content has nothing to
        # do without this, since JP's own row never references them.
        english.set_operand(
            _address_of_line(
                english,
                "SheetsTable_AA1",
                "db $00, $39, $39, $72, $40, $41, $39, $0F",
            ),
            "db $16, $39, $1D, $17, $40, $41, $39, $1E",
        )
        # SheetsTable_AA1's row-3 slot ($72 above) is itself only a fallback:
        # InitializeTilesets prefers SheetsTable_AA2's row $51 ($0AA2's
        # value) for that slot when it is nonzero, and it is ($72 in JP) --
        # so the row-$23 rewrite alone left this one slot still loading JP's
        # sheet. Same story here: the code is shared, this row's data is
        # not (JP $72, US $17); the other 3 bytes of the row are already
        # identical between the disassemblies, so only the first is rewritten.
        english.set_operand(
            _address_of_line(
                english, "SheetsTable_AA2", "db $72, $40, $41, $39"
            ),
            "db $17, $40, $41, $39",
        )
        # SheetsTable_AA3 row $7D (read by InitializeTilesets via $0AA3,
        # which Intro_InitializeDefaultGFX sets to $7D in both
        # disassemblies): this row feeds LoadSpriteGraphics/
        # Decompress_sprite_arbitrary, i.e. OBJ (sprite) tiles rather than
        # BG ones -- confirmed by live-tracing VRAM writes in the real US
        # ROM, which land exactly on this row's first slot's sheet and
        # exactly on the sword-blade OBJ tiles (VRAM word $5000+, name
        # table 1 per this intro's OBSEL=$02) that were rendering blank in
        # our build. JP's row leaves that slot unchanged ($00, inheriting
        # whatever the previous caller left in $7EC2FC -- nothing sword-
        # shaped, since JP's title screen has no sword); US's row loads
        # sheet $32 there explicitly. The other 3 bytes already match.
        english.set_operand(
            _address_of_line(
                english, "SheetsTable_AA3", "db $00, $00, $00, $08"
            ),
            "db $32, $00, $00, $08",
        )
        # Intro_LoadAllPalettes_long (bank $02) redirected to the pulled US
        # Intro_LoadAllPalettes (see title_screen()'s comment on that pull):
        # JP's own version sets a different PaletteLoad_OWBGMain "area"
        # ($0AB3=$04, not US's $05) and skips a PaletteLoad_SpritePal0Left
        # call US has, both feeding the same live-CGRAM-diff-confirmed
        # colors the sword-stab scene reveal uses. The wrapper being
        # replaced is exactly 4 bytes (JSR Intro_LoadAllPalettes / RTL),
        # matching a JML's own footprint, so this is a same-size swap: no
        # orphan bytes, nothing after it shifts.
        english.relocate_block(
            0x02802A,
            "EN_TitleScreenUS_LoadAllPalettes",
            resume=0x02802E,
            comment=(
                "; [ENG-TITLE] Intro_LoadAllPalettes_long -> the pulled US "
                "Intro_LoadAllPalettes (bank $28, title_screen.asm);",
                "; JP's own version loads different colors for the "
                "sword-stab scene reveal. US title screen only (default;",
                "; --title-screen jp opts out).",
            ),
        )
        # relocate_block's replaced range runs from `address` up to (not
        # including) the first *anchored* line at or past `resume` -- the
        # unanchored "AnimatedTileSheets:" label line, which sits between
        # Intro_LoadAllPalettes_long's body and its own first data line
        # ($02802E), has no address of its own and gets swept up and
        # dropped along with it. AnimatedTileSheets>0 bytes themselves are
        # untouched (they start exactly at resume); only the label needs
        # restoring.
        english.insert_before(0x02802E, ["AnimatedTileSheets:"])
        # PaletteData (bank $1B): the pulled Intro_LoadAllPalettes reaches
        # PaletteLoad_OWBG1/OWBG2/OWBGMain/HUD by plain (unprefixed) name, so
        # those calls resolve against JP's own bank $1B, unmodified -- same
        # trap as SheetsTable_AA1/AA2 above, but for palette *data* this
        # time. PaletteLoad_OWBG1/OWBGMain/HUD's own code is byte-identical
        # between the disassemblies (confirmed directly), but with
        # $0AB1=$05/$0AB4=$03 (the values the pulled routine sets -- also
        # shared by JP's own dead copy of Intro_LoadAllPalettes, so this
        # slot is intro-only, never touched by real overworld-area palette
        # loads), some of the "owaux"/OW-area sub-tables it indexes into
        # hold genuinely different colors in JP's disassembly (found by
        # tracing every PaletteLoadMultiple source pointer live during this
        # exact call, then diffing JP's vs US's raw ROM bytes at each one).
        # Two rows here are otherwise-unused JP filler ($7FFF/$0000
        # repeats), not real JP-game colors, so rewriting them in place
        # can't affect anything else -- unlike the four rows above, these
        # aren't shared with Module14_Attract's own background: the second
        # one feeds PaletteLoad_OWBG3 ($0AB8-indexed, CGRAM $71-$77), whose
        # *only* caller in the whole game is FileSelect_InitializeGFX, not
        # anything in Module14_Attract -- confirmed live, its title-screen
        # value (loaded once, area 5) simply persists unchanged through
        # attract in both JP and the real US ROM, so an in-place edit here
        # doesn't leak the way the four rows above did. (Left mostly at
        # JP's own $0000 filler until the credits' triforce-room curtain
        # investigation traced a visible black strip in the attract
        # scene's own scroll graphic back to this same row.)
        for old, new in (
            (
                "dw  $4DAD,  $4DAD,  $4DAD,  $4DAD,  $4DAD,  $4DAD,  $4DAD",
                "dw  $377F,  $54E9,  $165F,  $1016,  $2C43,  $6B18,  $61EF",
            ),
            (
                "dw  $14A5,  $0000,  $0000,  $0000,  $0000,  $0000,  $0000",
                "dw  $190A,  $3549,  $45EC,  $6E50,  $258D,  $3A32,  $5F3A",
            ),
        ):
            english.set_operand(
                _address_of_line(english, "PaletteData", old), new
            )
        english.set_operand(
            _address_of_line(
                english,
                "PaletteData",
                f"#_{0x1BE906:06X}: dw  $7FFF,  $7FFF,  $7FFF,  $7FFF,  "
                "$7FFF,  $7FFF,  $7FFF",
            ),
            "dw  $7FFF,  $3DEF,  $14A5,  $14A5,  $0000,  $6318,  $4E73",
        )
        english.set_operand(
            _address_of_line(
                english,
                "PaletteData",
                "#_1BE85E: dw  $0000,  $0000,  $0000,  $0000,  $0000,  "
                "$0000,  $0000",
            ),
            "dw  $1084,  $1908,  $258C,  $4273,  $52F7,  $188B,  $1532",
        )
        english.set_operand(
            _address_of_line(
                english,
                "PaletteData",
                "dw  $7FFF,  $005C,  $0000,  $015C,  $021F,  $02BF,  $033F",
            ),
            "dw  $044E,  $0009,  $0CFC,  $63DF,  $4E73,  $323F,  $0C75",
        )
        # The other four rows of this same owmain_05 block (CGRAM
        # $21-$27/$31-$37/$41-$47/$51-$57, at PaletteData addresses
        # $1BE826/$1BE834/$1BE842/$1BE850 -- also originally JP filler,
        # $7FFF/$0000 repeats) are the title screen's own background during
        # the sword-reveal/logo phase (confirmed live: a fresh CGRAM dump
        # taken on that exact screen read back this table's raw, un-faded
        # filler verbatim -- a solid white background with a black
        # rectangle blending into it, not "slightly off colors"). An
        # earlier attempt here corrected them to the real US ROM's values
        # and was reverted, based on a steady-state attract-mode CGRAM read
        # that showed the *old*, unedited JP filler still winning there --
        # but that predates TitleScreenUS_AttractInitializePalettes (this
        # module's own Attract_Initialize hook, above): it already issues
        # its own PaletteLoad_OWBGMain call (area 4) every time attract
        # starts, which overwrites these exact same five CGRAM destinations
        # (the destination is fixed at $0042/+$20 per row; only the source
        # area differs) with attract's own colors regardless of whatever
        # the title screen left behind -- so fixing the title screen's own
        # copy here no longer has anywhere to leak into.
        for old, new in (
            (
                f"#_{0x1BE826:06X}: dw  $7FFF,  $7FFF,  $7FFF,  $7FFF,  "
                "$7FFF,  $7FFF,  $7FFF",
                "dw  $5D8C,  $558A,  $76B3,  $7AF4,  $0D23,  $11C4,  $2287",
            ),
            (
                f"#_{0x1BE834:06X}: dw  $7FFF,  $7FFF,  $7FFF,  $7FFF,  "
                "$7FFF,  $7FFF,  $7FFF",
                "dw  $4908,  $558A,  $76B3,  $7AF4,  $2CA3,  $3584,  $3E09",
            ),
            (
                f"#_{0x1BE842:06X}: dw  $7FFF,  $7FFF,  $7FFF,  $7FFF,  "
                "$7FFF,  $7FFF,  $7FFF",
                "dw  $6A0F,  $6E50,  $6E71,  $76B3,  $558A,  $69EE,  $7B17",
            ),
            (
                f"#_{0x1BE850:06X}: dw  $0000,  $0000,  $0000,  $0000,  "
                "$0000,  $0000,  $0000",
                "dw  $190A,  $3549,  $45EC,  $6E50,  $258D,  $3A32,  $5F3A",
            ),
        ):
            english.set_operand(
                _address_of_line(english, "PaletteData", old), new
            )
        # InitializeSceneSprite_Copyright (bank $0C): sets the copyright
        # line's starting X position. US starts 12px further left ($4C,
        # not JP's $58) to keep the longer "1991,1992" text centered --
        # AnimateSceneSprite_DrawCopyright (the routine that actually draws
        # it) is hooked to the pulled US version in title_screen(); this
        # routine only sets up the position/counters, so a byte-neutral
        # constant swap is enough, no relocation needed.
        english.set_operand(
            _address_of_line(
                english, "InitializeSceneSprite_Copyright", "LDA.b #$58"
            ),
            "LDA.b #$4C",
        )
        # IntroTriangle_MoveIntoPlace's own pool (bank $0C): the three
        # title-screen triforce triangles' landing Y coordinates. US sets
        # them 8px lower than JP ($58/$30/$58 vs JP's $50/$28/$50, X
        # unchanged at $4B/$5F/$75) -- confirmed by diffing this pool
        # directly between the disassemblies, after the user reported the
        # triforce sitting noticeably higher than the real US ROM's, at
        # exactly JP's own title-screen height. This pool/routine is only
        # ever reached from this one title-screen sprite (grepped: every
        # reference is local to this routine's own dispatch table, bank
        # $0C only), so a byte-neutral constant swap is safe -- no
        # relocation needed. The two $50 values are edited by explicit
        # address since "db $50" alone is ambiguous (matches both).
        english.set_operand(
            _address_of_pool_line(
                english, "IntroTriangle_MoveIntoPlace", "#_0CC5A1: db $50"
            ),
            "db $58",
        )
        english.set_operand(
            _address_of_pool_line(
                english, "IntroTriangle_MoveIntoPlace", "db $28"
            ),
            "db $30",
        )
        english.set_operand(
            _address_of_pool_line(
                english, "IntroTriangle_MoveIntoPlace", "#_0CC5A3: db $50"
            ),
            "db $58",
        )
        # AnimateSceneSprite_DrawTriangle (bank $0C): JP shares this ONE
        # routine (and its .rightside_objects/.leftside_objects pool,
        # priority 2 -- $2B/$6B) between three different US ROM scenes: the
        # title-screen logo triangles, the credits' triforce-room scene,
        # and the rolling credits triangle. The real US ROM's own title
        # screen wants priority 1 ($1B/$5B) there (confirmed by live OAM
        # comparison) -- but its triforce-room/credits scenes use a
        # *separate* routine (DrawTriforceRoomTriangle) that stays at
        # priority 2 (confirmed live via a real credits save state: a
        # simple in-place priority-1 edit here left the credits' own
        # triforce sitting behind a priority-3 curtain sprite, wrong).
        # relocate_block replaces the routine itself with one that checks
        # the sprite's own subtype ($1E18,X, set once at init) and picks
        # the matching pool, so JP's original 3 callers (Triangle/
        # TriforceRoomTriangle/CreditsTriangle), still plain same-bank
        # JSR, unchanged, each land on the right one automatically.
        #
        # This crashed on every earlier attempt, during an entirely
        # ordinary cold boot (confirmed via this project's own fresh-boot
        # test harness) -- root-caused by directly inspecting the stack at
        # the crash site: this routine is reached via a bare JML (bank
        # $0C -> $28), which changes the program bank (K) but touches
        # nothing on the stack, yet it was ending in a plain RTS. RTS
        # only restores the 16-bit PC, never K -- so execution landed at
        # the right *address* but still in bank $28, running whatever
        # unrelated data happens to live there as code. The fix: JML back
        # to bank $0C first (restoring K), landing on the original
        # routine's own now-orphaned RTS (left intact in ROM by
        # relocate_block at the tail of the displaced block) -- that RTS,
        # executing with the correct bank already restored, then correctly
        # consumes the caller's own JSR-pushed return address. `LDA.b #$10
        # / STA.b $06` is exactly 4 bytes, matching a JML's own footprint:
        # no orphan bytes, and everything from STZ.b $07 onward (including
        # the now-unreferenced pool and the routine's own original RTS,
        # which the replacement's own ending jumps back to) is left as-is.
        english.relocate_block(
            0x0CC6DD,
            "EN_TitleScreenUS_DrawTriangle",
            resume=0x0CC6E1,
            comment=(
                "; [ENG-TITLE] AnimateSceneSprite_DrawTriangle -> a "
                "subtype-aware replacement (bank $28, title_screen.asm);",
                "; keeps the title screen at priority 1 without also "
                "changing the triforce-room/credits scenes. "
                "US title screen only (default; --title-screen jp opts"
                " out).",
            ),
        )
        # Attract_Initialize (bank $0C): the real US ROM's own version
        # additionally sets $0AB3=4 and calls PaletteLoad_OWBGMain here --
        # JP's original doesn't, leaving the attract background's own
        # CGRAM rows (e.g. $21-$27) stuck holding whatever the title
        # screen's own Intro_LoadAllPalettes (area 5) last put there,
        # confirmed live as raw, unprocessed JP filler (solid white).
        # `JSL TransferAttractPlaques` is exactly 4 bytes, matching a
        # JML's own footprint: no orphan bytes, and the routine's own
        # tail (from JSL PaletteLoad_LinkArmorAndGloves on, including its
        # same-bank JSR calls to Attract_BuildBackgrounds/
        # Attract_SetUpWindowingHDMA) is untouched, reached by the pulled
        # copy's own JML back.
        english.relocate_block(
            0x0CED7E,
            "EN_TitleScreenUS_AttractInitializePalettes",
            resume=0x0CED82,
            comment=(
                "; [ENG-TITLE] Attract_Initialize -> the pulled "
                "$0AB3=4/PaletteLoad_OWBGMain call (bank $28,",
                "; title_screen.asm); loads the attract background's own "
                "colors instead of leaving the title screen's. "
                "US title screen only (default; --title-screen jp opts"
                " out).",
            ),
        )
    # Attract mode's four auto-scrolling story captions -- see
    # attract_text_timing_fix (bank $20) for why these need widening at all:
    # JP 1.0 sized each one's on-screen duration for its own short text, and
    # this build's much longer English translation (grafted in behind the
    # same JP message IDs) now runs out of screen time before finishing.
    # Byte-neutral -- LDX.w #imm16 either way -- so no relocation needed.
    english.set_operand(
        _address_of_line(english, "Attract_Initialize", "LDX.w #$0E9C"),
        "LDX.w #$1010",
        comment="[ENG-ATTRACT] intro caption's own duration -> the US"
        " ROM's own value (4112 frames, was 3740).",
    )
    throne_room_duration = _address_of_line(
        english, "AttractScene_ThroneRoom", "LDA.b #$C0"
    )
    english.relocate_block(
        throne_room_duration,
        "EN_ThroneRoomDuration",
        resume=throne_room_duration + 4,
    )
    # ThroneRoom's own init falls straight through into the shared
    # AttractScene_AdvanceFromDungeon (no JMP.w, unlike Prison/Altar below)
    # -- its label sits exactly at the resume boundary, so relocate_block's
    # line-slice sweeps the (unanchored) label declaration away along with
    # the replaced span. Prison's and Altar's own untouched JMP.w
    # AttractScene_AdvanceFromDungeon still need it, so it's restored here.
    english.insert_before(
        throne_room_duration + 4, ["AttractScene_AdvanceFromDungeon:"]
    )
    for block, needle, target in (
        ("AttractScene_Prison", "LDA.b #$D0", "EN_PrisonDuration"),
        ("AttractScene_AgahnimAltar", "LDA.b #$C0", "EN_AltarDuration"),
    ):
        address = _address_of_line(english, block, needle)
        english.relocate_block(address, target, resume=address + 4)
    dungeon_scene_countdown = _address_of_line(
        english, "Attract_DoTextInDungeonScene", "LDA.b $64"
    )
    english.relocate_block(
        dungeon_scene_countdown,
        "EN_DungeonSceneCountdown",
        resume=dungeon_scene_countdown + 4,
    )
    for block, target in (
        ("AttractDramatize_ThroneRoom", "EN_ThroneRoomCheck"),
        ("Dramaghanim_MoveAndSpin", "EN_PrisonCheck"),
        ("Dramagahnim_IdleGuiltily", "EN_AltarCheck"),
    ):
        address = _address_of_line(english, block, "LDA.b $64")
        english.relocate_block(address, target, resume=address + 4)


def apply_intro_fix(english: Rom) -> None:
    """Fix JP 1.0's intro-cutscene guard-sprite bug in ``PuppetSoldier``
    (bank $1D). ``PuppetSoldier`` picks sprite $41 (sword) or $43 (spear)
    with ``LDA $05 / ORA #$30 / STA
    $0F50,X`` (the palette) followed by ``CMP #$39``; JP 1.0 sandwiches an
    unrelated ``LDA #$10 / STA $0E60,X`` between those two, clobbering A, so
    the CMP always fails and every guard draws as the spear sprite. The US
    ROM already has both stray writes moved earlier in the routine, keeping A
    intact for the check; this reorders JP 1.0 to match -- the same
    instructions, just resequenced, so the routine's size (and everything
    after it) is unaffected.
    """
    unit = english.units[english.unit_of("PuppetSoldier")]
    start = unit.labels["PuppetSoldier"]
    end = unit.labels["Overlord19_ArmosCoordinator"]
    original_size = sum(line.size for line in unit.lines[start:end])

    # A plain constructor, not from_lines(): that would re-run AnchorSizer
    # over just this slice, and the final RTL -- the slice's last anchor,
    # with no later anchor in range to measure against -- would collapse to
    # size 0. The lines already carry correct sizes from the full-file parse.
    block = Assembly(list(unit.lines[start:end]))
    block.delete("LDA.b #$10", until="STZ.w $0F70,X")
    block.delete("STZ.w $0F70,X", until="LDY.b #$41")
    block.delete("!USELESS", until="CMP.b #$39")
    block.insert_before(
        "JSL Get16BitSpriteCoords_long",
        [*instructions(["STZ.w $0F70,X"]), note("")],
    )
    block.insert_before(
        "STZ.w $0B89,X",
        [*instructions(["LDA.b #$10", "STA.w $0E60,X"]), note("")],
    )
    block.insert_before(
        "CMP.b #$39",
        notes(
            [
                "; [ENG-INTRO] JP 1.0 bug fix: LDA #$10/STA",
                "; $0E60,X used to sit between the palette store above and",
                "; this CMP, clobbering A so the check always failed and",
                "; this guard always drew with the spear sprite ($43).",
                "; Reordered (matching the US ROM) so A still holds the",
                "; palette value here. Skippable with --no-intro-fix.",
            ]
        ),
    )

    fixed_size = sum(line.size for line in block.lines)
    if fixed_size != original_size:
        msg = (
            f"apply_intro_fix: PuppetSoldier size changed "
            f"({original_size} -> {fixed_size} bytes)"
        )
        raise ValueError(msg)
    # render() re-stamps every #_<hex> anchor to its new (reordered) address
    # using each line's already-correct size; reparsed with a plain Line (not
    # from_content(), which would hit the same truncated-slice sizing issue
    # as above) since the addresses are now final text, not bookkeeping.
    unit.lines[start:end] = [
        Line.from_line(text) for text in block.render().splitlines()
    ]


# ---------------------------------------------------------------------------
# main.asm: pull in the graft banks + pad the ROM to a clean 2 MB
# ---------------------------------------------------------------------------
_MAIN_ANCHOR = 'incsrc "bank_1F.asm"'
_MAIN_MARKER = 'incsrc "bank_20.asm"'
# Inserted right after the last base-bank include: the graft-bank includes,
# then the 2 MB padding + SNES header size byte the expansion needs (so the
# checksum is a plain byte-sum every emulator agrees on). Every graft bank is
# unconditional here, including bank_28/bank_29 (the US title-screen sword
# code/graphics, on by default) -- present-but-inert like the rest of the
# graft when --title-screen jp opts out (see title_screen()/title_screen_
# graphics()), not spliced in conditionally.
_MAIN_BLOCK_HEAD = (
    "",
    'incsrc "bank_20.asm"',  # our VWF font + relocated TransferFontToVRAM
    'incsrc "bank_22.asm"',  # message data (main table)
    'incsrc "bank_23.asm"',  # message data (overflow)
    'incsrc "bank_26.asm"',  # US menu/HUD + file-select font & bg graphics
    'incsrc "bank_27.asm"',  # file-select US palette overlay + palette data
    # Sword animation; on by default, inert if --title-screen jp opts out.
    'incsrc "bank_28.asm"',
    # Logo/triforce/sword gfx; on by default, inert if --title-screen jp
    # opts out.
    'incsrc "bank_29.asm"',
)
_MAIN_BLOCK_TAIL = (
    'incsrc "bank_2C.asm"',  # file-select / copy / erase / name-entry
    'incsrc "bank_2D.asm"',  # item menu
    'incsrc "bank_2E.asm"',  # text engine (+ override stubs) and credits
    "",
    "; [ENG-FS] Pad the ROM up to a clean 2 MB (power-of-2). The English graft"
    " expands the ROM",
    "; into banks $20-$2E, leaving it at a non-power-of-2 size (~0x150000)."
    " The SNES header",
    "; checksum for a non-power-of-2 ROM is computed by a mirror-and-sum"
    " algorithm that asar and",
    "; some emulators (e.g. snes9x) disagree on, which made snes9x report"
    ' "invalid checksum".',
    "; Padding to exactly 2 MB makes the checksum a plain byte-sum that"
    " everyone agrees on, so",
    "; --fix-checksum writes a value snes9x accepts. The gaps between the"
    " graft banks ($21, $24-$25,",
    "; $2A-$2B, $2F-$3F) are unused ($00 fill) -- valid LoROM space in a",
    "; 2 MB ROM, free for future use.",
    "org $3FFFFF",
    "db $FF",
    "",
    "; Update the ROM-size byte in the SNES header to match the new 2 MB size"
    " ($0A=1 MB -> $0B=2 MB;",
    "; the field is 2^n KB). Keeps the header self-consistent with the padded"
    " file.",
    "org $00FFD7",
    "db $0B",
)


def patch_main_asm(english: Rom) -> None:
    """Wire the graft-bank includes + 2 MB padding into the entry ``main.asm``.

    Idempotent and located by the ``incsrc "bank_1F.asm"`` anchor, not a line
    number, so it survives upstream reformatting and fails loud if the anchor
    is gone.
    """
    if not english.order:
        msg = "patch_main_asm: Rom has no entry file"
        raise ValueError(msg)
    main = english.units[
        english.order[0]
    ]  # the entry (main.asm), loaded first
    if any(str(line).strip() == _MAIN_MARKER for line in main.lines):
        return
    anchor = next(
        (
            index
            for index, line in enumerate(main.lines)
            if str(line).strip() == _MAIN_ANCHOR
        ),
        None,
    )
    if anchor is None:
        msg = f"main.asm: anchor {_MAIN_ANCHOR!r} not found"
        raise ValueError(msg)
    block = (*_MAIN_BLOCK_HEAD, *_MAIN_BLOCK_TAIL)
    main.lines[anchor + 1 : anchor + 1] = [
        Line.from_line(text) for text in block
    ]
    main.resize()


def build(
    *,
    usdasm: Path,
    jpdasm: Path,
    changes: bool,
    intro_fix: bool = True,
    weathercock_fix: bool = True,
    keep_religious_imagery: bool = False,
    epilepsy_fix: bool = True,
    keep_jp_credits: bool = False,
    credits_font: str = "jp",
    us_title_screen: bool = True,
    yellow_counts_at_absolute_max: bool = False,
    gba_text_fixes: bool = True,
    low_health_beep: bool = True,
    flute_is_ocarina: bool = False,
    fix_mothula_bugs: bool = False,
    null_padbyte_threshold: int = DEFAULT_NULL_PADBYTE_THRESHOLD,
    nop_padbyte_threshold: int = DEFAULT_NOP_PADBYTE_THRESHOLD,
) -> Rom:
    """Assemble the whole English program as one editable :class:`Rom`.

    Loads the pristine US and JP disassemblies as whole-program
    :class:`Rom`\\ s, copies the JP into the working ``english`` program, then
    does everything on those objects -- the subsystems pull their code *by
    name* from ``us``/``jp`` (never by which bank file it lives in) and fold
    into ``english``, the base banks are hooked to reach them (unless
    ``changes`` is off, the change-free baseline), and ``main.asm`` is wired.
    Writing it back out -- base banks hooked in place, graft banks beside them
    -- is the caller's :meth:`Rom.write` (pass it the *same*
    ``null_padbyte_threshold``: this only covers a subsystem's own internal
    gaps, e.g. ``file_select``'s low-region/US-mirror bridge -- write()
    independently fills every *bank-level* gap). ``nop_padbyte_threshold`` is
    separate: it only governs the baseline's byte-neutral filler at a
    ``changes``-guarded edit site (see :func:`nop_fill`), never a bank-level
    free-ROM region. ``intro_fix`` (only meaningful alongside ``changes``)
    applies :func:`apply_intro_fix`; ``False`` leaves JP 1.0's intro
    guard-sprite bug exactly as it shipped. ``weathercock_fix`` (also only
    meaningful alongside ``changes``) closes off the animated weathercock's
    open-looking right end to match the EU release, in ``apply_base_edits``;
    ``False`` leaves it exactly as JP 1.0/US shipped it. ``keep_religious_
    imagery`` (also only meaningful alongside ``changes``) leaves Eastern
    Palace's Star-of-David floor tile exactly as JP 1.0 shipped it; by
    default ``apply_base_edits`` swaps it for the US ROM's generic tile.
    ``epilepsy_fix`` (also only meaningful alongside ``changes``) tones down
    every full-screen flash effect's brightness to match a later Japanese
    revision's photosensitive-epilepsy-safety pass; ``False`` leaves JP
    1.0's original (much brighter) flash intensity.
    ``yellow_counts_at_absolute_max`` (also only meaningful alongside
    ``changes``) makes the HUD bomb/arrow max-capacity gold color (see
    :func:`hud_max_capacity_color`) trigger only at the true, fully-upgraded
    50/70 instead of the player's *current* upgrade tier's own cap (the
    default, matching the GBA re-release's own behavior).
    ``keep_jp_credits`` (also only meaningful alongside ``changes``) skips
    :func:`credits_bank`'s handful of JP-mistake text fixes, leaving the
    (already JP-fonted) credits text exactly as JP 1.0 shipped it. Player
    names are a 6-character field. ``credits_font`` (also only meaningful
    alongside ``changes``) picks which font credits render with: ``"jp"``
    (default) is JP 1.0's own bolder font, computed offline from the JP ROM;
    ``"us"`` instead pulls the same character set from the US dialogue font,
    so credits match the look of the rest of the (US-fonted) game -- see
    :func:`credits_font_upload`. ``us_title_screen`` (meaningful even without
    ``changes``: its graft banks are always built, present-but-inert, like
    :func:`graphics`'s -- see :func:`title_screen`/
    :func:`title_screen_graphics`) swaps the title-screen logo for the US
    ROM's logo + animated sword, on by default (matching ``--title-screen
    us``, the CLI default); ``False`` (``--title-screen jp``) keeps JP 1.0's
    own title screen instead. Either way, the intro keeps JP 1.0's own
    press-to-skip timing (skippable as soon as the triforce forms, not
    gated behind the sword animation like the real US ROM).
    ``gba_text_fixes`` (also only meaningful alongside ``changes``) applies
    six GBA-era wording fixes to the dialogue/intro and credits text (see
    :func:`gba_dialogue_text_fixes`/``credits_bank``'s ``edit_smithery_to_
    smithy``/``edit_venus_faeries_to_fairies``) plus ``item_menu``'s "GOOD
    BEE" -> "GOLDEN"/"BEE" item-menu rename; ``False`` leaves the original
    US translation's wording. ``low_health_beep`` (also only meaningful
    alongside ``changes``) leaves the repeating low-health warning beep
    (``apply_base_edits``, ``RefillLogic`` in bank_0D) in place; ``False``
    (``--no-low-health-beep``) silences it. ``flute_is_ocarina`` (also
    only meaningful alongside ``changes``) renames "Flute" to "Ocarina"
    (the JP original's name for the item) in dialogue (:func:`flute_to_
    ocarina_dialogue_fixes`), the item menu (``item_menu``'s ``fix_flute_
    to_ocarina``), and the credits (skips ``credits_bank``'s
    ``edit_flute_boy``, leaving JP's own "OCARINA BOY PLAYS AGAIN" caption
    in place). ``fix_mothula_bugs`` (also only meaningful alongside
    ``changes``, default off) applies :func:`mothula_damage_bugfix`: JP
    1.0/US Mothula is immune to golden-sword swings/dashes, tempered-sword
    spin attacks, and golden-sword spin attacks (all zeroed out by its own
    damage-table row); ``True`` gives it the same sword-tier damage
    scaling every other class already has. Does not touch Mothula's own
    immunity to its room's spike hazards (see the function's docstring for
    why the two are linked) -- not a full GBA-parity fix, just this
    specific reported bug.
    """
    title_screen_on = us_title_screen and changes
    sources = Sources(
        us=Rom.load(usdasm / "main.asm"), jp=Rom.load(jpdasm / "main.asm")
    )
    english = sources.jp.copy()
    relocations = [
        text(
            sources,
            changes=changes,
            gba_text_fixes=gba_text_fixes,
            flute_is_ocarina=flute_is_ocarina,
            nop_padbyte_threshold=nop_padbyte_threshold,
        ),
        font_upload(sources, changes=changes),
        credits_font_upload(
            sources, changes=changes, credits_font=credits_font
        ),
        hud_max_capacity_color(
            changes=changes,
            yellow_counts_at_absolute_max=yellow_counts_at_absolute_max,
        ),
        mothula_damage_bugfix(
            changes=changes, fix_mothula_bugs=fix_mothula_bugs
        ),
        credits_bank(
            sources,
            changes=changes,
            keep_jp_credits=keep_jp_credits,
            gba_text_fixes=gba_text_fixes,
            flute_is_ocarina=flute_is_ocarina,
        ),
        attract_text_timing_fix(sources, changes=changes),
        item_menu(
            sources,
            changes=changes,
            gba_text_fixes=gba_text_fixes,
            flute_is_ocarina=flute_is_ocarina,
        ),
        file_select(
            sources,
            changes=changes,
            null_padbyte_threshold=null_padbyte_threshold,
            nop_padbyte_threshold=nop_padbyte_threshold,
            us_title_screen=title_screen_on,
        ),
        graphics(changes=changes),
        file_select_palette(changes=changes),
        title_screen(
            sources, changes=changes, us_title_screen=us_title_screen
        ),
        title_screen_graphics(
            changes=changes, us_title_screen=us_title_screen
        ),
    ]
    # Wire hooks first: it classifies each hook (alias vs pad) from the
    # pristine JP's callers and records the alias set the pieces then emit.
    if changes:
        _wire_hooks(english, sources.jp, relocations)
    for relocation in relocations:
        english.add(relocation)
    if changes:
        apply_base_edits(
            english,
            keep_jp_credits=keep_jp_credits,
            weathercock_fix=weathercock_fix,
            keep_religious_imagery=keep_religious_imagery,
            epilepsy_fix=epilepsy_fix,
            us_title_screen=title_screen_on,
            low_health_beep=low_health_beep,
            fix_mothula_bugs=fix_mothula_bugs,
            credits_font=credits_font,
        )
        if intro_fix:
            apply_intro_fix(english)
    patch_main_asm(english)
    return english


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="emit the change-free baseline (no graft edits or base hooks)",
    )
    parser.add_argument(
        "--no-save-compatibility",
        action="store_true",
        help="omit the on-entry US/Japanese save-slot migrator",
    )
    parser.add_argument("--usdasm", type=Path, default=USDASM)
    parser.add_argument("--jpdasm", type=Path, default=JPDASM)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(),
        help="jpdasm fork to write the whole English program into",
    )
    parser.add_argument(
        "--null-padbyte-threshold",
        type=int,
        default=DEFAULT_NULL_PADBYTE_THRESHOLD,
        help="free-ROM gaps over this many bytes use a padbyte/pad jump "
        f"instead of explicit db $FF rows; 0 disables padbyte entirely, "
        f"always explicit rows (default: {DEFAULT_NULL_PADBYTE_THRESHOLD})",
    )
    parser.add_argument(
        "--nop-padbyte-threshold",
        type=int,
        default=DEFAULT_NOP_PADBYTE_THRESHOLD,
        help="a baseline edit-site filler's interior dead bytes (past its "
        "JMP) over this many bytes use a padbyte/pad fill instead of one "
        "NOP per byte; 0 disables padbyte entirely, always explicit NOPs "
        f"(default: {DEFAULT_NOP_PADBYTE_THRESHOLD}). Separate from "
        "--null-padbyte-threshold, which only covers bank-level free-ROM "
        "regions",
    )
    args = parser.parse_args()
    english = build(
        usdasm=args.usdasm,
        jpdasm=args.jpdasm,
        changes=not args.baseline,
        null_padbyte_threshold=args.null_padbyte_threshold,
        nop_padbyte_threshold=args.nop_padbyte_threshold,
    )
    generated = english.write(
        args.out,
        bank_header=bank_header,
        null_padbyte_threshold=args.null_padbyte_threshold,
    )
    mode = "baseline" if args.baseline else "with changes"
    print(
        f"wrote English program ({mode}): "
        f"{len(generated)} graft banks + base banks -> {args.out}"
    )


if __name__ == "__main__":
    main()
