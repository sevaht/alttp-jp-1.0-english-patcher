#!/usr/bin/env python3
"""script_reviewer.py - a local web app for reviewing/editing the dialogue
script (US SNES original vs. a "Proposed" rewrite vs. GBA text) with a
pixel-accurate live preview using the ACTUAL in-game variable-width font.

Why this exists: gba_script_diff.py's curated GLOBAL_FIXES (propose()) have
never been checked against the real per-character pixel width the SNES VWF
engine uses -- an edit could silently overflow the textbox the exact same
way the vanilla US quote-character bug did (see WidthTable's docstring).
Reviewing 397 messages by pasting text back and forth doesn't scale either;
this serves an editable page so that work happens without a human/agent
round-trip per line.

Two hard limits, cross-verified two independent ways (ROM buffer geometry,
and hand-summing a known-overflowing vanilla line -- see WidthTable):

* MAX_LINE_WIDTH = 176px (22 tiles) per line -- generate.py's
  --wide-dialogue-lines (landed 2026-09-08) widens the VWF's per-line
  buffer to match, since the dialog box's own border was already drawn
  this wide and only the text engine's own buffer was narrower. This tool
  assumes that flag is always on, matching how the script is now authored.
* There is NO automatic line-wrap in the SNES engine. Line breaks are
  hardcoded per-message by $73/$74/$75/$76 control codes in the data; an
  edited line that's too wide silently corrupts into the next line's tile
  buffer rather than wrapping. This is why the frontend must flag overflow
  live, not just "look about right."

Usage:
    script_reviewer.py --gba-rom PATH.gba --snes-rom PATH.sfc \\
        --state PATH.json [--usdasm DIR] [--port 8000]

`--state` defaults to tools/reviewer_state.json, tracked in-repo alongside
this tool -- review progress is checked in like any other project file.
`--gba-rom` and `--snes-rom` are both required, no defaults -- ROMs are
never assumed.
"""
import argparse
import difflib
import http.server
import json
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gba_script_diff  # noqa: E402  (sibling tools/ script, not a package)

STATIC_DIR = Path(__file__).resolve().parent / "reviewer"


# ---------------------------------------------------------------------------
# WidthTable -- the SNES dialogue VWF engine's own per-character widths
# ---------------------------------------------------------------------------


class WidthTable:
    """Per-character pixel widths, read straight from the SNES ROM's own
    VWF width table (usdasm/bank_0E.asm's `.width` sublabel inside
    `RenderText_PerformVWFing`, ROM file offset 0x74ADF, 99 bytes, indexed
    directly by the raw dialogue character code -- no translation step).

    Applies the same quote-narrowing patch generate.py applies at build
    time (generate.py's `"db   8,  7,  7,  7,  7,  4" ->
    "db   6,  7,  7,  7,  7,  4"` replace, code $4C 8px -> 6px) so this
    tool's numbers match what actually ships, not the vanilla US table.

    MAX_LINE_WIDTH (176px = 22 tiles, with generate.py's
    --wide-dialogue-lines) is the VWF render buffer's real hard limit --
    there is no auto-wrap, a line past this corrupts into the next line's
    tile buffer (or, for a page's 3rd line, off the end of VRAM -- both
    guarded against by that same patch). The vanilla (pre-widening) limit
    was 168px = 21 tiles, verified two independent ways: the buffer's own
    per-line stride ($150 bytes / 16 bytes-per-tile-row = 21 tiles * 8px),
    and by hand-summing the known vanilla-overflow line `said, "Once I have
    finished with` using the table's UNPATCHED 8px quote width, which comes
    to exactly 169 -- one pixel over. Confirmed live (Mesen, a VRAM dump,
    and the disassembly) that the dialog box's own border was already
    drawn 22 tiles wide the whole time -- only the text engine's own
    buffer was narrower -- so widening it to match doesn't touch anything
    else on screen.

    `[LINK]` is the disassembly's stand-in for the $6A "insert player name"
    control code -- at runtime this is whatever 1-6 character name the
    player entered on the (US-grafted) name-entry screen, not literal text,
    so there's no single true width to measure. It's rendered/measured as
    NAME_PLACEHOLDER_WORD ("PLAYER" -- real letters, so it uses real font
    glyphs rather than needing yet another synthetic one; chosen instead of
    literally spelling "[LINK]" only because `[`/`]` have no ROM glyph at
    all), but each of its NAME_CHARS (6) letters advances by
    max_name_char_width -- the WIDEST width among every character the
    name-entry screen allows (A-Z, a-z, 0-9, space, '!') -- instead of its
    own natural width, so the total always equals the true worst-case name
    width regardless of which word is drawn. A line that fits is therefore
    guaranteed to fit for any name the player could actually choose.

    `[#0]`-`[#3]` is the disassembly's stand-in for the $6C "write BCD
    digit" control code (ParseText_WriteBCD, usdasm/bank_0E.asm:7625) --
    confirmed by reading the routine itself: the operand (0-3) selects
    ones/tens/hundreds/thousands from a 2-byte BCD buffer at WRAM
    $1CF2/$1CF3 (`Y = operand >> 1` picks the byte, the operand's low bit
    picks its low/high nibble), so it ALWAYS emits exactly one digit
    character -- never more, and only operands 0-3 are meaningful (a
    higher operand would read unrelated WRAM, so `[#4]` etc. is genuinely
    invalid, not just unsupported here). Rendered as its own slot digit
    ("0"-"3" -- so four adjacent placeholders read as four distinct digit
    positions of one shared runtime value, not four identical unknowns),
    but always at max_digit_width -- the widest width among 0-9 (every
    digit except '1', which is narrower) -- regardless of which digit is
    drawn, same worst-case-guarantee reasoning as the name placeholder.

    `[OP:...]` is gba_script_diff.py's own tag for a GBA control code it
    couldn't decode -- not real display text, and unlike [LINK]/[#N] there's
    no known real-world meaning to stand in for, so it's measured/rendered
    as one GBA_OP_PLACEHOLDER_CODE glyph -- a synthetic hatched box (not
    ROM data, see FontSheet) rather than any real character, so it reads
    unambiguously as "unresolved," not as actual text.

    "\U0001F62C" (the grimacing-face emoji usdasm's comment uses for
    Link's face icon) always expands to exactly two REAL codes, $4A then
    $4B ("Link face L"/"Link face R") -- confirmed by reading the only
    occurrence in the whole script (text.asm:2065): `db $8A, $4A, $4B,
    $8A, ...`. Both codes and their order are fixed, so unlike [LINK]/[#N]
    there's no worst-case guessing needed -- it's drawn/measured at its
    real width every time.

    "♥" (the heart-suit character) is NOT safe to map this way: the
    width table has SEVEN distinct heart-icon codes ($52-$58, "1L 2L ER 3L
    3R 4L 4R"), and usdasm's comment renders every one of them as the same
    "♥" -- confirmed by reading the raw bytes behind all 5 of its
    occurrences in the script (message $0155 is codes $52,$53; $0156 is
    $54,$53; $0157 is $55,$56; $0158 and $0159 are both $57,$58). A bare
    "♥" is therefore ambiguous and deliberately left unmapped (flagged as
    unmeasured) -- MessageStore substitutes an explicit RAW_CODE_TOKEN_RE
    tag (`[C:52]`, `[C:53]`, ...) for these 5 known messages before this
    tool ever measures/renders/persists their text, which resolves the
    ambiguity losslessly: the tag names the exact byte code, so it reads
    back unambiguously later.

    A literal `[` or `]` that isn't part of any of the forms above is
    flagged as invalid syntax rather than silently measured (or silently
    ignored) -- see tokenize().
    """

    ROM_OFFSET = 0x74ADF
    TABLE_SIZE = 0x63  # 99 bytes, codes $00-$62
    MAX_LINE_WIDTH = 176  # 22 tiles, with generate.py's --wide-dialogue-lines

    _QUOTE_CODE = 0x4C
    _PATCHED_QUOTE_WIDTH = 6  # generate.py's engine.replace(...) patch

    NAME_CHARS = 6  # this build's name field width (US font/name-entry graft)
    NAME_ENTRY_CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz0123456789 !")
    NAME_TOKEN = "[LINK]"
    NAME_PLACEHOLDER_WORD = "PLAYER"  # real glyphs; see docstring for why

    DIGIT_TOKEN_RE = re.compile(r"\[#([0-3])\]")
    GBA_OP_TAG_RE = re.compile(r"\[OP:[^\]]*\]")
    GBA_OP_PLACEHOLDER_CODE = 0x63  # one past the real table -- synthetic, not ROM data
    GBA_OP_PLACEHOLDER_WIDTH = 7  # a plain, unremarkable width; see docstring

    LINK_FACE_TOKEN = "\U0001F62C"  # 😬 -- always expands to codes $4A,$4B
    LINK_FACE_CODES = (0x4A, 0x4B)

    # explicit "draw exactly this ROM code" escape -- used where a comment
    # character is ambiguous (see the heart-icon docstring paragraph above)
    RAW_CODE_TOKEN_RE = re.compile(r"\[C:([0-9A-Fa-f]{2})\]")

    def __init__(self, snes_rom):
        if len(self.NAME_PLACEHOLDER_WORD) != self.NAME_CHARS:
            msg = "NAME_PLACEHOLDER_WORD must be exactly NAME_CHARS letters"
            raise ValueError(msg)
        table = bytearray(snes_rom[self.ROM_OFFSET:self.ROM_OFFSET + self.TABLE_SIZE])
        if len(table) != self.TABLE_SIZE:
            msg = "SNES ROM is too small to contain the VWF width table"
            raise ValueError(msg)
        table[self._QUOTE_CODE] = self._PATCHED_QUOTE_WIDTH
        self.widths = bytes(table) + bytes([self.GBA_OP_PLACEHOLDER_WIDTH])
        self.code_of = self._build_char_map()
        self.max_name_char_width = max(
            self.widths[self.code_of[ch]] for ch in self.NAME_ENTRY_CHARS
        )
        self.max_digit_width = max(self.widths[self.code_of[d]] for d in "0123456789")

    @staticmethod
    def _build_char_map():
        """ASCII/Unicode character -> dialogue byte code, derived from the
        width table's own annotated rows (usdasm/bank_0E.asm:8546-8583).
        Curly quotes/apostrophes map onto the same codes as their straight
        equivalents -- the font has no separate glyph for them."""
        code_of = {}
        for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            code_of[ch] = i
        for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
            code_of[ch] = 0x1A + i
        for i, ch in enumerate("0123456789"):
            code_of[ch] = 0x34 + i
        code_of.update({
            "!": 0x3E, "?": 0x3F, "-": 0x40, ".": 0x41, ",": 0x42, "…": 0x43,
            ">": 0x44, "(": 0x45, ")": 0x46,
            '"': 0x4C, "“": 0x4C, "”": 0x4C,
            "↑": 0x4D, "↓": 0x4E, "←": 0x4F, "→": 0x50,
            "'": 0x51, "’": 0x51,
            " ": 0x59,
            "<": 0x5A,
            "Ⓐ": 0x5B, "Ⓑ": 0x5C, "ⓧ": 0x5D, "ⓨ": 0x5E,
            # "ancient writing" decoration glyphs (usdasm/bank_0E.asm:8571,
            # "; > ( ) ☥ 𓈗 Ƨ LINKFACE") -- confirmed 1:1, no other code
            # renders as any of these three symbols anywhere in the script.
            "☥": 0x47, "𓈗": 0x48, "Ƨ": 0x49,
        })
        return code_of

    def char_width(self, ch):
        code = self.code_of.get(ch)
        return self.widths[code] if code is not None else None

    def tokenize(self, text):
        """text -> [("char", ch) | ("name", None) | ("digit", slot_digit) |
        ("op_tag", None) | ("link_face", None) | ("raw_code", hex_code) |
        ("bad_bracket", ch), ...]. An exact `[LINK]` run becomes one "name"
        token, `[#0]`-`[#3]` becomes one "digit" token (value = the slot
        digit itself, "0"-"3" -- rendered as that digit so four adjacent
        placeholders read as four distinct digit positions of one shared
        value, not four identical unknowns; the WIDTH is still always
        max_digit_width regardless of which digit is drawn), `[OP:...]`
        becomes one "op_tag" token, the Link-face emoji becomes one
        "link_face" token (expands to codes $4A,$4B), `[C:XX]` becomes one
        "raw_code" token (draws/measures exactly ROM code XX -- see the
        heart-icon docstring paragraph for why this exists); any OTHER `[`
        or `]` becomes "bad_bracket" (invalid syntax, not a measurable
        character)."""
        tokens = []
        i, n = 0, len(text)
        while i < n:
            if text.startswith(self.NAME_TOKEN, i):
                tokens.append(("name", None))
                i += len(self.NAME_TOKEN)
                continue
            if text.startswith(self.LINK_FACE_TOKEN, i):
                tokens.append(("link_face", None))
                i += len(self.LINK_FACE_TOKEN)
                continue
            digit_match = self.DIGIT_TOKEN_RE.match(text, i)
            if digit_match:
                tokens.append(("digit", digit_match.group(1)))
                i = digit_match.end()
                continue
            op_match = self.GBA_OP_TAG_RE.match(text, i)
            if op_match:
                tokens.append(("op_tag", None))
                i = op_match.end()
                continue
            raw_match = self.RAW_CODE_TOKEN_RE.match(text, i)
            if raw_match:
                tokens.append(("raw_code", raw_match.group(1)))
                i = raw_match.end()
                continue
            if text[i] in "[]":
                tokens.append(("bad_bracket", text[i]))
                i += 1
                continue
            tokens.append(("char", text[i]))
            i += 1
        return tokens

    def line_width(self, text):
        """(pixel width, sorted unmeasurable chars, bad brackets found).
        Unmeasurable characters count as 0px -- the caller should surface
        the unknown-character list rather than trust a silently-low width.
        """
        width = 0
        unknown = set()
        bad_brackets = []
        for kind, value in self.tokenize(text):
            if kind == "name":
                width += self.NAME_CHARS * self.max_name_char_width
            elif kind == "digit":
                width += self.max_digit_width
            elif kind == "link_face":
                width += sum(self.widths[c] for c in self.LINK_FACE_CODES)
            elif kind == "raw_code":
                width += self.widths[int(value, 16)]
            elif kind == "op_tag":
                width += self.widths[self.GBA_OP_PLACEHOLDER_CODE]
            elif kind == "bad_bracket":
                bad_brackets.append(value)
            else:
                w = self.char_width(value)
                if w is None:
                    unknown.add(value)
                else:
                    width += w
        return width, sorted(unknown), bad_brackets

    def widths_by_code(self):
        return {i: w for i, w in enumerate(self.widths)}


# ---------------------------------------------------------------------------
# FontSheet -- the actual dialogue font, decoded from the ROM into a PNG
# ---------------------------------------------------------------------------


class FontSheet:
    """Decodes TheFont (SNES ROM 0x70000, 0x1000 bytes = 256 tiles) into a
    single-row PNG sprite sheet, one 8x16 glyph per WidthTable code, so the
    frontend can crop real glyph bitmaps straight out of an <img>. Same
    2bpp interleave as tools/render_tiles.py's tile_pixels() (adapted here
    rather than imported -- tools/ scripts stay standalone, see
    tools/README.md), and the same code->tile formula as
    usdasm/bank_0E.asm:8651-8687 (each glyph is a top tile + tile+16 stacked
    for the bottom half)."""

    ROM_OFFSET = 0x70000
    ROM_SIZE = 0x1000
    GLYPH_W = 8
    GLYPH_H = 16
    # WidthTable.TABLE_SIZE real ROM glyphs, plus the one synthetic
    # GBA_OP_PLACEHOLDER_CODE slot for [OP:...] (see WidthTable's docstring).
    CODE_COUNT = WidthTable.TABLE_SIZE + 1
    OP_PLACEHOLDER_CODE = WidthTable.GBA_OP_PLACEHOLDER_CODE

    # (r, g, b) per 2bpp pixel value 0-3; 0 is rendered fully transparent.
    _PALETTE = ((0, 0, 0), (110, 110, 120), (200, 200, 210), (255, 255, 255))

    def __init__(self, snes_rom):
        self.data = snes_rom[self.ROM_OFFSET:self.ROM_OFFSET + self.ROM_SIZE]
        if len(self.data) != self.ROM_SIZE:
            msg = "SNES ROM is too small to contain TheFont"
            raise ValueError(msg)

    @staticmethod
    def _tile_pixels(data, off):
        rows = []
        for y in range(8):
            p0 = data[off + y * 2]
            p1 = data[off + y * 2 + 1]
            rows.append([((p0 >> (7 - x)) & 1) | (((p1 >> (7 - x)) & 1) << 1)
                         for x in range(8)])
        return rows

    @staticmethod
    def _tile_byte_offset(code):
        tile_index = ((code & 0xF0) << 1) | (code & 0x0F)
        return tile_index * 16

    @staticmethod
    def _placeholder_glyph_pixels():
        """Synthetic glyph for OP_PLACEHOLDER_CODE (not ROM data) -- a
        diagonally-hatched box in the same blocky, 3-shade pixel-art style
        as the font's other icon glyphs (buttons, arrows), marking
        "unresolved content" wherever an [OP:...] tag is rendered -- an
        undecoded GBA control code, not a stand-in for any known real
        character, unlike [LINK]/[#N]."""
        rows = []
        for y in range(16):
            row = []
            for x in range(8):
                if x in (0, 7) or y in (0, 15):
                    row.append(2)  # border
                elif (x + y // 2) % 3 == 0:
                    row.append(1)  # hatch fill
                else:
                    row.append(0)  # transparent
            rows.append(row)
        return rows

    def glyph_pixels(self, code):
        """16 rows x 8 cols of palette indices 0-3 (0 = transparent)."""
        if code == self.OP_PLACEHOLDER_CODE:
            return self._placeholder_glyph_pixels()
        top_off = self._tile_byte_offset(code)
        bottom_off = top_off + 0x100
        top = self._tile_pixels(self.data, top_off)
        bottom = self._tile_pixels(self.data, bottom_off)
        return top + bottom

    def sprite_sheet_png(self):
        """(CODE_COUNT * 8) x 16 RGBA PNG; code c's glyph occupies columns
        [c*8, c*8+8)."""
        w = self.CODE_COUNT * self.GLYPH_W
        h = self.GLYPH_H
        img = [[0] * w for _ in range(h)]
        for code in range(self.CODE_COUNT):
            glyph = self.glyph_pixels(code)
            x0 = code * self.GLYPH_W
            for y in range(h):
                for x in range(self.GLYPH_W):
                    img[y][x0 + x] = glyph[y][x]
        raw = bytearray()
        for y in range(h):
            raw.append(0)  # filter type: none
            for x in range(w):
                idx = img[y][x]
                r, g, b = self._PALETTE[idx]
                a = 0 if idx == 0 else 255
                raw += bytes((r, g, b, a))
        return _write_png_rgba(w, h, bytes(raw))


def _write_png_rgba(w, h, raw_with_filter_bytes):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_with_filter_bytes, 6))
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# MessageStore -- wraps gba_script_diff's already-working loaders
# ---------------------------------------------------------------------------


def _reconstruct_lines(original_lines, fixed_text):
    """Places the ORIGINAL per-line breaks back onto `fixed_text` (a
    GLOBAL_FIXES-substituted flattening of `original_lines`) so the
    Proposed textarea starts from a layout matching the real in-game
    lines, instead of one giant unbroken line -- gba_script_diff.py's
    propose() only ever returns a single flattened string, since its own
    report doesn't render line-by-line.

    This only decides WHERE to put newlines in text that's already
    finalized; it never changes which words are present -- unlike the
    word-level content merge that was tried and reverted earlier (see
    gba_script_diff.py's GLOBAL_FIXES module comment), an imperfect break
    here is just a starting point to hand-adjust, not a correctness risk.

    For the large majority of rows (no GLOBAL_FIXES pattern touched them,
    so fixed_text == the original flattened text) this reproduces the
    original lines exactly. For a row a fix DID touch, a word-level
    alignment (SequenceMatcher) maps each original line boundary onto its
    best-matching position in the fixed text; runs of unchanged words
    (the common case -- most fixes are small in-place swaps) carry their
    exact original break across untouched, and only text inside/after a
    substituted span gets an approximate placement.
    """
    old_words = " ".join(original_lines).split(" ")
    new_words = fixed_text.split(" ")
    if old_words == new_words:
        return list(original_lines)

    sm = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    # old word index -> new word index, for the word occupying that old
    # index (equal runs map 1:1; replaced spans map proportionally).
    old_to_new = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                old_to_new[i1 + k] = j1 + k
        elif tag == "replace":
            span = i2 - i1
            for k in range(span):
                frac = k / span
                old_to_new[i1 + k] = j1 + round(frac * (j2 - j1))
        # "delete" (old words with no new counterpart) intentionally left
        # unmapped -- the boundary walk below falls back to the previous
        # mapped word.

    boundaries = []
    idx = 0
    for line in original_lines:
        idx += len(line.split(" "))
        boundaries.append(idx)

    lines = []
    prev_new_idx = 0
    for i, boundary in enumerate(boundaries):
        if i == len(boundaries) - 1:
            new_idx = len(new_words)
        else:
            old_word_idx = boundary - 1
            while old_word_idx >= 0 and old_word_idx not in old_to_new:
                old_word_idx -= 1
            new_idx = old_to_new[old_word_idx] + 1 if old_word_idx >= 0 else prev_new_idx
            new_idx = max(new_idx, prev_new_idx)
        lines.append(" ".join(new_words[prev_new_idx:new_idx]))
        prev_new_idx = new_idx
    return lines


# The usdasm comment collapses all 7 heart-icon byte codes ($52-$58) into
# the same "♥" character (see WidthTable's docstring) -- confirmed by
# reading the raw `db` lines behind every occurrence of "♥♥" in the whole
# script (there are exactly 5, all listed here). Substituted for an
# explicit [C:XX] tag per message BEFORE anything else (width-checking,
# GLOBAL_FIXES, line reconstruction, persistence) ever sees this text, so
# it's never ambiguous downstream.
HEART_ICON_OVERRIDES = {
    0x155: ("52", "53"),
    0x156: ("54", "53"),
    0x157: ("55", "56"),
    0x158: ("57", "58"),
    0x159: ("57", "58"),
}


def _disambiguate_heart_icons(snes):
    for message_id, (code1, code2) in HEART_ICON_OVERRIDES.items():
        lines = snes.get(message_id)
        if lines is None:
            continue
        tag = f"[C:{code1}][C:{code2}]"
        snes[message_id] = [line.replace("♥♥", tag, 1) for line in lines]


class MessageStore:
    """Builds the reviewable row list by reusing gba_script_diff.py's
    loaders wholesale -- no message decoding/alignment logic is duplicated
    here."""

    def __init__(self, usdasm_dir, gba_rom_path):
        snes = gba_script_diff.load_snes(usdasm_dir)
        _disambiguate_heart_icons(snes)
        gba = gba_script_diff.load_gba(gba_rom_path)
        mapping, _gba_only = gba_script_diff.align(snes, gba)
        # gba_only messages (no SNES counterpart) are intentionally excluded
        # from this reviewer -- there's nothing to compare/edit against.
        self.rows = gba_script_diff.build_rows(snes, gba, mapping, [])

    @staticmethod
    def _key(row):
        return str(row.snes_id) if row.snes_id is not None else f"g{row.gba_id}"

    def to_json(self):
        out = []
        for row in self.rows:
            fixed_text = row.proposed_lines[0] if row.proposed_lines else ""
            proposed_lines = (
                _reconstruct_lines(row.snes_lines, fixed_text)
                if row.kind == "changed" else []
            )
            out.append({
                "key": self._key(row),
                "snes_id": row.snes_id,
                "gba_id": row.gba_id,
                "kind": row.kind,
                "snes_lines": row.snes_lines,
                "gba_lines": row.gba_lines,
                "proposed_default": "\n".join(proposed_lines),
            })
        return out


# ---------------------------------------------------------------------------
# ReviewState -- persisted edits + completed flags
# ---------------------------------------------------------------------------


class ReviewState:
    """{key: {"proposed": str | None, "completed": bool}}, persisted as
    JSON. `proposed: None` means "use the current curated default" -- so a
    row nobody has ever edited stays live-updated if GLOBAL_FIXES changes,
    rather than freezing in whatever it was the first time the file was
    written."""

    def __init__(self, path):
        self.path = path
        self.data = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def set(self, key, proposed, completed):
        self.data[key] = {"proposed": proposed, "completed": bool(completed)}
        self.save()

    def save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ReviewHandler(http.server.SimpleHTTPRequestHandler):
    # set on the class by main() before serve_forever(); one process, one
    # set of loaded ROM data, shared read-only across request handlers.
    message_store = None
    width_table = None
    font_sheet = None
    review_state = None
    _font_png_cache = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; errors still raise/print via BaseHTTPRequestHandler

    def do_GET(self):
        if self.path == "/api/data":
            self._send_json(self._data_payload())
        elif self.path == "/api/font.png":
            if ReviewHandler._font_png_cache is None:
                ReviewHandler._font_png_cache = self.font_sheet.sprite_sheet_png()
            self._send_bytes(ReviewHandler._font_png_cache, "image/png")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/state":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.review_state.set(body["key"], body.get("proposed"), body.get("completed", False))
        self._send_json({"ok": True})

    def _data_payload(self):
        return {
            "messages": self.message_store.to_json(),
            "state": self.review_state.data,
            "widths": self.width_table.widths_by_code(),
            "char_codes": self.width_table.code_of,
            "max_line_width": WidthTable.MAX_LINE_WIDTH,
            "glyph_width": FontSheet.GLYPH_W,
            "glyph_height": FontSheet.GLYPH_H,
            "name_token": WidthTable.NAME_TOKEN,
            "name_chars": WidthTable.NAME_CHARS,
            "name_placeholder_word": WidthTable.NAME_PLACEHOLDER_WORD,
            "max_name_char_width": self.width_table.max_name_char_width,
            "digit_operands": "0123",  # [#N] -- only these N are hardware-valid
            "max_digit_width": self.width_table.max_digit_width,
            "gba_op_placeholder_code": WidthTable.GBA_OP_PLACEHOLDER_CODE,
            "link_face_token": WidthTable.LINK_FACE_TOKEN,
            "link_face_codes": list(WidthTable.LINK_FACE_CODES),
        }

    def _send_json(self, obj):
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json")

    def _send_bytes(self, payload, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gba-rom", required=True, type=Path, help="GBA ALttP&FS ROM")
    ap.add_argument("--snes-rom", required=True, type=Path, help="US SNES ROM")
    ap.add_argument("--state", type=Path,
                     default=Path(__file__).resolve().parent / "reviewer_state.json",
                     help="review-progress JSON file (default: tools/reviewer_state.json)")
    ap.add_argument("--usdasm", type=Path, default=None,
                     help="usdasm checkout (default: cached clone, same as gba_script_diff.py)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    state_path = args.state.resolve()

    usdasm_dir = args.usdasm
    if usdasm_dir is None:
        usdasm_dir = gba_script_diff._default_usdasm_dir()  # noqa: SLF001 (sibling script)
        if usdasm_dir is None or not (usdasm_dir / "text.asm").exists():
            ap.error("--usdasm not given and no cached checkout found; pass it explicitly")

    print(f"loading {args.snes_rom} / {args.gba_rom} / {usdasm_dir} ...")
    snes_rom_bytes = args.snes_rom.read_bytes()

    ReviewHandler.message_store = MessageStore(usdasm_dir, args.gba_rom)
    ReviewHandler.width_table = WidthTable(snes_rom_bytes)
    ReviewHandler.font_sheet = FontSheet(snes_rom_bytes)
    ReviewHandler.review_state = ReviewState(state_path)
    print(f"{len(ReviewHandler.message_store.rows)} messages loaded; "
          f"state file: {state_path}")

    server = http.server.HTTPServer(("127.0.0.1", args.port), ReviewHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"reviewer running at {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
