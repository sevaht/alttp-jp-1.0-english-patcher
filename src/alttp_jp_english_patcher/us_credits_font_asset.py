#!/usr/bin/env python3
"""A US-styled alternate for credits' TOP/BOTTOM (2-tile, white) location
captions, in the same tile-slot layout as :mod:`jp_credits_font_asset`'s
``jp_credits_font.2bpp`` (see
``resources/binextract_us_credits_font.py.template``).

Credits actually has TWO fonts, not one, and they don't both vary by
region. The SMALL (1-tile, yellow) captions (e.g. "THE RETURN OF THE
KING") use a bold, purpose-built font that is pixel-identical between the
JP and US ROMs -- it was never regionalized, there is nothing to swap.
The TOP/BOTTOM (2-tile, white) location captions (e.g. "HYRULE CASTLE")
use a *different*, genuinely region-specific font: JP 1.0's own bolder
glyphs vs. the US ROM's own ``TheFont`` (its regular dialogue VWF font,
confirmed live: the real US ROM's ``Credits_InitializeTheActualCredits``,
usdasm bank_0E, calls ``JSL TransferFontToVRAM`` directly for credits,
same as normal dialogue -- no separate compressed credits-only asset on
the US side at all). ``--credits-font`` only ever needed to swap *this*
one font; the SMALL-role font was never in scope and should never be
touched, in either mode.

Built output is therefore **not a from-scratch font sheet**: ``build_output``
(see the template) starts from the ALREADY-BUILT ``bin/gfx/jp_credits_font
.2bpp`` (extraction order guarantees it exists first -- see
``deploy/binextract.py``) and only overwrites the TOP-role slots
:data:`TOP_TILE_MAP` lists (both halves, from US ``TheFont``) -- every
other slot, SMALL-role included, comes through unchanged from JP's own
font. This means ``us_credits_font.2bpp`` and ``jp_credits_font.2bpp``
are byte-identical outside the ~40 characters TOP-role captions actually
use, by construction, not by re-deriving JP's own font a second time.

:data:`TOP_TILE_MAP` (JP credits tile slot -> source US ``TheFont`` top-
half tile) was read directly off usdasm bank_0E's own
``Credits_CharacterToTile`` table, index-for-index against jpdasm's copy
(both ROMs share the same table shape and character order, just
different tile numbers) -- not recomputed from
``RenderText_PerformVWFing``'s own tile-number formula (``tile = ((code &
$F0) << 1) | (code & $0F)``), because the real game itself deviates from
that formula for a few characters (digits use tiles 230-239, not a
formula result; 'I' uses tile 175, a dedicated narrow-glyph slot, not the
formula's 8) -- reading the real table directly matches the shipped game
exactly; a formula guess doesn't, for those characters.

``TheFont`` is NOT a single-height font -- each dialogue letter is
genuinely 16px (2 VRAM tiles) tall: ``RenderText_PerformVWFing`` draws a
"top" tile at its own tile number, then a "bottom" tile at ``tile number +
16`` (its own ``LDA.b $0A / ADC.w #$0010`` right before the second half's
row loop). This exactly mirrors ``Credits_CharacterToTile``'s own
TOP/BOTTOM-slot-pairing convention (BOTTOM slot always == TOP slot + 16),
so ``build_output`` copies both halves verbatim for every
:data:`TOP_TILE_MAP` entry: JP slot -> US tile (top), JP slot + 16 -> US
tile + 16 (bottom). No downsampling, no squashing, no posterizing --
every earlier attempt at approximating a 2-tile source glyph into 1 tile
of room, or at collapsing 4-value shading into 1 solid color, was solving
a problem that doesn't exist once the SMALL role is simply left alone.

The real fix for TOP/BOTTOM's own shading (credits' font palettes being
degenerate for JP's own bold glyphs) is the CGRAM palette patch in
``generate.py``'s ``credits_font_upload`` (only when ``--credits-font
us``, leaving the shared ``Palettes_HUD`` table -- these CGRAM slots are
NOT credits-specific, general sprite/HUD accent palettes also used for
sword/shield glow effects elsewhere -- and JP's own degenerate-by-design
palette 3 (SMALL role, now never repointed at US data) completely
untouched) so ``TheFont``'s real shading renders correctly.

Mirrors :mod:`jp_credits_font_asset`'s template-rendering shape.
"""

from __future__ import annotations

from importlib import resources
from string import Template

from . import jp_credits_font_asset
from .us_assets import US_ROM_MD5, asset

#: Filename under ``bin/gfx/`` (both the graft's incbin path and the
#: extractor's output path).
FILENAME = "us_credits_font.2bpp"

#: Same layout/size as jp_credits_font.2bpp -- one 8192-byte flat sheet.
SIZE = 8192

#: Expected md5 of the built output.
OUTPUT_MD5 = "7c8da2ed3e128272e2337048653a1f5f"

#: (JP credits TOP-role tile slot -> source US TheFont top-half tile), one
#: entry per TOP half of each 2-tile location caption character JP's
#: credits actually reference. Read directly off usdasm bank_0E's own
#: ``Credits_CharacterToTile`` table (index-for-index against jpdasm's
#: copy -- both ROMs share the same table shape/character order, just
#: different tile numbers) rather than recomputed from
#: ``RenderText_PerformVWFing``'s formula: the real game deviates from
#: that formula for a few characters (digits use tiles 230-239, not a
#: formula result; 'I' uses tile 175, a dedicated narrow-glyph slot, not
#: the formula's 8) -- reading the table directly matches the real
#: shipped game exactly, formula guesses don't. The generated extractor
#: also copies each entry's BOTTOM half (JP slot + 16 -> US tile + 16) --
#: see module docstring. A slot not listed here is left blank.
TOP_TILE_MAP: dict[int, int] = {
    320: 230,
    321: 231,
    322: 232,
    323: 233,
    324: 234,
    325: 235,
    326: 236,
    327: 237,
    328: 238,
    329: 239,
    330: 0,
    331: 1,
    332: 2,
    333: 3,
    334: 4,
    335: 5,
    352: 6,
    353: 7,
    354: 175,
    355: 9,
    356: 10,
    357: 11,
    358: 12,
    359: 13,
    360: 14,
    361: 15,
    362: 32,
    363: 33,
    364: 34,
    365: 35,
    366: 36,
    367: 37,
    384: 38,
    385: 39,
    386: 40,
    387: 41,
    391: 110,
    424: 161,
}


def _load_template() -> Template:
    """The ``binextract-us-credits-font.py`` source template -- a bundled
    package resource (``.py.template``, so its name makes clear it is not
    itself runnable Python), filled in by
    :func:`render_binextract_us_credits_font`.
    """
    text = (
        resources.files("alttp_jp_english_patcher")
        .joinpath("resources", "binextract_us_credits_font.py.template")
        .read_text(encoding="utf-8")
    )
    return Template(text)


def render_binextract_us_credits_font() -> str:
    """The full source of ``binextract-us-credits-font.py``, generated from
    this module's constants -- the deployed extractor and :data:`OUTPUT_MD5`
    can never disagree, since both come from here. Reads US ``TheFont``'s
    own ROM slice directly (:func:`us_assets.asset`'s ``us_font.2bpp``
    offsets), not the already-extracted file, so extraction order between
    scripts is never a dependency -- unlike the JP credits font baseline
    (:data:`jp_credits_font_asset.FILENAME`), which this DOES read
    already-extracted (``deploy/binextract.py`` guarantees that script
    runs first).
    """
    font_asset = asset("us_font.2bpp")
    (font_slice,) = font_asset.slices
    tile_map_lines = "\n".join(
        f"    {slot}: {tile}," for slot, tile in sorted(TOP_TILE_MAP.items())
    )
    return _load_template().substitute(
        us_rom_md5=repr(US_ROM_MD5),
        output_md5=repr(OUTPUT_MD5),
        us_font_offset=hex(font_slice.offset),
        us_font_size=hex(font_slice.length),
        top_tile_map="{\n" + tile_map_lines + "\n}",
        jp_credits_font_filename=repr(jp_credits_font_asset.FILENAME),
    )
