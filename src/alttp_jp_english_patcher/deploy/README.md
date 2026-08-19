# A Link to the Past — JP 1.0 English Translation

An English build of the Japanese 1.0 version of the game. This is a fork of the
[JP 1.0 disassembly](https://github.com/spannerisms/jpdasm) with the US ROM's
English text, menu, and file-select subsystems grafted in — so it assembles to a
fully playable English ROM while staying byte-for-byte the JP 1.0 base
everywhere the translation does not touch.

The base banks are hooked in place; the graft lives beside them in expanded-ROM
banks `bank_20` .. `bank_2E`; `main.asm` is wired to include them and pad the
ROM to 2 MB. It carries a boot-time save migrator, so US and vanilla-Japanese
save files are converted to this build's format automatically the first time you
load them.

### Save-file names

When a foreign save is migrated, its player name is handled by what the two
fonts can actually show:

* **US saves** — the name is kept as-is (the English font already covers it).
* **Japanese saves** — the name is kept only if it is made up **entirely of
  uppercase Latin letters** (`A`–`Z`, the sole characters both the Japanese and
  English fonts share); those are remapped to this build's encoding. Any name
  containing kana or other glyphs the English font can't render is left
  blank instead. Selecting a save with a blank name drops you into the
  normal naming screen to pick one — the rest of the save (items, deaths,
  progress) is untouched, and confirming a name returns you to file select
  just like naming a brand-new file does.

### Intro cutscene guards

JP 1.0 has a programming bug where a stray write clobbers the register a
check relies on, so every guard in the opening cutscene draws with a spear
instead of its intended sprite. This build reorders the routine (matching
how the US ROM already does it) so the check works and guards draw
correctly; pass `--no-intro-fix` when generating to keep JP 1.0's original
(buggy) behavior instead.

### Credits

By default, the ending credits keep JP 1.0's own (bolder) font rather than
being converted to the US Latin font used everywhere else — JP's credits
text is already English, and its font reads better than a straight
conversion would, so the credits intentionally look different from the
rest of the game. Pass `--credits-font us` when generating to use the US
dialogue font there instead, matching the rest of the game's look; either
way, this is computed offline (no runtime decompression, no emulator) from
your own ROMs at extraction time (`binextract-jp-credits-font.py` /
`binextract-us-credits-font.py`), never committed. By default this build
also fixes a handful of JP 1.0 translation mistakes to match the US
release: "THE LOYAL PRIEST" → "THE LOYAL SAGE", "FINGER WEBS FOR SALE" →
"FLIPPERS FOR SALE" (centered — the US release left it off-center),
"OCARINA BOY PLAYS AGAIN" → "FLUTE BOY PLAYS AGAIN", "GANNON'S TOWER" →
"GANON'S TOWER", and adds the US-only "ENGLISH SCRIPT WRITERS" attribution.
Pass `--keep-jp-credits` when generating to leave the credits text exactly
as JP 1.0 shipped it (the font choice is independent either way).

### Weathercock

JP 1.0 and the US ROM both ship an animated overworld "weathercock" (windmill
vane) decoration whose right end is missing a bordering pixel, across all 3
of its animation frames — the tip looks left open instead of closed off. This
build adds that pixel to match the EU release; pass `--no-weathercock-fix`
when generating to keep JP 1.0/US's original (open-ended) look instead.

### Eastern Palace floor tile

JP 1.0 has a Star of David as a floor tile in Eastern Palace; the US ROM
replaced it with a generic tile. By default this build swaps in the US ROM's
version; pass `--keep-religious-imagery` when generating to keep JP 1.0's
original tile instead.

### Flash brightness (photosensitive-epilepsy safety)

JP 1.0's full-screen flash effects (Agahnim's and Vitreous's lightning
attacks, the Ether Medallion, the title screen's attract-mode cutscene, and
the Magic Bat's power-up flash all share one routine) boost each color
channel by 14 (of a maximum 31) every other frame while flashing -- a very
bright, high-contrast flicker. A later Japanese revision (matching a
Virtual Console ROM dump) tones this down to a boost of 2, as part of a
photosensitive-epilepsy-safety pass; this build applies that same reduction
by default. Pass `--no-epilepsy-fix` when generating to keep JP 1.0's
original (much brighter) flash intensity instead.

### Low-health warning beep

By default this build leaves the repeating low-health warning beep exactly
as JP 1.0/US shipped it (queued every 32 frames while current health is
below a fixed per-max-hearts threshold, silenced during cutscenes and
healing). Pass `--no-low-health-beep` when generating to silence it
entirely.

### Attract-mode caption timing

The attract-mode demo's four auto-scrolling story captions (the "Long ago,
in the beautiful kingdom of Hyrule..." intro, and the throne-room/prison/
altar dungeon captions) each stay on screen for a fixed duration before the
scene fades out, sized in JP 1.0 for its own short, kanji-dense text. This
build's English translation no longer fits in that time, so these durations
are widened to the US ROM's own values (and, for the three dungeon
captions, its wider counter) instead.

### GBA-era wording fixes

By default this build applies wording fixes to the dialogue, the intro, the
credits, and the item menu, matching the GBA re-release's translation:
"pegasus shoes" → "pegasus boots", "faerie"/"faeries" → "fairy"/"fairies",
"wise men"/"wise man" → "sages"/"sage" (the singular form never occurs), and
"smithery" → "smithy" (the credits' SMITHERY location caption, recentered;
"faeries" → "fairies" is also a credits-only occurrence, the "VENUS. QUEEN
OF FAERIES" caption). The bottled fairy item's own menu name also changes
from "FAERIE" to "FAIRY", and the bottled "GOOD BEE" item is renamed to
"GOLDEN"/"BEE" (two rows, matching the "MAGIC"/"MIRROR" layout) -- both in
the bottle-content menu and the equipped-item display. Pass
`--no-gba-text-fixes` when generating to keep the original US translation's
wording instead.

### Ocarina/Flute

The JP original calls this item the Ocarina, matching every other Zelda
game; the US translation renamed it "Flute". By default this build keeps
the US translation's "Flute" naming. Pass `--flute-is-ocarina` when
generating to rename it to "Ocarina" everywhere -- dialogue, the item
menu, and the credits, where JP's own "OCARINA BOY PLAYS AGAIN" caption is
kept instead of being converted to "FLUTE BOY PLAYS AGAIN" (this is
independent of `--keep-jp-credits`).

### Title screen

The US ROM's title screen plays a sword-reveal animation (the Triforce
splits open and the Master Sword rises out of it) before settling into the
logo; JP 1.0 skips straight to the logo without it. By default this build
uses the US ROM's version, including its attract-mode background colors
and title-logo/triforce OAM layering (keeping JP 1.0's own press-to-skip
timing -- a button press skips it as soon as the triforce forms, not
gated behind the sword animation like the real US ROM). Pass
`--title-screen jp` when generating to keep JP 1.0's own title screen
instead.

---

## Building

To assemble this code you need [Asar](https://github.com/RPGHacker/asar) — a
special pooling fork is included as `asarmon.exe` (run natively where a Linux
`asarmon` exists, else through `wine`).

Extraction of the ROM-derived binaries is a separate, one-time step (see
**Binaries** below); the build itself only assembles:

```sh
# 1. put both ROMs in this directory (see Binaries)
python3 binextract.py          # extract bin/gfx/* (JP + US), bin/brr/* (JP)
./_build.sh                    # -> alttp-english.sfc   (Linux/macOS)
```

On Windows, run `binextract.py`, then `_build.bat`. `make` also works on any
platform with `asarmon` on `PATH`. All three produce `alttp-english.sfc`; a
correct default build has MD5 `619c5348c7c5082dfe358c56af2b6104`.

## Binaries

Raw binaries are not included in this repository due to copyright. You extract
them from your **own** ROMs. This build draws data from **two** ROMs, so place
both in the base directory:

* `alttp.sfc` — JP 1.0 ROM (md5 `03a63945398191337e896e5771f77173`)
* `alttp-us.sfc` — US ROM (md5 `608c22b8ff930c62dc2de54bcd6eba72`)

Then run `python3 binextract.py`, which drives all four extractors:

* `binextract-jp.py` — the base disassembly's own extractor: the JP graphics and
  audio binaries under `bin/` (from `alttp.sfc`).
* `binextract-us.py` — the English font/menu assets, also under `bin/gfx/`
  (`us_*` files, alongside the JP ones there; from `alttp-us.sfc`): the US
  variable-width font and the file-select font/graphics/palette slices the
  graft repoints to.
* `binextract-jp-credits-font.py` — the credits' bold font
  (`jp_credits_font.2bpp`, from `alttp.sfc`): JP 1.0 ships this font
  pre-compressed (a bespoke bit-packed scheme the game unpacks at runtime);
  this script does that same unpacking once, offline, from the raw ROM
  bytes (no emulator), so the build just uploads the flat result instead of
  shipping the decompression routines. Only the ~69 glyphs credits actually
  display are unpacked (everything else — almost all of it, JP's kanji/kana
  that nothing in this build reads anymore — is skipped); the original
  compressed asset is untouched either way, still sitting in the base ROM.
* `binextract-us-credits-font.py` — a US-styled alternate for that same font
  (`us_credits_font.2bpp`, from `alttp-us.sfc`; only used with
  `--credits-font us`), same tile layout, each character's pixel data
  instead pulled from the US dialogue font.

Nothing copyrighted is committed here — everything under `bin/` is
regenerated from your ROMs.

An accurate assembly of the underlying JP 1.0 base has the following checksums:
* Internal (complement): `CDC8` (`3237`)
* CRC32: `3322EFFC`
* MD5: `03A63945398191337E896E5771F77173`
* SHA1: `E7E852F0159CE612E3911164878A9B08B3CB9060`

---

## Special Thanks
The JP 1.0 disassembly this builds on has massive shoutouts to give, in no
particular order:
* IsoFrieze, for creating [Diztinguish](https://github.com/Dotsarecool/DiztinGUIsh/releases).
* MathOnNapkins for his US disassembly, which served as an invaluable reference and sanity check. I took a lot of nomenclature from him, and when I didn't, I still checked my labels against his. The SPC engine is reformatted from his old work, which saved me the trouble of disassembling it.
* An extra, distinct thanks to MathOnNapkins for creating a fork of Asar with proper pool implementation.
* Zarby89, for his vast knowledge of the game's data, and for his direct contributions in parsing data (compressed graphics, overworld data, room objects).
* Myramong for identifying the Japanese kanji.
* Total for figuring out text compression and his direct contributions in parsing text graphics data.
* Lui for explaining Nintendo stripes. Also for being the patient victim of many ramblings about the code.
* Qwertymodo for the makefile.
* Aerinon for the Python extraction script.

---

## Using this disassembly
This disassembly was created with a number of specific guidelines

* Top-level labels use a mix of PascalCase and snake_case, where underscores will separate arbitrary hierarchies, such as `Sprite_MoveFunction`.
* Sublabels use pure snake_case. Some sublabels may redundantly include the top-level parent for explicit clarity; e.g. `Sprite_MoveFunction_continue`.
* The beginning of every line will have an address label of the form `#_AAAAAA:`, where `AAAAAA` is the 24-bit address in ROM in hexadecimal. The `#` prevents the label from creating a new hierarchy. The `_` is required as labels may not begin with numbers.
* APU labels will have `#_SSSS:` following the ROM label, where `SSSS` is the 16-bit address once transfered to the SPC in hexadecimal. To prevent name collision, song data will include an `o`, `u`, or `c` after the APU address, for the overworld, underworld, and credits banks, respectively.
* Code or data that appears unreachable is labelled `UNREACHABLE_AAAAAA`.
* Unreachable blocks of the filler byte `$FF` are labelled with `NULL_AAAAAA` and given a comment with `FREE ROM: <size>`.
* Lines contain 1 instruction each.
* For data bank and direct page changes, the full address will be written out.
* In data tables, the number of entries per line is determined by how they should logically be grouped. In all other cases, they are grouped in sets of four or eight.
* The MVN and MVP instructions are written with a macro so that writing them in the standard way assembles correctly.
* A list of standardized names for various entity classes is found in `values.asm`.
* My personal thoughts are noted in some comments with !WORD, where WORD is in all caps.
* The `.ly` files in the `resources/music/` directory can be compiled with [LilyPond](https://lilypond.org/).
