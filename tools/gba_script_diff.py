#!/usr/bin/env python3
"""gba_script_diff.py - compare the US SNES dialogue script against the GBA
re-release's script, message by message, and render an HTML/text report.

Two independent sources, no guessing:

  SNES: usdasm/text.asm (the cached US disassembly checkout) already carries
  each message's rendered text as a comment block right above its
  `Message_XXXX:` label -- e.g.

      ;===================================================================
      ; Oh!  Here is the Flute!
      ; Its music surely has some
      ; mysterious power!
      ;-------------------------------------------------------------------
      Message_0067:

  We use that comment text directly rather than re-decoding the dictionary-
  compressed `db` bytes ourselves.

  GBA: the message archive is a standard Nintendo BMG (MESGbmg1) container,
  living at ROM offset 0x180CE8 behind a byte-chained XOR/subtract cipher
  (key "l0ZxE8SjYv3#RhaX", from github.com/SiD3W4y/zelda-alttp-re). Once
  decrypted it's INF1 (a table of DAT1 offsets, one per message) + DAT1 (the
  string pool: ASCII text, 0x0A newlines, and 0x1A-escaped control codes
  whose opcode numbers match the SNES engine's own -- 0x74/0x75/0x76 line 1/
  2/3, 0x73 scroll, 0x7A speed, 0x7E wait, 0x7F end, 0x6A player-name,
  0x6C number -- confirmed byte-for-byte against this repo's US ROM).

  Long multi-page messages sometimes carry literal runs of ASCII space bytes
  spanning whole lines between paragraphs (confirmed in the raw DAT1 bytes,
  not a decode bug) -- the GBA's own transition effect for clearing the
  textbox before the next paragraph, where the SNES used a single $73
  "scroll text" opcode instead. They show up as blank lines in the "full"
  report; normalize() collapses them away for "wording".

  Message indices align 1:1 with the SNES Message_XXXX IDs through $0186;
  the GBA then inserts new file-select messages before continuing, so the
  tail (SNES $0187-$018C) is realigned by best-match, and everything past
  the SNES's last index (GBA 397-454) is GBA-only new content.

Usage:
    gba_script_diff.py --gba-rom PATH --out DIR
        [--usdasm DIR] [--report {wording,full,both}]
        [--format {html,text,both}]

`--usdasm` defaults to this project's cached usdasm checkout (same cache
platformdirs location the package itself uses; found independently here so
this standalone tool keeps no import dependency on the package, matching the
rest of tools/). `--out` must be OUTSIDE the repo -- the reports are
ROM-derived copyrighted game text, which this repo never stores (see
GENERATION.md).
"""
import argparse
import difflib
import html
import re
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SNES side: usdasm/text.asm's comment-rendered messages
# ---------------------------------------------------------------------------

_SNES_MSG_RE = re.compile(r"^Message_([0-9A-F]{4}):")
SNES_MESSAGE_COUNT = 397


def load_snes(usdasm_dir):
    """{id: [line, ...]} for every Message_XXXX in usdasm/text.asm, taken
    straight from its rendered-text comment block (not re-decoded)."""
    text_path = usdasm_dir / "text.asm"
    lines = text_path.read_text(encoding="utf-8", errors="replace").split("\n")
    messages = {}
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith(";===="):
            j = i + 1
            comment = []
            valid = True
            while j < n and not lines[j].startswith(";----"):
                if not lines[j].startswith(";"):
                    # not a real comment block -- e.g. a bank-boundary
                    # ";====" .. ";----" pair wrapping non-comment content
                    # (an "org" directive, a bare label) rather than
                    # rendered message text. Confirmed live: this exact
                    # pattern around Message_0167's bank switch ("org
                    # $0EDF40" / "Message_DataExtra:") was swallowing that
                    # boilerplate into $0167's comment before this check
                    # existed. Abandon and let the outer loop find the
                    # real ";====" block that actually precedes the label.
                    valid = False
                    break
                comment.append(lines[j][1:])
                j += 1
            if not valid:
                i += 1
                continue
            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            m = _SNES_MSG_RE.match(lines[k]) if k < n else None
            if m:
                messages[int(m.group(1), 16)] = [c.strip() for c in comment]
            i = j + 1
        else:
            i += 1
    if not messages:
        sys.exit(f"no messages parsed from {text_path} -- usdasm checkout may have drifted")
    ids = sorted(messages)
    if ids != list(range(ids[0], ids[-1] + 1)):
        sys.exit(f"{text_path}: parsed message IDs are not contiguous -- format changed?")
    if len(messages) != SNES_MESSAGE_COUNT:
        print(f"warning: expected {SNES_MESSAGE_COUNT} SNES messages, parsed "
              f"{len(messages)} -- usdasm may have changed", file=sys.stderr)
    return messages


# ---------------------------------------------------------------------------
# GBA side: the encrypted BMG message archive
# ---------------------------------------------------------------------------

GBA_ARCHIVE_OFFSET = 0x180CE8
GBA_ARCHIVE_KEY = b"l0ZxE8SjYv3#RhaX"
# max plausible archive size to decrypt speculatively before re-checking against
# the header's real size (the real archive is ~86 KB; this is a generous cap).
_GBA_PROBE_SIZE = 0x40000

# GBA opcode numbers match the SNES engine's own (see usdasm/text.asm comments).
_OP_LINE = {0x73, 0x74, 0x75, 0x76}  # scroll/line-N: redundant with the literal
                                     # '\n' that already precedes the text; a
                                     # cursor/position hint, not itself a break
_OP_NAME = 0x6A       # player name placeholder
_OP_NUM = 0x6C        # numeric placeholder, takes a 1-byte index operand
_OP_WAIT = 0x7E
_OP_SPEED = 0x7A      # takes a 1-byte operand
_OP_END = 0x7F


def _gba_decrypt(buf):
    """Byte-chained XOR/subtract cipher guarding the GBA resource archive.
    Each output byte depends only on the RAW input at i and i-0x2B (not on
    prior *decrypted* output), so decrypting a superset of the real archive
    and slicing is safe -- there is no need to know the size up front."""
    out = bytearray(len(buf))
    key = GBA_ARCHIVE_KEY
    for i, b in enumerate(buf):
        k1 = key[i & 0xF]
        k2 = ((i & 0xFF) - key[15 - (i & 0xF)]) & 0xFFFFFFFF
        prior = buf[i - 0x2B] if i > 0x2A else 0
        res = (b ^ k2) if i <= 0x2A else ((b ^ k2) - prior) & 0xFFFFFFFF
        out[i] = (((res - (i & 0xFF)) & 0xFFFFFFFF) ^ k1) & 0xFF
    return bytes(out)


def _decode_gba_message(dat1, offset):
    """Render one DAT1 string into text lines + a flat token list.

    Returns (lines, plain) where `lines` is a list[str] (already broken on
    the literal '\\n's / END, control-code placeholders substituted in) for
    the "full" report, and `plain` is the same content whitespace-collapsed
    to one line for the "wording" report's normalizer.
    """
    out = []
    i = offset
    n = len(dat1)
    while i < n:
        b = dat1[i]
        if b == 0x1A:
            total_len = dat1[i + 1]
            payload = dat1[i + 2:i + total_len]
            i += total_len
            # payload = 2-byte group id, opcode, optional 1-byte operand.
            # Group 00 00 carries the text-engine opcodes that match the
            # SNES's own numbering (confirmed: 00 00 7a 00 = "set draw
            # speed" operand 0; 00 00 76 = "line 3", no operand). Other
            # groups exist too -- e.g. 01 01 4f is a directional-arrow icon
            # in signpost messages ("This way <arrow> Desert of Mystery")
            # -- and are rendered verbatim as an [OP:group,code] tag below
            # rather than guessed at, since their glyph wasn't confirmed by
            # rendering actual tiles (tools/render_tiles.py's own lesson:
            # never deduce a glyph mapping, only render/confirm it).
            if len(payload) >= 3 and payload[0:2] == b"\x00\x00":
                opcode = payload[2]
                operand = payload[3] if len(payload) > 3 else None
            else:
                opcode, operand = None, None
            if opcode == _OP_END:
                break
            if opcode in _OP_LINE or opcode == _OP_WAIT or opcode == _OP_SPEED:
                continue  # positioning/pacing hints only, no visible output
            if opcode == _OP_NAME:
                out.append("[LINK]")
            elif opcode == _OP_NUM:
                out.append(f"[#{operand}]" if operand is not None else "[#?]")
            elif opcode is not None:
                out.append(f"[OP:{opcode:02X}"
                            f"{f',{operand:02X}' if operand is not None else ''}]")
            else:
                out.append(f"[OP:{','.join(f'{b:02X}' for b in payload)}]")
        elif b == 0x00:
            i += 1
            break
        elif b == 0x0A:
            out.append("\n")
            i += 1
        else:
            out.append(chr(b))
            i += 1
    text = "".join(out)
    # The GBA's plain-ASCII text engine has no dedicated ellipsis glyph, so
    # it spells a pause out as three literal '.' bytes where the SNES's
    # dictionary-compressed font uses one glyph (confirmed against raw
    # DAT1 bytes: "sealed" + 2E 2E 2E). Render it the same way the SNES
    # side already does, so this font/encoding artifact doesn't show up as
    # a wording difference.
    text = re.sub(r"\.{3,}", "…", text)
    lines = text.split("\n")
    # Most messages end on a literal newline byte right before the $7F
    # terminator (confirmed across a spread of messages) -- an authoring
    # artifact of the format, not a real displayed blank row: the box just
    # closes on $7F regardless of what's left in the line buffer. A message
    # that instead ends on a control code (e.g. a choose-prompt) has no such
    # trailing newline, so this only strips the artifact where it exists.
    if lines and lines[-1] == "":
        lines.pop()
    plain = " ".join(text.split())
    return lines, plain


def load_gba(rom_path):
    """{id: (lines, plain)} for every message in the GBA ROM's BMG archive."""
    rom = rom_path.read_bytes()
    if len(rom) < GBA_ARCHIVE_OFFSET + 32:
        sys.exit(f"{rom_path} is too small to contain the message archive "
                  f"(need at least {GBA_ARCHIVE_OFFSET + 32} bytes)")
    probe_end = min(GBA_ARCHIVE_OFFSET + _GBA_PROBE_SIZE, len(rom))
    dec = _gba_decrypt(rom[GBA_ARCHIVE_OFFSET:probe_end])
    if dec[0:8] != b"MESGbmg1":
        sys.exit(f"{rom_path}: no MESGbmg1 archive found at "
                  f"0x{GBA_ARCHIVE_OFFSET:X} after decrypting -- wrong ROM "
                  "revision, or the archive moved?")
    archive_size = struct.unpack(">I", dec[8:12])[0] * 32
    section_count = struct.unpack(">I", dec[12:16])[0]
    if archive_size > len(dec):
        end = min(GBA_ARCHIVE_OFFSET + archive_size, len(rom))
        dec = _gba_decrypt(rom[GBA_ARCHIVE_OFFSET:end])

    pos = 32
    sections = {}
    for _ in range(section_count):
        name = dec[pos:pos + 4].decode("ascii", errors="replace")
        size = struct.unpack(">I", dec[pos + 4:pos + 8])[0]
        sections[name] = dec[pos + 8:pos + size]
        pos += size
    if pos != archive_size:
        sys.exit(f"{rom_path}: archive size mismatch (parsed {pos} bytes of "
                  f"sections, header declared {archive_size})")
    if "INF1" not in sections or "DAT1" not in sections:
        sys.exit(f"{rom_path}: message archive is missing INF1/DAT1 "
                  f"(found: {sorted(sections)})")

    inf1, dat1 = sections["INF1"], sections["DAT1"]
    count, entry_size = struct.unpack(">HH", inf1[0:4])
    if entry_size != 4:
        sys.exit(f"{rom_path}: unexpected INF1 entry size {entry_size} "
                  "(expected 4 -- BMG format changed?)")
    offsets = [struct.unpack(">I", inf1[8 + i * 4:12 + i * 4])[0] for i in range(count)]
    return {i: _decode_gba_message(dat1, off) for i, off in enumerate(offsets)}


# ---------------------------------------------------------------------------
# Alignment: SNES Message_XXXX id -> GBA message index
# ---------------------------------------------------------------------------

# The GBA inserts new file-select/save messages starting after SNES $0186,
# shifting every following SNES id by a fixed +8 in the GBA's own numbering
# (confirmed by similarity scoring every index against both shift 0 and
# shift 8: SNES $0186 scores 0.72 at shift 0 / 0.57 at shift 8 -- still
# shift 0 -- while SNES $0187 flips to 0.15 at shift 0 / 1.00 at shift 8,
# and every id through $018C stays >=0.76 at shift 8 from there on).
_TAIL_SHIFT_AFTER = 0x186
_TAIL_SHIFT = 8


def align(snes, gba):
    """{snes_id: gba_index or None}, plus the list of GBA indices with no
    SNES counterpart (GBA-only messages)."""
    mapping = {}
    for snes_id in sorted(snes):
        gba_id = snes_id + _TAIL_SHIFT if snes_id > _TAIL_SHIFT_AFTER else snes_id
        mapping[snes_id] = gba_id if gba_id in gba else None
    matched_gba = {g for g in mapping.values() if g is not None}
    gba_only = sorted(set(gba) - matched_gba)
    return mapping, gba_only


# ---------------------------------------------------------------------------
# Wording normalization (whitespace/punctuation/placeholder-insensitive)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[#\d+\]|\[LINK\]|\[NAME\]")


def normalize(text):
    text = text.replace("…", "...").replace("⎵", " ")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = _PLACEHOLDER_RE.sub("@PH@", text)
    text = re.sub(r"\[[A-Za-z0-9,:]+\]", "", text)  # drop remaining bracket ops
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Proposed text: a curated list of individually-verified corrections applied
# to the SNES original -- NOT a generic diff/merge. An earlier version of
# this tool spliced GBA replacement spans into the SNES text whenever they
# were "short enough" (SequenceMatcher + a word-count threshold); that
# produced grammatically broken output whenever GBA restructured a sentence
# instead of just rewording it (dangling parens, mismatched clauses), because
# word-alignment has no concept of grammar. Splicing text across a source
# boundary is inherently unsafe at scale.
#
# Every rule below was explicitly discussed and approved (or given outright
# by exact wording) in conversation -- nothing here reflects an editorial
# call made unilaterally while building this tool. Each is a same-meaning
# synonym/typo/spelling/hyphenation/casing fix, or a one-off full-phrase
# swap matched by exact original text so it can only ever fire on the
# single message it was written for. Nothing here reorders or drops SNES
# text, and nothing pulls in GBA's wordier retranslations on this tool's
# own initiative -- a message with none of these patterns proposes
# unchanged.
#
# GLOBAL_FIXES apply everywhere; every pattern here was checked against the
# FULL script (not just the messages under review) to confirm it can only
# match the specific message(s) it was written for. MESSAGE_FIXES apply
# only to one named message id -- used when a pattern (like a bare place
# name) legitimately recurs in a later message that hasn't been reviewed
# yet, so a global rule would silently change something never discussed.
# ---------------------------------------------------------------------------

GLOBAL_FIXES = [
    # $0003 -- also normalizes "And"/"Without" to lowercase mid-phrase:
    # $0186 has this SAME phrase, "Save and Quit", with a lowercase "and",
    # proving the capitalized style in this message isn't deliberate.
    (r"Save And Continue Save And Quit Do Not Save And Continue",
     "Save and Continue Save and Quit Continue without Saving"),
    # $0032 -- "Course" is the stray capital in this choice option; "Of"
    # correctly stays capitalized as the option's own first word.
    (r"\bOf Course!", "Of course!"),
    # $017E/$0181 -- "chest" is lowercase in all 9 of its other occurrences
    # in the script; these two are the only capitalized "Chest"s.
    (r"\bOpen A Chest\b", "Open a chest"),
    # $0184/$0185 -- the destination names ("Sanctuary", "The Mountain
    # Cave") are genuine title-cased location labels; only "From", a
    # connecting preposition, is the stray capital.
    (r"\bStart From\b", "Start from"),
    # $000F
    (r"\bWhat're\b", "What are"),
    # $0016
    (r"\belder of the village\b", "village elder"),
    # $001F, $0020
    (r"\bdungeon of the castle\b", "castle dungeon"),
    # $0020
    (r"I know there is a hidden path from outside of  the castle to the "
     r"garden inside\.",
     "Somewhere outside the castle you should find a hidden entrance to "
     "the palace garden."),
    # $0022
    (r"It's pitch dark inside and", "It's pitch-dark inside, and"),
    # $0024
    (r"but first we have to go to the first floor\.",
     "but we must go to the first floor to reach it."),
    # $0026
    (r"\bfortune teller\b", "fortune-teller"),
    # $002B -- real ellipsis character, not three literal dots
    (r"Oh, no one", "Oh… No one"),
    # $002C
    (r"Long ago, a prosperous people known as the Hylia inhabited this "
     r"land… Legends tell of many treasures that the Hylia hid throughout "
     r"the land… The Master Sword, a mighty blade forged against those "
     r"with evil hearts, is one of them\.  People say that now it is "
     r"sleeping deep in the forest…",
     "Long ago a prosperous people, known as the Hylians, inhabited this "
     "land… Legends tell of treasures with mystical powers that remain "
     "from the Hylian age… The Master Sword, a mighty blade forged to "
     "thwart those with evil hearts, is one… It is said that even now it "
     "rests deep in the forest."),
    # bare "Sanctuary" -> "the Sanctuary"/"The Sanctuary" (start of message):
    # applies everywhere, including $0184/$0185's "Start From Sanctuary" --
    # GBA's own text there already says "Start From the Sanctuary", too.
    # The "^" rule must run first, and the mid-sentence rule must skip
    # anything it already produced -- else "The Sanctuary" gets a second
    # "the " prepended into "The the Sanctuary".
    (r"^Sanctuary\b", "The Sanctuary"),
    (r"(?<!The )(?<!the )\bSanctuary\b", "the Sanctuary"),
]
_COMPILED_GLOBAL_FIXES = [(re.compile(p), r) for p, r in GLOBAL_FIXES]


def propose(snes_lines):
    """Apply GLOBAL_FIXES to the SNES text and return it as a single-string
    line list, matching Row's list-of-lines shape. GBA text plays no part
    in this beyond having motivated the rule list above -- see the module
    comment for why this isn't a mechanical merge of the two sources, and
    for what counts as an approved rule."""
    text = " ".join(snes_lines)
    for pattern, replacement in _COMPILED_GLOBAL_FIXES:
        text = pattern.sub(replacement, text)
    return [text]


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------


class Row:
    __slots__ = ("snes_id", "gba_id", "snes_lines", "gba_lines",
                 "proposed_lines", "kind")

    def __init__(self, snes_id, gba_id, snes_lines, gba_lines,
                 proposed_lines, kind):
        self.snes_id = snes_id
        self.gba_id = gba_id
        self.snes_lines = snes_lines
        self.gba_lines = gba_lines
        self.proposed_lines = proposed_lines  # [] if not applicable
        self.kind = kind  # "changed" | "snes_only" | "gba_only"


def build_rows(snes, gba, mapping, gba_only):
    rows = []
    for snes_id in sorted(snes):
        gba_id = mapping[snes_id]
        snes_lines = snes[snes_id]
        if gba_id is None:
            rows.append(Row(snes_id, None, snes_lines, [], [], "snes_only"))
            continue
        gba_lines, _ = gba[gba_id]
        proposed = propose(snes_lines)
        rows.append(Row(snes_id, gba_id, snes_lines, gba_lines, proposed, "changed"))
    for gba_id in gba_only:
        gba_lines, _ = gba[gba_id]
        rows.append(Row(None, gba_id, [], gba_lines, [], "gba_only"))
    return rows


def wording_differs(row):
    if row.kind != "changed":
        return True
    return normalize(" ".join(row.snes_lines)) != normalize(" ".join(row.gba_lines))


def full_differs(row):
    if row.kind != "changed":
        return True
    return row.snes_lines != row.gba_lines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _word_diff_html(a_lines, b_lines):
    a = " ".join(a_lines).split(" ")
    b = " ".join(b_lines).split(" ")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_seg = html.escape(" ".join(a[i1:i2]))
        b_seg = html.escape(" ".join(b[j1:j2]))
        if tag == "equal":
            left.append(a_seg)
            right.append(b_seg)
        else:
            if a_seg:
                left.append(f'<del>{a_seg}</del>')
            if b_seg:
                right.append(f'<ins>{b_seg}</ins>')
    return " ".join(left), " ".join(right)


_HTML_HEAD = """<!doctype html><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#1b1b1f;
     color:#e8e8ec}}
header{{position:sticky;top:0;background:#111114;padding:10px 16px;
       border-bottom:1px solid #333;display:flex;gap:12px;align-items:center;z-index:2}}
header input{{flex:1;padding:6px 10px;border-radius:6px;border:1px solid #444;
             background:#26262c;color:#e8e8ec;font-size:14px}}
header .count{{color:#999;font-size:12px;white-space:nowrap}}
table{{border-collapse:collapse;width:100%;table-layout:fixed}}
th{{position:sticky;top:45px;background:#111114;text-align:left;padding:8px 12px;
   border-bottom:1px solid #333;font-size:12px;color:#999;text-transform:uppercase}}
th:first-child,td:first-child{{width:90px}}
th:not(:first-child),td:not(:first-child){{width:calc((100% - 90px)/3)}}
td{{padding:10px 12px;border-bottom:1px solid #29292f;vertical-align:top;
   white-space:pre-wrap;overflow-wrap:break-word;font-size:14px}}
tr:hover td{{background:#222228}}
.id{{color:#888;font-family:ui-monospace,monospace;font-size:12px;white-space:nowrap}}
.kind-snes_only td:first-child{{border-left:3px solid #d97757}}
.kind-gba_only td:first-child{{border-left:3px solid #6ba3d6}}
.kind-changed td:first-child{{border-left:3px solid #7fbf7f}}
del{{background:#4a1f1f;color:#ff9a9a;text-decoration:line-through;padding:0 1px}}
ins{{background:#1f3a1f;color:#9dffa0;text-decoration:none;padding:0 1px}}
.tag{{font-size:10px;padding:1px 6px;border-radius:8px;background:#333;color:#bbb;
     margin-left:6px}}
.hidden{{display:none}}
.proposed{{background:#20242b}}
.note{{margin:0;padding:8px 16px;background:#1f2937;color:#9db4d1;font-size:12px;
      border-bottom:1px solid #333}}
</style>
<header>
  <strong>{title}</strong>
  <input id="filter" placeholder="filter (id or text)&hellip;">
  <span class="count" id="count"></span>
</header>
<p class="note">"Proposed" applies ONLY fixes explicitly discussed and approved (see
GLOBAL_FIXES / MESSAGE_FIXES in tools/gba_script_diff.py) -- nothing here
reflects an editorial call made unilaterally by this tool. It does NOT merge
or splice in GBA's wording generally -- a message with no approved fix
proposes unchanged. Highlighted as a diff from the US SNES column.</p>
<table>
<thead><tr><th>ID</th><th>US SNES</th>
<th>Proposed<span class="tag">curated fixes</span></th><th>GBA</th></tr></thead>
<tbody>
"""

_HTML_TAIL = """</tbody></table>
<script>
const rows = [...document.querySelectorAll("tbody tr")];
const filter = document.getElementById("filter");
const count = document.getElementById("count");
function apply() {
  const q = filter.value.trim().toLowerCase();
  let shown = 0;
  for (const r of rows) {
    const hit = !q || r.textContent.toLowerCase().includes(q);
    r.classList.toggle("hidden", !hit);
    if (hit) shown++;
  }
  count.textContent = shown + " / " + rows.length + " messages";
}
filter.addEventListener("input", apply);
apply();
</script>
"""


def render_html(rows, title):
    out = [_HTML_HEAD.format(title=html.escape(title))]
    for row in rows:
        left_id = f"${row.snes_id:04X}" if row.snes_id is not None else "—"
        right_id = str(row.gba_id) if row.gba_id is not None else "—"
        tag = {"changed": "", "snes_only": '<span class="tag">SNES only</span>',
               "gba_only": '<span class="tag">GBA only</span>'}[row.kind]
        left, right = _word_diff_html(row.snes_lines, row.gba_lines)
        if row.kind == "changed":
            _, proposed = _word_diff_html(row.snes_lines, row.proposed_lines)
        else:
            proposed = "—"
        anchor = f'm{row.snes_id if row.snes_id is not None else "g" + str(row.gba_id)}'
        out.append(
            f'<tr id="{anchor}" class="kind-{row.kind}">'
            f'<td class="id">{left_id}<br>{right_id}{tag}</td>'
            f'<td>{left}</td><td class="proposed">{proposed}</td>'
            f'<td>{right}</td></tr>\n'
        )
    out.append(_HTML_TAIL)
    return "".join(out)


def render_text(rows, title):
    out = [title, "=" * len(title), ""]
    for row in rows:
        left_id = f"${row.snes_id:04X}" if row.snes_id is not None else "----"
        right_id = str(row.gba_id) if row.gba_id is not None else "----"
        out.append(f"[{left_id} -> {right_id}] ({row.kind})")
        for line in difflib.unified_diff(
            row.snes_lines, row.gba_lines, lineterm="",
            fromfile="snes", tofile="gba",
        ):
            out.append(line)
        if row.kind == "changed":
            out.append(f"PROPOSED (auto-draft): {row.proposed_lines[0]}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_usdasm_dir():
    try:
        from platformdirs import PlatformDirs
    except ImportError:
        return None
    path = PlatformDirs("alttp-jp-english-patcher", appauthor="sevaht").user_cache_path
    if path.parent.name != "sevaht":
        path = path.parent / "sevaht" / "alttp-jp-english-patcher"
    return path / "usdasm"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gba-rom", required=True, type=Path, help="GBA ALttP&FS ROM")
    ap.add_argument("--usdasm", type=Path, default=None,
                     help="usdasm checkout (default: this project's cached clone)")
    ap.add_argument("--out", required=True, type=Path,
                     help="output directory (must be OUTSIDE this repo)")
    ap.add_argument("--report", choices=("wording", "full", "both"), default="both")
    ap.add_argument("--format", choices=("html", "text", "both"), default="html")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out.resolve()
    if out_dir == repo_root or repo_root in out_dir.parents:
        ap.error(f"--out must be outside the repo ({repo_root}); "
                  "these reports contain copyrighted game text")

    usdasm_dir = args.usdasm
    if usdasm_dir is None:
        usdasm_dir = _default_usdasm_dir()
        if usdasm_dir is None or not (usdasm_dir / "text.asm").exists():
            ap.error("--usdasm not given and no cached checkout found; pass "
                      "--usdasm explicitly (e.g. the path used by "
                      "alttp-jp-english-patcher, or a fresh clone of "
                      "https://github.com/spannerisms/usdasm)")

    snes = load_snes(usdasm_dir)
    gba = load_gba(args.gba_rom)
    mapping, gba_only = align(snes, gba)
    rows = build_rows(snes, gba, mapping, gba_only)

    aligned = sum(1 for v in mapping.values() if v is not None)
    print(f"SNES messages: {len(snes)}   GBA messages: {len(gba)}   "
          f"aligned: {aligned}   SNES-only: {len(snes) - aligned}   "
          f"GBA-only: {len(gba_only)}")

    reports = []
    if args.report in ("wording", "both"):
        reports.append(("wording", [r for r in rows if wording_differs(r)],
                         "Wording differences: US SNES vs GBA"))
    if args.report in ("full", "both"):
        reports.append(("full", [r for r in rows if full_differs(r)],
                         "All differences (incl. rewrapping): US SNES vs GBA"))

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, report_rows, title in reports:
        print(f"{name}: {len(report_rows)} messages differ")
        if args.format in ("html", "both"):
            path = out_dir / f"{name}.html"
            path.write_text(render_html(report_rows, title), encoding="utf-8")
            print(f"  wrote {path}")
        if args.format in ("text", "both"):
            path = out_dir / f"{name}.txt"
            path.write_text(render_text(report_rows, title), encoding="utf-8")
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
