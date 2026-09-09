"""Generate a patched (English) jpdasm into a target directory (the CLI).

The ``--target`` is treated as an *output* directory, not an existing checkout:
it is created if missing, and any files this produces are overwritten if it
exists (anything else -- your ROMs, extracted binaries, built ROM -- is left
alone). So every run starts from the pristine disassemblies and is fully
predictable/repeatable.

Each run: clones (or reuses) pristine ``usdasm`` + ``jpdasm``; copies the JP
disassembly's support files (``asarmon.exe``, the reference ``.asm``, ``bin/``
dirs, ``binextract.py`` -> ``binextract-jp.py``) into the target; writes the
whole English program over it (base banks hooked in place, graft banks beside
them, ``main.asm`` wired); drops in the build tooling (``binextract-us.py`` +
the ``binextract.py`` stub + ``_build.sh`` + README, plus English-targeting
``Makefile`` / ``_build.bat`` overwriting the JP-only pristine ones) and the
``.gitignore``; and copies the two ROMs into place (``alttp.sfc`` /
``alttp-us.sfc``) if given.

The two disassembly *sources* are plain git checkouts, cloned on demand into
the platformdirs user cache (see
:func:`~alttp_jp_english_patcher.user_cache_path`); override with ``--usdasm``
/ ``--jpdasm`` or ``USDASM_DIR`` / ``JPDASM_DIR``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from . import user_cache_path, verify_base
from .generate import (
    DEFAULT_NOP_PADBYTE_THRESHOLD,
    DEFAULT_NULL_PADBYTE_THRESHOLD,
    build,
)
from .gitignore import patch as patch_gitignore
from .graft import bank_header
from .jp_credits_font_asset import render_binextract_jp_credits_font
from .us_assets import render_binextract_us
from .us_credits_font_asset import render_binextract_us_credits_font

if TYPE_CHECKING:
    from collections.abc import Sequence

# disassembly sources: label -> (git url, ref, a file that marks a good clone).
_SOURCES = {
    "usdasm": (
        "https://github.com/spannerisms/usdasm.git",
        "main",
        "bank_0E.asm",
    ),
    "jpdasm": (
        "https://github.com/spannerisms/jpdasm.git",
        "master",
        "bank_00.asm",
    ),
}

# bundled deploy artifacts -> written verbatim into the target fork.
# binextract-us.py / binextract-jp-credits-font.py /
# binextract-us-credits-font.py are NOT here: they are generated (see
# _deploy_tooling), not static files, so their asset data can never drift
# from what generate.py's incbin sizing assumes -- each reads its own
# module's constants (us_assets.US_ASSETS, jp_credits_font_asset,
# us_credits_font_asset).
#
# Makefile / _build.bat ARE here even though jpdasm ships its own: those
# pristine copies build alttp_reasm.sfc with the checksum fix off (they
# reassemble unmodified JP 1.0), so they're overwritten with versions that
# build alttp-english.sfc with the checksum fix on.
_DEPLOY_FILES = (
    "binextract.py",
    "_build.sh",
    "Makefile",
    "_build.bat",
    "README.md",
)


def _clone(label: str, dest: Path) -> None:
    """Clone the ``label`` source into ``dest`` if missing; else leave it."""
    url, ref, sentinel = _SOURCES[label]
    if (dest / sentinel).exists():
        print(f"{label}: using existing {dest}")
        return
    print(f"{label}: cloning ({ref})")
    command = ["git", "clone", "--quiet", "--depth", "1", "--branch", ref]
    command += [url, str(dest)]
    subprocess.run(command, check=True)  # noqa: S603
    if not (dest / sentinel).exists():
        msg = f"{dest / sentinel} missing after clone"
        raise RuntimeError(msg)


def _resolve_source(label: str, override: Path | None, env_var: str) -> Path:
    """A --flag override / $ENV path (used as-is), else a cached clone."""
    if override is not None:
        return override.resolve()
    env = os.environ.get(env_var)
    if env:
        return Path(env).resolve()
    dest = user_cache_path() / label
    dest.parent.mkdir(parents=True, exist_ok=True)
    _clone(label, dest)
    return dest


# ROMs the target needs: (destination filename, args attr, --flag).
_ROMS = (
    ("alttp.sfc", "jp_rom", "--jp-rom"),
    ("alttp-us.sfc", "us_rom", "--us-rom"),
)


def _populate_from_jpdasm(jpdasm: Path, target: Path) -> None:
    """Fill ``target`` with the pristine jpdasm's support files.

    Copies everything except its git metadata and its own ``binextract.py``
    (which becomes ``binextract-jp.py``); the generated ``.asm`` files are
    overwritten by :meth:`Rom.write` right after, and ``Makefile`` /
    ``_build.bat`` by :func:`_deploy_tooling` right after that (see
    ``_DEPLOY_FILES``). ``asarmon.exe``, the reference ``.asm``, the ``bin/``
    directory scaffolding, and ``LICENSE`` come along so the target is a
    self-contained buildable fork.
    """
    ignore = shutil.ignore_patterns(
        ".git", ".github", "__pycache__", "binextract.py"
    )
    shutil.copytree(jpdasm, target, dirs_exist_ok=True, ignore=ignore)
    jp_extractor = jpdasm / "binextract.py"
    if jp_extractor.is_file():
        shutil.copy2(jp_extractor, target / "binextract-jp.py")


def _deploy_tooling(target: Path) -> None:
    """Drop the English build tooling into the target (overwriting)."""
    root = resources.files("alttp_jp_english_patcher").joinpath("deploy")
    for name in _DEPLOY_FILES:
        (target / name).write_bytes(root.joinpath(name).read_bytes())
    (target / "binextract-us.py").write_text(render_binextract_us())
    (target / "binextract-jp-credits-font.py").write_text(
        render_binextract_jp_credits_font()
    )
    (target / "binextract-us-credits-font.py").write_text(
        render_binextract_us_credits_font()
    )
    (target / "_build.sh").chmod(0o755)
    (target / "binextract.py").chmod(0o755)
    (target / "binextract-jp-credits-font.py").chmod(0o755)
    (target / "binextract-us-credits-font.py").chmod(0o755)


def _place_roms(target: Path, args: argparse.Namespace) -> None:
    """Copy any supplied ROMs into the target under their expected names."""
    for dest_name, attr, _flag in _ROMS:
        source = getattr(args, attr)
        if source is not None:
            shutil.copy2(source, target / dest_name)
            print(f"copied {source} -> {dest_name}")


def _update_gitignore(target: Path) -> None:
    path = target / ".gitignore"
    original = path.read_text() if path.exists() else ""
    patched = patch_gitignore(original)
    if patched != original:
        path.write_text(patched)
        print(f"{path}: added English binary ignore rules")


def _run_verify(jpdasm: Path, usdasm: Path) -> int:
    got = verify_base.compute(jpdasm, usdasm)
    reference = verify_base.load_reference()
    mismatched = [b for b in sorted(got) if got[b] != reference.get(b)]
    for bank in sorted(got):
        print(f"  {bank}: {'MISMATCH' if bank in mismatched else 'match'}")
    if mismatched:
        print(f"FAIL: {', '.join(mismatched)}")
        return 1
    print("base edits OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alttp-jp-english-patcher",
        description="Generate a patched (English) jpdasm into a directory.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="output directory (created if missing; our files overwritten)",
    )
    parser.add_argument(
        "--jp-rom",
        type=Path,
        help="JP 1.0 ROM, copied in as alttp.sfc (required if not already "
        "in the target)",
    )
    parser.add_argument(
        "--us-rom",
        type=Path,
        help="US ROM, copied in as alttp-us.sfc (required if not already "
        "in the target)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="emit the change-free baseline (no graft edits or base hooks)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after deploying, check the base edits against the frozen hashes",
    )
    parser.add_argument(
        "--no-intro-fix",
        action="store_true",
        help="keep JP 1.0's intro-cutscene guard-sprite bug (a clobbered "
        "register makes every guard draw with a spear instead of its "
        "intended sprite); by default this patcher reorders the routine to "
        "match the US ROM's already-correct behavior",
    )
    parser.add_argument(
        "--no-weathercock-fix",
        action="store_true",
        help="keep the animated weathercock (windmill vane) tile's right "
        "end looking open (missing a bordering pixel), exactly as JP "
        "1.0/US shipped it; by default this patcher closes it off to "
        "match the EU release",
    )
    parser.add_argument(
        "--keep-religious-imagery",
        action="store_true",
        help="leave Eastern Palace's floor tile as JP 1.0's Star of David; "
        "by default this patcher swaps it for the US ROM's generic tile",
    )
    parser.add_argument(
        "--no-epilepsy-fix",
        action="store_true",
        help="keep JP 1.0's original (much brighter) full-screen flash "
        "effects (Agahnim's and Vitreous's lightning, the Ether Medallion, "
        "the title screen's attract-mode cutscene, the Magic Bat); by "
        "default this patcher tones down the flash brightness to match a "
        "later Japanese revision's photosensitive-epilepsy-safety pass",
    )
    parser.add_argument(
        "--yellow-counts-at-absolute-max",
        action="store_true",
        help="make the HUD bomb/arrow counters' max-capacity gold color "
        "trigger only at the true, fully-upgraded maximum (50 bombs, 70 "
        "arrows) instead; by default it triggers at the player's current "
        "upgrade tier's own cap, matching the GBA re-release's own "
        "behavior. Rupees are unaffected either way (999 is always the "
        "true max, no tiers)",
    )
    parser.add_argument(
        "--no-low-health-beep",
        action="store_true",
        help="silence the repeating low-health warning beep (queued every "
        "32 frames while current health is below a low threshold); by "
        "default this patcher leaves it exactly as JP 1.0/US shipped it",
    )
    parser.add_argument(
        "--flute-is-ocarina",
        action="store_true",
        help="rename 'Flute' to 'Ocarina' -- the JP original's name for "
        "the item, matching every other Zelda game -- in dialogue, the "
        "item menu, and the credits (JP's own 'OCARINA BOY PLAYS AGAIN' "
        "caption is kept instead of converting it to 'FLUTE BOY PLAYS "
        "AGAIN'); by default this patcher keeps the US translation's "
        "'Flute' naming",
    )
    parser.add_argument(
        "--fix-mothula-bugs",
        action="store_true",
        help="fix Mothula's (Skull Woods boss) damage table so it takes "
        "proper damage from golden-sword swings/dashes, tempered-sword "
        "spin attacks, and golden-sword spin attacks (all currently "
        "zeroed out, matching a longtime community-documented JP 1.0/US "
        "bug); by default this patcher leaves it exactly as JP 1.0/US "
        "shipped it. Does not touch Mothula's own immunity to its room's "
        "spike hazards (the two are linked -- see the tool's own docs) "
        "so this is not a full GBA-parity fix, just this specific bug",
    )
    parser.add_argument(
        "--wide-dialogue-lines",
        action="store_true",
        help="widen every dialogue line's pixel budget from 168px (21 "
        "tiles) to 176px (22 tiles), reclaiming a column the dialog box's "
        "own border was already drawn wide enough for but the text engine "
        "never used; by default this patcher leaves the original 168px "
        "limit in place",
    )
    parser.add_argument(
        "--no-gba-text-fixes",
        action="store_true",
        help="keep the original US translation's wording in dialogue, the "
        "intro, the credits, and the item menu; by default this patcher "
        "applies GBA-era wording fixes: 'pegasus shoes' -> 'pegasus "
        "boots', 'faerie'/'faeries' -> 'fairy'/'fairies', 'wise men' -> "
        "'sages', 'smithery' -> 'smithy', and the bottled 'GOOD BEE' item "
        "-> 'GOLDEN'/'BEE'",
    )
    parser.add_argument(
        "--keep-jp-credits",
        action="store_true",
        help="leave the (JP-fonted) credits text exactly as JP 1.0 shipped "
        "it; by default this patcher fixes a handful of JP 1.0 mistakes to "
        "match the US release (THE LOYAL SAGE, FLIPPERS FOR SALE, FLUTE "
        "BOY, GANON'S TOWER) and adds the US-only ENGLISH SCRIPT WRITERS "
        "attribution",
    )
    parser.add_argument(
        "--credits-font",
        choices=("jp", "us"),
        default="jp",
        help="which font credits render with: 'jp' (default) is JP 1.0's "
        "own bolder font, computed offline from the JP ROM; 'us' pulls the "
        "same character set from the US dialogue font instead, matching "
        "the rest of the (US-fonted) game's look",
    )
    parser.add_argument(
        "--title-screen",
        choices=("us", "jp"),
        default="us",
        help="which title screen to use: 'us' (default) swaps in the US "
        "ROM's logo and animated sword, keeping JP 1.0's own skip timing "
        "(a button press skips it immediately, as soon as the triforce "
        "forms, instead of the real US ROM's wait for the sword animation "
        "to finish); 'jp' keeps JP 1.0's own title screen instead",
    )
    parser.add_argument(
        "--usdasm", type=Path, help="US disassembly checkout (default: cached)"
    )
    parser.add_argument(
        "--jpdasm", type=Path, help="JP disassembly checkout (default: cached)"
    )
    parser.add_argument(
        "--null-padbyte-threshold",
        type=int,
        default=DEFAULT_NULL_PADBYTE_THRESHOLD,
        help="free-ROM gaps over this many bytes use a padbyte/pad jump "
        "instead of explicit db $FF rows; 0 disables padbyte entirely, "
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    target = args.target.resolve()

    # fail fast if a ROM is neither supplied nor already present
    for dest_name, attr, flag in _ROMS:
        source = getattr(args, attr)
        if source is not None and not source.is_file():
            parser.error(f"{flag} {source} not found")
        if source is None and not (target / dest_name).is_file():
            parser.error(f"{dest_name} is not in the target; pass {flag}")

    usdasm = _resolve_source("usdasm", args.usdasm, "USDASM_DIR")
    jpdasm = _resolve_source("jpdasm", args.jpdasm, "JPDASM_DIR")

    print(f"==> generating a patched jpdasm -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    _populate_from_jpdasm(jpdasm, target)
    english = build(
        usdasm=usdasm,
        jpdasm=jpdasm,
        changes=not args.baseline,
        intro_fix=not args.no_intro_fix,
        weathercock_fix=not args.no_weathercock_fix,
        keep_religious_imagery=args.keep_religious_imagery,
        epilepsy_fix=not args.no_epilepsy_fix,
        keep_jp_credits=args.keep_jp_credits,
        credits_font=args.credits_font,
        us_title_screen=args.title_screen == "us",
        yellow_counts_at_absolute_max=args.yellow_counts_at_absolute_max,
        gba_text_fixes=not args.no_gba_text_fixes,
        low_health_beep=not args.no_low_health_beep,
        flute_is_ocarina=args.flute_is_ocarina,
        fix_mothula_bugs=args.fix_mothula_bugs,
        wide_dialogue_lines=args.wide_dialogue_lines,
        null_padbyte_threshold=args.null_padbyte_threshold,
        nop_padbyte_threshold=args.nop_padbyte_threshold,
    )
    english.write(
        target,
        bank_header=bank_header,
        null_padbyte_threshold=args.null_padbyte_threshold,
    )

    print("==> deploying build tooling")
    _deploy_tooling(target)
    _update_gitignore(target)
    _place_roms(target, args)

    if args.verify:
        print("==> verifying base edits")
        if _run_verify(jpdasm, usdasm) != 0:
            return 1

    print("\nDone.\n")
    print(f"In {target} run these commands:")
    print("  python3 binextract.py")
    print("  ./_build.sh          (or: make / _build.bat)")
    return 0
