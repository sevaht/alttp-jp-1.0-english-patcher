#!/usr/bin/env python3
"""Regression guard for the generated base banks.

Builds the whole English program from pristine jpdasm + usdasm checkouts and
compares each hooked base bank's *assembler-relevant signature* (label|opcode|
args per content line -- comments and blank lines ignored, since they do not
affect assembly) against a frozen hash in ``reference_hashes.txt``. A mismatch
means either the edits changed or an upstream bank drifted; a clean run means
the generated base edits still reproduce the reviewed result.

    python3 verify_base.py --src /path/to/jpdasm --usdasm /path/to/usdasm
    python3 verify_base.py --src ... --usdasm ... --freeze  # rewrite reference
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from importlib import resources
from pathlib import Path

from .generate import build
from .snes_assembly_parser import Assembly

# The frozen reference is an embedded package resource: read via
# importlib.resources; --freeze writes back to the source file (only meaningful
# in an editable/source checkout, which is where you would re-freeze).
_REFERENCE = ("resources", "reference_hashes.txt")


def _reference_text() -> str:
    return (
        resources.files("alttp_jp_english_patcher")
        .joinpath(*_REFERENCE)
        .read_text(encoding="utf-8")
    )


def _reference_source_path() -> Path:
    return Path(__file__).resolve().parent.joinpath(*_REFERENCE)


# The base banks the graft hooks (the only base banks it changes).
BASE_BANKS = (
    "bank_00",
    "bank_02",
    "bank_09",
    "bank_0C",
    "bank_0D",
    "bank_0E",
    "bank_13",
    "bank_18",
    "bank_1C",
    "bank_1D",
)


def signature_hash(path: Path) -> str:
    """SHA-256 of a bank's content-line signature (comments/blanks ignored)."""
    lines = [
        f"{line.label or ''}|{line.opcode or ''}|{','.join(line.arguments)}"
        for line in Assembly.from_path(path).lines
        if line.has_content
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def compute(jpdasm: Path, usdasm: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        english = build(usdasm=usdasm, jpdasm=jpdasm, changes=True)
        english.write(Path(temp_dir))
        return {
            bank: signature_hash(Path(temp_dir) / f"{bank}.asm")
            for bank in BASE_BANKS
        }


def load_reference() -> dict[str, str]:
    reference: dict[str, str] = {}
    for raw in _reference_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        bank, digest = line.split(":", 1)
        reference[bank.strip()] = digest.strip()
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True, help="pristine jpdasm checkout"
    )
    parser.add_argument(
        "--usdasm", type=Path, required=True, help="pristine usdasm checkout"
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="write current hashes to reference_hashes.txt, don't check",
    )
    args = parser.parse_args()
    for label, path in (("jpdasm", args.src), ("usdasm", args.usdasm)):
        if not path.exists():
            print(f"error: pristine {label} {path} not found", file=sys.stderr)
            return 2

    got = compute(args.src, args.usdasm)
    if args.freeze:
        body = "".join(f"{bank}: {got[bank]}\n" for bank in sorted(got))
        destination = _reference_source_path()
        destination.write_text(
            "# base-bank hook signature hashes (see verify_base.py)\n" + body
        )
        print(f"froze {len(got)} hashes to {destination.name}")
        return 0

    reference = load_reference()
    mismatched = []
    for bank in sorted(got):
        if got[bank] == reference.get(bank):
            print(f"  {bank}: match")
        else:
            print(f"  {bank}: MISMATCH")
            mismatched.append(bank)
    if mismatched:
        print(f"FAIL: {', '.join(mismatched)}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
