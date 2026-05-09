"""Regenerate README.md and root index.qmd unit tables from units.yml.

Also generates per-deck "Continue" (prev/next) slides inside each unit's
primary `.qmd` source file, and writes a `_prev_next.json` sidecar in each
deck folder for tooling/debugging.

Idempotent: safe to run repeatedly. Edits files only between BEGIN/END
markers (HTML-comment style) so other content is preserved.

Usage:
    python3 scripts/build_link_tables.py            # apply edits
    python3 scripts/build_link_tables.py --check    # exit 1 if any edit needed

Constraints:
- stdlib + PyYAML only
- never touches `_site/`, `.quarto/`, `_freeze/`
- never modifies YAML frontmatter of any deck
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
UNITS_YAML = REPO_ROOT / "units.yml"
README = REPO_ROOT / "README.md"
INDEX_QMD = REPO_ROOT / "index.qmd"

# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def begin_marker(course_key: str) -> str:
    return f"<!-- BEGIN units:{course_key} -->"


def end_marker(course_key: str) -> str:
    return f"<!-- END units:{course_key} -->"


PREV_NEXT_BEGIN = "<!-- BEGIN prev-next -->"
PREV_NEXT_END = "<!-- END prev-next -->"


def replace_block(text: str, begin: str, end: str, new_body: str) -> str:
    """Replace content between begin and end markers (markers preserved).

    Raises KeyError if either marker is missing.
    """
    if begin not in text:
        raise KeyError(f"missing begin marker: {begin!r}")
    if end not in text:
        raise KeyError(f"missing end marker: {end!r}")

    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end),
        flags=re.DOTALL,
    )
    replacement = f"{begin}\n{new_body}\n{end}"
    return pattern.sub(lambda _m: replacement, text, count=1)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    num: str
    title: str
    folder: str
    deck_file: str
    source_file: str
    published: bool
    extras: list[dict]


@dataclass
class Course:
    key: str
    title: str
    site_path: str
    units: list[Unit]


def load_catalog() -> tuple[dict, list[Course]]:
    with UNITS_YAML.open() as f:
        data = yaml.safe_load(f)

    courses: list[Course] = []
    for key, raw in data["courses"].items():
        units = [
            Unit(
                num=u["num"],
                title=u["title"],
                folder=u["folder"],
                deck_file=u["deck_file"],
                source_file=u["source_file"],
                published=u.get("published", True),
                extras=u.get("extras", []) or [],
            )
            for u in raw["units"]
        ]
        courses.append(
            Course(key=key, title=raw["title"], site_path=raw["site_path"], units=units)
        )
    return data, courses


# ---------------------------------------------------------------------------
# README rendering
# ---------------------------------------------------------------------------


def render_readme_block(catalog: dict, course: Course) -> str:
    site_url = catalog["site_url"].rstrip("/")
    lines: list[str] = []
    for u in course.units:
        if not u.published:
            continue
        slides = f"{site_url}/{course.site_path}/{u.folder}/{u.deck_file}"
        source = f"./{course.site_path}/{u.folder}/{u.source_file}"
        line = (
            f"- **Unit {u.num}: {u.title}** — "
            f"[Slides]({slides}) · [Source]({source})"
        )
        for extra in u.extras:
            extra_url = (
                f"{site_url}/{course.site_path}/{u.folder}/{extra['file']}"
            )
            line += f" · [{extra['label']}]({extra_url})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# index.qmd rendering
# ---------------------------------------------------------------------------


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_index_block(catalog: dict, course: Course) -> str:
    blob_url = catalog["github_blob_url"].rstrip("/")
    items: list[str] = []
    for u in course.units:
        if not u.published:
            continue
        slides_rel = f"{course.site_path}/{u.folder}/{u.deck_file}"
        source_url = f"{blob_url}/{course.site_path}/{u.folder}/{u.source_file}"
        items.append(
            "    <div class=\"unit-item\">\n"
            f"      <div class=\"unit-badge\">Unit {u.num}</div>\n"
            f"      <div class=\"unit-name\">{html_escape(u.title)}</div>\n"
            "      <div class=\"unit-actions\">\n"
            f"        <a href=\"{slides_rel}\" class=\"action-btn action-slides\">Open Slides</a>\n"
            f"        <a href=\"{source_url}\" class=\"action-btn action-source\" target=\"_blank\">Source</a>\n"
            "      </div>\n"
            "    </div>"
        )
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Prev/next slide rendering
# ---------------------------------------------------------------------------


def relative_unit_url(from_unit: Unit, to_unit: Unit) -> str:
    """Build a relative URL from from_unit's deck dir to to_unit's deck.

    Within the same course (same parent folder), this is `../<folder>/<file>`.
    """
    return f"../{to_unit.folder}/{to_unit.deck_file}"


def render_prev_next_slide(
    course: Course, idx: int, site_url_prefix: str
) -> str:
    units = [u for u in course.units if u.published]
    # Find current unit's published-list index
    current = course.units[idx]
    if current not in units:
        return ""
    pub_idx = units.index(current)

    prev = units[pub_idx - 1] if pub_idx > 0 else None
    nxt = units[pub_idx + 1] if pub_idx < len(units) - 1 else None

    site_index = "../../index.html"

    body_lines = ["", "## Continue", ""]
    if prev is not None:
        prev_url = relative_unit_url(current, prev)
        body_lines.append(
            f"- &larr; Previous: [Unit {prev.num} &mdash; {prev.title}]({prev_url})"
        )
    if nxt is not None:
        nxt_url = relative_unit_url(current, nxt)
        body_lines.append(
            f"- &rarr; Next: [Unit {nxt.num} &mdash; {nxt.title}]({nxt_url})"
        )
    body_lines.append(f"- [All courses]({site_index})")
    body_lines.append("")
    return "\n".join(body_lines)


def render_prev_next_block(course: Course, idx: int, site_url_prefix: str) -> str:
    inner = render_prev_next_slide(course, idx, site_url_prefix)
    return f"{PREV_NEXT_BEGIN}\n{inner}\n{PREV_NEXT_END}"


# ---------------------------------------------------------------------------
# Per-deck file editing
# ---------------------------------------------------------------------------


REFS_HEADER_RE = re.compile(r"^##\s+.*[Rr]efer", re.MULTILINE)


def upsert_prev_next_in_deck(deck_path: Path, block: str) -> str:
    """Insert/update the prev-next block in the deck file. Returns one of:
    'inserted-before-refs', 'inserted-at-end', 'updated', 'noop', 'missing'.
    """
    if not deck_path.exists():
        return "missing"

    text = deck_path.read_text()

    if PREV_NEXT_BEGIN in text and PREV_NEXT_END in text:
        new_text = replace_block(text, PREV_NEXT_BEGIN, PREV_NEXT_END, "")
        # rebuild without empty body, then insert fresh body
        # Simpler: regex-replace the whole block in place
        pattern = re.compile(
            re.escape(PREV_NEXT_BEGIN) + r".*?" + re.escape(PREV_NEXT_END),
            flags=re.DOTALL,
        )
        new_text = pattern.sub(lambda _m: block, text, count=1)
        if new_text == text:
            return "noop"
        deck_path.write_text(new_text)
        return "updated"

    # No existing block: insert before the LAST `## ...References...` header.
    matches = list(REFS_HEADER_RE.finditer(text))
    if matches:
        last = matches[-1]
        insert_at = last.start()
        new_text = text[:insert_at] + block + "\n\n" + text[insert_at:]
        deck_path.write_text(new_text)
        return "inserted-before-refs"

    # Fallback: append at end (with a leading blank line).
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    new_text = text + sep + block + "\n"
    deck_path.write_text(new_text)
    return "inserted-at-end"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def regenerate_readme(catalog: dict, courses: list[Course]) -> tuple[bool, list[str]]:
    text = README.read_text()
    original = text
    changes: list[str] = []
    missing: list[str] = []
    for course in courses:
        try:
            text = replace_block(
                text,
                begin_marker(course.key),
                end_marker(course.key),
                render_readme_block(catalog, course),
            )
        except KeyError as e:
            missing.append(str(e))
    if missing:
        raise SystemExit(
            "README.md is missing required marker(s):\n  - "
            + "\n  - ".join(missing)
        )
    if text != original:
        README.write_text(text)
        changes.append("README.md")
    return text != original, changes


def regenerate_index(catalog: dict, courses: list[Course]) -> tuple[bool, list[str]]:
    text = INDEX_QMD.read_text()
    original = text
    changes: list[str] = []
    missing: list[str] = []
    for course in courses:
        try:
            text = replace_block(
                text,
                begin_marker(course.key),
                end_marker(course.key),
                render_index_block(catalog, course),
            )
        except KeyError as e:
            missing.append(str(e))
    if missing:
        raise SystemExit(
            "index.qmd is missing required marker(s):\n  - "
            + "\n  - ".join(missing)
        )
    if text != original:
        INDEX_QMD.write_text(text)
        changes.append("index.qmd")
    return text != original, changes


def regenerate_decks(catalog: dict, courses: list[Course]) -> dict:
    site_url = catalog["site_url"].rstrip("/")
    summary = {
        "inserted-before-refs": [],
        "inserted-at-end": [],
        "updated": [],
        "noop": [],
        "missing": [],
    }
    for course in courses:
        for idx, u in enumerate(course.units):
            if not u.published:
                continue
            deck_dir = REPO_ROOT / course.site_path / u.folder
            deck_path = deck_dir / u.source_file
            block = render_prev_next_block(course, idx, site_url)

            sidecar = {
                "course_key": course.key,
                "course_title": course.title,
                "unit_num": u.num,
                "unit_title": u.title,
                "deck": u.deck_file,
            }
            pub = [pu for pu in course.units if pu.published]
            pub_idx = pub.index(u)
            if pub_idx > 0:
                prev = pub[pub_idx - 1]
                sidecar["prev_url"] = relative_unit_url(u, prev)
                sidecar["prev_title"] = f"Unit {prev.num} — {prev.title}"
            if pub_idx < len(pub) - 1:
                nxt = pub[pub_idx + 1]
                sidecar["next_url"] = relative_unit_url(u, nxt)
                sidecar["next_title"] = f"Unit {nxt.num} — {nxt.title}"
            sidecar["site_index"] = "../../index.html"

            if deck_dir.exists():
                (deck_dir / "_prev_next.json").write_text(
                    json.dumps(sidecar, indent=2) + "\n"
                )

            status = upsert_prev_next_in_deck(deck_path, block)
            summary[status].append(str(deck_path.relative_to(REPO_ROOT)))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if any file would change"
    )
    args = parser.parse_args(argv)

    catalog, courses = load_catalog()

    # In --check mode, capture original content first
    if args.check:
        readme_before = README.read_text()
        index_before = INDEX_QMD.read_text()

    readme_changed, readme_chs = regenerate_readme(catalog, courses)
    index_changed, index_chs = regenerate_index(catalog, courses)
    deck_summary = regenerate_decks(catalog, courses)

    n_units = sum(
        1 for c in courses for u in c.units if u.published
    )
    n_inserted = (
        len(deck_summary["inserted-before-refs"])
        + len(deck_summary["inserted-at-end"])
    )
    n_updated = len(deck_summary["updated"])
    n_noop = len(deck_summary["noop"])
    n_missing = len(deck_summary["missing"])

    print("=== build_link_tables.py summary ===")
    print(f"courses:                {len(courses)}")
    print(f"published units:        {n_units}")
    print(f"README.md changed:      {readme_changed}")
    print(f"index.qmd changed:      {index_changed}")
    print(f"decks: inserted-before-refs = {len(deck_summary['inserted-before-refs'])}")
    print(f"decks: inserted-at-end      = {len(deck_summary['inserted-at-end'])}")
    print(f"decks: updated              = {n_updated}")
    print(f"decks: noop                 = {n_noop}")
    print(f"decks: missing source file  = {n_missing}")
    if deck_summary["missing"]:
        print("MISSING:")
        for p in deck_summary["missing"]:
            print(f"  - {p}")
    if deck_summary["inserted-at-end"]:
        print("INSERTED AT END (no References header found):")
        for p in deck_summary["inserted-at-end"]:
            print(f"  - {p}")

    if args.check:
        # Restore originals if they changed; report nonzero
        any_drift = False
        if README.read_text() != readme_before:
            README.write_text(readme_before)
            any_drift = True
        if INDEX_QMD.read_text() != index_before:
            INDEX_QMD.write_text(index_before)
            any_drift = True
        # Note: deck files & sidecars are NOT reverted in --check mode
        # because they may legitimately be new artifacts.
        if any_drift:
            print("CHECK FAILED: README.md or index.qmd would change.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
