from pathlib import Path
import re

root = Path(r"C:/Users/braun/Documents/public_presentations/mathematical_foundations_of_ai_and_ml")
files = sorted(root.rglob("*.qmd"))
footer_value = '"\N{COPYRIGHT SIGN} Philipp Pelz - Mathematical Foundations of AI & ML"'
updated = []

for path in files:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        continue
    end = text.find("\n---\n", 4)
    if end == -1:
        continue

    fm = text[:end + 5]
    body = text[end + 5:]
    lines = fm.splitlines(True)

    reveal_idx = None
    reveal_indent_str = ""
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)revealjs:\s*\n$', line)
        if m:
            reveal_idx = i
            reveal_indent_str = m.group(1)
            break
    if reveal_idx is None:
        continue

    reveal_indent = len(reveal_indent_str)
    j = reveal_idx + 1
    while j < len(lines):
        s = lines[j]
        if s.strip() == "":
            j += 1
            continue
        leading = len(s) - len(s.lstrip(" "))
        if leading <= reveal_indent:
            break
        j += 1

    block = lines[reveal_idx + 1:j]
    child_indents = []
    for bl in block:
        if bl.strip() == "" or bl.lstrip().startswith("#"):
            continue
        m = re.match(r'^(\s*)([A-Za-z0-9_-]+):', bl)
        if m:
            ind = len(m.group(1))
            if ind > reveal_indent:
                child_indents.append(ind)
    child_indent = min(child_indents) if child_indents else reveal_indent + 2
    child = " " * child_indent

    new_block = []
    for bl in block:
        m = re.match(r'^(\s*)footer:\s*.*$', bl)
        if m and len(m.group(1)) == child_indent:
            continue
        new_block.append(bl)

    new_block.append(f"{child}footer: {footer_value}\n")

    new_fm = ''.join(lines[:reveal_idx + 1] + new_block + lines[j:])
    if new_fm != fm:
        path.write_text(new_fm + body, encoding="utf-8")
        updated.append(path)

print(f"updated {len(updated)} files")
for p in updated:
    print(p)
