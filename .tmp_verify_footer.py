from pathlib import Path
import re

root = Path(r"C:/Users/braun/Documents/public_presentations/mathematical_foundations_of_ai_and_ml")
files = sorted(root.rglob("*.qmd"))
expected = '"\N{COPYRIGHT SIGN} Philipp Pelz - Mathematical Foundations of AI & ML"'

ok = []
bad = []
for p in files:
    t = p.read_text(encoding="utf-8")
    if not t.startswith("---\n"):
        bad.append((p, "missing front matter"))
        continue
    end = t.find("\n---\n", 4)
    if end == -1:
        bad.append((p, "unterminated front matter"))
        continue
    fm = t[: end + 5]
    m = re.search(r'^\s*revealjs:\s*$', fm, re.M)
    if not m:
        bad.append((p, "missing revealjs"))
        continue
    if expected in fm:
        ok.append(p)
    else:
        bad.append((p, "footer mismatch or missing"))

print("ok", len(ok))
print("bad", len(bad))
for p, reason in bad:
    print(reason, p)
