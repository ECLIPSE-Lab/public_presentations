import glob
import re

units = sorted(glob.glob("*/01_intro.qmd"))

for u in units:
    if not any(u.startswith(f"{i:02d}_") for i in range(6, 15)):
        continue
    
    with open(u, "r") as f:
        content = f.read()

    if "<!-- ===== §" in content:
        print(f"Sections already in {u}")
        continue
        
    lines = content.split("\n")
    new_lines = []
    
    sec_num = 1
    
    # We will add §1. Framing before the first ## 
    # And then increment every 3-4 headers or based on heuristics
    
    # Let's just group every 4 headers into a section for simplicity,
    # or just add sequential section comments
    
    header_count = 0
    in_code_block = False
    
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
            
        if line.startswith("## ") and not in_code_block:
            title = line[3:].strip()
            if header_count == 0:
                new_lines.append(f"\n<!-- ===== §1. Framing ===== -->\n")
            elif header_count % 4 == 0:
                sec_num += 1
                new_lines.append(f"\n<!-- ===== §{sec_num}. {title} ===== -->\n")
            header_count += 1
            new_lines.append(line)
        else:
            new_lines.append(line)

    with open(u, "w") as f:
        f.write("\n".join(new_lines))
        
    print(f"Added sections to {u}")
