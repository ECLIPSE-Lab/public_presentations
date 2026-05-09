import os
import glob
import re

units = sorted(glob.glob("*/01_intro.qmd"))
for u in units:
    if not any(u.startswith(f"{i:02d}_") for i in range(6, 15)):
        continue
    
    with open(u, "r") as f:
        lines = f.readlines()

    new_lines = []
    stack = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("::: {"):
            if "{.columns}" in stripped or "{.column" in stripped or "{.incremental}" in stripped:
                line = line.replace(":::", "::::", 1)
                stack.append("::::")
            else:
                stack.append(":::")
            new_lines.append(line)
        elif stripped == ":::":
            if stack:
                tag = stack.pop()
                if tag == "::::":
                    new_lines.append(line.replace(":::", "::::", 1))
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    content = "".join(new_lines)
    with open(u, "w") as f:
        f.write(content)
        
    print(f"Fixed columns in {u}")
