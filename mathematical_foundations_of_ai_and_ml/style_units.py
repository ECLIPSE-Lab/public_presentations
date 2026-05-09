import os
import glob
import re

units = sorted(glob.glob("*/01_intro.qmd"))
for u in units:
    if not any(u.startswith(f"{i:02d}_") for i in range(6, 15)):
        continue
    
    with open(u, "r") as f:
        content = f.read()

    # Fix Mermaid
    def fix_mermaid(match):
        block = match.group(0)
        if "%%| echo: false" not in block:
            block = block.replace("```{mermaid}", "```{mermaid}\n%%| echo: false\n%%| fig-align: center")
        
        def quote_bracket(m):
            node_id = m.group(1)
            inner = m.group(2)
            if inner.startswith('"') and inner.endswith('"'):
                return m.group(0)
            return f'{node_id}["{inner}"]'
            
        def quote_circle(m):
            node_id = m.group(1)
            inner = m.group(2)
            if inner.startswith('"') and inner.endswith('"'):
                return m.group(0)
            return f'{node_id}(("{inner}"))'
            
        block = re.sub(r'(\w+)\[([^\]]+)\]', quote_bracket, block)
        block = re.sub(r'(\w+)\(\(([^)]+)\)\)', quote_circle, block)
        return block

    content = re.sub(r'```\{mermaid\}.*?```', fix_mermaid, content, flags=re.DOTALL)

    # Convert ::: {.fragment} around lists to :::: {.incremental}
    # A block starting with ::: {.fragment} and containing primarily bullet points
    # ending with :::
    def fix_fragments(match):
        inner = match.group(1)
        # Check if it's mostly a list
        if bool(re.search(r'^\s*-\s+', inner, flags=re.MULTILINE)):
            return f":::: {{.incremental}}\n{inner}\n::::"
        else:
            return match.group(0)
            
    content = re.sub(r':::\s*\{\.fragment\}\n(.*?)\n:::', fix_fragments, content, flags=re.DOTALL)

    # Also, some lists are not wrapped in anything.
    # Let's wrap contiguous bullet points in :::: {.incremental} if they aren't already
    
    with open(u, "w") as f:
        f.write(content)
        
    print(f"Processed {u}")
