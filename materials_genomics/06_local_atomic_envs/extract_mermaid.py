import re
import os
import subprocess

with open('01_intro.qmd', 'r') as f:
    content = f.read()

# Pattern for mermaid blocks
pattern = re.compile(r'```\{mermaid\}\n(.*?)```', re.DOTALL)
matches = pattern.findall(content)

for i, match in enumerate(matches):
    mmd_path = f'diagram_{i}.mmd'
    # Remove Quarto directives like %%| echo: false
    cleaned_match = '\n'.join([line for line in match.split('\n') if not line.startswith('%%|')])
    
    with open(mmd_path, 'w') as f:
        f.write(cleaned_match)
    
    print(f"Extracted {mmd_path}")
