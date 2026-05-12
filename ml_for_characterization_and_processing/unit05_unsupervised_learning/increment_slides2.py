import re

def increment_headers(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('## '):
            m = re.match(r'^## (\d+)\. (.*)', line)
            if m:
                num = int(m.group(1))
                title = m.group(2)
                if num >= 41 and 'Denoising Low-Dose TEM with SHINE (Results)' not in title:
                    line = f"## {num+1}. {title}\n"
        new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("".join(new_lines))

increment_headers('01_intro.qmd')
