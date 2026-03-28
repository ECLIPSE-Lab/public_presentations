import os
import re
import yaml

base_dir = r'c:\Users\braun\Documents\public_presentations'

# Define the sections and their respective folder names
sections = [
    ("Data Science for Electron Microscopy", "data_science_for_em"),
    ("Mathematical Foundations of AI & ML", "mathematical_foundations_of_ai_and_ml"),
    ("Materials Genomics", "materials_genomics"),
    ("Machine Learning for Characterization and Processing", "ml_for_characterization_and_processing"),
    ("Conference Talks", "conference_talks")
]

def get_title_from_qmd(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
            if match:
                y = yaml.safe_load(match.group(1))
                t = y.get('title', '')
                if isinstance(t, str):
                    t = t.replace('\n', ' ')
                    t = re.sub(r'<br>', ' - ', t)
                    
                    # Clean up things like "Mathematical Foundations ... | Unit 1: ..."
                    if "Unit" in t:
                        m = re.search(r'Unit \d+:\s*(.*)', t)
                        if m:
                            return m.group(1).strip()
                    
                    return t.strip()
    except Exception as e:
        pass
    return None

def find_units(folder):
    units = []
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        return units
    
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # Sort them by their leading number
    def sort_key(x):
        m = re.search(r'^(\d+)', x)
        if m: return int(m.group(1))
        m = re.search(r'^unit(\d+)', x)
        if m: return int(m.group(1))
        return 9999
        
    subdirs.sort(key=sort_key)
    
    for d in subdirs:
        # ignore _unused_ templates etc
        if d.startswith('_') or d == 'data' or d == 'images' or d == 'assets' or d == 'example_notebooks':
            continue
            
        d_path = os.path.join(folder_path, d)
        qmd_files = [f for f in os.listdir(d_path) if f.endswith('.qmd')]
        
        target_qmd = None
        if 'template.qmd' in qmd_files:
            target_qmd = 'template.qmd'
        elif '01_intro.qmd' in qmd_files:
            target_qmd = '01_intro.qmd'
        elif len(qmd_files) > 0:
            target_qmd = qmd_files[0]
            
        if target_qmd:
            qmd_path = os.path.join(d_path, target_qmd)
            title = get_title_from_qmd(qmd_path)
            
            # Determine badge
            badge = "Talk"
            num_match = re.search(r'^(\d+)', d) or re.search(r'^unit(\d+)', d)
            if num_match:
                badge = f"Unit {int(num_match.group(1)):02d}"
                
            if not title:
                # fallback
                title = d.replace('_', ' ').title()
                
            # If there's a docs folder with template.html, use that (like data science for EM)
            html_link = f"{folder}/{d}/{target_qmd.replace('.qmd', '.html')}"
            if os.path.exists(os.path.join(d_path, 'docs', target_qmd.replace('.qmd', '.html'))):
                html_link = f"{folder}/{d}/docs/{target_qmd.replace('.qmd', '.html')}"
                
            source_link = f"https://github.com/ECLIPSE-Lab/public_presentations/blob/main/{folder}/{d}/{target_qmd}"
            
            units.append({
                'badge': badge,
                'title': title,
                'slides': html_link,
                'source': source_link
            })
            
    return units

out_path = os.path.join(base_dir, 'index.qmd')
output = []
output.append("""---
title: "ECLIPSE Lab — Presentations & Teaching"
description: "Lecture slides, conference talks, and course materials from the ECLIPSE Lab at FAU Erlangen-Nürnberg."
format: html
---

```{=html}
<style>
  .site-hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(139,92,246,0.14));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin: 2rem 0 3rem 0;
    text-align: left;
  }
  .site-hero h1 { color: #f8fafc; font-family: Outfit, sans-serif; font-weight: 800; margin-bottom: 0.5rem; border: none; }
  .site-hero p { color: #94a3b8; font-size: 1.1rem; margin: 0; }
  
  .lecture-container {
    background: rgba(20,25,40,0.65);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    margin-bottom: 2.5rem;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }
  .unit-list {
    display: flex;
    flex-direction: column;
  }
  .unit-item {
    display: flex;
    align-items: center;
    padding: 1rem 2.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.2s;
  }
  .unit-item:last-child {
    border-bottom: none;
  }
  .unit-item:hover {
    background: rgba(255,255,255,0.04);
  }
  .unit-badge {
    background: rgba(59,130,246,0.15);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-size: 0.85rem;
    font-family: monospace;
    font-weight: 600;
    margin-right: 1.5rem;
    min-width: 70px;
    text-align: center;
  }
  .unit-name {
    flex: 1;
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 500;
  }
  .unit-actions {
    display: flex;
    gap: 0.75rem;
  }
  .action-btn {
    text-decoration: none !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .action-slides {
    background: rgba(139,92,246,0.15);
    color: #c4b5fd;
    border: 1px solid rgba(139,92,246,0.3);
  }
  .action-slides:hover {
    background: rgba(139,92,246,0.3);
    color: #fff;
    transform: translateY(-1px);
  }
  .action-source {
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .action-source:hover {
    background: rgba(255,255,255,0.15);
    color: #fff;
    transform: translateY(-1px);
  }
  .back-lab {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
  }
  .back-lab a { color: #94a3b8; text-decoration: none; font-size: 0.95rem; display: inline-flex; align-items: center; gap: 0.5rem; }
  .back-lab a:hover { color: #93c5fd; }
  
  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .site-hero { padding: 2rem; }
    .unit-item {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.8rem;
    }
    .unit-badge { margin-right: 0; }
    .unit-actions { margin-top: 0.5rem; width: 100%; justify-content: flex-start; }
  }
</style>
```

:::{.site-hero}
# Presentations & Teaching Materials

Lecture slides, course decks, and conference presentations from the computational materials microscopy group at FAU Erlangen-Nürnberg.
:::
""")

import markdown

for sec_title, folder in sections:
    units = find_units(folder)
    if not units:
        continue
        
    output.append(f'\n## {sec_title}\n')
    output.append('```{=html}\n<div class="lecture-container">\n  <div class="unit-list">')
    
    for u in units:
        b = u['badge']
        t = u['title']
        sl = u['slides']
        so = u['source']
        output.append(f'''    <div class="unit-item">
      <div class="unit-badge">{b}</div>
      <div class="unit-name">{t}</div>
      <div class="unit-actions">
        <a href="{sl}" class="action-btn action-slides">Open Slides</a>
        <a href="{so}" class="action-btn action-source" target="_blank">Source</a>
      </div>
    </div>''')
    
    output.append("  </div>\n</div>\n```")

output.append("""
:::{.back-lab}
[← Back to ECLIPSE Lab](https://pelzlab.science)
:::
""")

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Successfully rebuilt index.qmd from actual directories!")
