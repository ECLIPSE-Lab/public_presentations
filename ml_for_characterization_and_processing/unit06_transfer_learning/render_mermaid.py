import re
import os
import subprocess
import json

# create puppeteer config
config = {
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
with open("puppeteer-config.json", "w") as f:
    json.dump(config, f)

qmd_file = "/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit06_transfer_learning/06_transfer_learning.qmd"
with open(qmd_file, 'r') as f:
    content = f.read()

pattern = re.compile(r'```\{mermaid\}\n(.*?)```', re.DOTALL)
matches = pattern.findall(content)

images_dir = "/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit06_transfer_learning/images"
os.makedirs(images_dir, exist_ok=True)

new_content = content

for i, match in enumerate(matches):
    mmd_path = f"diagram_{i}.mmd"
    png_path = f"{images_dir}/diagram_{i}.png"
    rel_png_path = f"images/diagram_{i}.png"
    
    with open(mmd_path, 'w') as f:
        f.write(match)
        
    print(f"Rendering {mmd_path} to {png_path}...")
    subprocess.run([
        "npx", "-y", "@mermaid-js/mermaid-cli", 
        "-i", mmd_path, 
        "-o", png_path, 
        "-t", "dark", 
        "-b", "transparent",
        "-p", "puppeteer-config.json"
    ])
    
    # We replaced it previously? Wait, my previous script did:
    # new_content.replace(old_block, new_block)
    # BUT since subprocess failed, it might have replaced the text anyway!
    # Let me check if `content` has mermaid blocks.
    
