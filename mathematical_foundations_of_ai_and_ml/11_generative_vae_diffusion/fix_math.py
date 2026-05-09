import sys
import re

file_path = "01_intro.qmd"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace \\( with $
text = text.replace("\\\\(", "$")
# Replace \\) with $
text = text.replace("\\\\)", "$")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done replacing \\\\( and \\\\) with $ in 01_intro.qmd.")
