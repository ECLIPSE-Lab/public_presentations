import sys

file_path = "01_intro.qmd"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace all occurrences of \$ with single $
text = text.replace(r"\$", "$")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done replacing \$ with $ in 01_intro.qmd.")
