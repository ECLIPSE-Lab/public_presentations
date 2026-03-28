import sys

file_path = "01_intro.qmd"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_inc = False

for line in lines:
    stripped = line.strip()
    
    # Check if this line is a list item ending with {.fragment}
    if stripped.startswith("- ") and " {.fragment}" in stripped:
        if not in_inc:
            new_lines.append("::: {.incremental}\n")
            in_inc = True
        
        # Remove the {.fragment} from this line specifically only from the end
        new_line = line.replace(" {.fragment}", "")
        new_lines.append(new_line)
    
    else:
        # If we are in an incremental block, should we close it?
        # A block of list items can contain empty lines or indented text.
        # We close it if we see a line that is NOT an empty line, NOT a list item, NOT indented text.
        # Or if it's a heading.
        if in_inc:
            if stripped == "":
                # Could be a space between list items. We'll append it.
                new_lines.append(line)
            elif stripped.startswith("- ") and not " {.fragment}" in stripped:
                # A regular list item. Actually, if a list suddenly has no fragment, we either
                # add it to the incremental block anyway (which animates it), or we close the block.
                # Let's close the block if it doesn't have a {.fragment}.
                if len(new_lines) > 0 and new_lines[-1].strip() == "":
                    # The empty line should be outside the block
                    new_lines.insert(-1, ":::\n")
                else:
                    new_lines.append(":::\n")
                in_inc = False
                new_lines.append(line)
            elif stripped.startswith("<!--") or stripped.startswith(":::"):
                # If we encounter another ::: block or a comment
                if len(new_lines) > 0 and new_lines[-1].strip() == "":
                    new_lines.insert(-1, ":::\n")
                else:
                    new_lines.append(":::\n")
                in_inc = False
                new_lines.append(line)
            elif not stripped.startswith("-") and not stripped.startswith(" "):
                # Any other regular text or heading closes the list.
                if len(new_lines) > 0 and new_lines[-1].strip() == "":
                    new_lines.insert(-1, ":::\n")
                else:
                    new_lines.append(":::\n")
                in_inc = False
                new_lines.append(line)
            else:
                # Indented text or something belonging to the list
                new_lines.append(line)
        else:
            new_lines.append(line)

# Handle EOF
if in_inc:
    if len(new_lines) > 0 and new_lines[-1].strip() == "":
        new_lines.insert(-1, ":::\n")
    else:
        new_lines.append(":::\n")

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Done converting {.fragment} to ::: {.incremental}.")
