import re

def increment_slide_numbers(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace `## X. ` with `## X+1. `
    # But only for X >= 33 (since we are making a new 33)
    def replacer(match):
        num = int(match.group(1))
        if num >= 33:
            return f"## {num+1}. "
        return match.group(0)

    # Note: we should also check if the user referenced these slides by number in the text!
    # Let's see if the user referenced "slide 33", "slide 43", etc. in the notes.
    # The text usually has "slide X" or "slide X+".
    # Let's do a regex for that as well.
    def slide_ref_replacer(match):
        prefix = match.group(1)
        num = int(match.group(2))
        suffix = match.group(3)
        if num >= 33:
            return f"{prefix}{num+1}{suffix}"
        return match.group(0)

    # Replace headings
    new_content = re.sub(r'## (\d+)\. ', replacer, content)

    # Replace references in text (e.g. "slide 33", "slides 36–43", "slide 43)")
    new_content = re.sub(r'(slide[s]?\s+)(\d+)([^0-9])', slide_ref_replacer, new_content, flags=re.IGNORECASE)
    # also handle things like "slides 36-43"
    # let's do a second pass for the second number in a range
    new_content = re.sub(r'(slide[s]?\s+\d+(?:–|-| and ))(\d+)([^0-9])', slide_ref_replacer, new_content, flags=re.IGNORECASE)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

increment_slide_numbers('01_intro.qmd')
