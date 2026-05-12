with open('01_intro.qmd', 'r') as f:
    text = f.read()

# I will just write a list of (old, new) based on the lines I found via grep.
replacements = [
    ("visualisation of an AE bottleneck (slide 45)", "visualisation of an AE bottleneck (slide 46)"),
    ("we'll pick it back up in slide 44", "we'll pick it back up in slide 46"),
    ("Compression (slide 41)", "Compression (slide 43)"),
    ("revisited in slide 32", "revisited in slide 34"),
    ("Bridge to slide 31", "Bridge to slide 32"),
    ("aloud at slide 34 and again at slide 54", "aloud at slide 35 and again at slide 57"),
    ("The math (slide 35) and the architecture (slide 36)", "The math (slide 36) and the architecture (slide 37)"),
    ("slides 37–44 are applications", "slides 38–46 are applications"),
    ("denoising (slide 39)", "denoising (slide 40)"),
    ("compression (slide 41)", "compression (slide 43)"),
    ("feature extraction (slide 42)", "feature extraction (slide 44)"),
    ("anomaly detection (slide 46)", "anomaly detection (slide 47)"),
    ("Hold this thought for slide 45", "Hold this thought for slide 46"),
    ("Structured noise, see slide 39", "Structured noise, see slide 40"),
    ("predict properties (slide 42)", "predict properties (slide 44)"),
    ("data ($z$, slide 44)", "data ($z$, slide 46)"),
    ("clean data (slide 39)", "clean data (slide 40)"),
    ("frame-level scoring (slide 46)", "frame-level scoring (slide 48)"),
    ("Recap of slides 51–52", "Recap of slides 54–55"),
    ("AE features (slide 42)", "AE features (slide 44)"),
    ("I said it at slide 02 and slide 34", "I said it at slide 02 and slide 35"),
    ("I say it again at slide 54", "I say it again at slide 57"),
]

for old, new in replacements:
    text = text.replace(old, new)

with open('01_intro.qmd', 'w') as f:
    f.write(text)
