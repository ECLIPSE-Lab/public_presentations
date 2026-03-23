import os
import glob
from pathlib import Path

base_dir = "/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations"
index_file = os.path.join(base_dir, "index.qmd")

courses = {
    "Mathematical Foundations of AI & ML": "mathematical_foundations_of_ai_and_ml",
    "ML for Characterization and Processing": "ml_for_characterization_and_processing"
}

markdown_append = "\n\n# Code Examples (Materials Data Science Book)\n\n"

for course_name, course_folder in courses.items():
    markdown_append += f"## {course_name}\n\n"
    course_path = os.path.join(base_dir, course_folder)
    
    # get all units
    units = sorted([d for d in os.listdir(course_path) if os.path.isdir(os.path.join(course_path, d))])
    
    for unit in units:
        example_folder = os.path.join(course_path, unit, "example_notebooks")
        if os.path.exists(example_folder):
            notebooks = sorted(glob.glob(os.path.join(example_folder, "*.ipynb")))
            if notebooks:
                markdown_append += f"### {unit}\n"
                for nb in notebooks:
                    nb_name = os.path.basename(nb)
                    nb_html = nb_name.replace(".ipynb", ".html")
                    rel_path = f"{course_folder}/{unit}/example_notebooks/{nb_html}"
                    markdown_append += f"- [{nb_name}]({rel_path})\n"
                markdown_append += "\n"

# read existing content
with open(index_file, "r") as f:
    content = f.read()

# insert right before Additional resources, or just append
if "# Additional Resources" in content:
    content = content.replace("# Additional Resources", markdown_append + "# Additional Resources")
else:
    content += markdown_append

with open(index_file, "w") as f:
    f.write(content)

print("index.qmd updated.")
