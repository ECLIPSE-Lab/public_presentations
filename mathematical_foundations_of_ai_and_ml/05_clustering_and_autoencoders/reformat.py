import re

with open("01_intro.qmd", "r") as f:
    content = f.read()

# Separate yaml frontmatter
frontmatter_match = re.match(r"^---\n.*?\n---\n", content, flags=re.DOTALL)
frontmatter = frontmatter_match.group(0)
body = content[len(frontmatter):]

sections = re.split(r"(?=^## )", body, flags=re.MULTILINE)

new_body = ""

for i, sec in enumerate(sections):
    if not sec.strip():
        continue
    
    lines = sec.split('\n')
    title = lines[0]
    rest = '\n'.join(lines[1:])
    
    # Custom sections handling
    if "Computational graph intuition" in title:
        sec = f"<!-- ===== §2. Chain Rule & Computational Graphs ===== -->\n\n{title}\n\n:::: {{.columns}}\n:::: {{.column width=\"50%\"}}\n:::: {{.incremental}}\n- Any computation can be represented as a **directed acyclic graph** (DAG) of elementary operations.\n- Nodes: variables (inputs, intermediates, outputs). Edges: operations (add, multiply, activate).\n- The chain rule follows the graph structure — derivatives propagate along edges.\n::::\n::::\n:::: {{.column width=\"50%\"}}\n```{{mermaid}}\ngraph LR\n    x((x)) --> plus[+]\n    y((y)) --> plus\n    plus --> mult[*]\n    z((z)) --> mult\n    mult --> J((J))\n    style plus fill:#f9f,stroke:#333,stroke-width:2px\n    style mult fill:#f9f,stroke:#333,stroke-width:2px\n```\n::::\n::::\n"
        new_body += sec + "\n"
        continue
    elif "Backpropagation algorithm: pseudocode" in title:
        sec = f"{title}\n\n:::: {{.columns}}\n:::: {{.column width=\"60%\"}}\n:::: {{.incremental}}\n1. **Forward pass**: compute and store all \\(\\mathbf{{z}}^{{(\\ell)}}, \\mathbf{{a}}^{{(\\ell)}}\\) for \\(\\ell = 1, \\dots, L\\).\n2. **Output delta**: \\(\\boldsymbol{{\\delta}}^{{(L)}} = \\nabla_{{\\mathbf{{a}}}} L \\odot \\sigma'(\\mathbf{{z}}^{{(L)}})\\).\n3. **Backward loop**: for \\(\\ell = L-1, \\dots, 1\\):\n   - \\(\\boldsymbol{{\\delta}}^{{(\\ell)}} = \\sigma'(\\mathbf{{z}}^{{(\\ell)}}) \\odot \\big((\\mathbf{{W}}^{{(\\ell+1)}})^T \\boldsymbol{{\\delta}}^{{(\\ell+1)}}\\big)\\)\n4. **Gradient accumulation**: \\(\\nabla_{{\\mathbf{{W}}^{{(\\ell)}}}} J = \\boldsymbol{{\\delta}}^{{(\\ell)}} (\\mathbf{{a}}^{{(\\ell-1)}})^T\\), \\(\\nabla_{{\\mathbf{{b}}^{{(\\ell)}}}} J = \\boldsymbol{{\\delta}}^{{(\\ell)}}\\).\n::::\n::::\n:::: {{.column width=\"40%\"}}\n```{{mermaid}}\ngraph TD\n    subgraph ForwardPass [Forward]\n        direction TB\n        F1[Input x] --> F2[Layer 1]\n        F2 --> F3[...]\n        F3 --> F4[Layer L]\n    end\n    ForwardPass --> Loss[Loss J]\n    Loss --> BP1[Output Delta δL]\n    subgraph BackwardPass [Backward]\n        direction TB\n        BP1 --> BP2[Layer L-1 δ]\n        BP2 --> BP3[...]\n        BP3 --> BP4[Layer 1 δ]\n    end\n    BackwardPass --> Grad[Accumulate Gradients]\n    Grad --> Update[Update Weights]\n    style ForwardPass fill:#e1f5fe,stroke:#01579b\n    style BackwardPass fill:#fff3e0,stroke:#e65100\n```\n::::\n::::\n"
        new_body += sec + "\n"
        continue
    elif "Backprop vs finite differences" in title:
        sec = f"{title}\n\n:::: {{.columns}}\n:::: {{.column width=\"50%\"}}\n| Method | Forward passes | Backward passes | Total cost |\n|--------|:--------------:|:---------------:|:----------:|\n| Finite differences | \\(W + 1\\) | 0 | \\(O(W^2)\\) |\n| Backpropagation | 1 | 1 | \\(O(W)\\) |\n::::\n\n:::: {{.column width=\"50%\"}}\n:::: {{.incremental}}\n- For \\(W = 10^6\\): backprop is \\(\\sim 10^6\\times\\) faster.\n- This efficiency gap is the reason deep learning is practical [@bishop2006pattern].\n::::\n::::\n::::\n"
        new_body += sec + "\n"
        continue
    elif "Checkpoint MCQ slide" in title:
        sec = f"<!-- ===== §6. Practical Considerations ===== -->\n\n{title}\n\n**Question**: A 10-layer network uses sigmoid activations and weights initialized with \\(\\|\\mathbf{{W}}^{{(\\ell)}}\\| \\approx 1\\). What happens to the gradient at layer 1 during the backward pass?\n\n:::: {{.incremental}}\n- A) **Exploding gradient**: It grows exponentially due to weight magnitude.\n- B) **Stability**: It stays approximately constant because weights are near 1.\n- C) **Vanishing gradient**: It shrinks exponentially due to the sigmoid derivative $\\sigma'(z) \\leq 0.25$.\n- D) **Oscillation**: It oscillates unpredictably depending on the input data.\n::::\n\n:::: {{.fragment}}\n- **Answer**: **C** — Since $\\sigma'(z) \\leq 0.25$, the product of 10 such terms is at most $0.25^{10} \\approx 9.5 \\times 10^{-7}$.\n::::\n"
        new_body += sec + "\n"
        continue
    elif "Interactive: Activation Functions & Derivatives" in title:
        # We need to wrap the OJS blocks in columns
        sec = title + "\n\n- Select an activation function to see its shape and derivative. Notice the maximum value of the derivative!\n\n:::: {.columns}\n:::: {.column width=\"25%\"}\n"
        # find the first ojs block
        blocks = re.split(r"(```{ojs}.*?```)", rest, flags=re.DOTALL)
        # blocks[1] is first ojs
        sec += blocks[1] + "\n::::\n:::: {.column width=\"75%\"}\n" + blocks[3] + "\n\n" + blocks[5] + "\n::::\n::::\n"
        new_body += sec + "\n"
        continue
    elif "Interactive: Vanishing & Exploding Gradient Simulator" in title:
        sec = title + "\n\n- Explore how activation choices and initialization scale affect gradient flow in a deep (15-layer) network.\n- **Notice:** Sigmoid shrinks gradients factorially back toward layer 1. ReLU sustains them much better!\n\n:::: {.columns}\n:::: {.column width=\"25%\"}\n"
        blocks = re.split(r"(```{ojs}.*?```)", rest, flags=re.DOTALL)
        sec += blocks[1] + "\n::::\n:::: {.column width=\"75%\"}\n" + blocks[3] + "\n\n" + blocks[5] + "\n::::\n::::\n"
        new_body += sec + "\n"
        continue
    elif "References + reading assignment for next unit" in title:
        sec = f"{title}\n\n:::: {{.columns}}\n:::: {{.column width=\"50%\"}}\n**Required reading before Unit 6:**\n- Neuer: Ch. 4.5.4–4.5.5\n- McClarren: Ch. 5.2–5.3.2\n\n**Optional depth:**\n- Bishop: Ch. 5.3 (error backpropagation)\n- Goodfellow et al.: Ch. 6.5 (computational graphs, backpropagation)\n::::\n\n:::: {{.column width=\"50%\"}}\n**Next unit:**\n- **Loss Landscapes and Optimization Behavior**\n- What does the surface we are descending on actually look like?\n::::\n::::\n\n::: {{#refs}}\n:::\n"
        new_body += sec + "\n"
        continue
    elif "Example Notebook" in title:
        new_body += sec + "\n"
        continue
    elif "Interactive: Forward & Backward Pass" in title:
        sec = f"<!-- ===== §4. Interactive Forward & Backward Pass ===== -->\n\n{title}\n\n- A minimal network: 2 inputs $\\rightarrow$ 1 hidden (ReLU) $\\rightarrow$ 1 output (Linear). Target $y=1$.\n- Adjust inputs/weights to see how the Loss $J$ changes, and how gradients backpropagate!\n\n:::: {{.columns}}\n:::: {{.column width=\"30%\"}}\n"
        blocks = re.split(r"(```{ojs}.*?```)", rest, flags=re.DOTALL)
        sec += blocks[1] + "\n::::\n:::: {.column width=\"70%\"}\n" + blocks[3] + "\n\n" + blocks[5] + "\n::::\n::::\n"
        new_body += sec + "\n"
        continue
        
    # Default processing
    if "Title + Unit 5 positioning" in title:
        new_body += "<!-- ===== §1. Framing ===== -->\n\n"
    elif "Forward pass: layer-by-layer computation" in title:
        new_body += "<!-- ===== §3. Forward Pass & Delta Recursion ===== -->\n\n"
    elif "Multiple outputs and loss functions" in title:
        new_body += "<!-- ===== §5. Gradient Flow & Depth ===== -->\n\n"
    elif "Lecture-essential vs exercise content split" in title:
        new_body += "<!-- ===== §7. Summary & Next Steps ===== -->\n\n"

    # Convert simple bullet lists to incremental
    # We find contiguous blocks of lines starting with -
    lines = sec.split('\n')
    in_list = False
    
    new_sec = lines[0] + "\n"
    
    i_line = 1
    while i_line < len(lines):
        line = lines[i_line]
        
        # Check if we should ignore converting to incremental for Multivariate chain rule, The delta recursion
        # They have ::: {.fragment} already in the original file! Let's strip the ::: {.fragment} and make them incremental.
        if ":::" in line:
            # just skip existing fragment tags so we can re-wrap properly if needed, but let's just strip them
            if "fragment" in line or line.strip() == ":::":
                i_line += 1
                continue

        if line.startswith("- ") and not in_list:
            # start incremental
            new_sec += "\n:::: {.incremental}\n"
            in_list = True
            new_sec += line + "\n"
        elif line.startswith("- ") and in_list:
            new_sec += line + "\n"
        elif in_list and (line.startswith("  ") or line.startswith("$$") or line.strip() == ""):
            # inside a list, a formula or continuation
            # check if it's the end of the list
            if line.strip() == "" and i_line + 1 < len(lines) and not lines[i_line+1].startswith("- ") and not lines[i_line+1].startswith("$$") and not lines[i_line+1].startswith("  ") and not lines[i_line+1].startswith("where"):
                new_sec += "::::\n"
                in_list = False
                new_sec += line + "\n"
            else:
                new_sec += line + "\n"
        else:
            if in_list:
                # also check if the current line is a formula that belongs to the previous bullet
                if line.startswith("$$") or line.startswith("where") or line.startswith("  "):
                    new_sec += line + "\n"
                else:
                    new_sec += "::::\n\n" + line + "\n"
                    in_list = False
            else:
                new_sec += line + "\n"
        i_line += 1
        
    if in_list:
        new_sec += "::::\n"
        
    new_body += new_sec + "\n"

# Fix some spacing issues and math block enclosures
new_body = new_body.replace("::::\n\n\n::::", "::::\n")
new_body = re.sub(r"::::\n\n(\$\$[\s\S]*?\$\$)\n\n", r"\n\1\n", new_body) # formulas inside lists

with open("01_intro_styled.qmd", "w") as f:
    f.write(frontmatter + new_body)

