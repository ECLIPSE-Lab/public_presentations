import sys

file_path = '/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit02_physics_of_data/01_intro.qmd'

with open(file_path, 'r') as f:
    text = f.read()

parts = text.split('\n## ')

notes = {
    "01. From Physical Sensing to ML": """::: {.notes}
**Pedagogical Goal:** Hook the students by framing the "Data Generation Process" as the crucial bridge between abstract physics and ML engineering.
- **Key Message:** Data is not a given; it's a consequence of a physical interaction. If we ignore how data is formed, our ML models will learn artifacts and noise instead of physics.
- **Transition:** Ask the students, "If you want to probe a material, what can you actually hit it with?"
:::""",
    "02. The Five Fundamental Probes": """::: {.notes}
**Pedagogical Goal:** Give a bird's-eye view of materials characterization.
- **Walkthrough:** Briefly contrast the probes. Electrons for high-res surface structure, photons for chemistry and bulk, ions for milling/mass, neutrons for deep magnetic/isotopic probing. Physical forces for surface topography.
- **ML Context:** Emphasize that the choice of probe completely restricts the *dimensionality* and *format* of the resulting data, which in turn dictates the ML architecture we'll need (e.g., 2D CNNs vs 1D RNNs/1D CNNs).
:::""",
    "03. Electron Interactions": """::: {.notes}
**Pedagogical Goal:** Deep dive into the most common high-res probe (electrons).
- **Core Physics:** Electrons are charged, so they interact strongly. High interaction means we get lots of signal from a very thin slice of the sample.
- **Data Implications:** Differentiate between elastic (diffraction/images) and inelastic (spectra). Mention that 4D-STEM will be a recurring example in this course because it produces massive, complex datasets ideal for ML.
:::""",
    "04. Photon Interactions (Vis & X-Ray)": """::: {.notes}
**Pedagogical Goal:** Contrast photons with electrons.
- **Core Physics:** Uncharged and massless. They interact with the electron cloud differently, penetrating much deeper.
- **Data Implications:** You generally don't get direct real-space sub-nm images easily because we lack "magnetic lenses" for X-rays. Instead, the data is usually 1D spectra or macroscopic averages. 
- **Analogy:** "If electrons are a scalpel, X-rays are a long needle."
:::""",
    "05. Massive Particles: Ions & Neutrons": """::: {.notes}
**Pedagogical Goal:** Complete the probe picture with niche but powerful techniques.
- **Ions:** Explain momentum transfer. When a heavy ion hits, it physically knocks atoms out (sputtering). This is the basis of Atom Probe Tomography—a true 3D atomic dataset.
- **Neutrons:** Highlight the "ghost-like" nature of neutrons. Explain how they are vital for distinguishing isotopes (like H vs D) which X-rays can't do due to low electron density.
:::""",
    "06. Physical Probes & Forces": """::: {.notes}
**Pedagogical Goal:** Explain proximal probes.
- **Concept:** Instead of shooting particles, we drag a tiny, sharp needle across the surface.
- **Data:** The data is a literal height map $Z(X,Y)$ or a force curve. Explain that this is distinct from a "projection" image like in TEM. In ML terms, AFM data is a true 2.5D surface, not an integration over depth.
:::""",
    "07. Sensors as Transducers": """::: {.notes}
**Pedagogical Goal:** Introduce the concept of the transfer function.
- **Concept:** We've discussed the physics; now how does that become a `.csv` or `.tiff` file? Sensors translate physics (photons/electrons) into digital voltage.
- **Emphasis:** The transfer function is never perfect. Every sensor stamps its own "signature" and noise onto the data. An ML model must either learn this signature or be designed to ignore it.
:::""",
    "08. Example: CMOS Detectors for Optical Imaging": """::: {.notes}
**Pedagogical Goal:** Ground the concept of sensors in something familiar (cell phone cameras).
- **Explanation:** Photons liberate charge carriers in silicon. Readout happens parallelizing per-pixel.
- **Trade-offs:** Fast, but reading every pixel amplifies thermal noise (dark current).
:::""",
    "09. Example: Scintillation Detectors (Indirect)": """::: {.notes}
**Pedagogical Goal:** Explain the challenge of detecting high-energy particles.
- **Analogy:** How do you catch a cannonball without breaking the net? You put a barrier (scintillator) that shatters into many small pebbles (visible light) which are easier to catch.
- **Data Consequence:** The scattering of light blurs the image. Introduce the concept of the Point Spread Function (PSF) here. The ML model might need to perform *deconvolution* to undo this physical blurring.
:::""",
    "10. Example: Direct Detectors": """::: {.notes}
**Pedagogical Goal:** Introduce a revolutionary technology that transformed structural biology and materials science (Cryo-EM).
- **Breakthrough:** Thinning the silicon so the primary electron goes straight through, creating a sharp, localized signal—no blurring scintillator needed.
- **Impact:** We can now count *individual* electrons. This incredibly high Signal-to-Noise Ratio (SNR) is what enables modern atomic-resolution reconstructions.
:::""",
    "11. Spectrometers: Energy Resolution in EDXS": """::: {.notes}
**Pedagogical Goal:** Explain how we get chemistry from X-rays using "Energy to Charge" conversion.
- **Math Intuition:** The number of electron-hole pairs created tells us the energy of the incoming X-ray. It's a simple proportional relationship: big energy, big charge pulse.
- **Limitations:** The counting process has statistical uncertainty (Fano noise). This sets a physical limit on how sharp the peaks can be, resulting in overlapping elemental peaks that ML models might need to untangle.
:::""",
    "12. Spectrometers: Energy Resolution in EELS": """::: {.notes}
**Pedagogical Goal:** Contrast EDXS with EELS using the "Energy to Space" concept.
- **Physics:** The magnetic prism acts like a glass prism for light, bending slower electrons more than faster ones.
- **Result:** We literally project an energy spectrum across a spatial camera. 
- **Comparison:** EELS provides much higher energy resolution than EDXS, allowing us to see not just *which* atoms are present, but their *chemical bonding* state.
:::""",
    "13. Continuous to Discrete Mapping": """::: {.notes}
**Pedagogical Goal:** Formalize the discretization process mathematically.
- **Walkthrough:** Write down the equation. $\\xi$ is the absolute ground truth we want to know. $x_i$ is what the computer actually gets.
- **Crucial Point:** Emphasize that $\\delta t$ (when we measure) and $u_x$ (how accurately we measure amplitude) are where the physics of the sensor ruins our perfect mathematical world. ML has to invert this process.
:::""",
    "14. Temporal and Spatial Sampling": """::: {.notes}
**Pedagogical Goal:** Relate the math to common experimental parameters.
- **Translation:** Make sure students understand that "sampling rate" isn't just about time. Pixel pitch is spatial sampling. Energy binning is spectral sampling. The math of signal processing applies equally to all of them.
:::""",
    "15. The Nyquist-Shannon Theorem": """::: {.notes}
**Pedagogical Goal:** Introduce the most important theorem in signal processing.
- **Rule of Thumb:** You must measure at least twice as fast as the fastest thing you want to see.
- **Interactive Component:** Use the widget to demonstrate what happens when you sample just slightly too slowly. The wave will appear to be a completely different, lower frequency.
:::""",
    "16. Aliasing - When Resolution Fails": """::: {.notes}
**Pedagogical Goal:** Show the physical consequences of violating Nyquist.
- **Examples:** Moiré patterns in TEM are a classic materials science example where the atomic lattice acts as a high-frequency signal that aliases against the pixel grid. The wagon-wheel effect is intuitive for temporal aliasing.
- **Interactive Component:** Let them play with the widget. Explain that aliasing generates *fake* signals that ML models could mistakenly interpret as real physical phenomena!
:::""",
    "17. Physical Resolution Limits": """::: {.notes}
**Pedagogical Goal:** Connect diffraction limits to the Point Spread Function (PSF).
- **Concept:** Even with infinite pixels, you can't see infinitely small features because light itself spreads out.
- **ML Application:** When building CNNs, knowing the PSF allows us to build physically-informed priors into the convolutional layers, directly embedding optics into the network architecture.
:::""",
    "18. Jitter and Temporal Resolution": """::: {.notes}
**Pedagogical Goal:** Highlight timing uncertainties.
- **Explanation:** Jitter is when the "clock ticks" of the sensor aren't perfectly spaced. 
- **Consequence:** In ultrafast experiments (e.g., watching a phase transition via pump-probe), jitter smears out the precise moment an event occurs.
:::""",
    "19. Finite Rate of Innovation (FRI)": """::: {.notes}
**Pedagogical Goal:** Introduce the "cheat code" around the Nyquist theorem.
- **Key Insight:** Nyquist assumes the worst-case scenario (random signals). But physical signals usually have *structure*—they are sparse. 
- **Example:** A crystal lattice is just a few atoms surrounded by empty space. If we *know* it's sparse, we don't have to sample the empty space densely.
:::""",
    "20. Sparse Recovery Example (FISTA)": """::: {.notes}
**Pedagogical Goal:** Show a practical ML application of sparsity (Compressed Sensing).
- **Concept:** We deliberately take fewer pixels (under-sampling) to save time or avoid damaging the sample with the probe (electron dose).
- **The Magic:** We use optimization (FISTA) to mathematically reconstruct the missing pixels by leveraging the L1 norm, which forces the solution to be sparse.
:::""",
    "21. The FISTA Algorithm (Mathematics)": """::: {.notes}
**Pedagogical Goal:** Demystify the solver algorithm.
- **Walkthrough:** Don't get lost in the weeds; focus on the intuition. 
- $f(\\mathbf{x})$ keeps the image looking like the data it measured.
- $g(\\mathbf{x})$ forces the image to be simple/sparse.
- The "Soft-Shrinkage" step simply throws away very small pixel values (setting them to zero), enforcing the sparsity. The Nesterov momentum just makes it run faster.
:::""",
    "22. Modeling Sensor Noise": """::: {.notes}
**Pedagogical Goal:** Shift the conversation from "resolution/sampling" to "noise".
- **Perspective Shift:** Stop thinking of measurements as just "numbers". They are samples pulled from a probability distribution.
- **Context:** If your detector heats up, the Gaussian distribution widens. If you don't collect enough photons, the Poisson distribution dictates you get a grainy image.
:::""",
    "23. Noise as a Physical Process": """::: {.notes}
**Pedagogical Goal:** Reinforce that noise is physics, not magic.
- **Key Message:** Noise isn't just an annoying artifact we throw moving averages at. It has physical origins (heat, particle counting). Understanding those origins lets us model the noise perfectly in our Bayesian or ML frameworks.
:::""",
    "24. Aleatory vs. Epistemic Uncertainty": """::: {.notes}
**Pedagogical Goal:** Give students the vocabulary to talk about uncertainty properly.
- **Aleatory:** Irreducible physical randomness. (The dice roll). E.g., Quantum nature of photon emission. No amount of ML changing will fix this.
- **Epistemic:** Ignorance. We don't know the instrument's exact calibration, or our ML model is too simple. We *can* fix this with better data or models.
:::""",
    "25. Physical Noise Models": """::: {.notes}
**Pedagogical Goal:** Map the physical processes to the statistical distributions.
- **Summary:** Thermal = Gaussian. Counting particles = Poisson. Material breaking/defects = Weibull. Knowing which one applies to your data dictates the Loss Function you must use.
:::""",
    "26. From Gaussian Noise to MSE Loss": """::: {.notes}
**Pedagogical Goal:** Provide an "Aha!" moment connecting statistics to deep learning.
- **The Derivation:** Walk through the math. Taking the negative log of the Gaussian PDF drops the exponential, leaving a squared difference.
- **The Takeaway Box:** This is crucial! When people blindly use Mean Squared Error (MSE) in PyTorch, they are implicitly telling the model, "My data is contaminated exclusively by Gaussian noise." If it's not, MSE is the wrong loss function!
:::""",
    "27. From Poisson Noise to Poisson Loss": """::: {.notes}
**Pedagogical Goal:** Show the alternative for Poisson statistics.
- **Application:** For very dark images (low dose TEM), the noise is incredibly asymmetric and non-Gaussian. 
- **The Takeaway:** If you train an autoencoder to denoise low-dose images using MSE, it will fail. You *must* derive and use the Poisson Loss logically.
:::""",
    "28. Bayesian Inference & MAP Estimation": """::: {.notes}
**Pedagogical Goal:** Formally unite physics, statistics, and machine learning.
- **Connections:** 
  - Likelihood = How well the model fits the data = The Loss Function.
  - Prior = What we know about physics beforehand (e.g., sparsity, smoothness) = Regularization in ML.
- **Conclusion:** MAP estimation shows that ML training is fundamentally just Bayesian inference in disguise.
:::""",
    "29. Notation Standards": """::: {.notes}
**Pedagogical Goal:** Establish baseline notation for the rest of the semester.
- **Practical Note:** Remind students to pay attention to bolded vs unbolded letters. Point out that the "Data Matrix" X is usually rows = samples, columns = features. This is the PyTorch/Scikit-Learn standard.
:::""",
    "30. Characterizing Data: Expected Value & Variance": """::: {.notes}
**Pedagogical Goal:** Quick review of fundamental statistics.
- **Relevance:** These aren't just abstract formulas. Mean is your "signal". Variance is essentially a measure of your "noise" or data spread. SNR (Signal to Noise Ratio) is roughly mean divided by standard deviation.
:::""",
    "31. Higher Moments: Skewness & Kurtosis": """::: {.notes}
**Pedagogical Goal:** Move beyond Gaussian thinking.
- **Application:** If your data is perfectly Gaussian, skewness is 0. If your detector frequently suffers from heavy cosmic ray hits (bright pixel spikes), your distribution will have very high Kurtosis (fat tails). This immediately tells you that standard MSE will struggle.
:::""",
    "32. The Curse of Dimensionality": """::: {.notes}
**Pedagogical Goal:** Explain why we can't just throw raw data into simple models.
- **The Curse:** In high dimensions, everything is far away from everything else. A 1-megapixel image is a 1-million dimensional space. We don't have enough atomic data in the universe to densely sample a 1-million dimensional cube.
- **The Solution:** We must find the lower-dimensional manifold where the actual physical physics "lives".
:::""",
    "33. Singular Value Decomposition (SVD)": """::: {.notes}
**Pedagogical Goal:** Introduce the workhorse of linear algebra for data science.
- **Concept:** SVD tears apart any data matrix into building blocks. 
- **U and V:** U contains the combinations of samples, V contains the combinations of features. S tells you how "important" each combination is.
:::""",
    "34. Principal Component Analysis (PCA)": """::: {.notes}
**Pedagogical Goal:** Connect SVD to PCA.
- **The Difference:** PCA is just SVD applied to *mean-centered* data. 
- **Geometric Intuition:** It rotates the coordinate system so that the first axis points along the direction of maximum spread (variance).
:::""",
    "35. Variance Explained & Dimensionality Reduction": """::: {.notes}
**Pedagogical Goal:** How to practically choose the number of dimensions.
- **The Scree Plot:** Explain that we look for the "elbow" or we set a threshold (like 95% variance). This allows us to compress a 1000-dimensional spectrum into just 3 or 4 numbers, removing noise and keeping the physics.
:::""",
    "36. Case Study: Time Series (McClarren)": """::: {.notes}
**Pedagogical Goal:** Concrete engineering example of PCA.
- **Analysis:** We took an entire time-evolving simulation and compressed it.
- **Physical Interpretation:** PC1 always captures the most obvious variance (overall amplitude). PC2 often captures derivatives or shifts (the timing of the peak). We've extracted physics from raw pixels!
:::""",
    "37. Case Study: Hyperspectral Foliage": """::: {.notes}
**Pedagogical Goal:** Concrete scientific example of PCA.
- **Analysis:** A hyperspectral pixel has 2000 channels. We humans can only see 3 (RGB). PCA distills those 2000 channels into the 4 most statistically significant combinations, showing us plant stress that is physically invisible to our eyes.
:::""",
    "38. PCA on Images: Eigenmicrostructures": """::: {.notes}
**Pedagogical Goal:** Apply PCA to 2D image domains.
- **Concept:** If you run PCA on a stack of images of grains, the Principal Components themselves will look like ghostly, average grains ("Eigen-grains"). 
- **Application:** You can reconstruct any complex microstructure by linearly summing these basic building block images.
:::""",
    "39. K-means Clustering": """::: {.notes}
**Pedagogical Goal:** Introduce unsupervised grouping.
- **Workflow:** First we use PCA to reduce the 1000 dimensions to 3. *Then* we use K-means to group the data points in that 3D space. 
- **Materials Context:** K-means is fantastic for automatic phase mapping in EDS or EELS datasets. It automatically segments the image into regions of similar chemistry.
:::""",
    "40. t-SNE: Visualizing High-Dim Manifolds": """::: {.notes}
**Pedagogical Goal:** Give students a tool for visualization.
- **Why t-SNE?:** PCA is linear. If the data lies on a curved sheet (a manifold), PCA will squish it badly. t-SNE preserves *local neighborhoods* (points that are close in high-dim stay close in 2D), allowing us to see complex, non-linear cluster structures.
- **Warning:** Remind them that t-SNE distances are distorted; it's for human eyeballs and visualization, not for downstream quantitative math.
:::""",
    "41. Summary & Key Takeaways": """::: {.notes}
**Pedagogical Goal:** Wrap up the unit.
- **Final Message:** A good ML materials engineer understands the entire pipeline: from the photon hitting the detector, to the Nyquist limits, the statistical noise model, and the dimensionality reduction. 
- **Next Steps:** Preview that next week we will start building models that actually learn these mappings.
:::"""
}

new_parts = [parts[0]]

for part in parts[1:]:
    lines = part.split('\n')
    title = lines[0].strip()
    if title in notes:
        part = part.rstrip() + '\n\n' + notes[title] + '\n'
    new_parts.append(part)

final_text = '\n## '.join(new_parts)

# Add title slide notes. The frontmatter ends with `---` on a line by itself.
if '---\n\n## ' in final_text:
    final_text = final_text.replace('---\n\n## ', '---\n\n::: {.notes}\n**Pedagogical Goal:** Welcome the students to Unit 2. Set the tone that we are bridging physics and data engineering today, explaining where the numbers actually come from before we build ML out of them.\n:::\n\n## ', 1)

with open(file_path, 'w') as f:
    f.write(final_text)

print("Notes added successfully.")

