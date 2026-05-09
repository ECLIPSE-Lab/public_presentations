# Unit 11: Automation in Microscopy and Characterization (50 Slide Draft)

## Section 1: Intro & Motivation (Slides 1-5)

1. **Slide 1: Title Slide**
   - Title: Unit 11: Automation in Microscopy and Characterization
   - Course: ML for Characterization and Processing (SS26)
   - Prof. Dr. Philipp Pelz
   - Focus: Reinforcement Learning, Control, and Autonomous Lab

2. **Slide 2: Recap: Deep Learning for Analysis**
   - We've seen how to analyze images (CNNs, AEs) and time-series (RNNs).
   - But: Who collected the data?
   - Usually a human spending hours at a microscope.

3. **Slide 3: The Manual Bottleneck**
   - Materials science is becoming high-throughput.
   - 1000s of samples need characterization.
   - Human operators are expensive, prone to fatigue, and introduce bias.
   - We need self-operating instruments.

4. **Slide 4: What can we automate?**
   - **Low-level**: Autofocus, stage movement, beam alignment.
   - **Mid-level**: Identifying regions of interest (ROI), grain selection.
   - **High-level**: Deciding the next experiment based on previous results.

5. **Slide 5: The "Self-Driving" Microscope**
   - Goal: From "Pressing Buttons" to "Defining Objectives."
   - Instrument as an autonomous agent.

## Section 2: Instrumentation & Control Basics (Slides 6-12)

6. **Slide 6: Control Theory: Refresher**
   - Feedback: Measure error, adjust control (e.g., thermostat).
   - Sensors: Detectors, beam current meters.
   - Actuators: Lenses, deflector coils, stage motors.

7. **Slide 7: Why is Microscopy Control hard?**
   - Non-linear response (magnetic lenses, saturation).
   - Hysteresis (magnetic fields).
   - High-dimensional parameter space (Aligning an EM has 50+ knobs).

8. **Slide 8: The Microscope as a State-Machine**
   - State $\mathbf{x}_t$: Position, focus, stigmation, illumination.
   - Action $\mathbf{a}_t$: Changing currents or voltages.
   - Measurement $y_t$: The captured image or signal.

9. **Slide 9: Classical Control: PID**
   - Proportional-Integral-Derivative control.
   - Works for simple, well-defined loops (e.g., stage position).
   - Fails for complex objectives (e.g., "Find a specific precipitate").

10. **Slide 10: The Transition to ML-based Control**
    - Instead of hard-coding the "knob-to-result" mapping, we learn it.

11. **Slide 11: Objectives vs. Commands**
    - High-level: "Maximize image sharpness."
    - ML translates this into specific lens current changes.

12. **Slide 12: Summary: Control**
    - Instrument = Sensors + Actuators + Intelligence (Agent).

## Section 3: Reinforcement Learning (RL) Foundations (Slides 13-25)

13. **Slide 13: What is Reinforcement Learning?**
    - (Ref: McClarren Ch 9.1).
    - Learning by Trial and Error.
    - No labels needed! Only a "Reward Signal."

14. **Slide 14: The RL Loop: Agent & Environment**
    - Agent (ML Model) $\leftrightarrow$ Environment (The Microscope).
    - Agent takes an Action $\to$ Environment returns a Reward and New State.

15. **Slide 15: Key Components: State, Action, Reward**
    - **State**: What the microscope "sees" now.
    - **Action**: What the agent "does" (Change lens, move stage).
    - **Reward**: A scalar number indicating how good the action was.

16. **Slide 16: Designing Reward Functions**
    - This is the "Teacher's" job.
    - Example for Focus: Sharpness index or FFT peak intensity.
    - Example for Alignment: Centering of the beam on the detector.

17. **Slide 17: The Reward Surface**
    - (Graphic: 3D plot of Reward vs. Lens Current).
    - The Agent wants to find the peak (The optimal focus).

18. **Slide 18: Policies: The Decision Maker**
    - A Policy $\pi(s)$ is a mapping from state $s$ to action $a$.
    - Goal: Find the optimal policy $\pi^*$ that maximizes total reward.

19. **Slide 19: Exploration vs. Exploitation**
    - (Ref: McClarren 9.3).
    - Exploration: Trying new, random actions to see what happens.
    - Exploitation: Using the best known actions to get a reward.
    - The balance is crucial!

20. **Slide 20: Policy Gradients (The Strategy)**
    - (Ref: McClarren 9.2).
    - Turning the decision into a probability distribution.
    - Update the NN to make "good" decisions more likely.

21. **Slide 21: Training by Weighting the Loss**
    - Supervised Loss: $\mathcal{L} = -\log \pi(a|s)$.
    - RL Loss: $\mathcal{L} = - R \cdot \log \pi(a|s)$.
    - Actions that give large $R$ get their probability boosted.

22. **Slide 22: Cumulative Rewards**
    - Summing the reward over many steps.
    - Important for delayed rewards (e.g., moving the stage to a better ROI).

23. **Slide 23: From Games to Labs**
    - RL learned to play Go and Atari; now it learns to operate microscopes.

24. **Slide 24: Implementation: Sampling the Policy**
    - Outputting probabilities (Softmax) and sampling.

25. **Slide 25: Summary: RL Foundations**
    - No labels, only rewards.
    - Trial and error is the primary teacher.

## Section 4: Automation in Microscopy (Slides 26-35)

26. **Slide 26: Low-Level Automation: Autofocus**
    - Traditional: Sweep lens current, pick the one with max sharpness.
    - ML: Learn to jump directly to the optimal current based on the blurry image.

27. **Slide 27: Defining the Focus Reward**
    - Laplacian variance.
    - Fourier Transform (FFT) high-frequency content.
    - Sharpness metrics from Unit 10.

28. **Slide 28: Beam Alignment and Stigmation**
    - Correcting for non-circular beams.
    - Agent learns to adjust deflector currents by looking at the beam shape.

29. **Slide 29: Mid-Level: Anomaly Detection and ROI**
    - Instrument scans the sample quickly at low-mag.
    - Anomaly detection (Unit 9) flags "interesting" regions.
    - Instrument automatically zooms in for high-res imaging.

30. **Slide 30: Case Study: Grain Boundary Analysis**
    - Agent moves the stage to find grain boundaries.
    - Goal: Collect 1000 boundaries automatically.

31. **Slide 31: Integration with Characterization Signals**
    - Using EDS or EELS to guide the search.
    - "Find all regions with high Ni concentration."

32. **Slide 32: Dealing with Sample Damage**
    - For beam-sensitive materials, the agent must minimize "dose."
    - Reward = Image Quality / Dose.

33. **Slide 33: Multi-Agent Systems**
    - Different models for focus, alignment, and navigation.

34. **Slide 34: Robustness in Characterization**
    - Handling noisy detectors and vibrations.

35. **Slide 35: Summary: Microscopy Automation**
    - Moving from reactive to proactive control.

## Section 5: Case Study: Industrial Glass Cooling (Slides 36-45)

36. **Slide 36: Why Process Control?**
    - Automation isn't just for characterization; it's for manufacturing.
    - (Ref: McClarren Ch 9.4).

37. **Slide 37: The Problem: Glass Cooling**
    - Cooling rate controls chemical reactions and physical stress.
    - We want the glass to follow a specific "Target Temperature" profile.

38. **Slide 38: The Physics (McClarren 9.4)**
    - Coupled Radiation and Diffusion PDEs.
    - Boundary temperature $u$ is the only control.

39. **Slide 39: RL Control Strategy**
    - Inputs: Current temperature, Target temperature (future).
    - Outputs: Change in boundary temperature ($\Delta u$).
    - Reward: Inverse of squared difference from target.

40. **Slide 40: Policy Gradient Implementation**
    - Small Neural Network (1 hidden layer).
    - Weighted by cumulative reward.

41. **Slide 41: Results: RL vs. Simple Control**
    - Simple Control: Set $u = T_{target}$. (Slow response).
    - RL: Learns to "overheat" slightly to reach the target faster.

42. **Slide 42: Learning System Dynamics**
    - The RL model has no knowledge of PDEs.
    - It "discovers" the thermal lag of the glass sheet.

43. **Slide 43: Tweaking the Reward**
    - Penalizing rapid temperature changes to avoid glass cracking.
    - Multi-objective rewards.

44. **Slide 44: From Simulations to Production**
    - Training on a simulator (The PDE model) then deploying on the real furnace.
    - Transfer Learning (Unit 6).

45. **Slide 45: Summary: Glass Cooling**
    - Control policies can be learned from data.
    - RL handles time-delays naturally.

## Section 6: Synthesis & Self-Driving Labs (Slides 46-50)

46. **Slide 46: Integration: Automation + Characterization + Analysis**
    - The complete feedback loop.
    - Automated synthesis $\to$ Automated characterization $\to$ ML analysis $\to$ Next experiment.

47. **Slide 47: The "Self-Driving Lab" Framework**
    - (Graphic: Circular loop of materials discovery).
    - Integration of Units 1-14.

48. **Slide 48: Challenges to Automation**
    - Software interfaces (Closed-source instrument software).
    - Data standardization.
    - Trust: Can we leave the lab unattended?

49. **Slide 49: Take-Home Messages**
    - Reinforcement Learning is the engine of automation.
    - Policy Gradients bridge control and deep learning.
    - Reward design is critical.
    - Future: Autonomous discovery in materials science.

50. **Slide 50: References & Further Reading**
    - McClarren (2021), Ch 9.
    - Sutton & Barto: Reinforcement Learning (The Bible of RL).
    - Next Unit: Uncertainty & GPs.
