# Unit 7: Time-series and Process Monitoring - Slide Content (50 Slides)

## Part 1: Sequences in Materials Science (8 slides)

**Slide 1: Title Slide**
- **Unit 7: Time-series and Process Monitoring**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: Learning from Sequential Data and Dynamic Processes

**Slide 2: Beyond Static Images**
- So far: CNNs for "snapshots" of microstructures.
- Reality: Materials are *made* via dynamic processes.
- Examples: Solidification, Additive Manufacturing, Rolling, Heat Treatment.

**Slide 3: Process Logs as Data**
- Temperature sensors (thermocouples).
- Gas pressure and flow rate logs.
- Mechanical load/stress over time (Fatigue cycles).
- These are **1D Sequences**.

**Slide 4: Additive Manufacturing (AM) Example**
- Laser Powder Bed Fusion (LPBF).
- Melt pool intensity fluctuations (Photodiode signals).
- Thousands of points per second.
- The signal pattern tells us if a "pore" is about to form.

**Slide 5: Why CNNs/MLPs Fail (I)**
- CNNs assume spatial local correlation but have no inherent "before" or "after."
- MLPs assume all inputs are independent.
- Neither handles **variable length** sequences easily.

**Slide 6: Why CNNs/MLPs Fail (II)**
- Lack of **Memory**: A process state at $t$ depends on $t-1$.
- We need a model that "remembers" the past to predict the future.

**Slide 7: Time-Series Characteristics**
- **Sampling Rate**: Does it match the physics? (e.g., cooling rate).
- **Non-stationarity**: Sensor drift or gradual degradation of equipment.
- **Autocorrelation**: The strongest predictor of tomorrow's temperature is often today's.

**Slide 8: Summary: Part 1**
- Sequences are ubiquitous in manufacturing.
- We need architectures built for temporal dependencies.

---

## Part 2: Recurrent Neural Networks (RNNs) (12 slides)

**Slide 9: The Concept of Recursion**
- Normal Neuron: $y = \sigma(Wx + b)$.
- Recurrent Neuron: Output at $t-1$ becomes an input at $t$.
- "A loop in the computational graph." (McClarren 7.1).

**Slide 10: The Hidden State $h_t$**
- $h_t$ is the "Internal Memory" of the network.
- It stores a summary of everything seen up to time $t$.

**Slide 11: RNN Architecture (McClarren Fig 7.1)**
- [Diagram of a single RNN cell with a self-loop]
- Inputs $x_t$, Outputs $y_t$, State $h_t$.

**Slide 12: Unrolling the RNN (I)**
- Visualizing the loop as a sequence of identical layers.
- One layer for each time step.

**Slide 13: Unrolling the RNN (II) (McClarren Fig 7.2)**
- [Diagram of unrolled RNN showing $t-1, t, t+1$]
- Key: The **Weights are shared** across all time steps.

**Slide 14: The Forward Pass Equations**
- Hidden update: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
- Output: $y_t = \sigma(W_{hy} h_t + b_y)$.

**Slide 15: Handling Variable Length**
- RNNs can process 10 steps or 1000 steps using the same weights.
- Simply keep iterating the same cell.

**Slide 16: Training: Backpropagation Through Time (BPTT)**
- Compute the error at the final time step.
- Backpropagate the gradient "back in time" through the unrolled graph.

**Slide 17: The Problem: Vanishing Gradients (I)**
- During BPTT, we multiply by the same weight $W$ many times.
- If $|W| < 1$, the gradient shrinks exponentially as it goes back.
- Result: The network "forgets" the early parts of the sequence (McClarren 7.1.1).

**Slide 18: Exploding Gradients**
- If $|W| > 1$, the gradient grows exponentially.
- Leads to "NaN" weights or numerical instability.
- Solution: **Gradient Clipping**.

**Slide 19: Long-Term Dependencies**
- Does the heating in Step 1 affect the final grain size in Step 100?
- Standard RNNs struggle to link these distant events.

**Slide 20: Summary: Part 2**
- RNNs use shared weights and a hidden state for memory.
- BPTT is the training engine, but suffers from vanishing gradients.

---

## Part 3: LSTMs and GRUs (12 slides)

**Slide 21: Solving the Memory Problem**
- We need a better way to "transport" information over long distances.
- Solution: **Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997).

**Slide 22: The LSTM Cell State $C_t$**
- The "Conveyor Belt" or "High-Speed Rail" of information.
- Runs through the entire sequence with minimal interaction.

**Slide 23: The Forget Gate**
- Decides which information to "throw away" from the memory.
- Example: If a new batch starts, forget the previous batch's specifics.

**Slide 24: The Input Gate**
- Decides which new information to "store" in the cell state.
- Filtering the current sensor reading for useful features.

**Slide 25: The Output Gate**
- Decides what to "output" based on the current memory.
- Providing the prediction for the next time step.

**Slide 26: Why LSTMs Don't Vanish**
- The additive nature of the cell state update allows gradients to flow through many steps without being repeatedly multiplied by small weights.

**Slide 27: Gated Recurrent Units (GRU)**
- A modern, simplified version of LSTM.
- Combines Forget and Input gates into a single **Update Gate**.
- Fewer parameters, often similar performance.

**Slide 28: RNN vs. LSTM vs. GRU**
- [Visual comparison table]
- Complexity vs. Memory capacity.

**Slide 29: Bidirectional RNNs**
- Processing the sequence from start-to-end AND end-to-start.
- Useful for post-experiment analysis (e.g., analyzing a full log).
- *Not* for real-time control (we can't see the future).

**Slide 30: Deep (Stacked) RNNs**
- Layer 1 extracts low-level trends (noise removal).
- Layer 2 extracts high-level process patterns.

**Slide 31: Sequence-to-Sequence (Seq2Seq)**
- Input a sequence (Process logs) $\to$ Output a sequence (Expected microstructure evolution).

**Slide 32: Summary: Part 3**
- Gates (LSTM/GRU) enable long-term memory.
- Cell states act as a gradient highway.

---

## Part 4: Case Studies in Process Monitoring (10 slides)

**Slide 33: Additive Manufacturing Anomaly Detection**
- Task: Is the laser current fluctuating abnormally?
- Data: High-speed photodiode.
- Model: LSTM trained on "normal" builds.

**Slide 34: Predicting Pore Formation**
- [Show signal vs. Pore location graph]
- LSTMs capture the "pre-pore" oscillations that static filters miss.

**Slide 35: Material Degradation & RUL**
- Remaining Useful Life (RUL) prediction.
- Forecasting when a turbine blade will fail based on vibration logs.

**Slide 36: Creep and Corrosion Modeling**
- Non-linear evolution of damage.
- RNNs as "learned" differential equation solvers.

**Slide 37: Signal Frequency Extraction (McClarren)**
- Example: Recovering frequency $\omega$ from $y(t) = \sin(\omega t) + \text{noise}$.
- RNN learns to "track" the phase.

**Slide 38: Phase Shift Prediction**
- Predicting how a signal shifts due to temperature changes in the instrument.

**Slide 39: Cart-Mounted Pendulum (McClarren)**
- Using RNNs to predict the physics of a dynamic system.
- Learning the "hidden" dynamics from observations.

**Slide 40: Sensor Fusion over Time**
- Combining 10 different thermocouples into one LSTM.
- Mapping spatial-temporal heat distributions.

**Slide 41: Real-time Feedback Loops**
- RNN outputs a "danger" signal $\to$ PLC reduces laser power instantly.
- The goal of **Autonomous Processing**.

**Slide 42: Summary: Part 4**
- Sequential models bridge the gap between "making" and "measuring."

---

## Part 5: Practical Implementation (8 slides)

**Slide 43: Preparing Sequential Data**
- **Sliding Window**: Converting a long log into many small "training sequences."
- Window size $T$: How far back should the model look?

**Slide 44: Padding and Masking**
- Experiments have different durations.
- Padding short sequences with zeros.
- Masking: Telling the network to ignore the padded zeros.

**Slide 45: Feature Scaling for RNNs**
- Standardization is crucial!
- RNNs are very sensitive to input magnitude due to repeated multiplications.

**Slide 46: Horizon-based Error**
- **1-step ahead**: Predicting $t+1$ from $t$.
- **N-step ahead**: Predicting the next 10 minutes.
- Error accumulates quickly in recursive models.

**Slide 47: Dynamic Time Warping (DTW)**
- Measuring similarity between two signals that might be shifted or stretched in time.
- Better than MSE for comparing process logs.

**Slide 48: The Rise of Transformers (Bonus)**
- Attention mechanisms are replacing RNNs for very long sequences.
- "Attention is all you need" (Unit 14 teaser).

**Slide 49: Summary & Key Takeaways**
1. Time-series capture the **physics of processing**.
2. RNNs provide **memory** via hidden states.
3. LSTMs/GRUs use **gates** to prevent forgetting.
4. Applications: AM monitoring, failure prediction, signal tracking.

**Slide 50: References & Exercises**
- McClarren Ch 7.
- Exercise: Predict the "Tensile Test Stress-Strain" curve evolution.
