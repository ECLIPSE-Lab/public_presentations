# Unit 7: Time-series and Process Monitoring - Lecture Plan

## Objective
The objective of this unit is to introduce the concepts and architectures necessary for handling sequential data in materials processing. Students will learn how to move from "static" snapshots (images) to "dynamic" process logs (time-series) using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.

## Slide Structure (approx. 50 slides)

### Part 1: Sequences in Materials Science (8 slides)
- **The Third Dimension: Time (3 slides):**
  - Process logs: Temperature, pressure, gas flow.
  - Signal fluctuations: Melt pool emissions in Additive Manufacturing (AM).
  - Why standard CNNs/MLPs fail here (lack of memory).
- **Time-Series Characteristics (3 slides):**
  - Sampling rate vs. physical timescales.
  - Non-stationarity and drift in sensors.
  - Autocorrelation: The past influences the future.
- **Goal: Forecasting and Anomaly Detection (2 slides):**
  - Predicting the next state.
  - Detecting when a process "goes off the rails."

### Part 2: Recurrent Neural Networks (RNNs) (12 slides)
- **The Concept of Recursion (3 slides):**
  - Neurons with self-loops (McClarren 7.1).
  - Encoding history in a "hidden state" $h_t$.
- **Unrolling the RNN (3 slides):**
  - Visualizing the computational graph over time (McClarren Fig 7.2).
  - Shared weights across all time steps.
- **The Forward Pass (3 slides):**
  - $h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$.
  - $y_t = \phi(W_y h_t + b_y)$.
- **The Training Challenge (3 slides):**
  - Backpropagation Through Time (BPTT).
  - The Vanishing Gradient Problem: Why RNNs "forget" the distant past (McClarren 7.1.1).

### Part 3: LSTMs and GRUs (12 slides)
- **Long Short-Term Memory (LSTM) (5 slides):**
  - The "Cell State" as a conveyor belt.
  - The Gates: Forget, Input, and Output.
  - How gates prevent the gradient from vanishing.
- **Gated Recurrent Units (GRU) (3 slides):**
  - A simplified LSTM: Update and Reset gates.
  - Efficiency vs. Performance.
- **Bidirectional RNNs (2 slides):**
  - Looking into the future (only for offline analysis).
- **Deep RNNs (2 slides):**
  - Stacking recurrent layers for hierarchical time-features.

### Part 4: Case Studies in Process Monitoring (10 slides)
- **Melt Pool Monitoring in AM (4 slides):**
  - Using RNNs to predict pore formation from photodiode signals.
  - Real-time feedback loops.
- **Predicting Material Degradation (3 slides):**
  - Corrosion or creep logs: Forecasting the remaining useful life (RUL).
- **Signal Frequency and Shift (3 slides):**
  - McClarren's example: Recovering frequency from a noisy sine wave.

### Part 5: Practical Implementation (8 slides)
- **Data Preparation for Sequences (3 slides):**
  - Sliding windows and sequence length.
  - Padding sequences of varying lengths.
- **Evaluation Metrics for Time-Series (3 slides):**
  - Dynamic Time Warping (DTW) for similarity.
  - Horizon-based error (1-step vs. N-step ahead).
- **Summary & Takeaways (2 slides):**
  - RNNs for short memory, LSTMs for long memory.

## Materials Sources
- **McClarren (2021):** Ch 7 (RNNs, LSTMs, Vanishing gradients, Examples).
- **Sandfeld (2024):** Reference for general sequence concepts.
- **Neuer (2024):** Reference for process control context.
