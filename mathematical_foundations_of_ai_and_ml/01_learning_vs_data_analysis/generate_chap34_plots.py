import numpy as np
import matplotlib.pyplot as plt
import os

IMG_DIR = '/home/philipp/projects/_public_presentations/mathematical_foundations_of_ai_and_ml/01_learning_vs_data_analysis/img'
os.makedirs(IMG_DIR, exist_ok=True)
plt.style.use('fivethirtyeight')
plt.rcParams['text.color'] = '#cccccc'
plt.rcParams['axes.labelcolor'] = '#cccccc'
plt.rcParams['xtick.color'] = '#cccccc'
plt.rcParams['ytick.color'] = '#cccccc'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.titlecolor'] = '#cccccc'
plt.rcParams['legend.labelcolor'] = '#cccccc'

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename), transparent=True)
    plt.close()

# 1. Normalization Plot
np.random.seed(12)
x = np.linspace(0, 10, 100)
y_raw = 50 * np.sin(x) + 200 + np.random.normal(0, 5, 100)
y_minmax = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
y_zscore = (y_raw - y_raw.mean()) / y_raw.std()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x, y_raw, color='#8e44ad')
axes[0].set_title("Raw Data (Arbitrary Scale)")

axes[1].plot(x, y_minmax, color='#2980b9')
axes[1].set_title("Min-Max Normalization (0 to 1)")
axes[1].set_ylim([-0.1, 1.1])

axes[2].plot(x, y_zscore, color='#27ae60')
axes[2].set_title("Z-Score / Sub-mean ($μ=0, σ=1$)")
axes[2].axhline(0, color='gray', linestyle='--')
save_plot('normalization.svg')

# 2. Filtering Plot (Moving Average)
np.random.seed(42)
y_noisy = np.sin(x) + np.random.normal(0, 0.4, 100)
y_smooth = np.convolve(y_noisy, np.ones(10)/10, mode='same')

plt.figure(figsize=(10, 5))
plt.plot(x, y_noisy, color='gray', alpha=0.6, label='Raw Noisy Data')
plt.plot(x, y_smooth, color='#c0392b', linewidth=4, label='Moving Average (Window=10)')
plt.title("Filtering Noise from a Signal")
plt.legend(facecolor='#333333', edgecolor='none')
plt.xlabel("Time")
plt.ylabel("Signal Amplitude")
save_plot('filtering.svg')

# 3. Train Test Split Visual
fig, ax = plt.subplots(figsize=(10, 2))
# Bar segments
ax.barh([1], [80], left=[0], color='#3498db', label='Training Set (80%)', height=0.5)
ax.barh([1], [20], left=[80], color='#e74c3c', label='Test Set (20%)', height=0.5)
ax.text(40, 1, 'Algorithm "Learns" from this', ha='center', va='center', color='white', fontweight='bold')
ax.text(90, 1, 'Final Evaluation', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
ax.set_yticks([])
ax.set_xticks([0, 80, 100])
ax.set_xlim(0, 100)
ax.set_title("The Golden Rule: Test Data Never Enters Training")
save_plot('train_test_split.svg')

# 4. Triggering Result Plot
np.random.seed(42)
time = np.arange(0, 1000)
signal = np.random.normal(0, 0.1, 1000)
event_starts = [100, 300, 450, 700, 850]
for start in event_starts:
    signal[start:start+50] += np.sin(np.linspace(0, np.pi, 50)) * 2 + np.random.normal(0, 0.1, 50)

threshold = 0.5
window = 60

# Plot the raw signal with threshold
plt.figure(figsize=(10, 3))
plt.plot(time, signal, color='gray', alpha=0.9, label='Sensor Stream')
plt.axhline(threshold, color='#e74c3c', linestyle='--', linewidth=2, label='Trigger Threshold')
plt.title("Continuous Time-Series Signal")
plt.xlabel("Continuous Time")
plt.ylabel("Signal Amplitude")
plt.legend(facecolor='#333333', edgecolor='none')
save_plot('triggering_raw_signal.svg')

events = []
i = 0
while i < len(signal) - window:
    if signal[i] > threshold:
        start_idx = max(0, i - 10)
        events.append(signal[start_idx : start_idx + window])
        i += window
    else:
        i += 1

plt.figure(figsize=(8, 4))
for idx, ev in enumerate(events):
    if len(ev) == window:
        plt.plot(ev, linewidth=2, alpha=0.8, label=f'Event {idx+1}')
plt.title("Overlay of Triggered Windows")
plt.xlabel("Time relative to trigger")
plt.ylabel("Signal Amplitude")
plt.legend(facecolor='#333333', edgecolor='none')
save_plot('triggering_result.svg')

# 5. Differentiation Plot
np.random.seed(10)
time_diff = np.linspace(0, 10, 200)
signal_drift = 5 * time_diff + 2 * np.sin(3 * time_diff) + np.random.normal(0, 0.5, 200)
signal_diff = np.diff(signal_drift)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time_diff, signal_drift, color='#3498db', linewidth=2)
axes[0].set_title("Original Signal with Linear Drift")
axes[0].set_ylabel("Amplitude")

axes[1].plot(time_diff[1:], signal_diff, color='#e74c3c', linewidth=2)
axes[1].set_title("Differentiated Signal (Drift Removed)")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Δ Amplitude")
plt.tight_layout()
save_plot('differentiating.svg')

# 6. Fourier Transform Plot
np.random.seed(20)
time_fft = np.linspace(0, 1, 500)
freq1, freq2 = 10, 50
signal_fft = 3 * np.sin(2 * np.pi * freq1 * time_fft) + 1.5 * np.sin(2 * np.pi * freq2 * time_fft) + np.random.normal(0, 1, 500)

fft_vals = np.abs(np.fft.rfft(signal_fft))
fft_freqs = np.fft.rfftfreq(len(time_fft), time_fft[1] - time_fft[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(time_fft, signal_fft, color='#9b59b6', linewidth=1)
axes[0].set_title("Complex Noisy Signal")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

axes[1].plot(fft_freqs, fft_vals, color='#f1c40f', linewidth=2)
axes[1].set_title("Frequency Spectrum (FFT)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_xlim(0, 100)
plt.tight_layout()
save_plot('fourier.svg')

# 7. Discrete Wavelet Transform (DWT) Plot
try:
    import pywt
    np.random.seed(30)
    time_dwt = np.linspace(0, 1, 400)
    
    # A piecewise signal with jumps and local oscillation
    signal_part1 = np.sin(20 * np.pi * time_dwt[:200])
    signal_part2 = np.sin(5 * np.pi * time_dwt[200:]) + 2
    signal_dwt = np.concatenate([signal_part1, signal_part2])
    signal_dwt += np.random.normal(0, 0.2, 400)
    
    coeffs = pywt.dwt(signal_dwt, 'db4')
    cA, cD = coeffs # Approximation and Detail coefficients
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    axes[0].plot(signal_dwt, color='#34495e', linewidth=1.5)
    axes[0].set_title("Original Non-Stationary Signal")
    
    axes[1].plot(cA, color='#e67e22', linewidth=1.5)
    axes[1].set_title("Approximation (cA) - Low Frequency Trend")
    
    axes[2].plot(cD, color='#16a085', linewidth=1.5)
    axes[2].set_title("Detail (cD) - High Frequency Edges")
    
    plt.tight_layout()
    save_plot('wavelet.svg')
except ImportError:
    print("PyWavelets not installed, skipping DWT plot.")

# 8. Classification vs Regression Examples
np.random.seed(42)
time_peak = np.linspace(0, 10, 200)
voltage = np.sin(time_peak) + np.random.normal(0, 0.2, 200)
voltage[90:110] += 4 * np.exp(-0.1 * (np.arange(20)-10)**2)  # dangerous peak

# Classification Example
fig, axes = plt.subplots(2, 1, figsize=(6, 5))
axes[0].text(0.5, 0.5, "Image Recognition:\n'Motorcycle' vs 'Traffic Light'", 
             ha='center', va='center', fontsize=18, color='#2ecc71',
             bbox=dict(boxstyle="round,pad=1.5", facecolor="#222222", edgecolor="#2ecc71"))
axes[0].axis('off')
axes[0].set_title("Ex 1: Target is a discrete class $d \in \mathcal{Y}$", pad=15)

axes[1].plot(time_peak, voltage, color='#e74c3c')
axes[1].axhline(3, color='gray', linestyle='--')
axes[1].text(2, 4, "$d = 1$ (Danger)", color='#e74c3c', fontsize=16, fontweight='bold')
axes[1].text(8, 0, "$d = 0$", color='#2ecc71', fontsize=16, fontweight='bold')
axes[1].set_title("Ex 2: Binary Alarm Classification", pad=10)
axes[1].axis('off')
plt.tight_layout()
save_plot("classification_examples.svg")

# Regression Example
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(time_peak, voltage, color='#f1c40f')
ax.axhline(3, color='gray', linestyle='--')
ax.annotate('Risk Output: $y = 0.8$ (80%)', xy=(5, 4.5), xytext=(1, 6),
            arrowprops=dict(facecolor='#f1c40f', shrink=0.05),
            fontsize=18, color='#f1c40f', fontweight='bold')
ax.set_title("Ex 3: Probability of Danger ($y \in [0, 1]$)", pad=20)
ax.axis('off')
plt.tight_layout()
save_plot("regression_example.svg")

print("Generated plots successfully.")
