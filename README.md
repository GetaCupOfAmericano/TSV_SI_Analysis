# TSV_SI_Analysis
Python-based Signal Integrity (SI) simulator for 3D IC Through-Silicon Vias (TSVs). Models frequency-dependent channel loss, crosstalk (NEXT/FEXT), and high-speed eye diagram analysis.
# 3D IC TSV Signal Integrity Analyzer

## Project Overview

This project conducted in ASTRI (Hong Kong Applied Science and Techonology Research Institute) provides a comprehensive Python-based simulation tool for analyzing the signal integrity of Through-Silicon Vias (TSVs) in 3D Integrated Circuits (ICs). It allows users to model TSV performance under various physical parameters, material properties, and signal characteristics, offering both frequency-domain (S-parameters, RLGC) and time-domain (transient waveforms, eye diagrams) analysis. A key feature is the ability to evaluate the impact of Transmit (TX) Pre-emphasis on improving channel performance and opening closed eye diagrams.

## Features

*   **Physics-Based TSV Modeling:** Calculates frequency-dependent R, L, G, C parameters for accurate channel representation.
*   **S-Parameter Generation:** Computes 4-port S-parameters for single or coupled TSV configurations, enabling analysis of insertion loss (S31), reflection (S11), near-end crosstalk (NEXT, S21), and far-end crosstalk (FEXT, S41).
*   **Transient Simulation:** Simulates the propagation of high-speed digital signals through the modeled TSV channel.
*   **Eye Diagram Analysis:** Generates eye diagrams from the output waveform, providing critical metrics like eye height and eye width for signal quality assessment. Includes FEXT waveform visualization.
*   **TX Pre-emphasis Implementation:** Allows users to enable and tune pre-emphasis to compensate for channel losses and improve eye opening.
*   **Interactive GUI:** (Assumes your front-end is a Tkinter or similar GUI) User-friendly graphical interface for easy parameter input and real-time visualization of results.
*   **Touchstone Export:** Exports S-parameter data in standard `.sNp` (Touchstone) format for interoperability with commercial EDA tools.

## Getting Started

### Prerequisites

*   Python 3.x
*   Required Python libraries: `numpy`, `scipy`, `matplotlib`
*   (If using Tkinter for GUI, it's usually included with Python standard distribution)

### Installation

1.  **Clone this repository** (or download the script files):
    ```bash
    git clone https://github.com/GetCupOfAmericano/TSV_SI_Analysis.git
    cd your-repo-name
    ```

2.  **Install the dependencies:**
    ```bash
    pip install numpy scipy matplotlib
    ```

### Usage

1.  **Run the main script:**
    ```bash
    python TSV_SI_Touchbench.py
    ```

2.  **Interactive GUI:**
    *   A graphical user interface (GUI) will appear.
    *   Input your desired TSV and signal parameters into the respective fields.
    *   Click the "Compute" or "Run Simulation" button to initiate the analysis.
    *   The results (S-parameters, RLGC, transient waveforms, eye diagrams) will be displayed directly within the GUI.

## Key Input Parameters & Their Impact

Understanding these parameters is crucial for effective use of the simulator:

*   **Analysis Mode (Single/Coupled):** Determines whether to simulate an isolated TSV or two adjacent, coupled TSVs. "Coupled" mode introduces crosstalk analysis (NEXT & FEXT).
*   **Number of TSVs:** Simulates the effect of cascading multiple TSVs, increasing total channel loss and delay.
*   **Height (μm), Radius (μm), T Ox (μm), Pitch (μm):** Core geometric parameters that directly influence R, L, C, G values and therefore channel loss, delay, and crosstalk.
*   **Data Rate (Gbps):** Defines the speed of the digital signal. Higher data rates demand wider signal bandwidth, leading to more severe high-frequency loss and increased eye closure.
    *   *Application:* Used to generate input bit sequence, determine frequency range for FFT/IFFT, and set the unit interval (UI) for eye diagram plotting.
*   **N_a (cm⁻³) - Substrate Doping Concentration:** The impurity concentration in the silicon substrate. Higher `N_a` leads to higher substrate conductivity, significantly increasing high-frequency `G` (conductance) and substrate loss.
    *   *Common Values:* Low-doped: ~1e14-1e15 cm⁻³; Medium-doped (common logic): ~1e15-1e16 cm⁻³; High-doped/Low-resistivity: ~1e17-1e19 cm⁻³.
*   **V_bias (V) - Bias Voltage & V_FB (V) - Flatband Voltage:** These parameters control the formation of a depletion region around the TSV. A positive `V_bias` (relative to `V_FB`) creates an insulating depletion zone, drastically reducing high-frequency substrate loss (`G`).
    *   *V_bias:* Typically 0V to chip supply voltage (e.g., 1.2V, 2.5V).
    *   *V_FB:* Small negative value, typically -1V to 0V.
*   **Temperature (C):** Affects copper resistivity (increasing R) and silicon conductivity (increasing G), leading to increased loss at higher temperatures.
*   **Roughness (μm):** Surface roughness of the conductor. At high frequencies, it increases effective resistance due to the skin effect.
*   **Tan Delta (Oxide) - Dielectric Loss Tangent:** A measure of energy dissipation in the oxide dielectric. Higher `Tan Delta` values indicate more dielectric loss (`G`), particularly significant at high frequencies.
    *   *Common Values for SiO2:* 0.001 - 0.005.
*   **Use TX Pre-emphasis & Tap Weight:** Enables a digital pre-compensation technique at the transmitter. Pre-emphasis boosts high-frequency components of the signal *before* it enters the lossy channel, helping to counteract channel attenuation and "open" the eye diagram at the receiver. `Tap Weight` controls the strength of this boost.

## Understanding Output Results

### S-Parameters (Frequency Domain)

*   **S11 (Reflection):** Indicates how much signal power is reflected back from the input port. Lower (more negative dB) is better.
*   **S31 (Insertion Loss/Thru):** Represents the signal power transmitted through the main channel (from input TSV's near end to its far end). Lower (more negative dB) values indicate higher loss.
*   **S21 (NEXT - Near-End Crosstalk):** The coupled noise observed at the *near end* of the victim TSV when signal is applied to the aggressor.
*   **S41 (FEXT - Far-End Crosstalk):** The coupled noise observed at the *far end* of the victim TSV when signal is applied to the aggressor.

### RLGC Parameters (Frequency Domain)

*   Plots showing the frequency dependency of Resistance (R), Inductance (L), Conductance (G), and Capacitance (C) per unit length of the TSV. These are the fundamental building blocks of the TSV channel model.

### Transient Waveforms (Time Domain)

*   **Input Waveform:** The ideal (or pre-emphasized) signal generated at the transmitter.
*   **Output Waveform:** The signal received at the far end of the main TSV after propagating through the channel. Shows attenuation, delay, and distortion.
*   **FEXT Waveform:** The noise induced on the victim TSV's far end due to crosstalk. This noise directly interferes with data if the victim line is also carrying a signal.

### Eye Diagram (Time Domain)

The ultimate visual representation of signal quality at the receiver:
*   **Eye Height:** Vertical opening of the eye, indicating the voltage margin for distinguishing '0's from '1's (noise margin).
*   **Eye Width:** Horizontal opening of the eye, indicating the time margin for reliable data sampling (timing margin/jitter tolerance).
*   A **wide and tall eye** signifies good signal integrity and low bit error rate. A **closed eye** indicates severe signal degradation and unreliable communication.


