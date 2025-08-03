# Neural Dynamics of Object Recognition Under Sensory Uncertainty

*Research Pproject for the Neuromatch Academy's Computational Neuroscience Course.*

This repository contains the complete Python pipeline for analyzing intracranial electrocorticography (ECoG) data to investigate how the human brain represents visual objects under varying levels of sensory noise. The project uses a combination of traditional signal analysis (High-Gamma Activity, ERPs) and machine learning (Multivariate Pattern Analysis - MVPA) to characterize the spatio-temporal dynamics of face and house perception.

## 🎯 Project Goal

The primary goal of this research is to map the evolution of categorical information in the brain and understand how these neural codes are systematically altered by sensory uncertainty. We analyze ECoG data from subjects performing a visual task where they identify images of faces and houses obscured by different levels of noise.

---

## 📈 Key Findings

Our analysis reveals a robust yet nuanced process of object recognition in the brain. While neural representations are highly specialized, they are impacted by noise in distinct ways.

### 1. Selective High-Gamma Activity (HGA)

Initial qualitative analysis of high-gamma activity (HGA) shows a strong, localized neural response to specific visual categories. As hypothesized, certain cortical areas are highly selective, with some channels showing a dramatic increase in activity specifically for face stimuli, and others for houses.

*The figure below shows the average broadband response across 50 ECoG channels for one subject. Note the clear selectivity: channel **ECOG_047** responds strongly to faces (orange line), while channel **ECOG_031** responds to houses (blue line).*

<p align="center">
  <img src="https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty/blob/master/High%20Gamma%20Activity%20%26%20ERP%20Plots/SubjectIndx1%20-%20NoiseLevel%20All.png" width="800">
</p>

### 2. The "Breaking Point" in Neural Decoding

Using a sliding-window MVPA approach with an SVM classifier, we decoded the object category (face vs. house) from the neural data at different noise levels. Our results show that the brain's ability to represent these categories is remarkably resilient up to a certain point, after which it degrades abruptly.

- **Decreasing Accuracy with Noise**: As sensory noise increases, the peak decoding accuracy systematically decreases. The neural representation is strongest in low-noise conditions and weakens as the stimulus becomes more ambiguous.

- **The 40-50% Noise "Breaking Point"**: Neural representation does not degrade linearly. Instead, it remains significantly above chance until the noise level approaches a 40-50% threshold. Beyond this point, the decoding accuracy drops abruptly to chance level, suggesting a "breaking point" in the evidence accumulation process.

*Left: Decoding accuracy curves for different noise bins. Note that the purple (0-20% noise) and blue (20-40% noise) curves show significant decoding, while higher noise levels do not. Right: Peak decoding accuracy drops as noise increases, becoming non-significant (red squares) after the 0.3 noise level center.*

<p align="center">
  <img src="https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty/blob/master/Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_comparison.png" width="400">
  <img src="https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty/blob/master/Across%20Noise%20Levels%20Across%20Subjects%20Results/PeakDecodingAccuracy%20vs%20NoiseLevel.png" width="400">
</p>

This abrupt cutoff is also visible in the channel-wise MVPA results for a single subject, where the robust decoding seen in low-noise bins (1 and 2) vanishes in higher-noise bins.

<p align="center">
  <img src="https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty/blob/master/Across%20Noise%20Levels%20Across%20Subjects%20Results/photo_2025-08-03_06-38-05.jpg" width="800">
</p>

### 3. Time Delay in Neural Processing

Increased sensory noise also introduces a delay in the peak neural representation. The grand-average heatmap across all subjects shows that as noise increases, the "hotspot" of maximum decoding accuracy not only gets weaker but also shifts rightward in time. This indicates that the brain requires more time to accumulate sufficient evidence to represent the object when the signal is noisy.

*Decoding accuracy heatmap averaged across subjects. The peak accuracy (yellow/orange) occurs later for the 0.2-0.4 noise level compared to the 0.0-0.2 level, demonstrating a temporal delay in processing.*

<p align="center">
  <img src="https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty/blob/master/Across%20Noise%20Levels%20Across%20Subjects%20Results/noise_levels_heatmap.png" width="700">
</p>

---

## 📂 Repository Structure

This repository is organized into several key scripts:

| File                               | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                          | **Main executable script.** Configures and runs the entire MVPA pipeline across all specified noise ranges and subjects.              |
| `utils.py`                         | Core utility functions, including data loading, preprocessing, and the `MVPAAnalyzer` class which implements the decoding analysis. |
| `generate_visualizations.py`       | A standalone script to generate summary plots and visualizations (Levels 1, 2, and 3) from the saved results of the `main.py` pipeline. |
| `Highy Gamma Activity & ERP Analysis.py` | Script for exploratory data analysis, including plotting HGA and ERPs for specific channels and noise levels.               |
| `Loading & Preprocessing Functions.py` | Contains helper functions for loading and preprocessing the ECoG data, designed to be make preprocessing pipeline easier to understand.              |
| `requirements.txt`                 | A list of all the necessary Python packages to run the code in this repository.                                                           |

---

## 🚀 How to Use

Follow these steps to replicate the analysis.

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Data Source

The raw ECoG data (`faceshouses.npz`) used in this project can be downloaded from the Stanford Digital Repository:
- **Link**: [https://exhibits.stanford.edu/data/catalog/zk881ps0522](https://exhibits.stanford.edu/data/catalog/zk881ps0522)

### 3. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/mohammadrezashahsavari/ECoG-Object-Recognition-Uncertainty.git
cd ECoG-Object-Recognition-Uncertainty
```

Next, install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Running the MVPA Pipeline

The entire MVPA is controlled by the `main.py` script.

1.  **Update the Data Path**: Open `main.py` and **you must update the file path** in the `config` dictionary to point to the location where you saved `faceshouses.npz`.

    ```python
    # In main.py
    config = {
        # ...
        'filepath': r'C:\path\to\your\data\faceshouses.npz',
        # ...
    }
    ```

2.  **Configure the Analysis**: You can modify the rest of the `config` dictionary to set your desired parameters. This includes preprocessing steps, MVPA windowing parameters, or processing mode (`ecog` vs. `high_gamma`).

3.  **Define Noise Bins**: Modify the `noise_ranges_to_analyze` list to specify which noise intervals you want to analyze. The script will run a separate, complete analysis for each entry.

    ```python
    # In main.py
    noise_ranges_to_analyze = [
        [0.0, 0.2],
        [0.2, 0.4],
        # ... and so on
    ]
    ```

4.  **Execute the Script**: Run the analysis from your terminal.

    ```bash
    python main.py
    ```

The script will create output directories based on the configuration for each noise range (e.g., `mvpa_results - ecog - ... - noise0.00to0.20`). If results for a subject already exist, that subject will be skipped to allow for easy resumption of a long analysis.

### 5. Generating Summary Visualizations

After the MVPA pipeline has finished and the result files (`.pkl`) are generated, you can create the summary figures.

1.  **Configure the Script**: Open `generate_visualizations.py` and set the `base_results_dir` to the parent directory containing all your analysis folders. In most cases, this will just be the project's root directory.

    ```python
    # In generate_visualizations.py
    base_results_dir = r"."
    ```

2.  **Execute the Script**: Run the script from your terminal.

    ```bash
    python generate_visualizations.py
    ```

This will create a new folder named `NoiseRange_Visualizations` containing three levels of summary plots, including the heatmaps and comparison plots shown in the findings section.

---
## 🛠️ Dependencies

The main libraries used in this project are:
- `numpy`
- `mne`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `nilearn`
- `plotly`

