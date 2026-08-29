# 🌍 PM2.5 Air Pollution Forecasting System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning-based spatiotemporal forecasting system for predicting PM2.5 air pollution concentrations using meteorological and emission data. The project implements a physics-informed ConvLSTM encoder-decoder architecture enhanced with multi-scale spatial feature extraction, wind-aware transport modeling, spatial attention, and episode-aware forecasting.

Originally developed for the ANRF AISE Hack PM2.5 Forecasting Challenge, the solution has been converted from a research notebook into a modular, reproducible, and production-ready Python codebase.

---

## ✨ Features

* 📈 16-step autoregressive PM2.5 forecasting
* 🧠 ConvLSTM encoder-decoder architecture
* 🌪️ WindWarp module for wind-driven pollutant transport
* 🎯 Spatial Attention mechanism
* 🔥 Episode Detection and Amplification modules
* 🏗️ Multi-scale Spatial Encoder using dilated convolutions
* ⚙️ Configuration-driven workflow using `params.yaml`
* 🚀 GPU-accelerated training and inference with Automatic Mixed Precision (AMP)
* 📦 Modular project structure for training and deployment
* 🔄 Reproducible training pipeline with checkpointing and early stopping

---

## 🏗️ Model Architecture

The forecasting pipeline consists of:

1. **Spatial Encoder**

   * Multi-scale dilated convolutions
   * Captures local and regional spatial patterns

2. **ConvLSTM Encoder**

   * Processes historical PM2.5 and meteorological observations
   * Learns spatiotemporal dependencies

3. **WindWarp Module**

   * Models pollutant transport using wind fields
   * Physics-inspired spatial warping mechanism

4. **Episode Detector**

   * Identifies potential pollution episode regions
   * Guides the decoder during forecasting

5. **Spatial Attention**

   * Focuses the model on important spatial regions

6. **Autoregressive Decoder**

   * Generates future PM2.5 forecasts step-by-step
   * Produces 16 forecast horizons

---

## 📂 Project Structure

```text
pm25-forecasting-system/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── dataset.py
│   │   └── episodes.py
│   │
│   ├── models/
│   │   └── phase2model.py
│   │
│   ├── training/
│   │   └── losses.py
│   │
│   └── inference/
│       └── predict.py
│
├── data/
│   ├── raw/
│   └── test_in/
│
├── models/
│   └── (generated model checkpoints)
│
├── artifacts/
│   └── (generated normalization statistics)
│
├── params.yaml
├── requirements.txt
├── train.py
└── README.md
```

---

## 🛠️ Tech Stack

**Programming Language**
- Python

**Deep Learning**
- PyTorch

**Scientific Computing**
- NumPy
- SciPy

**Configuration**
- YAML

**Acceleration**
- CUDA
- Automatic Mixed Precision (AMP)

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/07MEET/pm25-forecasting-system.git
cd pm25-forecasting-system
```

### Create Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Structure

Training data:

```text
data/raw/
├── APRIL_16/
├── JULY_16/
├── OCT_16/
└── DEC_16/
```

Inference data:

```text
data/test_in/
```

---

## 🏋️ Model Training

Generate normalization statistics and train the model:

```bash
python train.py
```

Training includes:

* Mixed Precision Training (AMP)
* Gradient Accumulation
* Gradient Clipping
* Learning Rate Warmup
* Cosine Annealing Scheduler
* Early Stopping
* Automatic Checkpoint Saving

Best model checkpoint:

```text
models/best_model_p2.pt
```

---

## 🔮 Inference

Run prediction on test data:

```bash
python -m src.inference.predict
```

Output:

```text
outputs/preds.npy
```

Shape:

```text
(218, 140, 124, 16)
```

--- 

## 📈 Results

| Property | Value |
|----------|-------|
| Forecast Horizon | 16 Steps |
| Input History | 10 Steps |
| Features | 15 |
| Spatial Resolution | 140 × 124 |
| Trainable Parameters | ~3.8 Million |
| Framework | PyTorch |
| Inference | GPU Accelerated |

The modular implementation reproduces the inference outputs of the original competition notebook while providing a clean, reusable, and production-oriented codebase.

---

## 🚀 Future Enhancements

* 🔄 FastAPI deployment (future enhancement)
* 🔄 Dockerization (future enhancement)
* 🔄 Experiment tracking (future enhancement)

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
