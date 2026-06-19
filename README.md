# 🌍 PM2.5 Air Pollution Forecasting System

A production-oriented deep learning project for forecasting PM2.5 concentrations using spatiotemporal meteorological and emission data. The system is based on a ConvLSTM encoder-decoder architecture enhanced with multi-scale spatial encoding, attention mechanisms, wind-aware transport modeling, and autoregressive forecasting.

The project is being refactored from a research notebook into a modular, configurable, and deployment-ready Python codebase.

## ✨ Features

- 📈 16-hour PM2.5 forecasting
- 🧠 ConvLSTM-based spatiotemporal deep learning model
- 🌪️ Wind-aware transport (WindWarp) module
- 🎯 Spatial attention and episode detection
- ⚙️ Configuration-driven pipeline using `params.yaml`
- 📦 Modular project structure for training and inference
- 🚀 FastAPI deployment support (planned)
- 🐳 Docker and MLOps integration (planned)

## 📂 Project Structure

```
pm25-air-pollution-forecasting/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── api/
│
├── data/
│   ├── raw/
│   ├── test_in/
│   └── stats/
│
├── models/
├── artifacts/
├── tests/
│
├── params.yaml
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- SciPy
- FastAPI
- YAML Configuration
- Docker *(planned)*
- MLflow *(planned)*
- DVC *(planned)*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/pm25-air-pollution-forecasting.git
cd pm25-air-pollution-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Organize the dataset

Place the competition data under:

```
data/
└── raw/
    ├── APRIL_16/
    ├── JULY_16/
    ├── OCT_16/
    └── DEC_16/
```

Test inputs should be stored in:

```
data/test_in/
```

### 5. Generate normalization statistics

```bash
python src/data/loader.py
```

This creates:

```
artifacts/norm_stats.pkl
```

### 6. Run inference

```bash
python src/inference/predict.py
```

## 📌 Project Status

- ✅ Notebook implementation completed
- ✅ Trained model available
- 🚧 Refactoring into modular package
- 🚧 FastAPI integration
- 🚧 Docker support
- 🚧 MLflow experiment tracking
- 🚧 DVC pipeline

## 📄 License

This project is intended for educational and research purposes.
