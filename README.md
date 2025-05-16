# 🌡️ Wet Bulb Temperature Analysis

This project analyzes the relationship between wet-bulb temperature in Singapore and various climate variables, including greenhouse gases. Wet bulb temperature is a crucial indicator that combines temperature and humidity to measure how effectively the human body can cool through sweating.

## ⚡ Quick Start Guide

New to this project? Here's how to get started in minutes:

1. **Setup Environment**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd wet-bulb-temperature-analysis

   # Create and activate environment (choose one)
   python -m venv venv && venv\Scripts\activate  # Option 1: Using pip
   # OR
   conda env create -f environment.yml && conda activate wet-bulb-temp  # Option 2: Using conda

   # Install dependencies
   pip install -r requirements.txt  # If using pip
   ```

2. **Verify Setup**
   ```bash
   python scripts/verify_environment.py
   ```

3. **Process Data**
   ```bash
   python scripts/preprocess_data.py
   ```

4. **Launch Dashboard**
   ```bash
   python run_dashboard.py
   ```
   Then open your browser at http://localhost:8501

5. **Run Sample Analysis**
   ```bash
   python scripts/analyze.py
   ```

6. **Explore Notebooks**
   - View `notebooks/project_evolution.ipynb` for a guided introduction to the project
   - Check `notebooks/sample_analysis.ipynb` for example usage of the modules

## 📘 Overview

Commissioned as a hypothetical policy study for the Singapore government, this project investigates the **relationship between wet-bulb temperature (WBT)**—a crucial indicator of heat stress—and climate change drivers such as greenhouse gases and meteorological factors. Using time-series regression modeling, we aim to identify key contributors to extreme heat conditions in tropical environments.

## 📌 Background & Objective

The wet-bulb temperature (WBT) is the lowest temperature that can be reached by evaporating water into the air. When WBT exceeds 35°C, the human body can no longer cool itself through sweating, which can be fatal. With climate change, parts of the world are approaching dangerous WBT levels.

Singapore's tropical climate and high humidity make it particularly vulnerable to high wet bulb temperatures. This analysis helps identify trends and relationships that could inform climate adaptation and mitigation policies.

**Key Objectives**:
- To model and predict WBT in Singapore using multivariate regression  
- To assess the impact of greenhouse gases and meteorological variables on heat stress  
- To derive actionable public health and policy recommendations

## 📊 Features

- **Data Processing**: Automated scripts to clean and prepare climate datasets
- **Exploratory Analysis**: Visualizations and statistical tools to analyze wet bulb temperature patterns
- **Time Series Analysis**: Tools for decomposing time series and identifying trends
- **Correlation Analysis**: Methods for examining relationships between climate variables
- **Regression Modeling**: Linear regression models to understand the impact of different variables
- **Interactive Dashboard**: Streamlit application for exploring the data and models

## 📂 Project Structure

```
wet-bulb-temperature-analysis/
├── dashboard/           # Streamlit dashboard application
│   └── app.py           # Main Streamlit app
├── data/                # Data files
│   ├── raw/             # Original data files
│   ├── processed/       # Cleaned data
│   └── output/          # Results and visualizations
├── notebooks/           # Jupyter notebooks
│   ├── data_analysis_of_wet_bulb_temperature.ipynb  # Original analysis notebook
│   └── sample_analysis.ipynb                        # Generated sample notebook
├── scripts/             # Utility scripts
│   ├── analyze.py                  # Run sample analysis
│   ├── preprocess_data.py          # Process raw data files
│   ├── create_sample_notebook.py   # Generate sample notebook
│   └── verify_environment.py       # Check environment setup
├── src/                 # Source code modules
│   ├── app_pages/       # Modular Streamlit pages
│   ├── data_processing/ # Data loading functions
│   ├── features/        # Feature engineering
│   ├── models/          # Regression models
│   ├── utils/           # Custom statistical functions
│   └── visualization/   # Plotting functions
├── run_dashboard.py     # Script to run the Streamlit dashboard
├── requirements.txt     # Python dependencies (pip)
├── environment.yml      # Python dependencies (conda)
└── README.md            # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Conda (optional, for environment setup)

### Installation

#### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd wet-bulb-temperature-analysis

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Using conda

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate wet-bulb-temp
```

### Verify Environment Setup

```bash
# Run the verification script to make sure everything is set up correctly
python scripts/verify_environment.py
```

### Preprocess Data

```bash
# Process the raw data files and generate the analysis dataset
python scripts/preprocess_data.py
```

### Running the Dashboard

```bash
# Run using the convenience script
python run_dashboard.py

# Alternatively, run Streamlit directly
streamlit run dashboard/app.py
```

## 🧭 Using the Dashboard

The interactive dashboard includes several pages:

1. **Home**: Overview of the wet bulb temperature analysis with key metrics
2. **Data Explorer**: Tools to explore the datasets and examine variable distributions
3. **Time Series Analysis**: Analysis of trends and seasonal patterns in wet bulb temperature
4. **Correlation Analysis**: Examination of relationships between climate variables
5. **Regression Modeling**: Build and evaluate models to predict wet bulb temperature
6. **About**: Information about the project and methodology

## 📊 Data Dictionary

This study integrates datasets from [Data.gov.sg](https://data.gov.sg) and [NOAA](https://gml.noaa.gov):

| Feature                     | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `month`                     | Monthly timestamp in `YYYY-MM` format                        |
| `mean_surface_airtemp`      | Mean surface air temperature (°C)                            |
| `mean_wet_bulb_temperature` | Derived monthly WBT from hourly readings (°C)                |
| `total_rainfall`            | Total rainfall (mm)                                          |
| `daily_mean_sunshine`       | Daily mean sunshine hours                                    |
| `mean_relative_humidity`    | Mean relative humidity (%)                                   |
| `average_co2_ppm`           | Atmospheric CO₂ concentration (ppm)                          |
| `average_ch4_ppb`           | Atmospheric CH₄ concentration (ppb)                          |
| `average_n2o_ppb`           | Atmospheric N₂O concentration (ppb)                          |
| `average_sf6_ppt`           | Atmospheric SF₆ concentration (ppt)                          |

## 🧪 Methods & Key Findings

### Exploratory Data Analysis (EDA)
- Correlation matrices and time-series visualizations
- Seasonal decomposition of WBT and meteorological variables
- Outlier detection and trend profiling

### Feature Engineering
- Time alignment and cleaning of multi-source datasets
- Lag variables to account for delayed atmospheric effects
- Standardization of greenhouse gas units for integration

### Modeling & Evaluation
- Trained a **Multiple Linear Regression** model to predict WBT
- Evaluated via **R² score**, **RMSE**, and residual diagnostics
- Assessed feature importance and multicollinearity patterns

### Key Findings
- **Positive correlation with WBT:** Mean air temperature, nitrous oxide (N₂O), sulfur hexafluoride (SF₆), sunshine, and rainfall
- **Negative correlation with WBT:** Relative humidity
- Greenhouse gases exhibit high multicollinearity, reflecting shared anthropogenic sources
- No clear year-over-year WBT trend, but potential rise in **extreme values** linked to compound heat effects

## 📂 Project Modules

### Data Processing
- `data_loader.py`: Functions to load and preprocess data
- `preprocess_data.py`: Script to run the complete preprocessing workflow

### Visualization
- `exploratory.py`: Functions for creating visualizations
- Interactive visualizations in the Streamlit dashboard

### Analysis
- `statistics.py`: Custom statistical functions
- `regression.py`: Functions for regression modeling and evaluation

### Feature Engineering
- `feature_engineering.py`: Functions to create derived features from the data

## 🧠 Notebooks vs Scripts

The project contains both notebooks and scripts:

1. **Original Analysis Notebook** (`notebooks/data_analysis_of_wet_bulb_temperature.ipynb`):  
   The initial exploratory data analysis with detailed explanations and visualizations.

2. **Sample Notebook** (`notebooks/sample_analysis.ipynb`):  
   A streamlined version generated using `scripts/create_sample_notebook.py` that demonstrates key analysis techniques.

3. **Analysis Scripts** (`scripts/analyze.py`, `scripts/preprocess_data.py`):  
   Command-line scripts that implement the analysis pipeline for automation and reproducibility.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Singapore Meteorological Service for climate data
- NOAA Global Monitoring Laboratory for greenhouse gas data
- All contributors to the open-source libraries used in this project

## 📚 References

### 📊 Data Sources
- [Wet-Bulb Temperature (Hourly) – data.gov.sg](https://data.gov.sg/dataset/wet-bulb-temperature-hourly)
- [Surface Air Temperature (Monthly Mean) – data.gov.sg](https://data.gov.sg/dataset/surface-air-temperature-monthly-mean)
- [Rainfall, Sunshine, Humidity – SingStat (Table M890081)](https://tablebuilder.singstat.gov.sg/table/TS/M890081)
- [Greenhouse Gas Trends – NOAA](https://gml.noaa.gov/ccgg/trends/data.html)

### 🌱 Selected Scientific References
- [Encyclopedia of Environmental Science – Wet-Bulb Temp](https://link.springer.com/referenceworkentry/10.1007/1-4020-3266-8_94)
- [NIH – Wet Bulb Temperature & Health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7209987/)
- [Journal of Applied Physiology – Heat Stress Studies](https://journals.physiology.org/doi/full/10.1152/japplphysiol.00738.2021)
- [CNA – Singapore & 40°C Heat Risk](https://www.channelnewsasia.com/singapore/singapore-weather-40-degrees-celsius-heatwave-global-warming-aircon-3597176)
