
# 🌡️ Predicting Wet-Bulb Temperature with Climate Variables  
> Exploring climate change implications for Singapore’s heat stress risk using regression analysis

## 📘 Overview  
Commissioned as a hypothetical policy study for the Singapore government, this project investigates the **relationship between wet-bulb temperature (WBT)**—a crucial indicator of heat stress—and climate change drivers such as greenhouse gases and meteorological factors. Using time-series regression modeling, we aim to identify key contributors to extreme heat conditions in tropical environments.

## 📌 Objective  
- To model and predict WBT in Singapore using multivariate regression  
- To assess the impact of greenhouse gases and meteorological variables on heat stress  
- To derive actionable public health and policy recommendations

## 📂 Data Dictionary  
This study integrates 7 datasets from [Data.gov.sg](https://data.gov.sg) and [NOAA](https://gml.noaa.gov):

| Feature                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `month`                    | Monthly timestamp in `YYYY-MM` format                                       |
| `mean_surface_airtemp`     | Mean surface air temperature (°C)                                           |
| `mean_wet_bulb_temperature`| Derived monthly WBT from hourly readings (°C)                               |
| `total_rainfall`           | Total rainfall (mm)                                                         |
| `daily_mean_sunshine`      | Daily mean sunshine hours                                                   |
| `mean_relative_humidity`   | Mean relative humidity (%)                                                  |
| `average_co2_ppm`          | Atmospheric CO₂ concentration (ppm)                                         |
| `average_ch4_ppb`          | Atmospheric CH₄ concentration (ppb)                                         |
| `average_n2o_ppb`          | Atmospheric N₂O concentration (ppb)                                         |
| `average_sf6_ppt`          | Atmospheric SF₆ concentration (ppt)                                         |

## 🧪 Methods  

### 🧭 Exploratory Data Analysis (EDA)  
- Correlation matrices and time-series visualizations  
- Seasonal decomposition of WBT and meteorological variables  
- Outlier detection and trend profiling

### 🛠 Feature Engineering  
- Time alignment and cleaning of multi-source datasets  
- Lag variables to account for delayed atmospheric effects  
- Standardization of greenhouse gas units for integration

### 📊 Modeling & Evaluation  
- Trained a **Multiple Linear Regression** model to predict WBT  
- Evaluated via **R² score**, **RMSE**, and residual diagnostics  
- Assessed feature importance and multicollinearity patterns

## 📈 Key Findings  
- **Positive correlation with WBT:** Mean air temperature, nitrous oxide (N₂O), sulfur hexafluoride (SF₆), sunshine, and rainfall  
- **Negative correlation with WBT:** Relative humidity  
- Greenhouse gases exhibit high multicollinearity, reflecting shared anthropogenic sources  
- No clear year-over-year WBT trend, but potential rise in **extreme values** linked to compound heat effects

## 🧠 Interpretation & Policy Implications  
- Climate change is **altering the heat-humidity dynamics** critical to human survivability  
- The **reduction in relative humidity**, while seemingly benign, may exacerbate heat stress under rising air temperatures  
- Policy actions may include:
  - Integrating WBT into **heatwave early warning systems**
  - **Public education** on wet-bulb safety thresholds
  - Tracking WBT alongside **CO₂-equivalent indices**

## 🚀 Future Work  
- Expand models to include non-linear regressors (Random Forest, XGBoost)  
- Integrate with **real-time APIs** for continuous monitoring  
- Cross-reference with **public health data** (e.g., ER visits, heat stroke rates)

## 🔍 Tools & Libraries  
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- Jupyter Notebook

## 📁 Files  
- `data_analysis_of_wet_bulb_temperature.ipynb`: full analysis notebook  
- `README.md`: project documentation and synthesis  

## 📚 References  
### 📊 Data Sources  
- [Wet-Bulb Temperature (Hourly) – data.gov.sg](https://data.gov.sg/dataset/wet-bulb-temperature-hourly)  
- [Surface Air Temperature (Monthly Mean) – data.gov.sg](https://data.gov.sg/dataset/surface-air-temperature-monthly-mean)  
- [Rainfall, Sunshine, Humidity – SingStat (Table M890081)](https://tablebuilder.singstat.gov.sg/table/TS/M890081)  
- [Greenhouse Gas Trends – NOAA (CO₂)](https://gml.noaa.gov/ccgg/trends/data.html)  
- [Methane Trends – NOAA](https://gml.noaa.gov/ccgg/trends_ch4/)  
- [Nitrous Oxide Trends – NOAA](https://gml.noaa.gov/ccgg/trends_n2o/)  
- [Sulfur Hexafluoride Trends – NOAA](https://gml.noaa.gov/ccgg/trends_sf6/)  

### 🌱 Scientific References and Background  
- [OpenStax Biology – Homeostasis](https://openstax.org/books/biology-2e/pages/33-3-homeostasis)  
- [Wet Bulb & Dry Bulb Illustration – Khan Academy](https://cdn.kastatic.org/ka-perseus-images/69ad70de87b8e05dcbd1708d4f48dc176e1276e9.png)  
- [Encyclopedia of Environmental Science – Wet-Bulb Temp](https://link.springer.com/referenceworkentry/10.1007/1-4020-3266-8_94)  
- [NOAA Heat Index Chart](https://www.weather.gov/images/safety/heatindexchart-650.jpg)  
- [LBL – Underestimating Heat Waves](https://newscenter.lbl.gov/2022/08/24/no-more-underestimating-heat-waves/)  
- [NOAA Wet Bulb Definition](https://www.weather.gov/source/zhu/ZHU_Training_Page/definitions/dry_wet_bulb_definition/dry_wet_bulb.html)  
- [PhysicsCalc – Wet Bulb Calculator](https://physicscalc.com/physics/wet-bulb-calculator/)  
- [The Guardian – Why WBT Matters](https://www.theguardian.com/science/2022/jul/31/why-you-need-to-worry-about-the-wet-bulb-temperature)  
- [NIH – Wet Bulb Temperature & Health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7209987/)  
- [Washington Post – Extreme Heat & WBT](https://www.washingtonpost.com/weather/2021/07/24/wet-bulb-temperature-extreme-heat/)  
- [NASA Climate – Heat Beyond Tolerance](https://climate.nasa.gov/explore/ask-nasa-climate/3151/too-hot-to-handle-how-climate-change-may-make-some-places-too-hot-to-live/)  
- [Washington Post – Explainer on WBT Mortality](https://www.washingtonpost.com/business/energy/2023/07/04/explainer-how-extreme-heat-and-high-wet-bulb-temperatures-kill-people/83bca63e-1ada-11ee-be41-a036f4b098ec_story.html)  
- [Journal of Applied Physiology – Heat Stress Studies](https://journals.physiology.org/doi/full/10.1152/japplphysiol.00738.2021)  
- [Phys.org – Fatal Combinations of Heat & Humidity](https://phys.org/news/2020-05-potentially-fatal-combinations-humidity-emerging.html)  

### 🧠 Broader Climate & Policy Context  
- [NRDC – Greenhouse Effect 101](https://www.nrdc.org/stories/greenhouse-effect-101#gases)  
- [EPA – Climate Change Basics](https://www.epa.gov/climatechange-science/basics-climate-change)  
- [Scientific American – What Causes Humidity?](https://www.scientificamerican.com/article/what-causes-humidity/)  
- [JPL NASA – Understanding Air & Water](https://sealevel.jpl.nasa.gov/ocean-observation/understanding-climate/air-and-water/)  
- [CNA – Singapore & 40°C Heat Risk](https://www.channelnewsasia.com/singapore/singapore-weather-40-degrees-celsius-heatwave-global-warming-aircon-3597176)  
- [NOAA – Labor Capacity & Heat Stress](https://www.gfdl.noaa.gov/research_highlight/heat-stress-reduces-labor-capacity-under-climate-warming/)  
- [ILO – Climate Risk for Workers](https://www.ilo.org/global/about-the-ilo/newsroom/news/WCMS_794475/lang--en/index.htm)  
- [Straits Times – NUS Study on Heat Stress](https://www.straitstimes.com/singapore/environment/nus-scientists-to-study-construction-workers-risk-of-heat-stress)  
- [OmniCalculator – Wet Bulb Calculator](https://www.omnicalculator.com/physics/wet-bulb)  
- [Carbon Brief – Humidity Paradox](https://www.carbonbrief.org/guest-post-investigating-climate-changes-humidity-paradox/)  
- [Singapore’s 2nd Nat’l Climate Change Study (ResearchGate)](https://www.researchgate.net/publication/339398733_Singapore%27s_Second_National_Climate_Change_Study_Climate_Projections_to_2100_-_Report_for_Stakeholders)  
- [IISD – Precautionary Principle](https://www.iisd.org/articles/deep-dive/precautionary-principle)  
- [Harvard – Tragedy of the Commons](https://online.hbs.edu/blog/post/tragedy-of-the-commons-impact-on-sustainability-issues)
---

**Author:** Wes Lee  
🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee) · 💻 Portfolio available upon request  
📜 License: MIT
