"""
About page for the Streamlit dashboard
"""
import streamlit as st

def show(df=None):
    """
    Display the about page with project information
    
    Creates an informational page with details about the project, methodology, and data sources.
    
    Parameters
    ----------
    df : pandas.DataFrame, optional
        DataFrame containing the analysis data, not used in this page but included
        for consistency with other page functions
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("About This Project")
    
    st.markdown("""
    ## Wet Bulb Temperature Analysis
    
    This project analyzes the relationship between wet-bulb temperature in Singapore and various climate variables,
    including greenhouse gases. The wet-bulb temperature is a critical measure of how well humans can cool down by
    sweating when it's hot and humid.

    ### Why Wet Bulb Temperature Matters
    
    The wet-bulb temperature (WBT) is the lowest temperature that can be reached by evaporating water into the air. 
    When the WBT exceeds 35°C, the human body can no longer cool itself through sweating, which can be fatal. 
    With climate change, parts of the world are approaching dangerous WBT levels.
    
    A normal internal human body temperature of 36.8° ± 0.5°C requires skin temperatures of around 35°C to maintain
    a gradient directing heat outwards from the core. Once the air (dry-bulb) temperature rises above 35°C, the human
    body can only cool down by sweating. But when the WBT exceeds 35°C, our ability to dissipate heat by sweating 
    breaks down entirely.
    
    ### Data Sources
    
    This analysis uses the following data:
    
    - **Wet Bulb Temperature Hourly** - Singapore hourly wet bulb temperature from 01/01/1982 to 31/05/2023
    - **Surface Air Temperature Monthly** - Singapore monthly mean surface air temperature from 01/1982 to 05/2023
    - **Climate Variables** - Singapore monthly data for rainfall, sunshine, and relative humidity from 01/1975 to 05/2023
    - **Greenhouse Gases** - Global monthly mean values for CO2, CH4, N2O, and SF6
    
    ### Analysis Methods
    
    The project includes:
    
    - Time series analysis of wet bulb temperature trends
    - Correlation analysis between climate variables and wet bulb temperature
    - Regression modeling to understand the relationships between variables
    - Seasonal pattern analysis of wet bulb temperature
    
    ### Project Structure
    
    This project is organized as a Python package with the following components:
    
    - **data_processing/** - Data loading and preprocessing modules
    - **utils/** - Custom statistical functions
    - **visualization/** - Data visualization tools
    - **models/** - Machine learning models
    - **features/** - Feature engineering tools
    - **app_pages/** - Streamlit dashboard pages
    
    ### Methodology
    
    The analysis follows these steps:
    
    1. **Data Collection**: Gathering data from government sources and climate databases
    2. **Data Cleaning**: Processing and standardizing the data for analysis
    3. **Exploratory Analysis**: Initial investigation of patterns and relationships
    4. **Feature Engineering**: Creating derived variables to improve analysis
    5. **Modeling**: Building regression models to understand relationships
    6. **Visualization**: Creating informative charts and graphs to communicate findings
    
    ### Future Considerations for Singapore
    
    Singapore is particularly vulnerable to climate change due to its tropical location and high humidity.
    The analysis of wet bulb temperature trends has critical implications for:
    
    - Public health and safety measures
    - Urban planning and infrastructure design
    - Workplace regulations for outdoor workers
    - Energy demand for cooling
    - Military and emergency response operations
    
    ### References
    
    - [Heat stress and human performance article](https://journals.physiology.org/doi/full/10.1152/japplphysiol.00738.2021)
    - [NASA Climate article on heat and habitability](https://climate.nasa.gov/explore/ask-nasa-climate/3151/too-hot-to-handle-how-climate-change-may-make-some-places-too-hot-to-live/)
    - [Singapore National Climate Change Strategy](https://www.nccs.gov.sg/)
    """)
    
    st.subheader("Project Contributors")
    
    st.markdown("""
    This project was developed as part of a data science portfolio project focusing on climate data analysis.
    
    **Technologies Used:**
    - Python for data processing and analysis
    - Pandas and NumPy for data manipulation
    - Matplotlib and Seaborn for visualization
    - Scikit-learn for machine learning models
    - Streamlit for interactive dashboard
    """)
    
    # Add contact information
    with st.expander("Contact Information"):
        st.write("""
        For questions or feedback about this project, please contact:
        - Email: weslee.qb@gamil.com
        - GitHub: https://github.com/Adredes-weslee
        """)
        
    # Add data acknowledgments
    with st.expander("Data Acknowledgments"):
        st.write("""
        Data for this project was sourced from:
        - [data.gov.sg](https://data.gov.sg/) - Singapore government open data
        - [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/) - Greenhouse gas data
        """)
        
    # Display project version
    st.sidebar.info("Project Version: 1.0.0")
