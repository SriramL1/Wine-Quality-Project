# description.py
import streamlit as st

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="Wine Quality Analysis", page_icon="🍷", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    div[data-testid="stAppViewContainer"] > div {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    .main-title {
        font-size: 36px !important;
        color: rgb(241, 22, 132) !important;
        text-align: center !important;
        margin-top: 0 !important;
        margin-bottom: 10px !important;
    }
    .subtitle {
        font-size: 18px !important;
        color: rgb(205, 127, 127) !important;
        text-align: center !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stSidebar"] {
        background-color: rgb(200, 160, 250);
    }
    body {
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<h1 class="main-title">Welcome to Wine Quality Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explore and predict wine quality with data-driven insights</p>', unsafe_allow_html=True)

# Layout with columns
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    st.write("""
    This app provides two powerful tools to analyze wine quality data:
    - **Correlation Analysis**: Discover relationships between wine features like acidity, alcohol, and quality.
    - **Quality Prediction**: Use a machine learning model to predict wine quality based on your input.
    Navigate using the sidebar on the left to get started!
    """)

with col2:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTOTZUFEKh8Jzo4bBA_A5a7YqL7z2GXeR-z_g&s", 
             caption="Wine Picture", use_container_width=True)



# Footer with Kaggle link
st.markdown("---")
st.write("""
### About This App
Built with Streamlit and Google BigQuery, this app leverages the Wine Quality dataset to provide actionable insights. 
Whether you're a wine enthusiast or a data scientist, dive into the numbers behind the flavors!
""")

if st.button("Learn More About Wine Quality Data", key="learn_more"):
    st.markdown("""
    Check out the [Kaggle Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data) 
    for more details on the data used in this app.
    """, unsafe_allow_html=True)