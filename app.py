#######################
# Import libraries
import streamlit as st
import requests
import pandas as pd
import altair as alt

#######################
# Page configuration
st.set_page_config(
    page_title="Proposal_3", 
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Initialize loading state
if 'loading' not in st.session_state:
    st.session_state.loading = False

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    st.title('Proposal_3')

    # Page Button
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. John Kenneth P. Alon\n2. Rob Eugene A. Dequinon\n3. Neil Andrew R. Mediavillo\n4. Benedict Ezekiel M. Martin\n5. Emmanuel D.  Villosa ")

#######################
#  Load Data 

def load_data():
    urls = [
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Arma_3.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Counter_Strike.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Counter_Strike_Global_Offensive.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Dota_2.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Football_Manager_2015.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Garrys_Mod.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Grand_Theft_Auto_V.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Sid_Meiers_Civilization_5.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Team_Fortress_2.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/The_Elder_Scrolls_V.jsonlines',
        'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Warframe.jsonlines'
    ]

    data = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            df = pd.read_json(response.text, lines=True)

            if not df.empty:
                data.append(df)
            else:
                st.warning(f"Warning: {url} contains no data.")
        except ValueError as e:
            st.error(f"Error reading data from {url}: {e}")
        except requests.RequestException as e:
            st.error(f"Request failed for {url}: {e}")

    if data:
        combined_df = pd.concat(data, ignore_index=True)
        return combined_df
    else:
        st.error("No valid data to concatenate.")
        return None

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    # Set loading state to True
    st.session_state.loading = True
    st.write("Loading data from multiple sources...")

    # Load the data
    combined_df = load_data()

    # Set loading state to False
    st.session_state.loading = False

    if combined_df is not None:
        st.write("Combined DataFrame created successfully.")
        st.dataframe(combined_df)  # Display the DataFrame
    else:
        st.write("No data available.")

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')

    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here