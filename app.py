#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import requests

#######################
# Page configuration
st.set_page_config(
    page_title="Steam Game Reviews", 
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    st.title('Steam Game Reviews')

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
    st.subheader("Abstract")
    st.write("A Streamlit dashboard highlighting the results of multiple Steam reviews using the datasets from Kaggle.")
    st.subheader("Members")
    st.markdown("1. John Kenneth P. Alon\n2. Rob Eugene A. Dequinon\n3. Benedict Ezekiel M. Martin\n4. Neil Andrew R. Mediavillo\n5. Emmanuel D.  Villosa ")

#######################
# Datasets

# URLs for each game dataset
arma_3_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Arma_3.jsonlines'
cs_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Counter_Strike.jsonlines'
csgo_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Counter_Strike_Global_Offensive.jsonlines'
dotes_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Dota_2.jsonlines'
football_manager_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Football_Manager_2015.jsonlines'
gmod_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Garrys_Mod.jsonlines'
gtaV_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Grand_Theft_Auto_V.jsonlines'
sid_meiers_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Sid_Meiers_Civilization_5.jsonlines'
tf2_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Team_Fortress_2.jsonlines'
elder_scrolls5_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/The_Elder_Scrolls_V.jsonlines'
warframe_url = 'https://raw.githubusercontent.com/mulhod/steam_reviews/refs/heads/master/data/Warframe.jsonlines'

jsonl_files = [arma_3_url, cs_url, csgo_url, dotes_url, football_manager_url, gmod_url, gtaV_url, sid_meiers_url, tf2_url, elder_scrolls5_url, warframe_url]

data = []
for url in jsonl_files:
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Load JSONL data into a DataFrame
        df = pd.read_json(response.text, lines=True)

        # Append the DataFrame to the list if it's not empty
        if not df.empty:
            data.append(df)
        else:
            print(f"Warning: {url} contains no data.")
    except ValueError as e:
        print(f"Error reading data from {url}: {e}")
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")

# Concatenate DataFrames if data is available
if data:
    combined_df = pd.concat(data, ignore_index=True)
    print("Combined DataFrame created successfully.")
else:
    combined_df = pd.DataFrame()  # If no data, initialize as empty DataFrame
    print("No valid data to concatenate.")

#######################
# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("""
    The Steam game reviews dataset includes data from several popular video games such as Arma 3, Counter-Strike, Dota 2, Grand Theft Auto V, and others. 
    These datasets, collected from Steam user reviews, provide valuable insights into player sentiment and game performance. 
    The dataset includes various features, such as review sentiment (positive or negative), review text, and user-related information.
    """)

    st.subheader("Content")
    st.write("""
    The combined dataset consists of several JSONL files from various games, each containing player reviews, their sentiment, and game-related details.
    These games include:

    - **Arma 3**
    - **Counter-Strike**
    - **Counter-Strike: Global Offensive**
    - **Dota 2**
    - **Football Manager 2015**
    - **Garry's Mod**
    - **Grand Theft Auto V**
    - **Sid Meier's Civilization V**
    - **Team Fortress 2**
    - **The Elder Scrolls V**
    - **Warframe**
    """)

    st.write("Link: [Steam Reviews Dataset on GitHub](https://github.com/mulhod/steam_reviews)")

    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(combined_df)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write("""
    The results from `df.describe()` provide some basic insights into the dataset, showing the distribution of numerical features like 
    review length, helpfulness ratings, and sentiment scores across different games. These statistical summaries help identify trends in 
    user feedback and highlight any variations between games.
    """)
    
    # Display Descriptive Statistics Table
    st.dataframe(combined_df.describe())

    st.write("""
    The results from `df.describe()` highlight the distribution of numerical columns like review length and helpfulness score. 
    For example, the **review length** column averages {:.2f} characters, with a standard deviation of {:.2f}, indicating moderate variation in how long users' reviews are. 
    **Helpfulness ratings**, on the other hand, have a lower mean of {:.2f} and a larger standard deviation of {:.2f}, suggesting some reviews are much more helpful than others.
    """.format(
        combined_df["review_length"].mean(), combined_df["review_length"].std(),
        combined_df["helpfulness_score"].mean(), combined_df["helpfulness_score"].std()
    ))

    # Calculating min and max values for specific insights
    min_review_length = combined_df["review_length"].min()
    max_review_length = combined_df["review_length"].max()
    min_helpfulness = combined_df["helpfulness_score"].min()
    max_helpfulness = combined_df["helpfulness_score"].max()

    # Displaying Minimum and Maximum Value Insights
    st.write("""
    Speaking of minimum and maximum values, **review length** ranges from {:.1f} characters up to {:.1f} characters, 
    and **helpfulness score** ranges from {:.1f} to {:.1f}, indicating significant variability across the dataset.
    """.format(min_review_length, max_review_length, min_helpfulness, max_helpfulness))

    # Displaying Percentile Insights
    st.write("""
    The 25th, 50th, and 75th percentiles reveal a general trend in the length and helpfulness of reviews, 
    suggesting that more detailed and helpful reviews tend to appear as the dataset grows.
    """)

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

# Decision Tree Classifier Explanation
if st.session_state.page_selection == "machine_learning":
    st.header("Decision Tree Classifier")

    # Display Explanation
    st.write("""
    **Decision Tree Classifier** from Scikit-learn library is a machine learning algorithm that is used primarily for classification tasks.
    Its goal is to categorize data points into specific classes. The process involves breaking down data into smaller subsets 
    based on questions which creates a "Tree" structure with each node representing a decision point.
    """)
    st.write("Reference: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)")

    # Show Training Code (if relevant)
    st.subheader("Training the Decision Tree Classifier")
    st.code("""
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    """)

    # Evaluation Code
    st.subheader("Model Evaluation")
    st.code("""
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    """)
    st.write("**Accuracy:** 100%")

    # Feature Importance
    st.subheader("Feature Importance")
    st.write("""
    Upon running .feature_importances in the Decision Tree Classifier Model to check how each feature influences the model's decisions, 
    it is clear that 'petal_length' holds the most influence with 89% importance, followed by 'petal_width' with 8.7% importance.
    """)
    st.code("""
    decision_tree_feature_importance = pd.Series(dt_classifier.feature_importances_, index=X_train.columns)
    decision_tree_feature_importance
    """)

    # Plot Decision Tree
    st.subheader("Decision Tree Classifier - Tree Plot")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(dt_classifier, filled=True, feature_names=X_train.columns, class_names=["Setosa", "Versicolor", "Virginica"], ax=ax)
    st.pyplot(fig)

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
