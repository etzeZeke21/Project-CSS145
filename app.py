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

    st.write("""
            This Streamlit Application is a Final Project in the course CSS145 Introduction to Data Science.
            
             """)
    

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write(
    """
    The **Steam Game Reviews** dataset contains detailed information on thousands of game reviews collected from Steam. This dataset includes variables such as gameplay hours, social engagement metrics, user ratings, and review helpfulness scores. It is useful for analyzing player behaviors, understanding gaming preferences, and measuring engagement.
    
    For each review, various features are measured, including **total_game_hours_last_two_weeks** (hours played in the last two weeks), **num_reviews** (total number of reviews written by the user), **num_friends** (number of friends the user has), and **found_helpful_percentage** (percentage of users who found the review helpful). This data offers insights into both game popularity and player interaction on Steam. The dataset was sourced from a GitHub repository dedicated to Steam game reviews.
    
    **Content**  
    The dataset contains approximately 80,000 rows with attributes related to gameplay activity, review metrics, and social engagement.
    """
)

    st.write("`Link:` [Steam Reviews Dataset on GitHub](https://github.com/mulhod/steam_reviews)")

    # Displaying the dataset as a Data Frame
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(combined_df)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write("Here’s a summary of some of the key attributes in the dataset:")

    # Displaying Descriptive Statistics Table
    st.dataframe(combined_df.describe()[[
        "total_game_hours_last_two_weeks", "num_groups", "num_badges", "num_found_funny", 
        "num_workshop_items", "found_helpful_percentage", "num_voted_helpfulness", "num_found_helpful", 
        "friend_player_level", "total_game_hours", "num_guides", "num_friends", 
        "num_screenshots", "num_comments", "num_reviews", "num_games_owned"
    ]])

    # Extracting relevant statistics from the describe() output
    mean_total_game_hours = combined_df["total_game_hours"].mean()
    max_total_game_hours = combined_df["total_game_hours"].max()
    std_total_game_hours = combined_df["total_game_hours"].std()

    mean_num_reviews = combined_df["num_reviews"].mean()
    std_num_reviews = combined_df["num_reviews"].std()
    max_num_reviews = combined_df["num_reviews"].max()

    mean_num_friends = combined_df["num_friends"].mean()
    std_num_friends = combined_df["num_friends"].std()

    mean_num_found_funny = combined_df["num_found_funny"].mean()
    max_num_found_funny = combined_df["num_found_funny"].max()

    mean_found_helpful_percentage = combined_df["found_helpful_percentage"].mean()

# Writing the paragraph with the statistics
    st.write(f"""
The results from `combined_df.describe()` provide valuable insights into player engagement, review activity, and social interaction in the Steam gaming community.

On average, players have spent approximately {mean_total_game_hours:.2f} hours on their games, with a wide range of engagement from 0.0 to {max_total_game_hours:.1f} hours (standard deviation of {std_total_game_hours:.2f}). This suggests that while some players dedicate a significant amount of time to their games, others engage minimally. When it comes to reviews, players have submitted an average of {mean_num_reviews:.2f} reviews, but there is a high variability (standard deviation of {std_num_reviews:.2f}), with the maximum number of reviews reaching {max_num_reviews}. This shows that some users are more active in providing feedback, while others are less engaged.

The social aspect of gaming is also evident, with players having an average of {mean_num_friends:.2f} friends (standard deviation of {std_num_friends:.2f}), demonstrating diverse levels of social engagement. In terms of humor, users have found an average of {mean_num_found_funny:.2f} reviews funny, with a broad range, as the highest number of humorous reviews found is {max_num_found_funny}. This highlights that while some players appreciate humor in reviews, others may not engage with it as much.

Finally, the **found_helpful_percentage**, which averages {mean_found_helpful_percentage:.2f}, shows that players generally find reviews to be helpful, with some reviews being rated as helpful by nearly every player. This indicates that feedback shared on the platform holds significant value within the community.

These insights provide a detailed look at how players engage with games and reviews on Steam.
""")

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Summary")
    st.write("Here’s a summary of the key features in the dataset:")
    st.dataframe(combined_df.describe())

    # Create the first row with columns 1 and 2
    col1, col2 = st.columns(2, gap='medium')

    with col1:
        st.subheader("Distribution of Total Game Hours")
        chart = alt.Chart(combined_df).mark_bar().encode(
            alt.X("total_game_hours", bin=True),
            y='count()'
        ).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)

        #Caption
        st.write("""
        **Interpretation:** This graph shows how many hours players have spent on their games. A peak at lower hours would suggest 
        that many users play only briefly, while a high frequency at higher hours could indicate a subset of dedicated players. 
        This distribution gives insight into user engagement levels.
        """)

    with col2:
        st.subheader("Helpfulness of Reviews")
        helpful_chart = alt.Chart(combined_df).mark_bar().encode(
            alt.X("found_helpful_percentage", bin=True),
            y='count()'
        ).properties(width=700, height=400)
        st.altair_chart(helpful_chart, use_container_width=True)

        # Caption
        st.write("""
        **Interpretation:** This graph shows the percentage of users who found reviews helpful. A high peak at higher percentages 
        indicates that most reviews are generally seen as useful by the community, while a diverse spread suggests mixed perceptions 
        of review helpfulness.
        """)

    col3, col4 = st.columns(2, gap='medium')

    with col3:
        combined_df['review_length'] = combined_df['review'].apply(lambda x: len(str(x)))
        st.subheader("Review Length Analysis")
        review_length_chart = alt.Chart(combined_df).mark_bar().encode(
            alt.X("review_length", bin=True),
            y='count()'
        ).properties(width=700, height=400)
        st.altair_chart(review_length_chart, use_container_width=True)
        # Caption
        st.write("""
        **Interpretation:** This graph shows the length of reviews. If most reviews are short, users may prefer giving quick 
        feedback, whereas longer reviews may indicate more detailed feedback. This analysis helps understand how deeply users engage 
        in reviews.
        """)

    with col4:
        st.subheader("Social Engagement: Number of Friends vs. Number of Reviews")
        social_engagement_chart = alt.Chart(combined_df).mark_circle().encode(
            x='num_friends',
            y='num_reviews',
            tooltip=['num_friends', 'num_reviews']
        ).properties(width=700, height=400)
        st.altair_chart(social_engagement_chart, use_container_width=True)
        # Caption for interpretation
        st.write("""
        **Interpretation:** This scatter plot shows the relationship between the number of friends and number of reviews. 
        A positive trend would suggest that social users are also active reviewers. Outliers may represent users who review a lot 
        without social engagement or vice versa.
        """)

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    st.subheader("Word Cloud of Most Common Words in Reviews")
    text = ' '.join(review for review in combined_df.review)
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Caption
    st.write("""
    **Interpretation:** This word cloud shows frequently mentioned words in reviews. Common words can reveal popular topics or 
    frequent concerns among players, such as game, fun, or play. Positive or negative words can indicate the 
    general sentiment.
    """)

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


    st.subheader("Sentimental Analysis")
    st.write("""
    The sentimental analysis model showed great results in predicting the review text if the review was recommended or not. By adding a few more data it could be used to differentiate which reviews are useful or helpful for other players that may seek information about the game.
    """)
    st.subheader("Classification Model")
    st.write("""
     The classification model used the data groups: number of voted helpfulness, number of voted funny, total game hours, and number of comments to try and predict the helpfulness percentage of the review. According to the random forest classification the only data group that had significant impact was number of voted helpfulness, and the other data groups had little to no importance in the helpfulness percentage. Though the classification model still had a high accuracy percentage.
            """)
    st.subheader("Time Analysis Model")
    st.write("""
    The time analysis model analyzed the trends between the games using the dates when the review was posted and the total game hours of the reviewers.The trends were all decreasing which also points to the fact that these games were released at around 10 years ago so after their initial spike and popularity, the reviews are slowing down. But it is still suprising that even after 5 years they are still getting reviews. The model can be used to try and predict the trends until now and it would show that the reviews are on a steady decline from its initial release.
            """)