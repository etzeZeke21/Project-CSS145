#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


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
    
    st.write("The dataset provides reviews from the Steam website across 11 games namely: Arma 3, Counter Strike, Counter Strike Global Offensive, Dota 2, Football Manager 2015, Garry’s Mod, Grand Theft Auto V, Sid Meiers Civilization 5, Team Fortress 2, The Elder Scrolls V, and Warframe. Each review comes alongside numerous pieces of data, including the number of people who marked the review helpful, the number of people who marked the review funny, the number of friends the reviewer has on the site, etc. One of the more important pieces of data, however, is the number of hours that the reviewer played the game that they are reviewing.\n\n Successfully created the following:")
    
    st.markdown("- A sentimental analysis model that will analyze a review and decide if the review is recommmended or not recommended, using the review text.")
    st.markdown("- A helpful review classification that can predict the helpfulness score, using the number of voted helpfulness, number of voted funny, total game hours, achievement progress, and number of comments.")
    st.markdown("- A time series analysis that uses the date posted and total game hours to track how the sentiment of reviews or helpfulness score changes over time.")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)
    

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
     # Load and show initial data
    st.subheader("Initial Dataset")
    st.dataframe(combined_df.head())

      # Display initial number of rows
    st.subheader("Initial DataFrame Statistics")
    st.write(f"Number of rows before cleaning: {len(df)}")

    # Display data types of columns
    st.subheader("Column Data Types")
    st.dataframe(df.dtypes)

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Display missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Display unique values in categorical columns
    for col in df.select_dtypes(include=['object']):
        st.subheader(f"Unique values in {col}")
        st.write(df[col].value_counts())

    # Display data cleaning steps
    st.subheader("Data Cleaning Steps")
    st.write("1. Removed null values")
    st.write("2. Converted 'rating' column to binary (recommended/not recommended)")
    st.write("3. Downsampled the 'recommended' class to match the number of 'not recommended' reviews")
    st.write("4. Processed text data using NLTK")
    st.write("5. Calculated review quality score")
    st.write("6. Created new features (log playtime, has significant playtime)")
    st.write("7. Removed rows with missing values in required columns")

    # Display final DataFrame statistics
    st.subheader("Final DataFrame Statistics")
    st.write(f"Number of rows after cleaning: {len(df)}")

    # Display sample of cleaned data
    st.subheader("CLEANED DATA")
    st.dataframe(df.head())

    # Display summary of cleaning process
    st.subheader("PROCESS")
    st.write("The data cleaning process involved several steps to prepare the dataset for analysis.")
    st.write("We removed null values, converted the rating column to binary, downsampled the recommended class, processed text data using NLTK, calculated a review quality score, created new features, and removed rows with missing values in required columns.")
    st.write("These steps helped ensure that the dataset was clean, balanced, and ready for analysis.")

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

# Decision Tree Classifier Explanation
if st.session_state.page_selection == "machine_learning":
    
    
    #1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
    # Classifier
    st.header("1. Classifier")

    # Classifier Display Explanation
    st.write("""
    **Classifier**, according to C3.AI, is a machine learning algorithm used to categorize data inputs into specific categories, 
    which gives great enterprise applications. Thry are trained using labeled data algorithms and that it employs complex methods 
    from mathematics and statistics and uses it to generate predictions.
    """)
    st.write("Reference: [C3.AI Glossary about Data Science](https://c3.ai/glossary/data-science/classifier/)")

    # Show Training Code (Classifier)
    st.subheader("Training the Classifier")
    with st.expander("See Code for the Training of Classifier"):
        st.code("""
        class SteamReviewClassifier:
        def __init__(self):
            self.text_vectorizer = TfidfVectorizer(max_features=5000)
            self.numeric_scaler = StandardScaler()
            self.model = None

        def create_feature_pipeline(self):

            numeric_features = ['review_quality_score', 'log_playtime',
                            'found_helpful_percentage', 'has_significant_playtime']

            text_transformer = ('text', self.text_vectorizer, 'processed_text')

            numeric_transformer = ('numeric', self.numeric_scaler, numeric_features)

            column_transformer = ColumnTransformer(
                transformers=[text_transformer, numeric_transformer],
                remainder='drop'
            )

            pipeline = Pipeline([
                ('features', column_transformer),
                ('classifier', LogisticRegression(max_iter=1000))
            ])

            return pipeline

        def train(self, df):

            prepared_df = prepare_data(df)

            if len(prepared_df) == 0:
                raise ValueError("No valid data remaining after preprocessing")

            X = prepared_df[['processed_text', 'review_quality_score', 'log_playtime',
                            'found_helpful_percentage', 'has_significant_playtime']]
            y = prepared_df['rating_binary']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = self.create_feature_pipeline()
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            return X_test, y_test, y_pred

        def predict(self, review_text, game_hours=0, helpful_votes=0,
                    unhelpful_votes=0, helpful_percentage=0):

            processed_text = text_processing(review_text)

            if processed_text is None:
                raise ValueError("Invalid review text")

            total_votes = helpful_votes + unhelpful_votes
            if total_votes > 0:
                review_quality_score = helpful_votes / total_votes
            else:
                review_quality_score = 0.5

            log_playtime = np.log1p(float(game_hours))
            has_significant_playtime = float(game_hours > 2)

            input_data = pd.DataFrame({
                'processed_text': [processed_text],
                'review_quality_score': [review_quality_score],
                'log_playtime': [log_playtime],
                'found_helpful_percentage': [helpful_percentage],
                'has_significant_playtime': [has_significant_playtime]
            }))
        """)

    # Evaluation Code (Classifier)
    st.subheader("Model Evaluation")
    with st.expander("See Code for the Model Evaluation of Classifier"):
        st.code("""
        classifier = SteamReviewClassifier()
    X_test, y_test, y_pred = classifier.train(combined_df)

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Recommended', 'Recommended'],
                yticklabels=['Not Recommended', 'Recommended'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    labels = ['Not Recommended', 'Recommended']
    actual_counts = [list(y_test).count(0), list(y_test).count(1)]
    predicted_counts = [list(y_pred).count(0), list(y_pred).count(1)]

    x = range(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x, actual_counts, width=0.4, label='Actual', color='b', align='center')
    plt.bar([p + 0.4 for p in x], predicted_counts, width=0.4, label='Predicted', color='r', align='center')

    plt.xlabel('Review Recommendation')
    plt.ylabel('Count')
    plt.title('Actual vs Predicted Review Recommendations')
    plt.xticks([p + 0.2 for p in x], labels)
    plt.legend()
    plt.show()
        """)

    pic11 = Image.open('assets/pics/1.1.png')
    st.image(pic11, caption='Confusion Matrix of Classifier')

    st.write(" ")

    pic12 = Image.open('assets/pics/1.2.png')
    st.image(pic12, caption='Bar Chart of the Review Recommendation')
    
  


    #2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    # Random Forest Classifier
    st.header(" ")
    st.header("2. Random Forest Classifier")

    # Random Forest Classifier Display Explanation
    st.write("""
    **Random Forest Classifier**  is a machine learning technique that uses a number of Decision Trees. By selecting a random 
    subset of the dataset and evaluating a random set of features at each split is how each tree is built. All the trees collectively 
    produce an output during prediction. For classification problems, this is done through majority voting. However, for the 
    regression, it is done by taking an average of all the outputs from the trees. This will go together with collective 
    insight, because the more insightful the trees, the more reliable the predictions. It is said to have a higher accuracy in
    predictions compared to Decision Tree Classifier.
    """)
    st.write("Reference: [Geeks for Geeks](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)")

    # Show Training Code (Random Forest Classifier)
    st.subheader("Training the Random Forest Classifier")
    with st.expander("See Code for the Training of Random Forest Classifier"):
        st.code("""
        def train_helpfulness_classifier(df_unclean):

        threshold = 0.7

        features = [
            'num_found_helpful',
            'num_found_funny',
            'total_game_hours',
            'num_comments'
        ]

        df = df_unclean.dropna(subset=['found_helpful_percentage'], inplace=False)

        X = df[features].copy()

        y = (df['found_helpful_percentage'] >= threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        rf_model.fit(X_train_scaled, y_train)

        y_pred = rf_model.predict(X_test_scaled)

        report = classification_report(y_test, y_pred)

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for Review Helpfulness Prediction')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return rf_model, scaler, report

    def predict_helpfulness(model, scaler, new_data):

        features = [
            'num_found_helpful',
            'num_found_funny',
            'total_game_hours',
            'num_comments'
        ]

        X_new = new_data[features].copy()
        X_new = X_new.fillna(0)
        X_new_scaled = scaler.transform(X_new)

        return model.predict(X_new_scaled)
        """)

    # Evaluation Code (Random Forest Classifier)
    st.subheader("Model Evaluation")
    with st.expander("See Code for the Model Evaluation of Random Forest Classifier"):
        st.code("""
        model, scaler, report = train_helpfulness_classifier(combined_df)
    print("Classification Report:")
    print(report)

    new_reviews = pd.DataFrame({
        'num_found_helpful': [10],
        'num_found_funny': [5],
        'total_game_hours': [100],
        'num_comments': [3]
    })

    predictions = predict_helpfulness(model, scaler, new_reviews)
    #produces 1 or 0, helpful is 1 while not helpful is 0
    print("\nPredicted helpful (1) or not helpful (0):", predictions)

    print("\nClassification Report:")
    print(report)
        """)

    # Feature Importance (Random Forest Classifier)
    st.subheader("Feature Importance")
    st.write("""
    When the .feature_importance ran in the Random Forest Classifier model, it showed that 'num_found_helpful' has the highest importance
    followed by 'total_game_hours', then 'num_found_funny', and lastly 'num_comments'. You can check the bar chart below the codes.
    """)
    with st.expander("See Code for the Feature Importance of Random Forest Classifier"):
        st.code("""
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for Review Helpfulness Prediction')
        plt.tight_layout()
        plt.show()
        """)


    pic21 = Image.open('assets/pics/2.1.png')
    st.image(pic21, caption='Bar Chart of the Feature Importance')

    st.write(" ")

    pic22 = Image.open('assets/pics/2.2.png')
    st.image(pic22, caption='Confusion Matrix of Random Forest Classifier')





    #3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    # Linear Regression
    st.header(" ")
    st.header("3. Linear Regression")

    # Linear Regression Display Explanation
    st.write("""
    **Linear Regression** is a type of supervised machine learning alogorithm by calculating the linear relationships between independent
    and dependent variables by the use of linear equation, hence its name. It is used because of the notable strength of linear regression,
    its interpretability. There are two main types of Linear Regression, Simple Linear Regression for only one independent feature, and
    Multiple Linear Regression for features that is more than one.
    """)
    st.write("Reference: [Geek for Geeks](https://www.geeksforgeeks.org/ml-linear-regression/)")

    # Show Training Code (Linear Regression)
    st.subheader("Training the Linear Regression")
    with st.expander("See Code for the Training of Linear Regression"):
        st.code("""
        def analyze_steam_reviews(df):

        # Convert date columns to datetime
        df['date_posted'] = pd.to_datetime(df['date_posted'])

        # Create month-year column for aggregation
        df['month_year'] = df['date_posted'].dt.to_period('M')

        # Calculate monthly metrics
        monthly_stats = pd.DataFrame({
            'avg_game_hours': df.groupby('month_year')['total_game_hours'].mean(),
            'review_count': df.groupby('month_year').size(),
            'avg_helpfulness': df.groupby('month_year')['found_helpful_percentage'].mean(),
            'positive_ratio': df.groupby('month_year')['rating'].apply(
                lambda x: (x == 'Recommended').mean() * 100
            )
        }).reset_index()

        # Convert period to datetime for plotting
        monthly_stats['month_year'] = monthly_stats['month_year'].dt.to_timestamp()

        # Create subplots for visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Steam Reviews Time Series Analysis', fontsize=16)

        # Plot 1: Average Game Hours Over Time
        axes[0, 0].plot(monthly_stats['month_year'], monthly_stats['avg_game_hours'])
        axes[0, 0].set_title('Average Game Hours Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Hours')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Review Volume Over Time
        axes[0, 1].plot(monthly_stats['month_year'], monthly_stats['review_count'])
        axes[0, 1].set_title('Number of Reviews Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Reviews')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Average Helpfulness Score Over Time
        axes[1, 0].plot(monthly_stats['month_year'], monthly_stats['avg_helpfulness'])
        axes[1, 0].set_title('Average Helpfulness Score Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average Helpfulness (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Positive Review Ratio Over Time
        axes[1, 1].plot(monthly_stats['month_year'], monthly_stats['positive_ratio'])
        axes[1, 1].set_title('Positive Review Ratio Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Positive Reviews (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        # Calculate correlation between game hours and other metrics
        correlations = pd.DataFrame({
            'Metric': ['Helpfulness Score', 'Positive Rating'],
            'Correlation with Game Hours': [
                df['total_game_hours'].corr(df['found_helpful_percentage']),
                df['total_game_hours'].corr(df['rating'].map({'Recommended': 1, 'Not Recommended': 0}))
            ]
        })

        # Calculate trend statistics
        trend_stats = {
            'avg_hours_trend': np.polyfit(range(len(monthly_stats)), monthly_stats['avg_game_hours'], 1)[0],
            'helpfulness_trend': np.polyfit(range(len(monthly_stats)), monthly_stats['avg_helpfulness'], 1)[0],
            'sentiment_trend': np.polyfit(range(len(monthly_stats)), monthly_stats['positive_ratio'], 1)[0]
        }

        return monthly_stats, correlations, trend_stats
        """)

    # Evaluation Code (Linear Regression)
    st.subheader("Model Evaluation")
    with st.expander("See Code for the Model Evaluation of Linear Regression"):
        st.code("""
    monthly_stats, correlations, trend_stats = analyze_steam_reviews(combined_df)

    print("\nCorrelations with Game Hours:")
    print(correlations)
    print("\nTrend Analysis:")
    print(f"Average Hours Trend: {'Increasing' if trend_stats['avg_hours_trend'] > 0 else 'Decreasing'}")
    print(f"Helpfulness Score Trend: {'Increasing' if trend_stats['helpfulness_trend'] > 0 else 'Decreasing'}")
    print(f"Positive Review Ratio Trend: {'Increasing' if trend_stats['sentiment_trend'] > 0 else 'Decreasing'}")
        """)

    pic31 = Image.open('assets/pics/3.1.png')
    st.image(pic31, caption='Multiple Line Graphs of Steam Reviews Time Series Analysis')



    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    
    # Define model training function
    def train_helpfulness_classifier(df_unclean):
        threshold = 0.7
        features = [
            'num_found_helpful',
            'num_found_funny',
            'total_game_hours',
            'num_comments'
        ]

        df = df_unclean.dropna(subset=['found_helpful_percentage'], inplace=False)

        X = df[features].copy()

        y = (df['found_helpful_percentage'] >= threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )

        rf_model.fit(X_train_scaled, y_train)

        y_pred = rf_model.predict(X_test_scaled)

        report = classification_report(y_test, y_pred, output_dict=False)

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return rf_model, scaler, report, feature_importance

    # Train model and scaler using the combined dataset
    model, scaler, report, feature_importance = train_helpfulness_classifier(combined_df)

    # User input for new data to predict
    st.subheader("Helpful Review Classification")
    num_found_helpful = st.number_input("Number of people who found the review helpful💡", min_value=0)
    num_found_funny = st.number_input("Number of people who found the review funny😂", min_value=0)
    total_game_hours = st.number_input("Total hours spent on the game⏳", min_value=0)
    num_comments = st.number_input("Number of comments on the review💬", min_value=0)

    new_review_data = pd.DataFrame({
        'num_found_helpful': [num_found_helpful],
        'num_found_funny': [num_found_funny],
        'total_game_hours': [total_game_hours],
        'num_comments': [num_comments]
    })

    # Prediction
    if st.button("Predict Helpfulness"):
        prediction = model.predict(scaler.transform(new_review_data))[0]
        if prediction == 1:
            st.success("Prediction: The review is likely to be marked as helpful.")
        else:
            st.info("Prediction: The review is less likely to be marked as helpful.")
    
    # Checkbox to toggle feature importance graph
    show_graph = st.checkbox("Show Feature Importance Graph", value=False)

    # Display the feature importance graph if the checkbox is checked
    if show_graph:
        st.subheader("Feature Importance for Review Helpfulness Prediction")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance for Review Helpfulness Prediction')
        plt.tight_layout()
        st.pyplot(fig)

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
