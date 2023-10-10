import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6dd27;  /* Koyu sarı renk */
    </style>
    """,
    unsafe_allow_html=True
)

# Stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
nltk.download('punkt')


#Baslik ekliyoruz
st.markdown('<p style="background-color: #96e627; color: black; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">Sentiment Analysis for Women Clothes Review</p>', unsafe_allow_html=True)
st.image("https://npr.brightspotcdn.com/dims4/default/720110d/2147483647/strip/true/crop/900x534+0+0/resize/1760x1044!/format/webp/quality/90/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F03%2F9c%2F3a2e47fc412a857e60875267fc30%2Fclothing-istock-vectorikart-2021-0730.jpg", use_column_width=True)

# Side Bar
st.sidebar.title("Please Select a Method")
selected_model = st.sidebar.selectbox("Choose a Method", ['LogisticRegression', 'SVM', 'KNN', 'RandomForest', 'AdaBoost', 'NaiveBayes'])

positive_sample = "I absolutely love this product! It fits perfectly and the quality is amazing."
negative_sample = "This is the worst product I've ever purchased. It's cheaply made and doesn't work as advertised."

# Sample Buttons
if st.button("Sample Reviews"):
    st.write("Positive: The fit is perfect and the material feels luxurious. A must-have for any wardrobe")
    st.write("Negative: The dress came, but the color is faded and the pattern is small, it tore immediately")

# Text Area
user_input = st.text_area("Please enter a review without any punctual or number:", height=4)

# Data Cleaning
def cleaning(data):
    # Tokenize
    text_tokens = word_tokenize(data.lower())
    
    # Remove Puncs and numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    
    # Remove Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    
    # Lemmatization
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]
    
    # Joining
    return " ".join(text_cleaned)

# Load the selected model
try:
    model_filename = f"{selected_model}.pkl"
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.write(f"Model {model_filename} could not be found.")

st.sidebar.title("Dataset Information")
st.sidebar.info("The dataset used in this study contains 23486 rows and 10 feature variables. After preprocessing the data, 6 models were created with NLP on 22628 comments and each comment's tag. For each model, recommendation suggestions can be obtained from comments with an average success rate of 90 percent.")


# Submit Button
if st.button("Submit"):
    if user_input:
        st.write("Your review was received, and analysis began.")
                
        # Kaydedilmiş vektörleyiciyi pickle ile yükle
        with open('tf_idf_vectorizer.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        
        # Yalnızca transform kullan
        user_input_tf_idf = loaded_vectorizer.transform([user_input])  # Note the list

        try:
            result = model.predict(user_input_tf_idf)
            if result[0] == 0:
                st.info("Result: Positive")
                st.image("https://st2.depositphotos.com/1967477/6346/v/950/depositphotos_63469893-stock-illustration-happy-smiley-emoticon-cartoon-face.jpg", use_column_width=False, width=250)
                #st.image("https://st2.depositphotos.com/1967477/6346/v/950/depositphotos_63469893-stock-illustration-happy-smiley-emoticon-cartoon-face.jpg", use_column_width=True)
            elif result[0] == 1:
                st.info("Result: Negative")
                st.image("https://st.depositphotos.com/1001911/4372/v/950/depositphotos_43725677-stock-illustration-wiping-tear-emoticon.jpg", use_column_width=False, width=250)
            
        except Exception as e:
            st.write(f"An error occurred: {e}")

    else:
        st.write("Please enter a valid review.")
        
show_wordcloud = st.sidebar.button("Show WordCloud")

if show_wordcloud:
    st.subheader("WordCloud for Recommended Reviews")
    dff = pd.read_csv("clean.csv")
    data_recommended = dff[dff['recommend'] == 1]["text"]
    all_words = " ".join(data_recommended)
    recommended = WordCloud(background_color="white", max_words=500)
    recommended.generate(all_words)

    # WordCloud'u görüntüle
    plt.figure(figsize=(13, 13))
    plt.imshow(recommended, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    
    st.subheader("WordCloud of the Not Recommended Reviews")
    not_recommended = dff[dff['recommend'] == 0]["text"]
    all_words2 = " ".join(not_recommended)
    notrecommended = WordCloud(background_color="white", max_words=500)
    notrecommended.generate(all_words2)

    # WordCloud'u görüntüle
    plt.figure(figsize=(13, 13))
    plt.imshow(notrecommended, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
