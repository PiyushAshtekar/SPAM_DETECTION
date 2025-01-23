import string
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
nltk.data.path.append('./nltk_data')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pickle file correctly
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the pickle file correctly
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the Message Here")

if st.button('Predict'):

    # 1. preprocess

    transform_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")

