import string  # Importing string module for string operations
import pandas as pd  # Importing pandas for data manipulation and analysis
import chardet  # Importing chardet for character encoding detection
from nltk.corpus import stopwords  # Importing stopwords from NLTK for text preprocessing
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical labels
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import nltk  # Importing NLTK for natural language processing
import seaborn as sns  # Importing seaborn for enhanced data visualization
from wordcloud import WordCloud  # Importing WordCloud for word cloud generation
from nltk.stem.porter import PorterStemmer  # Importing PorterStemmer for stemming
ps = PorterStemmer()  # Creating an instance of PorterStemmer
from collections import Counter  # Importing Counter for counting word occurrences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Importing vectorizers

# Detecting the encoding of the CSV file
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
print(result)  # Printing the detected encoding

# Reading the CSV file into a DataFrame with the detected encoding
df = pd.read_csv('spam.csv', encoding="Windows-1252")
# print(df.sample(5))  # Uncomment to view a random sample of the DataFrame
# print(df.shape)  # Uncomment to view the shape of the DataFrame

# 1. Data cleaning
# 2. EDA (Exploratory Data Analysis)
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy

# 1. Data Cleaning

# print(df.info())  # Uncomment to view information about the DataFrame

# Dropping unnecessary columns from the DataFrame
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Renaming columns for better understanding
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

encoder = LabelEncoder()  # Creating an instance of LabelEncoder

# Encoding the target variable (ham/spam) into numerical values
df['target'] = encoder.fit_transform(df['target'])

# Checking for missing values
# print(df.isnull().sum())  # Uncomment to view missing values

# Checking for duplicate values in the DataFrame
df.duplicated().sum()  # Returns the count of duplicate rows

# Removing duplicate rows from the DataFrame
df = df.drop_duplicates(keep='first')

df.duplicated().sum()  # Returns the count of duplicate rows after removal

# 2. EDA (Exploratory Data Analysis)
# Counting the occurrences of each target class (ham/spam)
df['target'].value_counts()

# Creating a pie chart to visualize the distribution of ham and spam messages
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()  # Displaying the pie chart

# Note: The data is imbalanced

# Calculating the number of characters in each message
df['num_characters'] = df['text'].apply(len)

# Calculating the number of words in each message
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# Calculating the number of sentences in each message
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Displaying descriptive statistics for character, word, and sentence counts
df[['num_characters', 'num_words', 'num_sentences']].describe()

# Descriptive statistics for ham messages
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()

# Descriptive statistics for spam messages
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()


# Plotting histograms for character counts in ham and spam messages
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')

# Plotting histograms for word counts in ham and spam messages
plt.figure(figsize=(12, 6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='red')

# Creating pair plots to visualize relationships between features
sns.pairplot(df, hue='target')

# sns.heatmap(df.corr(), annot=True)  # Uncomment to visualize correlation heatmap

# 3. Data Preprocessing
# Steps include: Lower casing, Tokenization, Removing special characters, Removing stop words and punctuation, Stemming

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()  # Converting text to lowercase
    text = nltk.word_tokenize(text)  # Tokenizing the text into words

    y = []  # List to hold processed words
    for i in text:
        if i.isalnum():  # Keeping only alphanumeric characters
            y.append(i)

    text = y[:]  # Copying processed words to text
    y.clear()  # Clearing the list for the next step

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Removing stop words and punctuation
            y.append(i)

    text = y[:]  # Copying processed words to text
    y.clear()  # Clearing the list for the next step

    for i in text:
        y.append(ps.stem(i))  # Applying stemming to each word

    return " ".join(y)  # Joining the processed words back into a single string

# Applying the transform_text function to the 'text' column
df['transformed_text'] = df['text'].apply(transform_text)

# Creating a word cloud to visualize the most common words in spam messages
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

# Generating word cloud for spam messages
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)  # Displaying the spam word cloud

# Generating word cloud for ham messages
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)  # Displaying the ham word cloud

# Creating a list of words from spam messages for further analysis
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

# Creating a DataFrame of the 30 most common words in spam messages
most_common = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['word', 'count'])

# Plotting a bar chart of the most common words in spam messages
sns.barplot(x='word', y='count', data=most_common)
plt.xticks(rotation='vertical')  # Rotating x-axis labels for better readability
plt.show()  # Displaying the bar chart

# Creating a list of words from ham messages for further analysis
ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# Creating a DataFrame of the 30 most common words in ham messages
most_common = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['word', 'count'])

# Plotting a bar chart of the most common words in ham messages
sns.barplot(x='word', y='count', data=most_common)
plt.xticks(rotation='vertical')  # Rotating x-axis labels for better readability
plt.show()  # Displaying the bar chart

# Text Vectorization
# Using Bag of Words and TF-IDF for feature extraction
df.head()  # Displaying the first few rows of the DataFrame

# 4. Model Building

cv = CountVectorizer()  # Creating an instance of CountVectorizer
tfidf = TfidfVectorizer(max_features=3000)  # Creating an instance of TfidfVectorizer with a limit on features

# Transforming the text data into feature vectors
X = tfidf.fit_transform(df['transformed_text']).toarray()

# from sklearn.preprocessing import MinMaxScaler  # Uncomment to import MinMaxScaler for scaling features
# scaler = MinMaxScaler()  # Uncomment to create an instance of MinMaxScaler
# X = scaler.fit_transform(X)  # Uncomment to scale the features

# Appending the number of characters column to the feature set (if needed)
# X = np.hstack((X, df['num_characters'].values.reshape(-1, 1)))  # Uncomment to append character counts

y = df['target'].values  # Extracting the target variable

from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # Importing Naive Bayes classifiers
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score  # Importing metrics for evaluation

# Creating instances of the classifiers
gnb = GaussianNB()  # Gaussian Naive Bayes
mnb = MultinomialNB()  # Multinomial Naive Bayes
bnb = BernoulliNB()  # Bernoulli Naive Bayes

# Training and evaluating the Gaussian Naive Bayes classifier
gnb.fit(X_train, y_train)  # Fitting the model to the training data
y_pred1 = gnb.predict(X_test)  # Making predictions on the test data
print(accuracy_score(y_test, y_pred1))  # Printing accuracy score
print(confusion_matrix(y_test, y_pred1))  # Printing confusion matrix
print(precision_score(y_test, y_pred1))  # Printing precision score

# Training and evaluating the Multinomial Naive Bayes classifier
mnb.fit(X_train, y_train)  # Fitting the model to the training data
y_pred2 = mnb.predict(X_test)  # Making predictions on the test data
print(accuracy_score(y_test, y_pred2))  # Printing accuracy score
print(confusion_matrix(y_test, y_pred2))  # Printing confusion matrix
print(precision_score(y_test, y_pred2))  # Printing precision score

# Training and evaluating the Bernoulli Naive Bayes classifier
bnb.fit(X_train, y_train)  # Fitting the model to the training data
y_pred3 = bnb.predict(X_test)  # Making predictions on the test data
print(accuracy_score(y_test, y_pred3))  # Printing accuracy score
print(confusion_matrix(y_test, y_pred3))  # Printing confusion matrix
print(precision_score(y_test, y_pred3))  # Printing precision score

# tfidf --> MNB  # Placeholder for future model improvements

# Saving the trained TF-IDF vectorizer and model using pickle
import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))  # Saving the vectorizer
pickle.dump(mnb, open('model.pkl', 'wb'))  # Saving the model
