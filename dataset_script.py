import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load original CSV
df = pd.read_csv("netflix_titles_nov_2019.csv")

# Keep only relevant columns
df_revised = df[['title', 'rating', 'description']].drop_duplicates(subset='title').dropna(subset=['rating'])

# Lowercase all text
df_revised = df_revised.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Text cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words])

df_revised['description'] = df_revised['description'].apply(clean_text)

# Remove punctuation/numbers
df_revised['removed_punc'] = df_revised['description'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Tokenize
df_revised['tokenized_description'] = df_revised['removed_punc'].apply(lambda x: x.split())

# Save cleaned CSV
df_revised.to_csv("netflix_cleaned.csv", index=False)

print("netflix_cleaned.csv created successfully!")
