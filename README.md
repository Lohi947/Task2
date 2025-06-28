# Task2
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample FAQ Data
faq_data = {
    "What is your return policy?": "You can return any item within 30 days of purchase.",
    "How can I track my order?": "You can track your order using the tracking ID sent to your email.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, PayPal, and net banking.",
    "How do I reset my password?": "Click on 'Forgot Password' on the login page to reset your password.",
    "Do you ship internationally?": "Yes, we ship to over 100 countries worldwide."
}

# Preprocess text
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Prepare corpus
questions = list(faq_data.keys())
answers = list(faq_data.values())
processed_questions = [preprocess(q) for q in questions]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Chatbot function
def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_vec, X)
    max_score = np.max(similarity_scores)
    max_index = np.argmax(similarity_scores)

    if max_score > 0.3:  # threshold to ignore weak matches
        return answers[max_index]
    else:
        return "Sorry, I couldn't find an answer to your question."

# Example interaction
while True:
    user_q = input("\nYou: ")
    if user_q.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_q)
    print("Bot:", response)
