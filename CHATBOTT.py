import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
    """Preprocesses text by tokenizing, removing stopwords, and lemmatizing."""
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def create_response(user_input, dataset):
    """Creates a response based on user input."""
    user_input_processed = preprocess_text(user_input)

    # Preprocess the dataset into sentences
    sentences = nltk.sent_tokenize(dataset)
    sentences_processed = [preprocess_text(sentence) for sentence in sentences]

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences_processed)

    # Convert user input to a vector
    user_input_vector = vectorizer.transform([user_input_processed])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_input_vector, tfidf_matrix).flatten()
    idx = similarities.argsort()[-1]

    return sentences[idx]

if __name__ == "__main__":
    print("Hello! I'm a Python chatbot. Ask me anything about Python.")
    
    # Load the dataset
    with open('python_data.txt', 'r') as f:
        dataset = f.read()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = create_response(user_input, dataset)
        print("Chatbot:", response)
