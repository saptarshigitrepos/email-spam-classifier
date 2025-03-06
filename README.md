# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Make sure to adjust the path to where your spam.csv file is located
data = pd.read_csv(r"C:\Users\ghosh\OneDrive\c++\Desktop\spamdataset.csv", encoding='latin-1')


# Display the first few rows of the dataset to understand its structure
print("Sample data:")
print(data.head())

# Preprocess the data
# Select relevant columns: 'v1' for labels (spam/ham) and 'v2' for messages
data = data[['v1', 'v2']]
data.columns = ['label', 'message']  # Renaming columns for clarity

# Convert labels to binary values: spam = 1, ham = 0
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Display the distribution of labels in the dataset
print("\nLabel distribution:")
print(data['label'].value_counts())

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text messages using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_vectorized = vectorizer.transform(X_test)        # Only transform test data

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f'\nAccuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Function to predict if a new message is spam or ham
def predict_message(message):
    message_vectorized = vectorizer.transform([message])  # Vectorize the input message
    prediction = classifier.predict(message_vectorized)     # Predict using the trained model
    return "Spam" if prediction[0] == 1 else "Ham"

# Interactive user input for message prediction
if __name__ == "_main_":
    while True:
        user_input = input("\nEnter a message to check if it's spam or ham (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = predict_message(user_input)
        print(f'The message is classified as: {result}')
