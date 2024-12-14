# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import matplotlib.pyplot as plt

# Step 2: Load the dataset from your local system
# Specify the file path to your 'spam.csv' file
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')  # Adjust encoding if needed
df = df[['v1', 'v2']]  # Keep only the necessary columns ('v1' as label, 'v2' as message)
df.columns = ['label', 'message']  # Rename columns for clarity
print(df.head())

# Step 3: Preprocessing
# Encode labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Step 4: Build the model
model = Sequential([
    Dense(16, input_dim=X_train_vec.shape[1], activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train the model
history = model.fit(X_train_vec, y_train, epochs=5, batch_size=32, validation_data=(X_test_vec, y_test))

# Step 6: Evaluate the model
y_pred = (model.predict(X_test_vec) > 0.5).astype('int32').flatten()
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Step 7: Plot the accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 8: Classify User Input
def classify_message(message):
    # Preprocess and vectorize the input message
    message_vec = vectorizer.transform([message]).toarray()
    # Predict using the trained model
    prediction = model.predict(message_vec)
    # Determine if spam or ham based on prediction threshold
    return "Spam" if prediction > 0.5 else "Ham"

# Interactive user input for message classification
while True:
    user_input = input("Enter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    result = classify_message(user_input)
    print(f"The message is classified as: {result}")
