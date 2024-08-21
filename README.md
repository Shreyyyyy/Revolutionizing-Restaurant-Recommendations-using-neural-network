# Personalized Restaurant Recommendation System

This project demonstrates a modern way to recommend restaurants using Google APIs, spreadsheets, and neural networks, aimed at both seasoned coders and newcomers. By automating the data collection through Google Forms and training a neural network, the system provides personalized restaurant suggestions.

## Features

- **Google Forms Integration**: Collects user feedback on restaurants which is then stored in Google Sheets.
- **Neural Networks for Recommendations**: Personalizes recommendations for returning users based on their past ratings using a trained neural network.
- **Google Places API**: Provides top restaurant recommendations for new users based on location and cuisine preferences.
- **Streamlit App**: An interactive web application that makes it easy for users to receive recommendations.

## How It Works

### Step 1: Setting Up the Google Form
Users submit their restaurant experiences via a Google Form, automatically populating a Google Sheet with their data.

### Step 2: Training the Neural Network
The data from Google Sheets is pulled into a Google Colab environment to train a neural network that predicts user preferences. This includes:
- Label Encoding for categorical data transformation.
- Train-Test Split for model validation.
- Neural network training using TensorFlow and Keras.

### Step 3: For New Users - Google Places API
For users new to the system, the Google Places API is used to fetch top-rated restaurants based on the user's specified preferences.

### Step 4: Bringing it All Together in Streamlit
The system is wrapped up using Streamlit, providing a user-friendly interface where users can interactively input their preferences and receive recommendations.

## Code Example

```python
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_json', scope)
client = gspread.authorize(creds)
spreadsheet = client.open_by_url("your_google_sheet_url")
worksheet = spreadsheet.get_worksheet(0)
data = worksheet.get_all_values()
df = pd.DataFrame(data)

# Neural Network Model
model = Sequential([
    Dense(128, input_dim=2, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
