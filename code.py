%%writefile new.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import streamlit as st
import requests
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the scope of the API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Add the path to your credentials JSON file
creds = ServiceAccountCredentials.from_json_keyfile_name('/content/drive/MyDrive/micro-amplifier-428012-k8-3176f64f8e81.json', scope)

# Authorize the client
client = gspread.authorize(creds)

# Open the spreadsheet using the spreadsheet's URL
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1FQn7OjNZQwXRUqF3_QNNFpyQNU2I1UyRP1s5YZ_ISVQ/edit?usp=sharing")

# Select the first worksheet in the spreadsheet
worksheet = spreadsheet.get_worksheet(0)

# Fetch all data from the worksheet
data = worksheet.get_all_values()

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Optionally set the first row as the column headers (if your Google Sheet has headers)
df.columns = df.iloc[0]  # Set the first row as column names
df = df.drop(0)  # Drop the first row after setting headers

# Neural network model for personalized recommendations
def train_neural_network(user_data):
    # Ensure the rate column is numeric
    user_data['rate'] = pd.to_numeric(user_data['rate'], errors='coerce')

    # Drop any rows with NaN values after converting 'rate' to numeric
    user_data = user_data.dropna(subset=['rate'])

    user_encoder = LabelEncoder()
    restaurant_encoder = LabelEncoder()

    # Encode the categorical columns (user_name and name)
    user_data['user_id_encoded'] = user_encoder.fit_transform(user_data['user_name'])
    user_data['restaurant_id_encoded'] = restaurant_encoder.fit_transform(user_data['name'])

    # Ensure there are no missing values in the encoded data
    user_data.dropna(subset=['user_id_encoded', 'restaurant_id_encoded', 'rate'], inplace=True)

    # Prepare input features and target
    X = user_data[['user_id_encoded', 'restaurant_id_encoded']].values
    y = user_data['rate'].values

    # Convert target to float, if necessary
    y = y.astype(float)

    # Standardize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple neural network model
    model = Sequential([
        Dense(128, input_dim=2, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model with validation
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    return model, scaler, user_encoder, restaurant_encoder


# Function to get personalized recommendations using a neural network
def get_personalized_recommendations(name, df, user_data, model, scaler, user_encoder, restaurant_encoder):
    if name in user_data['user_name'].unique():
        user_id_encoded = user_encoder.transform([name])[0]
        restaurant_ids = df['name'].unique()

        # Create combinations of the current user with all restaurants
        user_restaurant_combinations = pd.DataFrame({
            'user_id_encoded': [user_id_encoded] * len(restaurant_ids),
            'restaurant_id_encoded': restaurant_encoder.transform(restaurant_ids)
        })

        # Standardize the input features
        X = scaler.transform(user_restaurant_combinations)

        # Predict the ratings using the neural network
        predicted_ratings = model.predict(X).flatten()

        # Attach the predicted ratings to the restaurants
        user_restaurant_combinations['predicted_rating'] = predicted_ratings
        top_recommendations = user_restaurant_combinations.sort_values(by='predicted_rating', ascending=False).head(3)

        # Return the details of the recommended restaurants
        recommended_restaurants = df[df['name'].isin(restaurant_encoder.inverse_transform(top_recommendations['restaurant_id_encoded']))]
        return recommended_restaurants
    else:
        return None

# Function to fetch data from Google Places API (Normal recommendation system)
def fetch_restaurants_from_google(location, cuisine, max_cost, online_order, book_table, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={cuisine}+restaurants+in+{location}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        filtered_results = []
        for result in results:
            place_id = result.get('place_id')
            place_details = fetch_place_details(place_id, api_key)
            simulated_cost = 1000  # Placeholder value for demonstration
            if simulated_cost <= max_cost:
                filtered_results.append({
                    'name': place_details['name'],
                    'location': place_details['location'],
                    'rating': place_details['rating'],
                    'votes': place_details['votes'],
                    'approx_cost(for two people)': simulated_cost
                })

        recommendations_df = pd.DataFrame(filtered_results)
        return recommendations_df.sort_values(by='rating', ascending=False).head(3)
    else:
        st.error(f"Error fetching data from Google Places API: {response.status_code}")
        return pd.DataFrame()  # Return empty DataFrame on error

def fetch_place_details(place_id, api_key):
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,rating,user_ratings_total,formatted_address&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result = data.get('result', {})
        return {
            'name': result.get('name', ''),
            'rating': result.get('rating', 'N/A'),
            'votes': result.get('user_ratings_total', 'N/A'),
            'location': result.get('formatted_address', ''),
        }
    else:
        st.error(f"Error fetching place details: {response.status_code}")
        return {}

# Streamlit UI with improved modern styling
def main():
    st.title("🍽️ Modern Restaurant Recommendation System")

    # Custom CSS for modern and clean design
    st.markdown("""
        <style>
        /* General styling for recommendation box with glassmorphism effect */
        .recommendation-box {
            padding: 30px;
            margin-bottom: 35px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
            transition: transform 0.4s ease, box-shadow 0.4s ease, background 0.4s ease;
            color: #f0f0f0;
        }

        /* Hover effect for recommendation box */
        .recommendation-box:hover {
            transform: scale(1.08) rotate(2deg);
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.5);
        }

        /* Styling for the second recommendation box */
        .recommendation-box:nth-child(2) {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.4);
        }

        /* Styling for the third recommendation box */
        .recommendation-box:nth-child(3) {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }

        /* Dynamic gradient animation for header */
        .header {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #f39c12, #e74c3c, #3498db);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-move 5s ease infinite;
            margin-bottom: 35px;
        }

        /* Text styling inside the recommendation boxes with smooth hover color transition */
        .recommendation {
            font-size: 24px;
            line-height: 1.8;
            transition: color 0.3s ease;
            color: #f0f0f0;
        }

        .recommendation:hover {
            color: #ffeb3b;
        }

        /* Styling for highlighted restaurant names */
        .recommendation b {
            color: #e74c3c;
            background: linear-gradient(90deg, #e74c3c, #f39c12);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            transition: background 0.5s ease;
        }

        /* Keyframes for dynamic gradient animation */
        @keyframes gradient-move {
            0% { background-position: 0%; }
            100% { background-position: 100%; }
        }

        /* Subtle pulsing effect for the recommendation box */
        .recommendation-box:hover::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.2), transparent 70%);
            border-radius: 20px;
            animation: pulse 1.5s infinite;
            z-index: 0;
        }

        /* Keyframes for pulse effect */
        @keyframes pulse {
            0% { opacity: 0.5; }
            100% { opacity: 0; }
        }

        </style>
    """, unsafe_allow_html=True)

    # Fixed list of random cuisines and locations
    random_cuisines = ['Indian', 'Chinese', 'Italian', 'Mexican', 'Thai', 'Japanese', 'Mediterranean', 'French', 'Spanish', 'Lebanese']
    random_locations = ['Indiranagar', 'Koramangala', 'MG Road', 'Whitefield', 'Jayanagar', 'HSR Layout', 'Marathahalli', 'JP Nagar', 'Electronic City', 'Bellandur']

    # Streamlit select boxes for cuisine and location
    selected_cuisine = st.selectbox("Select a Cuisine", random_cuisines)
    selected_location = st.selectbox("Select a Location", random_locations)

    # Check if selections are made to continue
    if selected_cuisine and selected_location:

        # Train the neural network
        user_data = df[['user_name', 'name', 'rate']].dropna()
        model, scaler, user_encoder, restaurant_encoder = train_neural_network(user_data)

        name = st.text_input("What's your name? ").strip()

        # Personalized recommendations for returning users
        personalized_recommendations = get_personalized_recommendations(name, df, user_data, model, scaler, user_encoder, restaurant_encoder)

        if personalized_recommendations is not None:
            st.write(f"<div class='header'>Personalized recommendations for {name}:</div>", unsafe_allow_html=True)
            for idx, row in personalized_recommendations.iterrows():
                st.markdown(f"""
                    <div class='recommendation-box'>
                        <div class='recommendation'>
                            <b>{row['name']}</b> <br>
                            Location: {row['location']} <br>
                            Rating: {row.get('rate', 'N/A')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Normal recommendation system for new users (API extraction)
            online_order = st.radio("Do you prefer a restaurant with online ordering?", ('yes', 'no')) == 'yes'
            book_table = st.radio("Would you like to book a table?", ('yes', 'no')) == 'yes'

            max_cost = st.slider("What's your maximum budget for two people?", 500, 3000, 1000)

            filters = {
                'cuisine': selected_cuisine,
                'location': selected_location,
                'online_order': online_order,
                'book_table': book_table,
                'max_cost': max_cost
            }

            if st.button('Get Recommendations'):
                recommendations = fetch_restaurants_from_google(
                    filters['location'],
                    filters['cuisine'],
                    filters['max_cost'],
                    filters['online_order'],
                    filters['book_table'],
                    'AIzaSyB7Epudt_9fxH8MXXVBGCiqQr9BsA3DeFM' 
                )

                if recommendations.empty:
                    st.write("Sorry, no restaurants found matching your preferences.")
                else:
                    st.write(f"<div class='header'>Top restaurant recommendations for you:</div>", unsafe_allow_html=True)
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                            <div class='recommendation-box'>
                                <div class='recommendation'>
                                    <b>{row['name']}</b> <br>
                                    Location: {row['location']} <br>
                                    Rating: {row.get('rating', 'N/A')}, Votes: {row.get('votes', 'N/A')} <br>
                                    Approx. Cost: ₹{row['approx_cost(for two people)']} for two <br>
                                    {'  **Table booking available**' if row.get('book_table') == 'yes' else ''}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

