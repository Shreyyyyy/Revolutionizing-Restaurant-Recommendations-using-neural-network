import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from fuzzywuzzy import process

def load_data(file_path):
    return pd.read_csv(file_path)

def fuzzy_match_user_input(user_input, options):
    match, score = process.extractOne(user_input, options)
    return match if score > 80 else user_input

def get_recommendations(df, cuisine, location):
    # Filter based on cuisine and location
    recommendations = df[(df['cuisine_type'] == cuisine) & (df['location'] == location)]
    return recommendations.sort_values(by='rating', ascending=False).head(3)

def greet_user(name, df):
    if name in df['customer_name'].unique():
        print(f"Hello {name}! Welcome back!")
        return True
    else:
        print(f"Hello {name}! Let's find you a great place to eat!")
        return False

def preprocess_data(df, user_name):
    user_data = df[df['customer_name'] == user_name]
    X = user_data[['cuisine_type', 'location', 'price_range']]
    y = user_data['liked']

    # Convert categorical features to numerical using LabelEncoder
    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, label_encoders, scaler

def train_neural_network(X, y):
    # Convert target variable to categorical
    y_categorical = to_categorical(y, num_classes=2)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Build the neural network model
    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with validation
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=10, verbose=1)

    return model

def predict_with_neural_network(model, user_input, label_encoders, scaler):
    # Encode the input features using the same label encoders
    for column, le in label_encoders.items():
        user_input[column] = le.transform([user_input[column]])[0]

    # Standardize the input features
    user_input = scaler.transform([list(user_input.values())])

    # Predict the user's preference
    prediction = model.predict(user_input)
    return prediction.argmax(axis=1)[0]

def predict_dishes(df, cuisine, location, dishes_df):
    # Find restaurants that match the criteria
    restaurants = get_recommendations(df, cuisine, location)
    
    # Extract restaurant names
    restaurant_names = restaurants['restaurant_name'].tolist()
    
    # Get dish recommendations from dishes_df
    dish_recommendations = dishes_df[dishes_df['restaurant_name'].isin(restaurant_names)]
    
    return dish_recommendations

def main():
    # Load data
    file_path = '/content/drive/MyDrive/Colab Notebooks/sproj_rrs.csv'
    df = load_data(file_path)

    # Ensure relevant columns are treated as strings and handle NaNs
    df['cuisine_type'] = df['cuisine_type'].astype(str).str.lower().fillna('')
    df['price_range'] = df['price_range'].astype(str).str.lower().fillna('')
    df['location'] = df['location'].astype(str).str.lower().fillna('')

    # Sample DataFrame for dishes (Replace with actual data or load from a file)
    dishes_df = pd.DataFrame({
        'restaurant_name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
        'dish': ['Pasta', 'Pizza', 'Risotto']
    })

    # Get user name and greet
    name = input("What's your name? ").strip()
    is_known_user = greet_user(name, df)

    if is_known_user:
        # Preprocess data and train neural network
        X, y, label_encoders, scaler = preprocess_data(df, name)
        model = train_neural_network(X, y)

        # Get user input with fuzzy matching
        cuisine_input = input("Cuisine you want to eat: ").lower()
        cuisine = fuzzy_match_user_input(cuisine_input, df['cuisine_type'].unique().tolist())

        location_input = input("What location are you considering? ").lower()
        location = fuzzy_match_user_input(location_input, df['location'].unique().tolist())

        # Prepare user input for prediction
        user_input = {
            'cuisine_type': cuisine,
            'location': location,
            'price_range': df[df['customer_name'] == name]['price_range'].iloc[0]
        }

        # Predict user's preference
        prediction = predict_with_neural_network(model, user_input, label_encoders, scaler)

        if prediction == 1:
            print(f"Even though you usually prefer other options, since you chose {cuisine}, I recommend you this restaurant in {location}.")
        else:
            print(f"Based on your preferences, I recommend avoiding {cuisine} in {location}.")

        # Get top 3 recommendations
        recommendations = get_recommendations(df, cuisine, location)
        print("\nTop 3 restaurant recommendations for you:")

        for idx, row in recommendations.iterrows():
            print(f"- {row['restaurant_name']} (Rating: {row['rating']})")
            print(f"  Reason: This restaurant offers {row['cuisine_type']} cuisine, which matches your preference, and is located in {row['location']}. It's also highly rated by other customers.")
        
        # Get dish recommendations
        dish_recommendations = predict_dishes(df, cuisine, location, dishes_df)
        print("\nRecommended dishes at these restaurants:")
        for idx, row in dish_recommendations.iterrows():
            print(f"- {row['dish']} at {row['restaurant_name']}")

    else:
        # General recommendation flow
        cuisine_input = input("Cuisine you want to eat: ").lower()
        cuisine = fuzzy_match_user_input(cuisine_input, df['cuisine_type'].unique().tolist())

        location_input = input("What location are you considering? ").lower()
        location = fuzzy_match_user_input(location_input, df['location'].unique().tolist())

        # Get top 3 recommendations
        recommendations = get_recommendations(df, cuisine, location)
        print("\nTop 3 restaurant recommendations for you:")

        for idx, row in recommendations.iterrows():
            print(f"- {row['restaurant_name']} (Rating: {row['rating']})")
            print(f"  Reason: This restaurant offers {row['cuisine_type']} cuisine, which matches your preference, and is located in {row['location']}. It's also highly rated by other customers.")
        
        # Get dish recommendations
        dish_recommendations = predict_dishes(df, cuisine, location, dishes_df)
        print("\nRecommended dishes at these restaurants:")
        for idx, row in dish_recommendations.iterrows():
            print(f"- {row['dish']} at {row['restaurant_name']}")

if __name__ == "__main__":
    main()
