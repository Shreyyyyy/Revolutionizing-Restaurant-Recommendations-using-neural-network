import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)  # Use .values to avoid feature name issue

    return X, y, label_encoders, scaler

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_models(X, y):
    y_categorical = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    models = []
    histories = []

    for i in range(5):
        model = build_model(X.shape[1])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=10, verbose=0)
        models.append(model)
        histories.append(history)
    
    return models, histories, X_test, y_test

def plot_histories(histories):
    plt.figure(figsize=(15, 10))
    
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Model {i+1} - Training Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Model {i+1} - Validation Accuracy', linestyle='--')

    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_models(models, X_test, y_test):
    results = []

    for model in models:
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_true = y_test.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred_classes)
        results.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'Model {i+1}' for i in range(5)], y=results)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()

def predict_dishes(df, cuisine, location, dishes_df):
    restaurants = get_recommendations(df, cuisine, location)
    if restaurants.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no recommendations are found

    restaurant_names = restaurants['restaurant_name'].tolist()
    dish_recommendations = dishes_df[dishes_df['restaurant_name'].isin(restaurant_names)]
    
    return dish_recommendations

def recommend_dishes_based_on_cuisine(dishes_df, cuisine):
    if dishes_df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if dishes_df is empty
    
    return dishes_df[dishes_df['dish'].str.contains(cuisine, case=False, na=False)]

def predict_with_neural_network(model, user_input, label_encoders, scaler):
    for column, le in label_encoders.items():
        user_input[column] = le.transform([user_input[column]])[0]

    user_input = scaler.transform([list(user_input.values())])
    prediction = model.predict(user_input)
    return prediction.argmax(axis=1)[0]

def main():
    file_path = '/content/drive/MyDrive/Colab Notebooks/sproj_rrs.csv'
    df = load_data(file_path)

    df['cuisine_type'] = df['cuisine_type'].astype(str).str.lower().fillna('')
    df['price_range'] = df['price_range'].astype(str).str.lower().fillna('')
    df['location'] = df['location'].astype(str).str.lower().fillna('')

    dishes_df = pd.DataFrame({
        'restaurant_name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
        'dish': ['Pasta', 'Pizza', 'Risotto']
    })

    name = input("What's your name? ").strip()
    is_known_user = greet_user(name, df)

    if is_known_user:
        X, y, label_encoders, scaler = preprocess_data(df, name)
        models, histories, X_test, y_test = train_and_evaluate_models(X, y)
        
        plot_histories(histories)
        evaluate_models(models, X_test, y_test)

        cuisine_input = input("Cuisine you want to eat: ").lower()
        cuisine = fuzzy_match_user_input(cuisine_input, df['cuisine_type'].unique().tolist())

        location_input = input("What location are you considering? ").lower()
        location = fuzzy_match_user_input(location_input, df['location'].unique().tolist())

        user_input = {
            'cuisine_type': cuisine,
            'location': location,
            'price_range': df[df['customer_name'] == name]['price_range'].iloc[0]
        }

        prediction = predict_with_neural_network(models[0], user_input, label_encoders, scaler)

        if prediction == 1:
            print(f"Even though you usually prefer other options, since you chose {cuisine}, I recommend you this restaurant in {location}.")
        else:
            print(f"Based on your preferences, I recommend avoiding {cuisine} in {location}.")

        recommendations = get_recommendations(df, cuisine, location)
        print("\nTop 3 restaurant recommendations for you:")

        for idx, row in recommendations.iterrows():
            print(f"- {row['restaurant_name']} (Rating: {row['rating']})")
            print(f"  Reason: This restaurant offers {row['cuisine_type']} cuisine, which matches your preference, and is located in {row['location']}. It's also highly rated by other customers.")
        
        dish_recommendations = predict_dishes(df, cuisine, location, dishes_df)
        if dish_recommendations.empty:
            print("\nNo dishes found for the recommended restaurants.")
        else:
            print("\nRecommended dishes at these restaurants:")
            for idx, row in dish_recommendations.iterrows():
                print(f"- {row['dish']} at {row['restaurant_name']}")

        dish_recommendations_based_on_cuisine = recommend_dishes_based_on_cuisine(dishes_df, cuisine_input)
        if dish_recommendations_based_on_cuisine.empty:
            print("\nNo dishes found for the selected cuisine.")
        else:
            print("\nDishes recommended for your selected cuisine:")
            for idx, row in dish_recommendations_based_on_cuisine.iterrows():
                print(f"- {row['dish']}")

    else:
        cuisine_input = input("Cuisine you want to eat: ").lower()
        cuisine = fuzzy_match_user_input(cuisine_input, df['cuisine_type'].unique().tolist())

        location_input = input("What location are you considering? ").lower()
        location = fuzzy_match_user_input(location_input, df['location'].unique().tolist())

        recommendations = get_recommendations(df, cuisine, location)
        print("\nTop 3 restaurant recommendations for you:")

        for idx, row in recommendations.iterrows():
            print(f"- {row['restaurant_name']} (Rating: {row['rating']})")
            print(f"  Reason: This restaurant offers {row['cuisine_type']} cuisine, which matches your preference, and is located in {row['location']}. It's also highly rated by other customers.")
        
        dish_recommendations = predict_dishes(df, cuisine, location, dishes_df)
        if dish_recommendations.empty:
            print("\nNo dishes found for the recommended restaurants.")
        else:
            print("\nRecommended dishes at these restaurants:")
            for idx, row in dish_recommendations.iterrows():
                print(f"- {row['dish']}")

if __name__ == "__main__":
    main()
