import pandas as pd
import numpy as np
import requests
from fuzzywuzzy import process

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Normalize text columns
    df['cuisines'] = df['cuisines'].astype(str).str.lower().str.replace('[^\w\s,]', '', regex=True).str.strip()
    df['location'] = df['location'].astype(str).str.lower().str.replace('[^\w\s,]', '', regex=True).str.strip()
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce').fillna(0)
    df['online_order'] = df['online_order'].astype(str).str.lower().str.strip()
    df['book_table'] = df['book_table'].astype(str).str.lower().str.strip()

def fuzzy_match_user_input(user_input, options):
    match, score = process.extractOne(user_input, options)
    return match if score > 80 else user_input

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
        print(f"Error fetching place details: {response.status_code}")
        return {}

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
        print("Error fetching data from Google Places API:", response.status_code)
        return pd.DataFrame()  # Return empty DataFrame on error

def get_recommendations(df, filters, api_key):
    google_recommendations = fetch_restaurants_from_google(
        filters['location'],
        filters['cuisine'],
        filters['max_cost'],
        filters['online_order'],
        filters['book_table'],
        api_key
    )

    if not google_recommendations.empty:
        return google_recommendations

    query = (df['cuisines'].str.contains(filters['cuisine'], case=False, na=False)) & \
            (df['location'].str.contains(filters['location'], case=False, na=False)) & \
            (df['approx_cost(for two people)'] <= filters['max_cost']) & \
            (df['online_order'] == filters['online_order']) & \
            (df['book_table'] == filters['book_table'])

    recommendations = df[query]
    if recommendations.empty:
        return recommendations
    else:
        return recommendations.sort_values(by='rate', ascending=False).head(3)

def greet_user(name, df):
    if name in df['name'].unique():
        print(f"Hello {name}! Welcome back!")
        return True
    else:
        print(f"Hello {name}! Let's find you a great place to eat!")
        return False

def fetch_dishes_from_spoonacular(cuisine, api_key):
    url = f"https://api.spoonacular.com/recipes/complexSearch?cuisine={cuisine}&number=5&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            return [(result['title'], result.get('sourceName', 'Unknown Source')) for result in data['results']]
        else:
            return []
    else:
        print("Error fetching data from Spoonacular:", response.status_code)
        return []

def main():
    file_path = '/content/drive/MyDrive/Colab Notebooks/res.csv'
    google_api_key = 'AIzaSyB7Epudt_9fxH8MXXVBGCiqQr9BsA3DeFM'
    spoonacular_api_key = '86a2c2d9bf5a414e8cece53436749a4d'

    df = load_data(file_path)
    clean_data(df)

    names = df['name'].unique().tolist()
    cuisines = df['cuisines'].unique().tolist()
    locations = df['location'].unique().tolist()
    online_order_options = ['yes', 'no']
    book_table_options = ['yes', 'no']
    max_cost_options = ['500', '1000', '2000', '3000']

    name = input("What's your name? ").strip()
    greet_user(name, df)

    print("\nAvailable cuisines:")
    print(cuisines)
    cuisine_input = input("Cuisine you want to eat: ").lower()
    cuisine = fuzzy_match_user_input(cuisine_input, cuisines)

    print("\nAvailable locations:")
    print(locations)
    location_input = input("Preferred location: ").lower()
    location = fuzzy_match_user_input(location_input, locations)

    print("\nDo you prefer a restaurant with online ordering? (yes/no)")
    online_order = input("Enter your choice: ").lower() == 'yes'

    print("\nWould you like to book a table? (yes/no)")
    book_table = input("Enter your choice: ").lower() == 'yes'

    print("\nWhat's your maximum budget for two people? Choose from the following:")
    print(max_cost_options)
    max_cost = float(input("Enter your budget: "))

    filters = {
        'cuisine': cuisine,
        'location': location,
        'online_order': online_order,
        'book_table': book_table,
        'max_cost': max_cost
    }

    print("\nApplied filters:")
    print(filters)

    recommendations = get_recommendations(df, filters, google_api_key)

    if recommendations.empty:
        print("Sorry, no restaurants found matching your preferences.")
    else:
        print("\nTop 3 restaurant recommendations for you:")
        for idx, row in recommendations.iterrows():
            print(f"- {row['name']} (Rating: {row.get('rating', 'N/A')}, Votes: {row.get('votes', 'N/A')})")
            print(f"  Location: {row['location']}, Approx. Cost: {row['approx_cost(for two people)']} for two.")

    dish_recommendations = fetch_dishes_from_spoonacular(cuisine, spoonacular_api_key)
    if dish_recommendations:
        print("\nDish recommendations based on your selected cuisine:")
        for dish, source in dish_recommendations:
            print(f"- {dish} from {source}")
    else:
        print("\nNo specific dish recommendations available.")

if __name__ == "__main__":
    main()
