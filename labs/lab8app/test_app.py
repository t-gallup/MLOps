import requests

def test_happiness_prediction():
    url = "http://localhost:8000/predict"
    
    test_data = {
        "Country": "Germany",
        "Year": 2023,
        "GDP_per_Capita": 45000,
        "Social_Support": 0.91,
        "Healthy_Life_Expectancy": 73.5,
        "Freedom": 0.88,
        "Generosity": 0.20,
        "Corruption_Perception": 0.85,
        "Unemployment_Rate": 5.2,
        "Education_Index": 0.85,
        "Population": 83000000,
        "Urbanization_Rate": 80.5,
        "Life_Satisfaction": 7.2,
        "Public_Trust": 0.67,
        "Mental_Health_Index": 70.3,
        "Income_Inequality": 33.8,
        "Public_Health_Expenditure": 9.2,
        "Climate_Index": 65.1,
        "Work_Life_Balance": 7.8,
        "Internet_Access": 93.5,
        "Crime_Rate": 32.1,
        "Political_Stability": 0.82,
        "Employment_Rate": 68.5
    }
    response = requests.post(url, json=test_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: Success")
        print(f"Happiness Score Prediction: {result['prediction']}")
    else:
        print(f"Error: Status code {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    test_happiness_prediction()