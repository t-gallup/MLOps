import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

happiness_df = pd.read_csv('data/world_happiness_report.csv')
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_countries = encoder.fit_transform(happiness_df[['Country']])
encoded_df = pd.DataFrame(
    encoded_countries,
    columns=encoder.get_feature_names_out(['Country'])
)
no_country = happiness_df.drop('Country', axis=1)
full_df = pd.concat([no_country, encoded_df], axis=1)
X = full_df.drop('Happiness_Score', axis=1)
y = full_df.Happiness_Score
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

X_train.to_csv('data/processed_X_train_data.csv')
y_train.to_csv('data/processed_y_train_data.csv')
X_val.to_csv('data/processed_X_val_data.csv')
y_val.to_csv('data/processed_y_val_data.csv')
X_test.to_csv('data/processed_X_test_data.csv')
y_test.to_csv('data/processed_y_test_data.csv')
