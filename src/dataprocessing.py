import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data():
    happiness_df = pd.read_csv('../data/world_happiness_report.csv')
    return happiness_df

def clean_data(happiness_df):
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_countries = encoder.fit_transform(happiness_df[['Country']])
    encoded_df = pd.DataFrame(
        encoded_countries,
        columns=encoder.get_feature_names_out(['Country'])
    )
    encoded_df.head()
    no_country = happiness_df.drop('Country', axis=1)
    full_df = pd.concat([no_country, encoded_df], axis=1)
    return full_df

def split_data(full_df):
    X = full_df.drop('Happiness_Score', axis=1)
    y = full_df.Happiness_Score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
