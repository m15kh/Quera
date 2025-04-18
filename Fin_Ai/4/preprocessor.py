
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df):
        # Initialize with a copy of the dataframe
        self.df = df.copy()

    def handle_missing_values(self):
        # Fill missing values with a placeholder (like 0 or any suitable method)
        self.df.fillna(0, inplace=True)

    def handle_date_column(self):
        # Convert 'DateTime_CartFinalize' to datetime format if it's not already
        self.df['DateTime_CartFinalize'] = pd.to_datetime(self.df['DateTime_CartFinalize'], errors='coerce')

    def handle_categorical_columns(self):
        # Encode categorical features (e.g., city names) as numeric
        self.df['city_name_fa'] = pd.factorize(self.df['city_name_fa'])[0]

    def feature_scaling(self):
        # Example of scaling numerical columns, e.g., 'Amount_Gross_Order' and 'Quantity_item'
        scaler = StandardScaler()
        self.df[['Amount_Gross_Order', 'Quantity_item']] = scaler.fit_transform(self.df[['Amount_Gross_Order', 'Quantity_item']])

    def transform(self):
        # Call all preprocessing functions in the desired order
        self.handle_missing_values()
        self.handle_date_column()
        self.handle_categorical_columns()
        self.feature_scaling()
        return self.df
