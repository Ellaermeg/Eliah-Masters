import zipfile
import sys
import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

class TraitManager:
    def __init__(self):
        self.default_value = 'unknown'

    def detect_delimiter(self, series):
        delimiters = [',', '|', ';']
        delimiter_counts = {delim: 0 for delim in delimiters}
        for entry in series.dropna():
            entry_str = str(entry)
            for delim in delimiters:
                if delim in entry_str:
                    delimiter_counts[delim] += 1
        max_count = max(delimiter_counts.values())
        if max_count == 0:
            return None
        for delim, count in delimiter_counts.items():
            if count == max_count:
                return delim
        return None

    def split_and_standardize(self, entry, delimiter):
        if pd.isna(entry):
            return []
        entry_str = str(entry).strip()
        if delimiter:
            tokens = entry_str.split(delimiter)
        else:
            tokens = [entry_str]
        standardized = []
        for token in tokens:
            token = token.strip().lower().replace(' ', '_')
            if token:
                standardized.append(token)
        return standardized

    def preprocess_traits(self, reduced_traits_data, trait_column, use_assembled_if_missing=False):
        if reduced_traits_data is None:
            print("Error: Reduced traits data is not available.")
            return None

        required_columns = ['key', 'speciesStrain', 'speciesStrainComp']
        missing_columns = [col for col in required_columns if col not in reduced_traits_data.columns]

        if missing_columns:
            print(f"Error: Missing columns {missing_columns} in the reduced traits data.")
            return None

        if trait_column not in reduced_traits_data.columns:
            print(f"Trait column '{trait_column}' not found in reduced traits data.")
            if use_assembled_if_missing:
                traits_assembled = self.load_assembled_traits_data()
                if traits_assembled is not None:
                    required_assembled_columns = [trait_column, 'speciesStrainComp', 'database']
                    missing_assembled_columns = [col for col in required_assembled_columns if col not in traits_assembled.columns]

                    if missing_assembled_columns:
                        print(f"Error: Missing columns {missing_assembled_columns} in the assembled traits data.")
                        return None

                    traits_assembled = traits_assembled.dropna(subset=[trait_column]).query("database == 'bacdive'")
                    reduced_traits_data = pd.merge(
                        reduced_traits_data[['key', 'speciesStrain', 'speciesStrainComp']],
                        traits_assembled[[trait_column, 'speciesStrainComp', 'database']],
                        on='speciesStrainComp',
                        how='inner'
                    )
                    print("Merged traits assembled data with reduced data.")
                else:
                    print("Assembled traits data is not available.")
                    return None
            else:
                print(f"Trait column '{trait_column}' not found, and use_assembled_if_missing is set to False.")
                return None

        if trait_column in reduced_traits_data.columns:
            reduced_traits_data = reduced_traits_data.dropna(subset=[trait_column])
            if reduced_traits_data.empty:
                print(f"Warning: No data available after dropping NA for trait column '{trait_column}'.")
                return None

            processed_data = self._process_trait_column(reduced_traits_data, trait_column)
            if processed_data is not None and not processed_data.empty:
                processed_df = pd.DataFrame({
                    'key': reduced_traits_data['key'],
                    trait_column: processed_data
                })
                aggregated = processed_df.groupby('key')[trait_column].agg(
                    lambda x: x.value_counts().idxmax() if not x.empty else None
                ).dropna()
                print(f"Aggregated labels for '{trait_column}'. Unique keys: {len(aggregated)}")
                return aggregated
            else:
                print(f"Error processing traits for column '{trait_column}'. Processed data is empty or None.")
                return None
        else:
            print(f"Error: Trait column '{trait_column}' not found in the reduced traits data after merging.")
            return None

    def _process_trait_column(self, data, trait_column):
        delimiter = self.detect_delimiter(data[trait_column])
        tokens = data[trait_column].apply(lambda x: self.split_and_standardize(x, delimiter))
        all_tokens = [token for sublist in tokens for token in sublist]
        unique_tokens = pd.unique(all_tokens)
        levels = {token: token for token in unique_tokens}
        processed = tokens.apply(lambda x: x[0] if x else self.default_value)
        return processed

class DataProcessor:
    def __init__(self, terms_zip_path=None, terms_csv_path=None, traits_reduced_zip_path=None, traits_reduced_csv_path=None, traits_assembled_zip_path=None, traits_assembled_csv_path=None):
        self.terms_zip_path = terms_zip_path
        self.terms_csv_path = terms_csv_path
        self.traits_reduced_zip_path = traits_reduced_zip_path
        self.traits_reduced_csv_path = traits_reduced_csv_path
        self.traits_assembled_zip_path = traits_assembled_zip_path
        self.traits_assembled_csv_path = traits_assembled_csv_path
        self.trait_manager = TraitManager()

    def load_data_from_zip(self, zip_path, csv_path):
        if not zip_path or not csv_path:
            print("Error: Zip path or CSV path not provided.")
            return None
        if not os.path.exists(zip_path):
            print(f"Error: The file {zip_path} does not exist.")
            return None
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(csv_path) as file:
                    data = pd.read_csv(file, sep=None, engine="python")
            if data is None or data.empty:
                print(f"Warning: The loaded data from {csv_path} is empty.")
            else:
                data.columns = data.columns.str.replace('^\ufeff', '', regex=True)
                print("Data loaded successfully:")
                print(data.head())
            return data
        except zipfile.BadZipFile:
            print("Error: Bad Zip")
        except FileNotFoundError:
            print(f"Error: File not found for {zip_path} or {csv_path}")
        except Exception as e:
            print(f'An error has occurred: {e}')
        return None

    def load_terms(self):
        return self.load_data_from_zip(self.terms_zip_path, self.terms_csv_path)

    def load_reduced_traits_data(self):
        return self.load_data_from_zip(self.traits_reduced_zip_path, self.traits_reduced_csv_path)

    def load_assembled_traits_data(self):
        return self.load_data_from_zip(self.traits_assembled_zip_path, self.traits_assembled_csv_path)

    def preprocess_features(self, terms_data, column_name):
        terms_data['value'] = 1
        X_terms = terms_data.pivot_table(index='key', columns=column_name, values='value', fill_value=0)
        return X_terms

    def align_data(self, X, y):
        y = y[~y.index.duplicated(keep='first')]
        common_keys = X.index.intersection(y.index)
        X_aligned = X.loc[common_keys]
        y_aligned = y.loc[common_keys]
        assert X_aligned.shape[0] == y_aligned.shape[0], "X and Y are not aligned"
        return X_aligned, y_aligned

    def data_checker(self, Y_aligned):
        class_distribution = pd.Series(Y_aligned).value_counts()
        print("Class distribution in Y_aligned:", class_distribution)
        unique_labels = np.unique(Y_aligned)
        print(f"Unique labels in Y_aligned: {unique_labels}")
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y_aligned)
        print("Encoded labels:", Y_encoded)
        return Y_encoded, class_distribution

class KOProcessor(DataProcessor): 
    def preprocess_terms(self, terms_data):
        return self.preprocess_features(terms_data, 'KO')

        """
        Processes traits data from the reduced dataset, optionally integrating the assembled dataset if the trait is missing.

        Parameters:
        - reduced_traits_data: DataFrame containing traits from the reduced dataset.
        - trait_column: The column of the trait that we want to process.
        - use_assembled_if_missing: Boolean to decide whether to use the assembled dataset if the trait column is missing.
        """

    def preprocess_traits(self, reduced_traits_data, trait_column, use_assembled_if_missing=False):
        if reduced_traits_data is None:
            print("Error: Reduced traits data is not available.")
            return None
        required_columns = ['key', 'speciesStrain', 'speciesStrainComp']
        missing_columns = [col for col in required_columns if col not in reduced_traits_data.columns]
        if missing_columns:
            print(f"Error: Missing columns {missing_columns} in the reduced traits data.")
            return None
        if trait_column not in reduced_traits_data.columns and use_assembled_if_missing:
            traits_assembled = self.load_assembled_traits_data()
            if traits_assembled is not None:
                required_assembled_columns = [trait_column, 'speciesStrainComp', 'database']
                missing_assembled_columns = [col for col in required_assembled_columns if col not in traits_assembled.columns]
                if missing_assembled_columns:
                    print(f"Error: Missing columns {missing_assembled_columns} in the assembled traits data.")
                    return None
                traits_assembled = traits_assembled.dropna(subset=[trait_column]).query("database == 'bacdive'")
                reduced_traits_data = pd.merge(
                    reduced_traits_data[['key', 'speciesStrain', 'speciesStrainComp']],
                    traits_assembled[[trait_column, 'speciesStrainComp', 'database']],
                    on='speciesStrainComp',
                    how='inner'
                )
        if trait_column in reduced_traits_data.columns:
            reduced_traits_data = reduced_traits_data.dropna(subset=[trait_column])
            return self.trait_manager.preprocess_traits(reduced_traits_data, trait_column)
        else:
            print(f"Error: Trait column '{trait_column}' not found in the reduced traits data after merging.")
            return None

# check error
if __name__ == "__main__":
    print("test")
