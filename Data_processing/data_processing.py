import zipfile
import sys
import pandas as pd
import os
import numpy as np
print(os.getcwd())
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

class TraitManager:
    def __init__(self):
        self.trait_mappings = {
            'gram': {
                'levels': {'positive': 'positive', 'negative': 'negative', 'nan': 'nan'},
                'default_value': 'unknown'
            },
            'oxygen': {
                'levels': {
                    'aerobic': 'aerobic', 'aerotolerant': 'aerotolerant', 'microaerophilic': 'microaerophilic',
                    'obligate aerobic': 'obligate aerobic', 'anaerobic': 'anaerobic', 'obligate anaerobic': 'obligate anaerobic',
                    'conflict': 'aerobic', 'facultative': 'facultative'
                },
                'default_value': 'unknown'
            },
            'trophy': {
                'levels': ['photo', 'chemo', 'litho', 'hetero', 'organo', 'auto']
            }
        }

    def preprocess_traits(self, reduced_traits_data, trait_column, use_assembled_if_missing=False):
        """
        Processes traits data from the reduced dataset, optionally integrating the assembled dataset if the trait is missing.

        Parameters:
        - reduced_traits_data: DataFrame containing traits from the reduced dataset.
        - trait_column: The column of the trait that we want to process.
        - use_assembled_if_missing: Boolean to decide whether to use the assembled dataset if the trait column is missing.
        """
        if reduced_traits_data is None:
            print("Error: Reduced traits data is not available.")
            return None

        required_columns = ['key', 'speciesstrain', 'speciesstraincomp']
        missing_columns = [col for col in required_columns if col not in reduced_traits_data.columns]

        if missing_columns:
            print(f"Error: Missing columns {missing_columns} in the reduced traits data.")
            return None

        # If the trait column does not exist in the reduced dataset, optionally use the assembled dataset
        if trait_column not in reduced_traits_data.columns:
            print(f"Trait column '{trait_column}' not found in reduced traits data.")
            if use_assembled_if_missing:
                traits_assembled = self.load_assembled_traits_data()
                if traits_assembled is not None:
                    required_assembled_columns = [trait_column, 'speciesstraincomp', 'database']
                    missing_assembled_columns = [col for col in required_assembled_columns if col not in traits_assembled.columns]

                    if missing_assembled_columns:
                        print(f"Error: Missing columns {missing_assembled_columns} in the assembled traits data.")
                        return None

                    traits_assembled = traits_assembled.dropna(subset=[trait_column]).query("database == 'bacdive'")
                    reduced_traits_data = pd.merge(
                        reduced_traits_data[['key', 'speciesstrain', 'speciesstraincomp']],
                        traits_assembled[[trait_column, 'speciesstraincomp', 'database']],
                        on='speciesstraincomp',
                        how='inner'
                    )
                    print("Merged traits assembled data with reduced data.")
                else:
                    print("Assembled traits data is not available.")
                    return None
            else:
                print(f"Trait column '{trait_column}' not found, and use_assembled_if_missing is set to False.")
                return None

        # If the trait column now exists, process it
        if trait_column in reduced_traits_data.columns:
            reduced_traits_data = reduced_traits_data.dropna(subset=[trait_column])
            if reduced_traits_data.empty:
                print(f"Warning: No data available after dropping NA for trait column '{trait_column}'.")
                return None

            processed_data = self.trait_manager.preprocess_trait(reduced_traits_data, trait_column)
            if processed_data is not None and not processed_data.empty:
                print(f"Processed traits for column '{trait_column}' successfully.")
                return processed_data
            else:
                print(f"Error processing traits for column '{trait_column}'. Processed data is empty or None.")
                return None
        else:
            print(f"Error: Trait column '{trait_column}' not found in the reduced traits data after merging.")
            return None

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
                # Remove BOM if present
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
        """Loads reduced traits data, which is the primary dataset."""
        return self.load_data_from_zip(self.traits_reduced_zip_path, self.traits_reduced_csv_path)

    def load_assembled_traits_data(self):
        """Loads assembled traits data if assembled dataset paths are provided."""
        return self.load_data_from_zip(self.traits_assembled_zip_path, self.traits_assembled_csv_path)

    def preprocess_features(self, terms_data, column_name):
        """Processes terms data to create a feature matrix and remove low variance features."""
        terms_data['value'] = 1
        X_terms = terms_data.pivot_table(index='key', columns=column_name, values='value', fill_value=0)
        return X_terms

    def align_data(self, X, y):
        """Aligns features and labels based on common keys."""
        common_keys = X.index.intersection(y.index)
        X_aligned = X.loc[common_keys]
        y_aligned = y.loc[common_keys]

        assert X_aligned.shape[0] == y_aligned.shape[0], "X and Y are not aligned"
        return X_aligned, y_aligned

    def data_checker(self, Y_aligned):
        # Calculate class distribution and print
        class_distribution = pd.Series(Y_aligned).value_counts()
        print("Class distribution in Y_aligned:", class_distribution)

        # Find unique labels and print
        unique_labels = np.unique(Y_aligned)
        print(f"Unique labels in Y_aligned: {unique_labels}")

        # Encoding labels
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y_aligned)

        # Print encoded labels to check the outcome
        print("Encoded labels:", Y_encoded)

        # Optionally return processed data, here returning the encoded labels and class distribution
        return Y_encoded, class_distribution

class KOProcessor(DataProcessor):
    def preprocess_terms(self, terms_data):
        """Processes KO terms using the generic feature preprocessing method."""
        return self.preprocess_features(terms_data, 'KO')

    def preprocess_traits(self, reduced_traits_data, trait_column, use_assembled_if_missing=False):
        """
        Processes traits data from the reduced dataset, optionally integrating the assembled dataset if the trait is missing.

        Parameters:
        - reduced_traits_data: DataFrame containing traits from the reduced dataset.
        - trait_column: The column of the trait that we want to process.
        - use_assembled_if_missing: Boolean to decide whether to use the assembled dataset if the trait column is missing.
        """
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
                required_assembled_columns = [trait_column, 'speciesstraincomp', 'database']
                missing_assembled_columns = [col for col in required_assembled_columns if col not in traits_assembled.columns]

                if missing_assembled_columns:
                    print(f"Error: Missing columns {missing_assembled_columns} in the assembled traits data.")
                    return None

                traits_assembled = traits_assembled.dropna(subset=[trait_column]).query("database == 'bacdive'")
                reduced_traits_data = pd.merge(
                    reduced_traits_data[['key', 'speciesstrain', 'speciesstraincomp']],
                    traits_assembled[[trait_column, 'speciesstraincomp', 'database']],
                    on='speciesstraincomp',
                    how='inner'
                )

        if trait_column in reduced_traits_data.columns:
            reduced_traits_data = reduced_traits_data.dropna(subset=[trait_column])
            return self.trait_manager.preprocess_trait(reduced_traits_data, trait_column)
        else:
            print(f"Error: Trait column '{trait_column}' not found in the reduced traits data after merging.")
            return None

class GOProcessor(DataProcessor):
    def preprocess_terms(self, terms_data):
        """Processes GO terms using the generic feature preprocessing method."""
        return self.preprocess_features(terms_data, 'GO')

# Example Usage
if __name__ == "__main__":
    # Paths to data files
    terms_zip_path = 'C:/Users/eliah/Documents/Master/Eliah-Masters/Datasets/terms_KO.zip'
    terms_csv_path = 'terms_KO.csv'
    traits_reduced_zip_path = 'C:/Users/eliah/Documents/Master/Eliah-Masters/Datasets/reducedDataset.zip'
    traits_reduced_csv_path = 'reducedDataset.csv'
    traits_assembled_zip_path = 'C:/Users/eliah/Documents/Master/Eliah-Masters/Datasets/assembledDataset.zip'
    traits_assembled_csv_path = 'assembledDataset.csv'

    # Instantiate KOProcessor and load data
    processor = KOProcessor(
        terms_zip_path=terms_zip_path, 
        terms_csv_path=terms_csv_path, 
        traits_reduced_zip_path=traits_reduced_zip_path, 
        traits_reduced_csv_path=traits_reduced_csv_path, 
        traits_assembled_zip_path=traits_assembled_zip_path, 
        traits_assembled_csv_path=traits_assembled_csv_path
    )

    # Load and preprocess data
    ko_terms = processor.load_terms()
    if ko_terms is None:
        raise FileNotFoundError("KO terms could not be loaded. Please check the file paths.")

    reduced_traits_data = processor.load_reduced_traits_data()
    if reduced_traits_data is None:
        raise FileNotFoundError("Reduced traits data could not be loaded. Please check the file paths.")

    # Preprocess KO terms and traits (trophy, gram, oxygen)
    X_terms = processor.preprocess_terms(ko_terms)
    y_traits = processor.preprocess_traits(reduced_traits_data, trait_column='trophy', use_assembled_if_missing=True)

    # Check if y_traits was processed correctly
    if y_traits is None:
        raise ValueError("Traits data could not be processed. Please check the log for errors.")

    # Align features and labels
    X_aligned, y_aligned = processor.align_data(X_terms, y_traits)

    # Feature Selection: Variance Threshold
    selector = VarianceThreshold(threshold=0.04)
    X_aligned = selector.fit_transform(X_aligned)

    # Check results
    print(f"Aligned X shape after feature selection: {X_aligned.shape}")
    print(f"Aligned Y shape: {y_aligned.shape}")
