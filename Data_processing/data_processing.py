import zipfile
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class DataProcessor:
    def __init__(self, terms_zip_path, terms_csv_path, traits_reduced_zip_path, traits_reduced_csv_path, traits_assembled_zip_path=None, traits_assembled_csv_path=None):
        self.terms_zip_path = terms_zip_path
        self.terms_csv_path = terms_csv_path
        self.traits_reduced_zip_path = traits_reduced_zip_path
        self.traits_reduced_csv_path = traits_reduced_csv_path
        self.traits_assembled_zip_path = traits_assembled_zip_path
        self.traits_assembled_csv_path = traits_assembled_csv_path

    def load_data_from_zip(self, zip_path, csv_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(csv_path) as file:
                    data = pd.read_csv(file, sep=None, engine="python")
                return data
        except zipfile.BadZipFile:
            print("Error: Bad Zip")
        except FileNotFoundError:
            print("Error: File not found")
        except Exception as e:
            print(f'An error has occurred: {e}')
    
    def load_terms(self):
        return self.load_data_from_zip(self.terms_zip_path, self.terms_csv_path)

    def load_reduced_traits_data(self):
        """Loads reduced traits data, which is the primary dataset."""
        return self.load_data_from_zip(self.traits_reduced_zip_path, self.traits_reduced_csv_path)

    def load_assembled_traits_data(self):
        """Loads assembled traits data if assembled dataset paths are provided."""
        if self.traits_assembled_zip_path and self.traits_assembled_csv_path:
            return self.load_data_from_zip(self.traits_assembled_zip_path, self.traits_assembled_csv_path)
        else:
            return None

    def preprocess_features(self, terms_data, column_name):
        """Processes terms data to create a feature matrix and remove low variance features."""
        terms_data['value'] = 1
        X_terms = terms_data.pivot_table(index='key', columns=column_name, values='value', fill_value=0)

        # Apply Variance Threshold
        selector = VarianceThreshold(threshold=0.04)
        X_filtered = selector.fit_transform(X_terms)
        X_filtered_df = pd.DataFrame(X_filtered, index=X_terms.index, columns=X_terms.columns[selector.get_support()])
        return X_filtered_df

    def align_data(self, X, y):
        """Aligns features and labels based on common keys."""
        common_keys = X.index.intersection(y.index)
        X_aligned = X.loc[common_keys]
        y_aligned = y.loc[common_keys]

        assert X_aligned.shape[0] == y_aligned.shape[0], "X and Y are not aligned"
        return X_aligned, y_aligned

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
        if trait_column not in reduced_traits_data.columns and use_assembled_if_missing:
            # Load assembled traits data if the trait is missing in the reduced dataset
            traits_assembled = self.load_assembled_traits_data()
            if traits_assembled is not None:
                # Merge assembled data to add the new trait
                traits_assembled = traits_assembled.dropna(subset=[trait_column]).query("database == 'bacdive'")
                reduced_traits_data = pd.merge(
                    reduced_traits_data[['key', 'speciesStrain', 'speciesStrainComp']],
                    traits_assembled[[trait_column, 'speciesStrainComp', 'database']],
                    on='speciesStrainComp'
                ) traits.to_csv('traits_indexed.csv', index=False)

        # Process the given trait column if available
        if trait_column == 'trophy':
            reduced_traits_data = reduced_traits_data.dropna(subset=[trait_column])
            reduced_traits_data[trait_column] = reduced_traits_data[trait_column].str.lower()

            # Define trophic levels
            trophic_levels = ['photo', 'chemo', 'litho', 'hetero', 'organo', 'auto']
            
            # Create binary labels for each trophic level
            binary_labels = {}
            for trophy in trophic_levels:
                binary_labels[trophy] = reduced_traits_data[trait_column].apply(lambda x: 1 if trophy in x else 0)

            y = pd.DataFrame(binary_labels)
            return y

        else:
            # For other trait columns, simply return the filtered data
            return reduced_traits_data

class GOProcessor(DataProcessor):
    def preprocess_terms(self, terms_data): 
        """Processes GO terms using the generic feature preprocessing method."""
        return self.preprocess_features(terms_data, 'GO')

# Example Usage
if __name__ == "__main__":
    # Paths to data files
    terms_zip_path = '../Datasets/terms_KO.zip'
    terms_csv_path = 'terms_KO.csv'
    traits_reduced_zip_path = '../Datasets/reducedDataset.zip'
    traits_reduced_csv_path = 'reducedDataset.csv'
    traits_assembled_zip_path = '../Datasets/assembledDataset.zip'
    traits_assembled_csv_path = 'assembledDataset.csv'

    # Instantiate KOProcessor and load data
    processor = KOProcessor(
        terms_zip_path, 
        terms_csv_path, 
        traits_reduced_zip_path, 
        traits_reduced_csv_path, 
        traits_assembled_zip_path=traits_assembled_zip_path, 
        traits_assembled_csv_path=traits_assembled_csv_path
    )

    # Load and preprocess data
    ko_terms = processor.load_terms()
    reduced_traits_data = processor.load_reduced_traits_data()

    # Preprocess KO terms and trophy traits
    X_terms = processor.preprocess_terms(ko_terms)
    y_traits = processor.preprocess_traits(reduced_traits_data, trait_column='trophy', use_assembled_if_missing=True)

    # Align features and labels
    X_aligned, y_aligned = processor.align_data(X_terms, y_traits)
    
    # Check results
    print(f"Aligned X shape: {X_aligned.shape}")
    print(f"Aligned Y shape: {y_aligned.shape}")
