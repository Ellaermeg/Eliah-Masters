import sys
sys.path.append('../Data_Feature')
sys.path.append('../Datasets')
print(sys.path)
import numpy as np
import os
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB # or GaussianNB if your data is normalized and continuous
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, make_scorer, matthews_corrcoef
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import zipfile 
from bioservices import KEGG
from K_func import translate_ko_terms

class DataProcessor:
    def __init__(self, terms_zip_path, terms_csv_path, traits_zip_path, traits_csv_path):
        self.terms_zip_path = terms_zip_path
        self.terms_csv_path = terms_csv_path
        self.traits_zip_path = traits_zip_path
        self.traits_csv_path = traits_csv_path

    def load_data_from_zip(self, zip_path, csv_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(csv_path) as file:
                    data = pd.read_csv(file, index_col=0, sep=None)
                print("Data loaded successfully:")
                print(data.head())
                return data
        except zipfile.BadZipFile:
            print("Error: Bad Zip")
        except FileNotFoundError:
            print("Error: file not found")
        except Exception as e:
            print(f'An error has occured: {e}')
    
    def preprocess_terms(self, terms_data):
        raise NotImplementedError("Use subclasses, eg: traits_data .")

    def preprocess_traits_oxygen(self, traits_data):
        raise NotImplementedError("Use subclasses, eg: terms_data")



class KOProcessor(DataProcessor):
    def preprocess_terms(self, terms_data):
        terms_data['value'] = 1
        X_terms = terms_data.pivot_table(index='key', columns='KO', values='value', fill_value=0)

        # Variance threshold for removal of features
        selector = VarianceThreshold(threshold=0.01)
        X_filtered = selector.fit_transform(X_terms)
        X_filtered_df = pd.DataFrame(X_filtered, index=X_terms.index, columns=X_terms.columns[selector.get_support()])
        return X_filtered_df

    def preprocess_traits_oxygen(self, traits_data):
        traits_data['oxygen'] = traits_data['oxygen'].str.lower()
        traits_data['oxygen'] = traits_data['oxygen'].map({
            'aerobic': 'aerobic',
            'aerotolerant': 'aerobic',
            'microaerophilic': 'aerobic',
            'obligate aerobic': 'aerobic',
            'anaerobic': 'anaerobic',
            'obligate anaerobic': 'anaerobic',
            'conflict': 'aerobic',  
            'facultative': 'aerobic'  
        })
        y = traits_data.dropna(subset=['oxygen']).groupby('key').agg({'oxygen': lambda x: x.value_counts().index[0]})
        return y
    
    def align_data(self, X, y):
        # Find common keys after removing missing values
        common_keys = X.index.intersection(y.index)

        # Align X (features) and Y (labels) based on common keys
        X_aligned = X.loc[common_keys]
        Y_aligned = y.loc[common_keys].values.ravel()

        # Ensures X_aligned and Y_aligned are aligned
        assert X_aligned.shape[0] == len(Y_aligned), "X and Y are not aligned"

        return X_aligned, Y_aligned



class GOProcessor(DataProcessor):

    def preprocess_terms(self, terms_data): 

        terms_data["value"] = 1
        X_terms = terms_data.pivot_table(index = "key", columns = "GO", values = "value", fill_value=0)

        # Variance threshold for feature removal
        selector = VarianceThreshold(threshold=0.01)
        X_filtered = selector.fit_transform(X_terms)
        X = pd.DataFrame(X_filtered, index=X_terms.index, columns= X_terms.columns[selector.get_support()])
        return X
    

    def preprocess_traits_oxygen(self, traits_data):
        traits_data['oxygen'] = traits_data['oxygen'].str.lower()
        traits_data['oxygen'] = traits_data['oxygen'].map({
            'aerobic': 'aerobic',
            'aerotolerant': 'aerobic',
            'microaerophilic': 'aerobic',
            'obligate aerobic': 'aerobic',
            'anaerobic': 'anaerobic',
            'obligate anaerobic': 'anaerobic',
            'conflict': 'aerobic',  
            'facultative': 'aerobic'  
        })
        y = traits_data.dropna(subset=['oxygen']).groupby('key').agg({'oxygen': lambda x: x.value_counts().index[0]})
        return y

    def align_data(self, X, y):
        # Find common keys after removing missing values
        common_keys = X.index.intersection(y.index)

        # Align X (features) and Y (labels) based on common keys
        X_aligned = X.loc[common_keys]
        Y_aligned = y.loc[common_keys].values.ravel()

        # Ensures X_aligned and Y_aligned are aligned
        assert X_aligned.shape[0] == len(Y_aligned), "X and Y are not aligned"

        return X_aligned, Y_aligned

'''class COGsProcessor(DataProcessor):
    #def preprocess_terms(self, terms_data):
        return X_filtered_df
    
    def preprocess_traits(self, traits_data):
       return y
    '''



# Something like this if i want to make a pipline method
class ModelPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def setup_pipeline(self):
        # Define the pipeline with steps and parameter grid
        pass

    def train_model(self):
        # Split data, run GridSearchCV, etc.
        pass

    def evaluate_model(self):
        # Calculate metrics, plot confusion matrix, etc.
        pass
