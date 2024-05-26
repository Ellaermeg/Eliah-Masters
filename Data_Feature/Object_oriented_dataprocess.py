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
            #print("THIS IS CSV PATH", csv_path) #Checking for path
            #print("THIS IS ZIP PATH", zip_path) #Checking for path
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(csv_path) as file:
                    data = pd.read_csv(file, sep=None) #Skjer noe wack her tror jeg
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
        y = traits_data.dropna(subset=['oxygen'])
        y = y.groupby(by = 'key', level = 0) # FUCK DETTE
        y = y.agg({'oxygen': lambda x: x.value_counts().index[0]}) # Something wierd is going on here
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
        terms_data["value"] = 0
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
        y = traits_data.dropna(subset=['oxygen'])
        y = y.groupby(by = 'key', level = 0)
        y = y.agg({'oxygen': lambda x: x.value_counts().index[0]})
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
    
    def data_checker(self, Y_aligned):
        class_distribution = pd.Series(Y_aligned).value_counts()
        assert "Class distribution in Y_aligned:", class_distribution

        unique_labels = np.unique(Y_aligned)
        assert "Unique labels in Y_aligned: {unique_labels}"

        label_encoder = LabelEncoder()
        Y_aligned = label_encoder.fit_transform(Y_aligned)
        Data_Y_aligned= {np.unique(Y_aligned)}

        return Data_Y_aligned


'''class COGsProcessor(DataProcessor):
    def preprocess_terms(self, terms_data):
        return X_filtered_df
    pass
    
    def preprocess_traits(self, traits_data):
       return y
    pass'''

    



# Something like this if i want to make a pipline method
class ModelPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
 
    def setup_pipeline(self):
        # Define the pipeline with steps and parameter grid
        pipeline = Pipeline([
            ("select_k", SelectKBest(f_classif)),
            ("estimator", None)
        ])
        pass

    def train_model(self):
        # Split data, run GridSearchCV, etc.
        pass

    def evaluate_model(self):
        # Calculate metrics, plot confusion matrix, etc.
        pass


'''
# Assuming X_aligned and Y_aligned are already defined and imported
def get_feature_importance(model, X, Y):
    model.fit(X, Y)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])
    sorted_indices = np.argsort(importances)[::-1]
    top_k_indices = sorted_indices[:10]
    selected_features = X.columns[top_k_indices]
    sorted_scores = importances[top_k_indices]
    return selected_features, sorted_scores

def visualize_importance(features, scores, model_name):
    translated_features = translate_ko_terms(list(features))  # Assuming translation function exists
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(features)), scores, tick_label=translated_features)
    plt.xticks(rotation='vertical', fontsize=8)
    plt.xlabel('KO Descriptions')
    plt.ylabel('Importance Scores' if model_name.startswith('RandomForest') else 'Coefficient Magnitudes')
    plt.title(f'Top 10 KO Descriptions by {model_name} Importance')
    plt.tight_layout()
    plt.show()

def map_ko_to_pathways(ko_terms):
    kegg = KEGG()
    pathways = {}
    for ko in ko_terms:
        gene_links = kegg.link("pathway", ko)
        if gene_links:
            for entry in gene_links.strip().split("\n"):
                split_entry = entry.split("\t")
                if len(split_entry) >= 2:
                    ko_id, pathway_id = split_entry[0], split_entry[1]
                    pathways.setdefault(pathway_id, set()).add(ko)
    return pathways

def visualize_network(ko_terms, pathways):
    G = nx.Graph()
    kegg = KEGG()
    for ko in ko_terms:
        G.add_node(ko, title=ko, label=ko, color='red', size=20)
    for pathway_id, kos in pathways.items():
        pathway_info = kegg.get(pathway_id)
        pathway_name = kegg.parse(pathway_info).get('NAME', ['Unknown'])[0]
        G.add_node(pathway_name, title=pathway_name, label=pathway_name, color='blue', size=30)
        for ko in kos:
            G.add_edge(ko, pathway_name)

    nt = Network("800px", "1200px", notebook=True, heading='Interactive Network of KO Terms and Pathways', bgcolor="#ffffff", font_color="black")
    nt.from_nx(G)
    nt.toggle_physics(True)
    nt.show_buttons(filter_=['physics'])
    nt.save_graph("ko_pathways_network.html")
    return nt

# Model Selection
model_choice = 'random_forest'  # 'random_forest' or 'logistic_regression'
model = RandomForestClassifier(n_estimators=100, random_state=42) if model_choice == 'random_forest' else LogisticRegression()
model_name = 'RandomForest Classifier' if model_choice == 'random_forest' else 'Logistic Regression'

selected_features, sorted_scores = get_feature_importance(model, X_aligned, Y_aligned)
visualize_importance(selected_features, sorted_scores, model_name)

# Pathway analysis test 
ko_terms = selected_features
pathways = map_ko_to_pathways(ko_terms)
nt = visualize_network(ko_terms, pathways)
nt.save_graph("ko_pathways_network.html")

'''