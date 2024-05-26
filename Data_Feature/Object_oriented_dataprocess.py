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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, make_scorer, matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
roc_auc_score, roc_curve, auc
import joblib
import logging
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
                    data = pd.read_csv(file, sep=None, engine="python") #Skjer noe wack her tror jeg
                print("Data loaded successfully:")
                print(data.head())
                print(data.columns)
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



class GOProcessor(DataProcessor):

    def preprocess_terms(self, terms_data): 
        terms_data["value"] = 1
        X_terms = terms_data.pivot_table(index ='key', columns ="GO", values = "value", fill_value=0)

        # Variance threshold for feature removal
        selector = VarianceThreshold(threshold=0.01) # gotta check this wierdness
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
        self.pipeline = None
        self.best_model = None
        logging.basicConfig(level=logging.INFO)

    def setup_pipeline(self, estimators):  
        self.pipeline = Pipeline(estimators)
        logging.info("Pipeline setup with estimators: {}".format(estimators))

    def train_model(self, param_grid, cv=5, n_jobs=-1, verbose=1):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X_train, Y_train)
        self.best_model = grid_search.best_estimator_
        logging.info("Best parameters: {}".format(grid_search.best_params_))

        Y_pred = self.best_model.predict(X_test)
        self.evaluate_model(Y_test, Y_pred)

    def evaluate_model(self, Y_test, Y_pred):
        mcc = matthews_corrcoef(Y_test, Y_pred)
        roc_score = roc_auc_score(Y_test, Y_pred)
        logging.info("MCC: {:.3f}, ROC AUC: {:.3f}".format(mcc, roc_score))
        self.plot_roc(Y_test, Y_pred)

    def plot_roc(self, Y_test, Y_pred):
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    '''def plot_feature_importance(self):
        # Check if the best model is RandomForest and has 'feature_importances_'
        if isinstance(self.best_model, RandomForestClassifier) and hasattr(self.best_model, 'feature_importances_'):
            feature_importances = self.best_model.feature_importances_
            title = 'Random Forest Feature Importances'
        # Check if the best model is Logistic Regression
        elif isinstance(self.best_model, LogisticRegression) and hasattr(self.best_model, 'coef_'):
            feature_importances = self.best_model.coef_[0]  # Logistic regression coefficients for the features
            title = 'Logistic Regression Coefficients'
        else:
            print("The best model does not support direct feature importance or coefficient extraction.")
            return

        # Proceed with extracting top 10 important features
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_k_indices = sorted_indices[:10]  # Get indices of top 10 features
        selected_features = self.X.columns[top_k_indices]
        sorted_scores = feature_importances[top_k_indices]

        # Translate selected features to their descriptions if function available
        try:
            translated_sorted_features = translate_ko_terms(list(selected_features))
            labels = [translated_sorted_features[ko] for ko in selected_features]
        except Exception as e:
            print(f"Could not translate KO terms: {e}")
            labels = selected_features  # Use original feature names if translation fails

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.bar(range(len(labels)), sorted_scores)
        plt.xticks(range(len(labels)), labels, rotation='vertical', fontsize=8)
        plt.xlabel('Feature Descriptions')
        plt.ylabel('Importance Scores')
        plt.title(title)
        plt.tight_layout()
        plt.show()'''


    def compare_models(self, k_range=(1, 1000, 20)):
        # Define the range of `k` values to explore
        k_values = range(*k_range)  # Unpack the range tuple

        # Define estimators to compare
        estimators = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'SupportVectorMachines': SVC(),
            'LogisticRegression': LogisticRegression(),
            'BernoulliNB': BernoulliNB()
        }

        # Prepare to store results for both F1 and MCC
        results = {name: {'f1': [], 'mcc': []} for name in estimators}

        # Initialize StratifiedKFold
        cv = StratifiedKFold(n_splits=5)

        # Loop over each estimator
        for name, estimator in estimators.items():
            logging.info(f"Processing estimator: {name}")
            # Loop over each `k` value
            for k in k_values:
                logging.info(f"Testing with k={k}")
                # Define the pipeline for the current estimator
                pipeline = Pipeline([
                    ('select_k', SelectKBest(f_classif, k=k)),
                    ('estimator', estimator)
                ])
                
                # Perform cross-validation for F1-score
                f1_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1)
                results[name]['f1'].append(f1_scores.mean())
                
                # Perform cross-validation for MCC
                mcc_scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring=make_scorer(matthews_corrcoef), n_jobs=-1)
                results[name]['mcc'].append(mcc_scores.mean())

        # Optional: Plotting can also be integrated here or can be done outside of this function
        self.plot_comparison_results(k_values, results)

    def plot_comparison_results(self, k_values, results):
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))
        for name, scores in results.items():
            k_values_list = list(k_values)  # Convert range to list for indexing
            ax[0].plot(k_values_list, scores['f1'], marker='o', linestyle='-', label=f'{name} F1 Score')
            ax[1].plot(k_values_list, scores['mcc'], marker='o', linestyle='-', label=f'{name} MCC')

        ax[0].set_title('F1 Score by Number of Selected Features (k) for Different Estimators')
        ax[0].set_xlabel('Number of Features (k)')
        ax[0].set_ylabel('F1 Score')
        ax[1].set_title('MCC by Number of Selected Features (k) for Different Estimators')
        ax[1].set_xlabel('Number of Features (k)')
        ax[1].set_ylabel('MCC Score')

        for a in ax:
            a.legend()
            a.grid(True)
        plt.show()

    def save_model(self, path):
        joblib.dump(self.best_model, path)
        logging.info("Model saved to {}".format(path))

    def load_model(self, path):
        self.best_model = joblib.load(path)
        logging.info("Model loaded from {}".format(path))



# Below here is something i want to implement later
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