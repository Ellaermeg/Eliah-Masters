# Dataprocessing
import sys
sys.path.append('../Data_Feature')
sys.path.append('../Datasets')
print(sys.path)
import numpy as np
import warnings
import os
import seaborn as sns
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, LeaveOneOut, cross_val_predict
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, make_scorer, matthews_corrcoef, roc_curve, auc
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import plotly.figure_factory as ff
import networkx as nx
from pyvis.network import Network
import zipfile 
from bioservices import KEGG
from K_func import translate_ko_terms

def load_data():
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
    return

def clean_data():
    return

def feature_extraction():
    return

def label_encoding():
    return


