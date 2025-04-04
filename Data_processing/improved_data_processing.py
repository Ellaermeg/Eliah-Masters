#!/usr/bin/env python3
"""
Enhanced Data Processing Module for Microbial Trait Prediction

This module provides classes and functions for processing microbial genomic data
and trait information from various sources including BacDive. It supports feature
extraction, data alignment, and preprocessing for machine learning pipelines.
"""

import os
import sys
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

class FeatureManager:
    """Manages feature extraction and preprocessing for genomic annotation data."""
    
    def __init__(self):
        self.encoders = {}
        self.imputers = {}
        self.scalers = {}
    
    def preprocess_features(self, data, feature_type):
        """
        Preprocesses feature data based on the feature type.
        
        Parameters:
        - data: DataFrame containing feature data
        - feature_type: Type of feature (e.g., 'KO', 'GO', 'COG')
        
        Returns:
        - Preprocessed feature matrix
        """
        if data is None or data.empty:
            logging.error(f"No {feature_type} data available for preprocessing")
            return None
        
        logging.info(f"Preprocessing {feature_type} features with {data.shape[0]} samples")
        
        # Check required columns
        required_columns = ['key', feature_type]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns {missing_columns} in {feature_type} data")
            return None
        
        # Convert feature data to binary presence/absence matrix
        try:
            # Pivot the data to create a binary feature matrix
            feature_matrix = pd.pivot_table(
                data, 
                values='value' if 'value' in data.columns else 1,
                index='key',
                columns=feature_type,
                aggfunc='max',
                fill_value=0
            )
            
            logging.info(f"Created feature matrix with shape {feature_matrix.shape}")
            
            # Handle missing values if any
            if feature_matrix.isna().any().any():
                if feature_type not in self.imputers:
                    self.imputers[feature_type] = SimpleImputer(strategy='constant', fill_value=0)
                
                feature_matrix = pd.DataFrame(
                    self.imputers[feature_type].fit_transform(feature_matrix),
                    index=feature_matrix.index,
                    columns=feature_matrix.columns
                )
            
            return feature_matrix
            
        except Exception as e:
            logging.error(f"Error preprocessing {feature_type} features: {str(e)}")
            return None

class TraitManager:
    """Manages trait data preprocessing and encoding."""
    
    def __init__(self):
        self.encoders = {}
    
    def preprocess_traits(self, data, trait_column):
        """
        Preprocesses trait data for machine learning.
        
        Parameters:
        - data: DataFrame containing trait data
        - trait_column: Column name of the trait to process
        
        Returns:
        - Series of preprocessed trait values
        """
        if data is None or data.empty:
            logging.error(f"No trait data available for preprocessing")
            return None
        
        if trait_column not in data.columns:
            logging.error(f"Trait column '{trait_column}' not found in data")
            return None
        
        logging.info(f"Preprocessing trait '{trait_column}' with {data.shape[0]} samples")
        
        # Extract trait values
        trait_values = data[['key', trait_column]].copy()
        trait_values = trait_values.drop_duplicates()
        
        # Check for duplicate keys
        duplicate_keys = trait_values['key'].duplicated()
        if duplicate_keys.any():
            logging.warning(f"Found {duplicate_keys.sum()} duplicate keys in trait data")
            # Keep the first occurrence of each key
            trait_values = trait_values.drop_duplicates(subset=['key'], keep='first')
        
        # Handle categorical traits
        if trait_values[trait_column].dtype == 'object':
            # Check if we need to encode this trait
            unique_values = trait_values[trait_column].unique()
            
            if len(unique_values) == 2:
                # Binary trait - use simple mapping
                logging.info(f"Binary trait detected with values: {unique_values}")
                
                if trait_column not in self.encoders:
                    # Create a binary encoder
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    self.encoders[trait_column] = mapping
                
                # Apply the mapping
                trait_values[trait_column] = trait_values[trait_column].map(self.encoders[trait_column])
                
            elif len(unique_values) > 2:
                # Multi-class trait - use label encoding
                logging.info(f"Multi-class trait detected with {len(unique_values)} values")
                
                if trait_column not in self.encoders:
                    # Create a label encoder
                    encoder = LabelEncoder()
                    self.encoders[trait_column] = encoder.fit(trait_values[trait_column])
                
                # Apply the encoding
                trait_values[trait_column] = self.encoders[trait_column].transform(trait_values[trait_column])
        
        # Set the key as index
        trait_values = trait_values.set_index('key')[trait_column]
        
        logging.info(f"Preprocessed trait '{trait_column}' with {len(trait_values)} samples")
        return trait_values

class DataProcessor:
    """Base class for data processing pipelines."""
    
    def __init__(self, 
                 terms_zip_path=None, 
                 terms_csv_path=None,
                 traits_reduced_zip_path=None, 
                 traits_reduced_csv_path=None,
                 traits_assembled_zip_path=None, 
                 traits_assembled_csv_path=None,
                 bacdive_data_path=None):
        """
        Initialize the data processor with paths to data files.
        
        Parameters:
        - terms_zip_path: Path to the zip file containing terms data
        - terms_csv_path: Path to the CSV file within the zip containing terms data
        - traits_reduced_zip_path: Path to the zip file containing reduced traits data
        - traits_reduced_csv_path: Path to the CSV file within the zip containing reduced traits data
        - traits_assembled_zip_path: Path to the zip file containing assembled traits data
        - traits_assembled_csv_path: Path to the CSV file within the zip containing assembled traits data
        - bacdive_data_path: Path to the merged BacDive data CSV file
        """
        self.terms_zip_path = terms_zip_path
        self.terms_csv_path = terms_csv_path
        self.traits_reduced_zip_path = traits_reduced_zip_path
        self.traits_reduced_csv_path = traits_reduced_csv_path
        self.traits_assembled_zip_path = traits_assembled_zip_path
        self.traits_assembled_csv_path = traits_assembled_csv_path
        self.bacdive_data_path = bacdive_data_path
        
        self.feature_manager = FeatureManager()
        self.trait_manager = TraitManager()
    
    def load_from_zip(self, zip_path, csv_path):
        """
        Loads a CSV file from a zip archive.
        
        Parameters:
        - zip_path: Path to the zip file
        - csv_path: Path to the CSV file within the zip
        
        Returns:
        - DataFrame containing the loaded data
        """
        if not zip_path or not os.path.exists(zip_path):
            logging.error(f"Zip file not found: {zip_path}")
            return None
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                if csv_path not in z.namelist():
                    logging.error(f"CSV file '{csv_path}' not found in zip '{zip_path}'")
                    return None
                
                with z.open(csv_path) as f:
                    df = pd.read_csv(BytesIO(f.read()))
                    logging.info(f"Loaded {len(df)} rows from {csv_path} in {zip_path}")
                    return df
        except Exception as e:
            logging.error(f"Error loading data from {zip_path}/{csv_path}: {str(e)}")
            return None
    
    def load_csv(self, csv_path, sep=','):
        """
        Loads a CSV file directly.
        
        Parameters:
        - csv_path: Path to the CSV file
        - sep: Separator character (default: ',')
        
        Returns:
        - DataFrame containing the loaded data
        """
        if not csv_path or not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path, sep=sep)
            logging.info(f"Loaded {len(df)} rows from {csv_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {csv_path}: {str(e)}")
            return None
    
    def load_terms(self):
        """
        Loads terms data from the specified zip file.
        
        Returns:
        - DataFrame containing terms data
        """
        return self.load_from_zip(self.terms_zip_path, self.terms_csv_path)
    
    def load_reduced_traits_data(self):
        """
        Loads reduced traits data from the specified zip file.
        
        Returns:
        - DataFrame containing reduced traits data
        """
        return self.load_from_zip(self.traits_reduced_zip_path, self.traits_reduced_csv_path)
    
    def load_assembled_traits_data(self):
        """
        Loads assembled traits data from the specified zip file.
        
        Returns:
        - DataFrame containing assembled traits data
        """
        return self.load_from_zip(self.traits_assembled_zip_path, self.traits_assembled_csv_path)
    
    def load_bacdive_data(self):
        """
        Loads merged BacDive data from the specified CSV file.
        
        Returns:
        - DataFrame containing BacDive data
        """
        if not self.bacdive_data_path:
            logging.error("BacDive data path not specified")
            return None
        
        # Try different separators if needed
        for sep in [',', ';', '\t']:
            try:
                df = self.load_csv(self.bacdive_data_path, sep=sep)
                if df is not None and not df.empty:
                    return df
            except:
                continue
        
        logging.error(f"Could not load BacDive data from {self.bacdive_data_path}")
        return None
    
    def preprocess_features(self, terms_data, feature_type):
        """
        Preprocesses feature data using the feature manager.
        
        Parameters:
        - terms_data: DataFrame containing feature data
        - feature_type: Type of feature (e.g., 'KO', 'GO', 'COG')
        
        Returns:
        - Preprocessed feature matrix
        """
        return self.feature_manager.preprocess_features(terms_data, feature_type)
    
    def align_data(self, X, y):
        """
        Aligns feature and label data based on common keys.
        
        Parameters:
        - X: Feature matrix with keys as index
        - y: Label series with keys as index
        
        Returns:
        - Tuple of aligned (X, y) data
        """
        if X is None or y is None:
            logging.error("Cannot align data: features or labels are missing")
            return None, None
        
        # Find common keys
        common_keys = X.index.intersection(y.index)
        
        if len(common_keys) == 0:
            logging.error("No common keys found between features and labels")
            return None, None
        
        logging.info(f"Aligning data on {len(common_keys)} common keys")
        
        # Align data
        X_aligned = X.loc[common_keys]
        y_aligned = y.loc[common_keys]
        
        return X_aligned, y_aligned
    
    def extract_bacdive_features(self, bacdive_data, feature_columns):
        """
        Extracts features from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        - feature_columns: List of column names to use as features
        
        Returns:
        - DataFrame containing extracted features
        """
        if bacdive_data is None or bacdive_data.empty:
            logging.error("No BacDive data available for feature extraction")
            return None
        
        # Check if all feature columns exist
        missing_columns = [col for col in feature_columns if col not in bacdive_data.columns]
        if missing_columns:
            logging.warning(f"Missing feature columns in BacDive data: {missing_columns}")
            # Use only available columns
            feature_columns = [col for col in feature_columns if col in bacdive_data.columns]
            
            if not feature_columns:
                logging.error("No valid feature columns found in BacDive data")
                return None
        
        logging.info(f"Extracting {len(feature_columns)} features from BacDive data")
        
        # Create a key column if not present
        if 'key' not in bacdive_data.columns:
            if 'strain' in bacdive_data.columns:
                # Use strain as key
                bacdive_data['key'] = bacdive_data['strain'].apply(
                    lambda x: x.split('/')[-1] if isinstance(x, str) and '/' in x else x
                )
            elif 'strain_id' in bacdive_data.columns:
                # Use strain_id as key
                bacdive_data['key'] = bacdive_data['strain_id']
            else:
                # Create a synthetic key
                bacdive_data['key'] = [f"BD{i}" for i in range(len(bacdive_data))]
        
        # Extract features
        features = bacdive_data[['key'] + feature_columns].copy()
        
        # Handle missing values
        features = features.fillna('-')
        
        return features
    
    def extract_bacdive_traits(self, bacdive_data, trait_column):
        """
        Extracts trait information from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        - trait_column: Column name of the trait to extract
        
        Returns:
        - DataFrame containing extracted trait data
        """
        if bacdive_data is None or bacdive_data.empty:
            logging.error("No BacDive data available for trait extraction")
            return None
        
        if trait_column not in bacdive_data.columns:
            logging.error(f"Trait column '{trait_column}' not found in BacDive data")
            return None
        
        logging.info(f"Extracting trait '{trait_column}' from BacDive data")
        
        # Create a key column if not present
        if 'key' not in bacdive_data.columns:
            if 'strain' in bacdive_data.columns:
                # Use strain as key
                bacdive_data['key'] = bacdive_data['strain'].apply(
                    lambda x: x.split('/')[-1] if isinstance(x, str) and '/' in x else x
                )
            elif 'strain_id' in bacdive_data.columns:
                # Use strain_id as key
                bacdive_data['key'] = bacdive_data['strain_id']
            else:
                # Create a synthetic key
                bacdive_data['key'] = [f"BD{i}" for i in range(len(bacdive_data))]
        
        # Extract trait data
        traits = bacdive_data[['key', trait_column]].copy()
        
        # Remove rows with missing trait values
        traits = traits.dropna(subset=[trait_column])
        
        # Handle duplicate keys
        if traits['key'].duplicated().any():
            logging.warning(f"Found duplicate keys in trait data, keeping first occurrence")
            traits = traits.drop_duplicates(subset=['key'], keep='first')
        
        return traits
    
    def preprocess_bacdive_traits(self, bacdive_data, trait_column):
        """
        Preprocesses trait data from BacDive.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        - trait_column: Column name of the trait to process
        
        Returns:
        - Series of preprocessed trait values
        """
        traits = self.extract_bacdive_traits(bacdive_data, trait_column)
        if traits is None:
            return None
        
        return self.trait_manager.preprocess_traits(traits, trait_column)
    
    def merge_with_existing_data(self, existing_data, bacdive_data, key_column='key'):
        """
        Merges existing data with new BacDive data.
        
        Parameters:
        - existing_data: DataFrame containing existing data
        - bacdive_data: DataFrame containing BacDive data
        - key_column: Column to use for merging
        
        Returns:
        - DataFrame containing merged data
        """
        if existing_data is None or existing_data.empty:
            logging.warning("No existing data available for merging, using only BacDive data")
            return bacdive_data
        
        if bacdive_data is None or bacdive_data.empty:
            logging.warning("No BacDive data available for merging, using only existing data")
            return existing_data
        
        # Check if key column exists in both dataframes
        if key_column not in existing_data.columns:
            logging.error(f"Key column '{key_column}' not found in existing data")
            return None
        
        if key_column not in bacdive_data.columns:
            logging.error(f"Key column '{key_column}' not found in BacDive data")
            return None
        
        logging.info(f"Merging existing data ({len(existing_data)} rows) with BacDive data ({len(bacdive_data)} rows)")
        
        # Merge data
        merged_data = pd.concat([existing_data, bacdive_data], ignore_index=True)
        
        # Handle duplicate keys
        if merged_data[key_column].duplicated().any():
            logging.warning(f"Found duplicate keys after merging, keeping first occurrence")
            merged_data = merged_data.drop_duplicates(subset=[key_column], keep='first')
        
        logging.info(f"Merged data has {len(merged_data)} rows")
        return merged_data

class KOProcessor(DataProcessor):
    """Processor for KEGG Orthology (KO) data."""
    
    def preprocess_terms(self, terms_data):
        """
        Processes KO terms using the generic feature preprocessing method.
        
        Parameters:
        - terms_data: DataFrame containing KO terms data
        
        Returns:
        - Preprocessed feature matrix
        """
        return self.preprocess_features(terms_data, 'KO')
    
    def extract_ko_from_bacdive(self, bacdive_data):
        """
        Extracts KO terms from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        
        Returns:
        - DataFrame containing KO terms
        """
        # Look for columns that might contain KO information
        ko_columns = [col for col in bacdive_data.columns if 'ko' in col.lower() or 'kegg' in col.lower()]
        
        if not ko_columns:
            logging.warning("No KO-related columns found in BacDive data")
            return None
        
        logging.info(f"Found potential KO columns: {ko_columns}")
        
        # Extract KO data
        ko_data = []
        
        for col in ko_columns:
            # Extract KO terms from the column
            for idx, row in bacdive_data.iterrows():
                if pd.isna(row[col]) or row[col] == '-':
                    continue
                
                # Create a key if not present
                key = row.get('key', row.get('strain_id', f"BD{idx}"))
                
                # Split KO terms if multiple are present
                ko_terms = str(row[col]).split(';')
                
                for ko_term in ko_terms:
                    ko_term = ko_term.strip()
                    if ko_term and ko_term != '-':
                        ko_data.append({
                            'key': key,
                            'KO': ko_term,
                            'value': 1
                        })
        
        if not ko_data:
            logging.warning("No KO terms extracted from BacDive data")
            return None
        
        # Create DataFrame
        ko_df = pd.DataFrame(ko_data)
        logging.info(f"Extracted {len(ko_df)} KO terms from BacDive data")
        
        return ko_df

class GOProcessor(DataProcessor):
    """Processor for Gene Ontology (GO) data."""
    
    def preprocess_terms(self, terms_data):
        """
        Processes GO terms using the generic feature preprocessing method.
        
        Parameters:
        - terms_data: DataFrame containing GO terms data
        
        Returns:
        - Preprocessed feature matrix
        """
        return self.preprocess_features(terms_data, 'GO')
    
    def extract_go_from_bacdive(self, bacdive_data):
        """
        Extracts GO terms from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        
        Returns:
        - DataFrame containing GO terms
        """
        # Look for columns that might contain GO information
        go_columns = [col for col in bacdive_data.columns if 'go' in col.lower() or 'gene_ontology' in col.lower()]
        
        if not go_columns:
            logging.warning("No GO-related columns found in BacDive data")
            return None
        
        logging.info(f"Found potential GO columns: {go_columns}")
        
        # Extract GO data
        go_data = []
        
        for col in go_columns:
            # Extract GO terms from the column
            for idx, row in bacdive_data.iterrows():
                if pd.isna(row[col]) or row[col] == '-':
                    continue
                
                # Create a key if not present
                key = row.get('key', row.get('strain_id', f"BD{idx}"))
                
                # Split GO terms if multiple are present
                go_terms = str(row[col]).split(';')
                
                for go_term in go_terms:
                    go_term = go_term.strip()
                    if go_term and go_term != '-':
                        go_data.append({
                            'key': key,
                            'GO': go_term,
                            'value': 1
                        })
        
        if not go_data:
            logging.warning("No GO terms extracted from BacDive data")
            return None
        
        # Create DataFrame
        go_df = pd.DataFrame(go_data)
        logging.info(f"Extracted {len(go_df)} GO terms from BacDive data")
        
        return go_df

class COGProcessor(DataProcessor):
    """Processor for Clusters of Orthologous Groups (COG) data."""
    
    def preprocess_terms(self, terms_data):
        """
        Processes COG terms using the generic feature preprocessing method.
        
        Parameters:
        - terms_data: DataFrame containing COG terms data
        
        Returns:
        - Preprocessed feature matrix
        """
        return self.preprocess_features(terms_data, 'COG')
    
    def extract_cog_from_bacdive(self, bacdive_data):
        """
        Extracts COG terms from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data
        
        Returns:
        - DataFrame containing COG terms
        """
        # Look for columns that might contain COG information
        cog_columns = [col for col in bacdive_data.columns if 'cog' in col.lower()]
        
        if not cog_columns:
            logging.warning("No COG-related columns found in BacDive data")
            return None
        
        logging.info(f"Found potential COG columns: {cog_columns}")
        
        # Extract COG data
        cog_data = []
        
        for col in cog_columns:
            # Extract COG terms from the column
            for idx, row in bacdive_data.iterrows():
                if pd.isna(row[col]) or row[col] == '-':
                    continue
                
                # Create a key if not present
                key = row.get('key', row.get('strain_id', f"BD{idx}"))
                
                # Split COG terms if multiple are present
                cog_terms = str(row[col]).split(';')
                
                for cog_term in cog_terms:
                    cog_term = cog_term.strip()
                    if cog_term and cog_term != '-':
                        cog_data.append({
                            'key': key,
                            'COG': cog_term,
                            'value': 1
                        })
        
        if not cog_data:
            logging.warning("No COG terms extracted from BacDive data")
            return None
        
        # Create DataFrame
        cog_df = pd.DataFrame(cog_data)
        logging.info(f"Extracted {len(cog_df)} COG terms from BacDive data")
        
        return cog_df

class BacDiveProcessor(DataProcessor):
    """Processor specifically for BacDive data."""
    
    def __init__(self, bacdive_data_path=None):
        """
        Initialize the BacDive processor.
        
        Parameters:
        - bacdive_data_path: Path to the merged BacDive data CSV file
        """
        super().__init__(bacdive_data_path=bacdive_data_path)
        self.ko_processor = KOProcessor()
        self.go_processor = GOProcessor()
        self.cog_processor = COGProcessor()
    
    def extract_all_features(self, bacdive_data=None):
        """
        Extracts all available features from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data (optional, will load from path if not provided)
        
        Returns:
        - Dictionary of feature DataFrames by type
        """
        if bacdive_data is None:
            bacdive_data = self.load_bacdive_data()
        
        if bacdive_data is None:
            logging.error("No BacDive data available for feature extraction")
            return None
        
        logging.info(f"Extracting features from BacDive data with {len(bacdive_data)} rows")
        
        # Extract features by type
        features = {}
        
        # Extract KO terms
        ko_features = self.ko_processor.extract_ko_from_bacdive(bacdive_data)
        if ko_features is not None:
            features['KO'] = ko_features
        
        # Extract GO terms
        go_features = self.go_processor.extract_go_from_bacdive(bacdive_data)
        if go_features is not None:
            features['GO'] = go_features
        
        # Extract COG terms
        cog_features = self.cog_processor.extract_cog_from_bacdive(bacdive_data)
        if cog_features is not None:
            features['COG'] = cog_features
        
        # Extract phenotype features
        phenotype_columns = [
            col for col in bacdive_data.columns 
            if any(term in col.lower() for term in [
                'phenotype', 'gram', 'shape', 'motility', 'pigment', 'spore', 
                'oxygen', 'temperature', 'ph', 'salinity', 'habitat'
            ])
        ]
        
        if phenotype_columns:
            logging.info(f"Found {len(phenotype_columns)} phenotype columns")
            features['phenotype'] = self.extract_bacdive_features(bacdive_data, phenotype_columns)
        
        # Extract genome features
        genome_columns = [
            col for col in bacdive_data.columns 
            if any(term in col.lower() for term in [
                'genome', 'sequence', 'gc', 'size', 'replicon', 'plasmid'
            ])
        ]
        
        if genome_columns:
            logging.info(f"Found {len(genome_columns)} genome columns")
            features['genome'] = self.extract_bacdive_features(bacdive_data, genome_columns)
        
        logging.info(f"Extracted features by type: {list(features.keys())}")
        return features
    
    def extract_all_traits(self, bacdive_data=None):
        """
        Extracts all potential trait columns from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data (optional, will load from path if not provided)
        
        Returns:
        - Dictionary of trait Series by trait name
        """
        if bacdive_data is None:
            bacdive_data = self.load_bacdive_data()
        
        if bacdive_data is None:
            logging.error("No BacDive data available for trait extraction")
            return None
        
        logging.info(f"Extracting traits from BacDive data with {len(bacdive_data)} rows")
        
        # Identify potential trait columns
        trait_columns = [
            col for col in bacdive_data.columns 
            if any(term in col.lower() for term in [
                'gram', 'oxygen', 'temperature', 'ph', 'salinity', 'habitat',
                'motility', 'spore', 'pigment', 'shape', 'pathogen'
            ])
        ]
        
        if not trait_columns:
            logging.warning("No potential trait columns found in BacDive data")
            return None
        
        logging.info(f"Found {len(trait_columns)} potential trait columns: {trait_columns}")
        
        # Extract and preprocess each trait
        traits = {}
        
        for trait_col in trait_columns:
            trait_series = self.preprocess_bacdive_traits(bacdive_data, trait_col)
            if trait_series is not None and len(trait_series) > 0:
                traits[trait_col] = trait_series
        
        logging.info(f"Extracted {len(traits)} traits: {list(traits.keys())}")
        return traits
    
    def prepare_ml_dataset(self, bacdive_data=None, feature_types=None, trait_column=None):
        """
        Prepares a complete machine learning dataset from BacDive data.
        
        Parameters:
        - bacdive_data: DataFrame containing BacDive data (optional, will load from path if not provided)
        - feature_types: List of feature types to include (default: all available)
        - trait_column: Column name of the trait to predict (required)
        
        Returns:
        - Dictionary containing features, labels, and metadata
        """
        if bacdive_data is None:
            bacdive_data = self.load_bacdive_data()
        
        if bacdive_data is None:
            logging.error("No BacDive data available for dataset preparation")
            return None
        
        if trait_column is None:
            logging.error("No trait column specified for prediction")
            return None
        
        logging.info(f"Preparing ML dataset for trait '{trait_column}'")
        
        # Extract features
        all_features = self.extract_all_features(bacdive_data)
        if all_features is None:
            logging.error("Failed to extract features from BacDive data")
            return None
        
        # Filter feature types if specified
        if feature_types is not None:
            all_features = {k: v for k, v in all_features.items() if k in feature_types}
        
        if not all_features:
            logging.error("No features available after filtering")
            return None
        
        # Preprocess features by type
        processed_features = {}
        
        for feature_type, feature_data in all_features.items():
            if feature_type == 'KO':
                processed_features[feature_type] = self.ko_processor.preprocess_terms(feature_data)
            elif feature_type == 'GO':
                processed_features[feature_type] = self.go_processor.preprocess_terms(feature_data)
            elif feature_type == 'COG':
                processed_features[feature_type] = self.cog_processor.preprocess_terms(feature_data)
            elif feature_type in ['phenotype', 'genome']:
                # These are already in the right format
                processed_features[feature_type] = feature_data.set_index('key')
        
        # Remove None values
        processed_features = {k: v for k, v in processed_features.items() if v is not None}
        
        if not processed_features:
            logging.error("No processed features available")
            return None
        
        # Extract and preprocess trait
        trait_series = self.preprocess_bacdive_traits(bacdive_data, trait_column)
        if trait_series is None:
            logging.error(f"Failed to extract trait '{trait_column}' from BacDive data")
            return None
        
        # Combine all feature matrices
        combined_features = None
        feature_names = []
        
        for feature_type, feature_matrix in processed_features.items():
            if combined_features is None:
                combined_features = feature_matrix
                feature_names = [f"{feature_type}_{col}" for col in feature_matrix.columns]
            else:
                # Find common keys
                common_keys = combined_features.index.intersection(feature_matrix.index)
                
                if len(common_keys) == 0:
                    logging.warning(f"No common keys between combined features and {feature_type} features")
                    continue
                
                # Align and concatenate
                combined_features = pd.concat(
                    [combined_features.loc[common_keys], feature_matrix.loc[common_keys]],
                    axis=1
                )
                feature_names.extend([f"{feature_type}_{col}" for col in feature_matrix.columns])
        
        if combined_features is None:
            logging.error("Failed to combine feature matrices")
            return None
        
        # Align features and trait
        X_aligned, y_aligned = self.align_data(combined_features, trait_series)
        
        if X_aligned is None or y_aligned is None:
            logging.error("Failed to align features and trait data")
            return None
        
        logging.info(f"Prepared ML dataset with {X_aligned.shape[0]} samples and {X_aligned.shape[1]} features")
        
        # Create final dataset
        dataset = {
            'features': X_aligned,
            'labels': y_aligned,
            'feature_names': feature_names[:X_aligned.shape[1]],
            'label_name': trait_column
        }
        
        return dataset

# Example Usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'bacdive_data_path': '../merged_data/merged_bacdive_data.csv',
        'target_trait': 'gramStain',
        'feature_types': ['KO', 'GO', 'COG', 'phenotype'],
        'variance_threshold': 0.04
    }
    
    # Initialize processor
    processor = BacDiveProcessor(
        bacdive_data_path=config['bacdive_data_path']
    )
    
    try:
        # Load BacDive data
        bacdive_data = processor.load_bacdive_data()
        
        if bacdive_data is not None:
            # Prepare ML dataset
            dataset = processor.prepare_ml_dataset(
                bacdive_data=bacdive_data,
                feature_types=config['feature_types'],
                trait_column=config['target_trait']
            )
            
            if dataset is not None:
                # Apply feature selection
                selector = VarianceThreshold(threshold=config['variance_threshold'])
                X_selected = selector.fit_transform(dataset['features'])
                
                # Update dataset
                dataset['features'] = X_selected
                dataset['feature_names'] = [
                    dataset['feature_names'][i] 
                    for i in range(len(dataset['feature_names'])) 
                    if selector.get_support()[i]
                ]
                
                print("\nPipeline executed successfully:")
                print(f"- Final features shape: {dataset['features'].shape}")
                print(f"- Final labels shape: {dataset['labels'].shape}")
                print(f"- Features retained: {len(dataset['feature_names'])}")
            else:
                print("\nFailed to prepare ML dataset")
        else:
            print("\nFailed to load BacDive data")
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
