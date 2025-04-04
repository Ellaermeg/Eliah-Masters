#!/usr/bin/env python3
"""
Improved BacDive TSV Merger

This script merges multiple TSV files from BacDive into a single comprehensive CSV file.
It ensures data integrity by properly handling strain identifiers and merging related data.
"""
import sys
sys.path.append("../Eliah-Masters")
import pandas as pd
import zipfile
import os
import sys
from io import BytesIO
import chardet
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tsv_merge.log"),
        logging.StreamHandler()
    ]
)

def clean_column_names(columns):
    """Clean column names by removing '?' prefix and standardizing format"""
    cleaned = []
    for col in columns:
        # Remove '?' prefix if present
        if col.startswith('?'):
            col = col[1:]
        # Convert to lowercase and replace spaces with underscores
        col = col.lower().replace(' ', '_')
        cleaned.append(col)
    return cleaned

def extract_strain_id(strain_uri):
    """Extract numeric ID from strain URI"""
    if pd.isna(strain_uri) or not isinstance(strain_uri, str):
        return None
    
    match = re.search(r'/strain/(\d+)', strain_uri)
    if match:
        return match.group(1)
    return strain_uri

def process_tsv(content, file_name):
    """Process TSV content with robust error handling"""
    file_base = os.path.basename(file_name).replace('bacdive_', '').replace('.tsv', '')
    logging.info(f"Processing {file_base}")
    
    # Detect encoding
    try:
        encoding = chardet.detect(content)['encoding']
        text_content = content.decode(encoding or 'utf-8')
    except UnicodeDecodeError:
        logging.warning(f"Unicode decode error in {file_name}, falling back to latin1")
        text_content = content.decode('latin1', errors='replace')
    
    # First count columns in header
    header = text_content.split('\n')[0]
    expected_columns = len(header.split('\t'))
    
    # Process content line-by-line
    lines = []
    for i, line in enumerate(text_content.split('\n')):
        if i == 0:  # Header row
            cols = clean_column_names(line.strip().split('\t'))
            continue
        
        if not line.strip():
            continue
            
        fields = line.strip().split('\t')
        if len(fields) != expected_columns:
            logging.warning(f"Line {i+1} in {file_name}: Expected {expected_columns} columns, found {len(fields)}")
            # Pad with empty fields or truncate as needed
            if len(fields) > expected_columns:
                fields = fields[:expected_columns]
            else:
                fields += [''] * (expected_columns - len(fields))
        
        lines.append(fields)
    
    # Create DataFrame
    df = pd.DataFrame(lines, columns=cols)
    
    # Clean data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Add file source column to track origin
    df['source_file'] = file_base
    
    # Extract strain ID for better joining
    if 'strain' in df.columns:
        df['strain_id'] = df['strain'].apply(extract_strain_id)
        
        # Group by strain to consolidate duplicate entries
        grouped = df.groupby('strain', as_index=False).agg(
            lambda x: '; '.join(x.dropna().astype(str).unique()) if x.notna().any() else '-'
        )
        
        # Keep the strain_id column for joining
        if 'strain_id' not in grouped.columns and 'strain_id' in df.columns:
            strain_ids = df[['strain', 'strain_id']].drop_duplicates()
            grouped = pd.merge(grouped, strain_ids, on='strain', how='left')
        
        return grouped
    
    logging.warning(f"No 'strain' column found in {file_name}, skipping")
    return None

def merge_tsv_files(tsv_dir, output_path, use_zip=False):
    """Merge all TSV files from a directory or zip archive into a single CSV file"""
    dfs = []
    file_count = 0
    
    if use_zip and os.path.isfile(tsv_dir):
        # Process files from zip archive
        with zipfile.ZipFile(tsv_dir, 'r') as z:
            tsv_files = [f for f in z.namelist() if f.endswith('.tsv') and not os.path.basename(f).startswith('._')]
            logging.info(f"Found {len(tsv_files)} TSV files in archive")
            
            for file_name in tsv_files:
                file_count += 1
                with z.open(file_name) as f:
                    content = f.read()
                    processed_df = process_tsv(content, file_name)
                    if processed_df is not None:
                        dfs.append(processed_df)
    else:
        # Process files from directory
        tsv_files = [f for f in os.listdir(tsv_dir) 
                    if f.endswith('.tsv') and not f.startswith('._') and os.path.isfile(os.path.join(tsv_dir, f))]
        logging.info(f"Found {len(tsv_files)} TSV files in directory")
        
        for file_name in tsv_files:
            file_count += 1
            file_path = os.path.join(tsv_dir, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
                processed_df = process_tsv(content, file_name)
                if processed_df is not None:
                    dfs.append(processed_df)
    
    if not dfs:
        logging.error("No valid TSV files found or processed.")
        return False
    
    logging.info(f"Successfully processed {len(dfs)} files out of {file_count}")
    
    # Merge all dataframes on 'strain'
    logging.info("Merging dataframes...")
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        logging.info(f"Merging dataframe {i+1}/{len(dfs)}")
        merged_df = pd.merge(
            merged_df,
            df,
            on='strain',
            how='outer',
            suffixes=('', f'_dup{i}'))
    
    # Remove duplicate columns
    duplicate_cols = [col for col in merged_df.columns if '_dup' in col]
    if duplicate_cols:
        logging.info(f"Removing {len(duplicate_cols)} duplicate columns")
        merged_df = merged_df.drop(columns=duplicate_cols)
    
    # Final cleaning
    logging.info("Performing final data cleaning")
    merged_df = merged_df.fillna("-")
    merged_df = merged_df.replace(["", "nan", "NaN"], "-")
    
    # Reorder columns to put strain and strain_id first
    if 'strain' in merged_df.columns:
        cols = ['strain']
        if 'strain_id' in merged_df.columns:
            cols.append('strain_id')
        cols.extend([col for col in merged_df.columns if col not in cols])
        merged_df = merged_df[cols]
    
    # Save result
    logging.info(f"Saving merged data to {output_path}")
    merged_df.to_csv(output_path, sep=',', index=False)
    
    # Also save as Excel for easier viewing
    excel_path = output_path.replace('.csv', '.xlsx')
    try:
        merged_df.to_excel(excel_path, index=False)
        logging.info(f"Excel version saved to {excel_path}")
    except Exception as e:
        logging.warning(f"Could not save Excel version: {e}")
    
    logging.info(f"Merge complete. {merged_df.shape[0]} rows and {merged_df.shape[1]} columns in final dataset.")
    return True

def main():
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use the original zip file with absolute path
    zip_path = os.path.join(current_dir, '..', 'Datasets', 'tsv.zip')
    
    if not os.path.isfile(zip_path):
        logging.error(f"Could not find TSV zip file at {zip_path}")
        sys.exit(1)
    
    logging.info(f"Found TSV zip file at {zip_path}")
    
    output_dir = os.path.join(current_dir, '..', 'merged_data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_bacdive_data.csv')
    
    logging.info(f"Starting TSV merge process")
    logging.info(f"Input: {zip_path}")
    logging.info(f"Output: {output_path}")
    
    success = merge_tsv_files(zip_path, output_path, use_zip=True)
    
    if success:
        print(f"\nMerge completed successfully!")
        print(f"Output saved to: {output_path}")
    else:
        print(f"\nMerge process failed. Check the logs for details.")

if __name__ == "__main__":
    main()
