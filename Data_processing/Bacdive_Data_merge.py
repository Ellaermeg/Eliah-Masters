import sys
sys.path.append("../Eliah-Masters")
import pandas as pd
import zipfile
import os
from io import BytesIO
import chardet

# Path to files
current_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(current_dir, '..', 'Datasets', 'tsv.zip')

def process_tsv(content, file_name):
    """Process TSV content with robust error handling"""
    # Detect encoding
    try:
        encoding = chardet.detect(content)['encoding']
        text_content = content.decode(encoding or 'latin1')
    except UnicodeDecodeError:
        text_content = content.decode('latin1', errors='replace')

    # First count columns in header
    header = text_content.split('\n')[0]
    expected_columns = len(header.split('\t'))
    
    # Process content line-by-line
    lines = []
    for i, line in enumerate(text_content.split('\n')):
        if i == 0:  # Header row
            cols = [col[1:] if col.startswith('?') else col 
                   for col in line.strip().split('\t')]
            continue
        
        if not line.strip():
            continue
            
        fields = line.strip().split('\t')
        if len(fields) != expected_columns:
            print(f"Line {i+1} in {file_name}: Expected {expected_columns} columns, found {len(fields)}")
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
    
    # Group by strain if exists
    if 'strain' in df.columns:
        grouped = df.groupby('strain', as_index=False).agg(
            lambda x: x.dropna().iloc[0] if x.notna().any() else 'unknown'
        )
        return grouped
    return None

# Process all files
dfs = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_name in z.namelist():
        if file_name.endswith('.tsv'):
            with z.open(file_name) as f:
                content = f.read()
                processed_df = process_tsv(content, file_name)
                if processed_df is not None:
                    dfs.append(processed_df)

if not dfs:
    print("No valid TSV files found.")
else:
    # Merge all dataframes on 'strain'
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='strain',
            how='outer',
            suffixes=('', '_DROP'))
        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_DROP')]
    
    # Final cleaning
    merged_df = merged_df.fillna("unknown")
    merged_df = merged_df.replace(["", "nan", "NaN"], "unknown")
    
    # Save result
    output_path = os.path.join(current_dir, 'merged_bacdive_data.csv')
    merged_df.to_csv(output_path, sep=';', index=False)
    print(f" Successfully merged {len(dfs)} files")
    print(f" Output saved to: {output_path}")
    print(f" Note: Some lines were malformed but were included with adjusted columns")