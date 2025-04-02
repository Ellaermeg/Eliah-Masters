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

def process_tsv(content):
    """Process TSV content with strain merging and data cleaning"""
    # Detect encoding
    encoding = chardet.detect(content)['encoding']
    try:
        text_content = content.decode(encoding or 'latin1')
    except UnicodeDecodeError:
        text_content = content.decode('latin1')

    # Read TSV with flexible parsing
    df = pd.read_csv(
        BytesIO(text_content.encode()),
        sep='\t',
        dtype=str,
        engine='python',
        quotechar='"',
        escapechar='\\',
        on_bad_lines='warn'
    )
    
    # Clean data
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.columns = df.columns.str.strip()
    
    # Group by strain and aggregate data
    grouped = df.groupby('?strain', as_index=False).agg(lambda x: x.dropna().iloc[0] if x.notna().any() else 'unknown')
    
    return grouped

# Open the zip file and process each TSV file
dfs = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for file_name in z.namelist():
        if file_name.endswith('.tsv'):
            with z.open(file_name) as f:
                content = f.read()
                processed_df = process_tsv(content)
                dfs.append(processed_df)

if not dfs:
    print("No TSV files found in the zip.")
else:
    # Merge all dataframes
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='?strain',
            how='outer',
            suffixes=('', '_DROP')
        )
        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_DROP')]
    
    # Final cleaning
    merged_df = merged_df.fillna("unknown")
    merged_df = merged_df.replace("", "unknown")
    
    # Save result
    output_path = os.path.join(current_dir, 'merged_bacdive_data.csv')
    merged_df.to_csv(output_path, sep=';', index=False)
    print(f"Merged data saved to '{output_path}'")