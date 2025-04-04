import sys
sys.path.append("../Eliah-Masters")
import pandas as pd
import zipfile
import os
import logging
from io import BytesIO
import chardet
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bacdive_merge.log'),
        logging.StreamHandler()
    ]
)

sys.path.append("../Eliah-Masters")
current_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(current_dir, '..', 'Datasets', 'tsv.zip')

def process_tsv(content, file_name):
    """Process TSV content with flexible column handling and logging"""
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Processing file: {file_name}")
        
        # Detect and decode content
        encoding = chardet.detect(content)['encoding']
        logger.debug(f"Detected encoding: {encoding}")
        
        text_content = content.decode(encoding or 'latin1')
    except UnicodeDecodeError:
        logger.warning(f"Decoding failed for {file_name}, using latin1 with error replacement")
        text_content = content.decode('latin1', errors='replace')

    # Process lines
    lines = text_content.split('\n')
    if not lines:
        logger.warning(f"Empty file: {file_name}")
        return None

    header = lines[0].strip()
    columns = [col[1:] if col.startswith('?') else col 
              for col in header.split('\t')]
    
    logger.info(f"File {file_name} has {len(columns)} columns")
    logger.debug(f"Columns detected: {columns}")

    data = []
    malformed_lines = 0
    for i, line in enumerate(lines[1:]):  # Skip header
        line_num = i + 2  # Account for header line
        if not line.strip():
            continue
            
        fields = line.strip().split('\t')
        
        # Handle column count mismatch
        if len(fields) != len(columns):
            malformed_lines += 1
            logger.debug(f"Line {line_num}: Expected {len(columns)} cols, found {len(fields)}")
            
            # Pad or truncate fields
            if len(fields) < len(columns):
                fields += ['-'] * (len(columns) - len(fields))
            else:
                fields = fields[:len(columns)]
            
        data.append(fields)
    
    if malformed_lines > 0:
        logger.warning(f"File {file_name} had {malformed_lines} malformed lines")

    # Create DataFrame
    try:
        df = pd.DataFrame(data, columns=columns)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        if 'strain' not in columns:
            logger.error(f"Skipping {file_name}: No 'strain' column found")
            return None
            
        logger.info(f"Processed {len(df)} rows from {file_name}")
        return df[['strain'] + [c for c in columns if c != 'strain']]
    
    except Exception as e:
        logger.error(f"Failed to process {file_name}: {str(e)}")
        return None

def merge_dataframes(df_list):
    """Merge dataframes dynamically on strain column with logging"""
    logger = logging.getLogger(__name__)
    
    if not df_list:
        logger.error("No dataframes to merge")
        return None
    
    logger.info("Starting merge process...")
    
    try:
        # Concatenate all dataframes
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)
        logger.info(f"Total rows before deduplication: {len(merged_df)}")
        
        # Group by strain and aggregate
        merged_df = merged_df.groupby('strain', as_index=False).agg(
            lambda x: x[x != '-'].dropna().iloc[0] if (x != '-').any() else '-'
        )
        logger.info(f"Final merged rows: {len(merged_df)}")
        logger.info(f"Final columns: {len(merged_df.columns)}")
        
        return merged_df
    except Exception as e:
        logger.error(f"Merge failed: {str(e)}")
        return None

def main():
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info(f"Starting BacDive merge process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all files
    dfs = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_count = len(z.namelist())
        logger.info(f"Found {file_count} files in zip archive")
        
        for idx, file_name in enumerate(z.namelist(), 1):
            if not file_name.endswith('.tsv'):
                logger.debug(f"Skipping non-TSV file: {file_name}")
                continue
                
            logger.info(f"Processing file {idx}/{file_count}: {file_name}")
            
            try:
                with z.open(file_name) as f:
                    content = f.read()
                    processed_df = process_tsv(content, file_name)
                    if processed_df is not None and not processed_df.empty:
                        dfs.append(processed_df)
            except Exception as e:
                logger.error(f"Failed to process {file_name}: {str(e)}")
    
    # Merge dataframes
    final_df = merge_dataframes(dfs)
    
    if final_df is not None:
        # Save result
        output_path = os.path.join(current_dir, 'merged_bacdive_data.csv')
        try:
            final_df.to_csv(output_path, sep=';', index=False)
            logger.info(f"Successfully saved merged data to {output_path}")
            logger.info(f"Final dataset size: {final_df.shape}")
        except Exception as e:
            logger.error(f"Failed to save output file: {str(e)}")
    else:
        logger.error("Merge process failed, no output generated")
    
    logger.info(f"Process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)

if __name__ == "__main__":
    main()