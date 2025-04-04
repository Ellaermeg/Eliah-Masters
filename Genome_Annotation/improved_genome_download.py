#!/usr/bin/env python3
"""
Improved Genome Download Script

This script downloads genome sequences from NCBI based on accession numbers
extracted from the merged BacDive data. It supports both protein and nucleotide
sequences, handles errors gracefully, and includes parallel download capabilities.
"""

import os
import sys
import time
import pandas as pd
import urllib.request
import urllib.parse
import urllib.error
import gzip
import shutil
import argparse
import concurrent.futures
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genome_download.log"),
        logging.StreamHandler()
    ]
)

def setup_argparse():
    """Set up command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Download genome sequences from NCBI')
    parser.add_argument('--input', '-i', required=True, 
                        help='Input CSV file with genome accession numbers')
    parser.add_argument('--output-dir', '-o', default='downloaded_genomes',
                        help='Directory to save downloaded genomes')
    parser.add_argument('--accession-col', default='sequenceAccession',
                        help='Column name containing accession numbers')
    parser.add_argument('--prefer-protein', action='store_true',
                        help='Prefer protein sequences over nucleotide sequences')
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Maximum number of parallel downloads')
    parser.add_argument('--retry', type=int, default=3,
                        help='Number of retry attempts for failed downloads')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--refseq-table', 
                        help='Path to NCBI RefSeq table (optional, will download if not provided)')
    return parser.parse_args()

def download_file(url, output_file, retry_count=3):
    """Download a file with retry logic"""
    for attempt in range(retry_count):
        try:
            logging.debug(f"Downloading {url} to {output_file}")
            _, result = urllib.request.urlretrieve(url, output_file)
            
            # Verify the content type
            content_type = result.get_content_type()
            if 'gzip' not in content_type and 'octet-stream' not in content_type:
                logging.warning(f"Unexpected content type: {content_type} for {url}")
                if os.path.exists(output_file):
                    os.remove(output_file)
                if attempt < retry_count - 1:
                    logging.info(f"Retrying download for {url} (attempt {attempt+2}/{retry_count})")
                    time.sleep(2)  # Wait before retrying
                    continue
                return False
            
            # Verify file size
            if os.path.getsize(output_file) < 100:  # Arbitrary small size check
                logging.warning(f"Downloaded file is too small: {output_file}")
                os.remove(output_file)
                if attempt < retry_count - 1:
                    logging.info(f"Retrying download for {url} (attempt {attempt+2}/{retry_count})")
                    time.sleep(2)  # Wait before retrying
                    continue
                return False
                
            return True
            
        except Exception as e:
            logging.warning(f"Download failed: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            if attempt < retry_count - 1:
                logging.info(f"Retrying download for {url} (attempt {attempt+2}/{retry_count})")
                time.sleep(2)  # Wait before retrying
            else:
                logging.error(f"Failed to download {url} after {retry_count} attempts")
                return False
    
    return False

def download_refseq_table(output_file="refseq_table.csv"):
    """Download the latest RefSeq table from NCBI"""
    logging.info("Downloading NCBI RefSeq table...")
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt"
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, output_file)
        
        # Process the file to convert it to a proper CSV
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (starts with #)
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("# assembly_accession"):
                header_idx = i
                break
        
        # Write the processed file
        with open(output_file, 'w') as f:
            # Write the header without the # prefix
            f.write(lines[header_idx][2:])
            # Write the data lines
            for line in lines[header_idx+1:]:
                f.write(line)
        
        logging.info(f"RefSeq table downloaded and saved to {output_file}")
        return output_file
    
    except Exception as e:
        logging.error(f"Failed to download RefSeq table: {e}")
        return None

def load_refseq_table(file_path=None):
    """Load the NCBI RefSeq table"""
    if file_path is None or not os.path.exists(file_path):
        file_path = download_refseq_table()
        if file_path is None:
            logging.error("Could not download or find RefSeq table")
            return None
    
    try:
        # Load the table
        table = pd.read_csv(file_path, sep='\t', low_memory=False)
        
        # Set the assembly_accession as the index
        if 'assembly_accession' in table.columns:
            table.set_index('assembly_accession', inplace=True)
        
        logging.info(f"Loaded RefSeq table with {len(table)} entries")
        return table
    
    except Exception as e:
        logging.error(f"Failed to load RefSeq table: {e}")
        return None

def extract_accession_from_genbank(accession):
    """Extract the base accession from a GenBank accession number"""
    # Handle different accession formats
    if accession.startswith('GCF_') or accession.startswith('GCA_'):
        return accession
    
    # Extract the base accession without version
    base_accession = accession.split('.')[0]
    return base_accession

def find_matching_refseq_entry(accession, refseq_table):
    """Find a matching entry in the RefSeq table for the given accession"""
    # Try direct match
    if accession in refseq_table.index:
        return accession
    
    # Try with GCF_ prefix
    if not accession.startswith('GCF_') and not accession.startswith('GCA_'):
        # Search for entries that might contain this accession
        for idx in refseq_table.index:
            if accession in idx:
                logging.info(f"Found matching RefSeq entry: {idx} for accession {accession}")
                return idx
    
    logging.warning(f"No matching RefSeq entry found for accession {accession}")
    return None

def download_genome(accession, refseq_table, output_dir, prefer_protein=True, overwrite=False, retry_count=3):
    """Download a genome from NCBI based on its accession number"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean and standardize the accession
    clean_accession = extract_accession_from_genbank(accession)
    if not clean_accession:
        logging.warning(f"Invalid accession format: {accession}")
        return None
    
    # Find matching entry in RefSeq table
    matching_accession = find_matching_refseq_entry(clean_accession, refseq_table)
    if not matching_accession:
        logging.warning(f"Accession not found in RefSeq table: {clean_accession}")
        return None
    
    # Get the entry from the table
    try:
        entry = refseq_table.loc[matching_accession]
    except KeyError:
        logging.warning(f"Entry not found in RefSeq table: {matching_accession}")
        return None
    
    # Determine FTP path
    try:
        ftp_path = entry['ftp_path']
        if pd.isna(ftp_path) or not ftp_path:
            logging.warning(f"No FTP path available for {matching_accession}")
            return None
        
        # Extract the filename base from the FTP path
        filename_base = ftp_path.split('/')[-1]
    except Exception as e:
        logging.warning(f"Error processing entry for {matching_accession}: {e}")
        return None
    
    # Try downloading protein sequence first if preferred
    downloaded_file = None
    if prefer_protein:
        protein_url = f"https://{ftp_path[6:]}/{filename_base}_protein.faa.gz"
        protein_output = os.path.join(output_dir, f"{clean_accession}_protein.faa.gz")
        
        if os.path.exists(protein_output) and not overwrite:
            logging.info(f"Protein file already exists for {clean_accession}, skipping")
            return protein_output
        
        logging.info(f"Downloading protein sequence for {clean_accession}")
        if download_file(protein_url, protein_output, retry_count):
            logging.info(f"Successfully downloaded protein sequence for {clean_accession}")
            downloaded_file = protein_output
    
    # If protein download failed or not preferred, try nucleotide sequence
    if downloaded_file is None:
        nucleotide_url = f"https://{ftp_path[6:]}/{filename_base}_genomic.fna.gz"
        nucleotide_output = os.path.join(output_dir, f"{clean_accession}_genomic.fna.gz")
        
        if os.path.exists(nucleotide_output) and not overwrite:
            logging.info(f"Nucleotide file already exists for {clean_accession}, skipping")
            return nucleotide_output
        
        logging.info(f"Downloading nucleotide sequence for {clean_accession}")
        if download_file(nucleotide_url, nucleotide_output, retry_count):
            logging.info(f"Successfully downloaded nucleotide sequence for {clean_accession}")
            downloaded_file = nucleotide_output
    
    return downloaded_file

def extract_accessions_from_csv(csv_file, accession_col='sequenceAccession'):
    """Extract accession numbers from a CSV file"""
    try:
        # Try with comma separator first
        df = pd.read_csv(csv_file, sep=',', low_memory=False)
        
        # If the accession column doesn't exist, try with semicolon separator
        if accession_col not in df.columns:
            df = pd.read_csv(csv_file, sep=';', low_memory=False)
        
        # If the column still doesn't exist, look for similar column names
        if accession_col not in df.columns:
            potential_cols = [col for col in df.columns if 'accession' in col.lower()]
            if potential_cols:
                accession_col = potential_cols[0]
                logging.info(f"Using column {accession_col} for accession numbers")
            else:
                logging.error(f"Could not find accession column in {csv_file}")
                return []
        
        # Extract accession numbers
        accessions = df[accession_col].dropna().unique().tolist()
        
        # Clean accessions (remove any non-alphanumeric characters)
        clean_accessions = []
        for acc in accessions:
            if isinstance(acc, str):
                # Split by common separators and take the first part
                parts = acc.replace(',', ';').split(';')
                for part in parts:
                    clean_part = part.strip()
                    if clean_part:
                        clean_accessions.append(clean_part)
        
        logging.info(f"Extracted {len(clean_accessions)} accession numbers from {csv_file}")
        return clean_accessions
    
    except Exception as e:
        logging.error(f"Failed to extract accessions from {csv_file}: {e}")
        return []

def download_worker(args):
    """Worker function for parallel downloads"""
    accession, refseq_table, output_dir, prefer_protein, overwrite, retry_count = args
    try:
        return download_genome(
            accession, refseq_table, output_dir, 
            prefer_protein, overwrite, retry_count
        )
    except Exception as e:
        logging.error(f"Error downloading {accession}: {e}")
        return None

def main():
    """Main function to download genomes"""
    # Parse command-line arguments
    args = setup_argparse()
    
    # Load the RefSeq table
    refseq_table = load_refseq_table(args.refseq_table)
    if refseq_table is None:
        logging.error("Failed to load RefSeq table. Exiting.")
        sys.exit(1)
    
    # Extract accession numbers from the input file
    accessions = extract_accessions_from_csv(args.input, args.accession_col)
    if not accessions:
        logging.error("No accession numbers found in the input file. Exiting.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download genomes in parallel
    logging.info(f"Starting download of {len(accessions)} genomes with {args.max_workers} workers")
    
    # Prepare arguments for the worker function
    worker_args = [
        (acc, refseq_table, args.output_dir, args.prefer_protein, args.overwrite, args.retry)
        for acc in accessions
    ]
    
    # Use ThreadPoolExecutor for parallel downloads
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all download tasks
        future_to_accession = {
            executor.submit(download_worker, arg): arg[0]
            for arg in worker_args
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_accession), 
                          total=len(accessions), 
                          desc="Downloading genomes"):
            accession = future_to_accession[future]
            try:
                result = future.result()
                results.append((accession, result))
            except Exception as e:
                logging.error(f"Error processing {accession}: {e}")
                results.append((accession, None))
    
    # Summarize results
    successful = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]
    
    logging.info(f"Download summary:")
    logging.info(f"  Total: {len(results)}")
    logging.info(f"  Successful: {len(successful)}")
    logging.info(f"  Failed: {len(failed)}")
    
    # Write summary to file
    summary_file = os.path.join(args.output_dir, "download_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Download summary:\n")
        f.write(f"  Total: {len(results)}\n")
        f.write(f"  Successful: {len(successful)}\n")
        f.write(f"  Failed: {len(failed)}\n\n")
        
        f.write("Successful downloads:\n")
        for acc, path in successful:
            f.write(f"  {acc}: {path}\n")
        
        f.write("\nFailed downloads:\n")
        for acc, _ in failed:
            f.write(f"  {acc}\n")
    
    logging.info(f"Download summary written to {summary_file}")
    
    # Create a list of downloaded files for further processing
    downloaded_files = [path for _, path in successful if path is not None]
    with open(os.path.join(args.output_dir, "downloaded_files.txt"), 'w') as f:
        for file_path in downloaded_files:
            f.write(f"{file_path}\n")
    
    logging.info(f"List of downloaded files written to {os.path.join(args.output_dir, 'downloaded_files.txt')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
