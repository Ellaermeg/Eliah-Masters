#!/usr/bin/env python3
"""
Custom Genome Annotation Pipeline for GPU Cluster

This script provides a wrapper for running various annotation tools on a GPU cluster,
with support for distributed processing and result aggregation.
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import glob
from pathlib import Path
import configparser
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("custom_annotation.log"),
        logging.StreamHandler()
    ]
)

def setup_argparse():
    """Set up command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Custom genome annotation pipeline for GPU clusters')
    parser.add_argument('--input', '-i', required=True, 
                        help='Input genome file (FASTA format, can be gzipped)')
    parser.add_argument('--output-dir', '-o', default='annotation_results',
                        help='Directory to save annotation results')
    parser.add_argument('--config', '-c', default='annotation_config.ini',
                        help='Configuration file for annotation pipeline')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help='Number of CPU threads to use')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force overwrite of existing results')
    parser.add_argument('--annotation-tools', nargs='+', 
                        default=['prodigal', 'hmmer', 'diamond'],
                        help='List of annotation tools to run')
    parser.add_argument('--databases', nargs='+',
                        default=['pfam', 'tigrfam', 'cog', 'ko'],
                        help='List of databases to use for annotation')
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from file"""
    if not os.path.exists(config_file):
        logging.warning(f"Config file {config_file} not found, using default settings")
        return {}
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Convert config to dictionary
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config[section].items():
            config_dict[section][key] = value
    
    return config_dict

def run_command(cmd, work_dir=None, env=None):
    """Run a shell command and return the output"""
    logging.debug(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=work_dir,
            env=env
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False, e.stderr

def decompress_if_needed(input_file, output_dir):
    """Decompress the input file if it's compressed"""
    if input_file.endswith('.gz'):
        base_name = os.path.basename(input_file)[:-3]
        output_file = os.path.join(output_dir, base_name)
        
        logging.info(f"Decompressing {input_file} to {output_file}")
        
        success, output = run_command(['gunzip', '-c', input_file], work_dir=output_dir)
        if success:
            with open(output_file, 'w') as f:
                f.write(output)
            return output_file
        else:
            logging.error(f"Failed to decompress {input_file}")
            return None
    else:
        # If not compressed, just return the original file
        return input_file

def run_prodigal(input_file, output_dir, threads=1, meta_mode=True):
    """Run Prodigal for gene prediction"""
    logging.info("Running Prodigal for gene prediction")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up output files
    base_name = os.path.basename(input_file).split('.')[0]
    protein_file = os.path.join(output_dir, f"{base_name}_proteins.faa")
    genes_file = os.path.join(output_dir, f"{base_name}_genes.gff")
    nucleotide_file = os.path.join(output_dir, f"{base_name}_genes.fna")
    
    # Build command
    cmd = [
        'prodigal',
        '-i', input_file,
        '-a', protein_file,
        '-o', genes_file,
        '-d', nucleotide_file
    ]
    
    # Add meta mode if specified
    if meta_mode:
        cmd.extend(['-p', 'meta'])
    
    # Run Prodigal
    success, output = run_command(cmd)
    
    if success:
        logging.info(f"Prodigal completed successfully")
        return {
            'protein_file': protein_file,
            'genes_file': genes_file,
            'nucleotide_file': nucleotide_file
        }
    else:
        logging.error("Prodigal failed")
        return None

def run_hmmer(protein_file, output_dir, database, threads=1):
    """Run HMMER for protein family annotation"""
    logging.info(f"Running HMMER with {database} database")
    
    # Create output directory
    hmmer_dir = os.path.join(output_dir, 'hmmer')
    os.makedirs(hmmer_dir, exist_ok=True)
    
    # Set up output file
    base_name = os.path.basename(protein_file).split('.')[0]
    output_file = os.path.join(hmmer_dir, f"{base_name}_{database}.txt")
    
    # Determine database path based on database name
    db_path = None
    if database.lower() == 'pfam':
        db_path = os.environ.get('PFAM_DB', '/path/to/pfam/Pfam-A.hmm')
    elif database.lower() == 'tigrfam':
        db_path = os.environ.get('TIGRFAM_DB', '/path/to/tigrfam/TIGRFAMs.hmm')
    else:
        db_path = database  # Assume the database parameter is a path
    
    if not os.path.exists(db_path):
        logging.error(f"Database file {db_path} not found")
        return None
    
    # Build command
    cmd = [
        'hmmsearch',
        '--cpu', str(threads),
        '-o', output_file,
        '--tblout', f"{output_file}.tbl",
        '--domtblout', f"{output_file}.domtbl",
        '--pfamtblout', f"{output_file}.pfamtbl",
        db_path,
        protein_file
    ]
    
    # Run HMMER
    success, output = run_command(cmd)
    
    if success:
        logging.info(f"HMMER completed successfully for {database}")
        return {
            'output_file': output_file,
            'tbl_file': f"{output_file}.tbl",
            'domtbl_file': f"{output_file}.domtbl",
            'pfamtbl_file': f"{output_file}.pfamtbl"
        }
    else:
        logging.error(f"HMMER failed for {database}")
        return None

def run_diamond(protein_file, output_dir, database, threads=1, sensitive=True):
    """Run DIAMOND for sequence similarity search"""
    logging.info(f"Running DIAMOND with {database} database")
    
    # Create output directory
    diamond_dir = os.path.join(output_dir, 'diamond')
    os.makedirs(diamond_dir, exist_ok=True)
    
    # Set up output file
    base_name = os.path.basename(protein_file).split('.')[0]
    output_file = os.path.join(diamond_dir, f"{base_name}_{database}.txt")
    
    # Determine database path based on database name
    db_path = None
    if database.lower() == 'nr':
        db_path = os.environ.get('NR_DIAMOND_DB', '/path/to/nr/nr.dmnd')
    elif database.lower() == 'swissprot':
        db_path = os.environ.get('SWISSPROT_DIAMOND_DB', '/path/to/swissprot/swissprot.dmnd')
    elif database.lower() == 'trembl':
        db_path = os.environ.get('TREMBL_DIAMOND_DB', '/path/to/trembl/trembl.dmnd')
    else:
        db_path = database  # Assume the database parameter is a path
    
    if not os.path.exists(db_path):
        logging.error(f"Database file {db_path} not found")
        return None
    
    # Build command
    cmd = [
        'diamond',
        'blastp',
        '--db', db_path,
        '--query', protein_file,
        '--out', output_file,
        '--threads', str(threads),
        '--outfmt', '6', 'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'stitle'
    ]
    
    # Add sensitivity option
    if sensitive:
        cmd.append('--sensitive')
    
    # Run DIAMOND
    success, output = run_command(cmd)
    
    if success:
        logging.info(f"DIAMOND completed successfully for {database}")
        return {
            'output_file': output_file
        }
    else:
        logging.error(f"DIAMOND failed for {database}")
        return None

def run_eggnog_mapper(protein_file, output_dir, threads=1, use_gpu=False):
    """Run eggNOG-mapper for functional annotation"""
    logging.info("Running eggNOG-mapper for functional annotation")
    
    # Create output directory
    eggnog_dir = os.path.join(output_dir, 'eggnog')
    os.makedirs(eggnog_dir, exist_ok=True)
    
    # Set up output file
    base_name = os.path.basename(protein_file).split('.')[0]
    output_prefix = os.path.join(eggnog_dir, base_name)
    
    # Build command
    cmd = [
        'emapper.py',
        '-i', protein_file,
        '--output', base_name,
        '--output_dir', eggnog_dir,
        '--cpu', str(threads),
        '--pfam',
        '--go_evidence',
        '--target_orthologs', 'all',
        '--seed_ortholog_evalue', '0.001',
        '--override'
    ]
    
    # Add GPU option if specified
    if use_gpu:
        cmd.append('--usemem')
    
    # Run eggNOG-mapper
    success, output = run_command(cmd)
    
    if success:
        logging.info("eggNOG-mapper completed successfully")
        return {
            'annotations_file': f"{output_prefix}.emapper.annotations",
            'hits_file': f"{output_prefix}.emapper.hits",
            'seed_orthologs_file': f"{output_prefix}.emapper.seed_orthologs"
        }
    else:
        logging.error("eggNOG-mapper failed")
        return None

def run_interproscan(protein_file, output_dir, threads=1):
    """Run InterProScan for protein domain annotation"""
    logging.info("Running InterProScan for protein domain annotation")
    
    # Create output directory
    interpro_dir = os.path.join(output_dir, 'interproscan')
    os.makedirs(interpro_dir, exist_ok=True)
    
    # Set up output file
    base_name = os.path.basename(protein_file).split('.')[0]
    output_prefix = os.path.join(interpro_dir, base_name)
    
    # Build command
    cmd = [
        'interproscan.sh',
        '-i', protein_file,
        '-d', interpro_dir,
        '-f', 'TSV,GFF3,JSON',
        '-goterms',
        '-pa',
        '-cpu', str(threads)
    ]
    
    # Run InterProScan
    success, output = run_command(cmd)
    
    if success:
        logging.info("InterProScan completed successfully")
        return {
            'tsv_file': f"{output_prefix}.tsv",
            'gff_file': f"{output_prefix}.gff3",
            'json_file': f"{output_prefix}.json"
        }
    else:
        logging.error("InterProScan failed")
        return None

def aggregate_results(results, output_dir):
    """Aggregate results from different annotation tools"""
    logging.info("Aggregating annotation results")
    
    # Create output directory for aggregated results
    aggregate_dir = os.path.join(output_dir, 'aggregated')
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Extract base name from results
    base_name = None
    for tool, tool_results in results.items():
        if tool == 'prodigal' and tool_results:
            protein_file = tool_results.get('protein_file')
            if protein_file:
                base_name = os.path.basename(protein_file).split('_proteins')[0]
                break
    
    if not base_name:
        logging.error("Could not determine base name for aggregated results")
        return None
    
    # Create aggregated annotation file
    aggregated_file = os.path.join(aggregate_dir, f"{base_name}_aggregated.tsv")
    
    # Load gene information from Prodigal
    genes = {}
    if results.get('prodigal'):
        protein_file = results['prodigal'].get('protein_file')
        if protein_file and os.path.exists(protein_file):
            with open(protein_file, 'r') as f:
                current_gene = None
                for line in f:
                    if line.startswith('>'):
                        # Parse gene ID and information
                        parts = line[1:].strip().split(' # ')
                        gene_id = parts[0]
                        if len(parts) >= 4:
                            start = int(parts[1])
                            end = int(parts[2])
                            strand = '+' if parts[3] == '1' else '-'
                            genes[gene_id] = {
                                'id': gene_id,
                                'start': start,
                                'end': end,
                                'strand': strand,
                                'annotations': {}
                            }
    
    # Add HMMER annotations
    if results.get('hmmer'):
        for db, hmmer_results in results['hmmer'].items():
            if hmmer_results and 'tbl_file' in hmmer_results:
                tbl_file = hmmer_results['tbl_file']
                if os.path.exists(tbl_file):
                    with open(tbl_file, 'r') as f:
                        for line in f:
                            if line.startswith('#'):
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                gene_id = parts[0]
                                hit_id = parts[2]
                                evalue = float(parts[4])
                                if gene_id in genes:
                                    if 'hmmer' not in genes[gene_id]['annotations']:
                                        genes[gene_id]['annotations']['hmmer'] = {}
                                    if db not in genes[gene_id]['annotations']['hmmer']:
                                        genes[gene_id]['annotations']['hmmer'][db] = []
                                    genes[gene_id]['annotations']['hmmer'][db].append({
                                        'hit_id': hit_id,
                                        'evalue': evalue
                                    })
    
    # Add DIAMOND annotations
    if results.get('diamond'):
        for db, diamond_results in results['diamond'].items():
            if diamond_results and 'output_file' in diamond_results:
                output_file = diamond_results['output_file']
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 12:
                                gene_id = parts[0]
                                hit_id = parts[1]
                                pident = float(parts[2])
                                evalue = float(parts[10])
                                bitscore = float(parts[11])
                                if gene_id in genes:
                                    if 'diamond' not in genes[gene_id]['annotations']:
                                        genes[gene_id]['annotations']['diamond'] = {}
                                    if db not in genes[gene_id]['annotations']['diamond']:
                                        genes[gene_id]['annotations']['diamond'][db] = []
                                    genes[gene_id]['annotations']['diamond'][db].append({
                                        'hit_id': hit_id,
                                        'pident': pident,
                                        'evalue': evalue,
                                        'bitscore': bitscore
                                    })
    
    # Add eggNOG-mapper annotations
    if results.get('eggnog') and 'annotations_file' in results['eggnog']:
        annotations_file = results['eggnog']['annotations_file']
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                header = None
                for line in f:
                    if line.startswith('#'):
                        if line.startswith('# query'):
                            header = line[1:].strip().split('\t')
                        continue
                    
                    if header:
                        parts = line.strip().split('\t')
                        if len(parts) >= len(header):
                            gene_id = parts[0]
                            if gene_id in genes:
                                if 'eggnog' not in genes[gene_id]['annotations']:
                                    genes[gene_id]['annotations']['eggnog'] = {}
                                
                                for i, field in enumerate(header[1:], 1):
                                    if i < len(parts):
                                        genes[gene_id]['annotations']['eggnog'][field] = parts[i]
    
    # Add InterProScan annotations
    if results.get('interproscan') and 'tsv_file' in results['interproscan']:
        tsv_file = results['interproscan']['tsv_file']
        if os.path.exists(tsv_file):
            with open(tsv_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 11:
                        gene_id = parts[0]
                        if gene_id in genes:
                            if 'interproscan' not in genes[gene_id]['annotations']:
                                genes[gene_id]['annotations']['interproscan'] = []
                            
                            genes[gene_id]['annotations']['interproscan'].append({
                                'analysis': parts[3],
                                'signature_acc': parts[4],
                                'signature_desc': parts[5],
                                'start': int(parts[6]),
                                'end': int(parts[7]),
                                'evalue': parts[8],
                                'status': parts[9],
                                'date': parts[10]
                            })
    
    # Write aggregated results to TSV file
    with open(aggregated_file, 'w') as f:
        # Write header
        f.write("gene_id\tstart\tend\tstrand")
        f.write("\thmmer_hits\tdiamond_hits\teggnog_cog\teggnog_ko\teggnog_go\tinterproscan_hits\n")
        
        # Write gene annotations
        for gene_id, gene in genes.items():
            # Basic gene information
            f.write(f"{gene_id}\t{gene['start']}\t{gene['end']}\t{gene['strand']}")
            
            # HMMER hits
            hmmer_hits = []
            if 'hmmer' in gene['annotations']:
                for db, hits in gene['annotations']['hmmer'].items():
                    for hit in hits:
                        hmmer_hits.append(f"{hit['hit_id']}:{hit['evalue']}")
            f.write(f"\t{';'.join(hmmer_hits)}")
            
            # DIAMOND hits
            diamond_hits = []
            if 'diamond' in gene['annotations']:
                for db, hits in gene['annotations']['diamond'].items():
                    for hit in hits:
                        diamond_hits.append(f"{hit['hit_id']}:{hit['evalue']}")
            f.write(f"\t{';'.join(diamond_hits)}")
            
            # eggNOG-mapper annotations
            eggnog_cog = "-"
            eggnog_ko = "-"
            eggnog_go = "-"
            if 'eggnog' in gene['annotations']:
                if 'COG_category' in gene['annotations']['eggnog']:
                    eggnog_cog = gene['annotations']['eggnog']['COG_category']
                if 'KEGG_ko' in gene['annotations']['eggnog']:
                    eggnog_ko = gene['annotations']['eggnog']['KEGG_ko']
                if 'GOs' in gene['annotations']['eggnog']:
                    eggnog_go = gene['annotations']['eggnog']['GOs']
            f.write(f"\t{eggnog_cog}\t{eggnog_ko}\t{eggnog_go}")
            
            # InterProScan hits
            interproscan_hits = []
            if 'interproscan' in gene['annotations']:
                for hit in gene['annotations']['interproscan']:
                    interproscan_hits.append(f"{hit['signature_acc']}:{hit['signature_desc']}")
            f.write(f"\t{';'.join(interproscan_hits)}\n")
    
    # Also save as JSON for easier parsing
    json_file = os.path.join(aggregate_dir, f"{base_name}_aggregated.json")
    with open(json_file, 'w') as f:
        json.dump(genes, f, indent=2)
    
    logging.info(f"Aggregated results saved to {aggregated_file} and {json_file}")
    
    return {
        'tsv_file': aggregated_file,
        'json_file': json_file
    }

def main():
    """Main function to run the annotation pipeline"""
    # Parse command-line arguments
    args = setup_argparse()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logging.error(f"Input file {args.input} not found")
        return 1
    
    # Decompress input file if needed
    input_file = decompress_if_needed(args.input, args.output_dir)
    if not input_file:
        logging.error("Failed to process input file")
        return 1
    
    # Initialize results dictionary
    results = {}
    
    # Run Prodigal for gene prediction
    if 'prodigal' in args.annotation_tools:
        results['prodigal'] = run_prodigal(
            input_file, 
            args.output_dir,
            threads=args.threads
        )
        
        if not results['prodigal']:
            logging.error("Prodigal failed, cannot continue with protein annotation")
            return 1
        
        protein_file = results['prodigal']['protein_file']
    else:
        # If not running Prodigal, assume input is protein file
        protein_file = input_file
    
    # Run HMMER for protein family annotation
    if 'hmmer' in args.annotation_tools:
        results['hmmer'] = {}
        for db in args.databases:
            if db in ['pfam', 'tigrfam']:
                results['hmmer'][db] = run_hmmer(
                    protein_file,
                    args.output_dir,
                    db,
                    threads=args.threads
                )
    
    # Run DIAMOND for sequence similarity search
    if 'diamond' in args.annotation_tools:
        results['diamond'] = {}
        for db in args.databases:
            if db in ['nr', 'swissprot', 'trembl']:
                results['diamond'][db] = run_diamond(
                    protein_file,
                    args.output_dir,
                    db,
                    threads=args.threads
                )
    
    # Run eggNOG-mapper for functional annotation
    if 'eggnog' in args.annotation_tools:
        results['eggnog'] = run_eggnog_mapper(
            protein_file,
            args.output_dir,
            threads=args.threads,
            use_gpu=args.gpu
        )
    
    # Run InterProScan for protein domain annotation
    if 'interproscan' in args.annotation_tools:
        results['interproscan'] = run_interproscan(
            protein_file,
            args.output_dir,
            threads=args.threads
        )
    
    # Aggregate results
    aggregated = aggregate_results(results, args.output_dir)
    if aggregated:
        logging.info("Annotation pipeline completed successfully")
    else:
        logging.error("Failed to aggregate annotation results")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
