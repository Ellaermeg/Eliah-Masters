# Modified Script to Download Genomes from BacDive using Multiple Genera for NCBI Access
import sys
sys.path.append("../Eliah-Masters")
import os
import time
from bacdive import BacdiveClient
from Bio import Entrez
from API_cred import APICredentials
import logging

# Initialize credentials from the class
creds = APICredentials()

# Initialize BacDive client
try:
    client = BacdiveClient(creds.EMAIL, creds.PASSWORD)
    print("-- Authentication successful --")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    exit(1)

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set NCBI Entrez email and API key parameters
Entrez.email = creds.EMAIL
Entrez.api_key = creds.NCBI_API_KEY

# List of genera to search for
GENERA_LIST = [
    'Bacillus', 'Escherichia', 'Streptococcus', 'Pseudomonas', 'Clostridium', 
    'Staphylococcus', 'Mycobacterium', 'Lactobacillus', 'Salmonella', 'Neisseria'
]

# Set to track downloaded BacDive IDs to avoid duplicates
downloaded_bacdive_ids = set()

def fetch_genome_from_ncbi(accession_number=None, taxonomy_id=None, genus=None, species=None):
    """Fetch genome sequence from NCBI using different search parameters."""
    try:
        if accession_number:
            # Fetch using accession number
            handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text", api_key=creds.NCBI_API_KEY)
        elif taxonomy_id:
            # Fetch using Taxonomy ID
            handle = Entrez.esearch(db="nucleotide", term=f"txid{taxonomy_id}[Organism]", retmax=1, api_key=creds.NCBI_API_KEY)
            result = Entrez.read(handle)
            handle.close()
            if result["IdList"]:
                id_to_fetch = result["IdList"][0]
                handle = Entrez.efetch(db="nucleotide", id=id_to_fetch, rettype="fasta", retmode="text", api_key=creds.NCBI_API_KEY)
            else:
                return None
        elif genus and species:
            # Fetch using genus and species name
            query = f"{genus} {species}"
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=1, api_key=creds.NCBI_API_KEY)
            result = Entrez.read(handle)
            handle.close()
            if result["IdList"]:
                id_to_fetch = result["IdList"][0]
                handle = Entrez.efetch(db="nucleotide", id=id_to_fetch, rettype="fasta", retmode="text", api_key=creds.NCBI_API_KEY)
            else:
                return None
        else:
            return None

        sequence_data = handle.read()
        handle.close()
        return sequence_data
    except Exception as e:
        print(f"Error fetching genome from NCBI: {e}")
        return None

def download_genomes():
    limit = 10  # Number of strains to retrieve per request
    offset = 0
    genomes_downloaded = 0
    max_genomes = 1000  # Set a maximum limit for total genomes to download

    # Iterate over the list of genera
    for genus in GENERA_LIST:
        offset = 0  # Reset offset for each genus
        while genomes_downloaded < max_genomes:
            try:
                print(f'Using taxonomy query: {genus}')
                count = client.search(taxonomy=genus, offset=offset, limit=limit)
                print(f'Fetching strains {offset} to {offset + limit} for genus {genus}...')
            except Exception as e:
                print(f"Error during search for genus {genus}: {e}")
                break

            if count == 0:
                print(f"No more strains found for genus: {genus}.")
                break

            for strain in client.retrieve():
                bacdive_id = strain.get('General', {}).get('BacDive-ID', 'Unknown_ID')

                # Skip if BacDive ID was already downloaded
                if bacdive_id in downloaded_bacdive_ids:
                    print(f"BacDive ID {bacdive_id} already downloaded. Skipping.")
                    continue

                sequence_info = strain.get('Sequence information', {})
                taxonomy_id = strain.get('General', {}).get('NCBI tax id', {}).get('NCBI tax id', None)
                genus = strain.get('Name and taxonomic classification', {}).get('genus')
                species = strain.get('Name and taxonomic classification', {}).get('species')

                accession_number = None
                if '16S sequences' in sequence_info:
                    sixteen_s_data = sequence_info['16S sequences']
                    if isinstance(sixteen_s_data, dict):
                        accession_number = sixteen_s_data.get('accession')
                    elif isinstance(sixteen_s_data, list) and len(sixteen_s_data) > 0:
                        accession_number = sixteen_s_data[0].get('accession')
                elif 'genome sequence' in sequence_info:
                    genome_seq_data = sequence_info['genome sequence']
                    if isinstance(genome_seq_data, dict):
                        accession_number = genome_seq_data.get('accession')
                    elif isinstance(genome_seq_data, list) and len(genome_seq_data) > 0:
                        accession_number = genome_seq_data[0].get('accession')

                # Fetch the genome data from NCBI
                genome_data = fetch_genome_from_ncbi(accession_number, taxonomy_id, genus, species)

                # Save the genome data if found
                if genome_data:
                    file_name = f"{genus}_{species}_{bacdive_id}.fasta".replace(' ', '_')
                    output_path = os.path.join(OUTPUT_DIR, file_name)
                    with open(output_path, 'w') as genome_file:
                        genome_file.write(genome_data)
                        print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                        genomes_downloaded += 1

                    # Add BacDive ID to set to avoid duplicates
                    downloaded_bacdive_ids.add(bacdive_id)

                # Add a delay to prevent NCBI rate limiting
                time.sleep(1)

            # Update offset for the next batch
            offset += limit

        # Break if maximum genome limit is reached
        if genomes_downloaded >= max_genomes:
            print(f"Reached the maximum limit of {max_genomes} genomes downloaded.")
            break

if __name__ == '__main__':
    download_genomes()
