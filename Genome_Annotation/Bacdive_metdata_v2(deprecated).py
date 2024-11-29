# Modified Script to Download Genomes from BacDive using Multiple Strategies for NCBI Access
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

def fetch_genome_from_ncbi(accession_number=None, taxonomy_id=None, genus=None, species=None):
    """Fetch genome sequence from NCBI using different search parameters."""
    try:
        if accession_number:
            # Fetch using accession number
            handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text",api_key=creds.NCBI_API_KEY)
        elif taxonomy_id:
            # Fetch using Taxonomy ID
            handle = Entrez.esearch(db="nucleotide", term=f"txid{taxonomy_id}[Organism]", retmax=1,api_key=creds.NCBI_API_KEY)
            result = Entrez.read(handle)
            handle.close()
            if result["IdList"]:
                id_to_fetch = result["IdList"][0]
                handle = Entrez.efetch(db="nucleotide", id=id_to_fetch, rettype="fasta", retmode="text",api_key=creds.NCBI_API_KEY)
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
    taxonomy_query = 'Bacillus'  # Set the taxonomy query for this run
    limit = 10  # Number of strains to retrieve per request
    offset = 0
    genomes_downloaded = 0

    while genomes_downloaded < 3:  # Stop after downloading 3 genomes for testing
        try:
            count = client.search(taxonomy=taxonomy_query)
            print(f'Fetching strains {offset} to {offset + limit}...')
        except Exception as e:
            print(f"Error during search: {e}")
            return

        if count == 0:
            print("No more strains found.")
            break

        for strain in client.retrieve():
            bacdive_id = strain.get('General', {}).get('BacDive-ID', 'Unknown_ID')
            sequence_info = strain.get('Sequence information', {})
            taxonomy_id = strain.get('General', {}).get('NCBI tax id', {}).get('NCBI tax id', None)
            genus = strain.get('Name and taxonomic classification', {}).get('genus')
            species = strain.get('Name and taxonomic classification', {}).get('species')

            accession_number = None
            if '16S sequences' in sequence_info:
                accession_number = sequence_info['16S sequences'].get('accession')
            elif 'genome sequence' in sequence_info:
                accession_number = sequence_info['genome sequence'].get('accession')

            genome_data = fetch_genome_from_ncbi(accession_number, taxonomy_id, genus, species)

            if genome_data:
                output_path = os.path.join(OUTPUT_DIR, f'{taxonomy_query}_{bacdive_id}.fasta')
                with open(output_path, 'w') as genome_file:
                    genome_file.write(genome_data)
                    print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                    genomes_downloaded += 1

            time.sleep(1)

        offset += limit

if __name__ == '__main__':
    download_genomes()
