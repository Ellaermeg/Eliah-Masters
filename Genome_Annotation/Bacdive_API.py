import os
import bacdive

# Replace with your BacDive login credentials
EMAIL = 'eliahmathias@gmail.com'
PASSWORD = 'Eliah123#'

# Initialize BacDive client
client = bacdive.BacdiveClient(EMAIL, PASSWORD)

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_genomes():
    # Search for all available strains
    count = client.search()
    print(f'{count} strains found.')

    # Iterate over each strain and save genome data if available
    for strain in client.retrieve():
        bacdive_id = strain.get('bacdive_id')
        genome_data = strain.get('seq_acc_num')  # Adjust the key based on actual data structure
        if genome_data:
            output_path = os.path.join(OUTPUT_DIR, f'{bacdive_id}.fasta')
            with open(output_path, 'w') as genome_file:
                genome_file.write(genome_data)
                print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')

if __name__ == '__main__':
    download_genomes()
