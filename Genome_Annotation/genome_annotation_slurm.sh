#!/bin/bash
#SBATCH --job-name=genome_annotation
#SBATCH --output=annotation_%A_%a.out
#SBATCH --error=annotation_%A_%a.err
#SBATCH --array=1-100%10
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Genome Annotation SLURM Script for GPU Cluster
# This script processes genome files in parallel using SLURM array jobs
# Each job annotates one genome file from the input list

# Load required modules (adjust based on your cluster's configuration)
module load python/3.8
module load hmmer/3.3
module load prodigal/2.6.3
module load blast/2.12.0
module load diamond/2.0.15

# Configuration variables
GENOME_LIST="$1"  # First argument: file containing list of genome files to process
OUTPUT_DIR="$2"   # Second argument: output directory
ANNOTATION_TOOL="$3"  # Third argument: annotation tool to use (prokka, bakta, etc.)
CONFIG_FILE="${4:-config.ini}"  # Fourth argument (optional): config file

# Default values if not provided
if [ -z "$GENOME_LIST" ]; then
    echo "Error: Genome list file not provided"
    echo "Usage: sbatch genome_annotation.slurm genome_list.txt output_dir annotation_tool [config_file]"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="annotation_results"
fi

if [ -z "$ANNOTATION_TOOL" ]; then
    ANNOTATION_TOOL="prokka"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get the genome file for this array job
GENOME_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$GENOME_LIST")
if [ -z "$GENOME_FILE" ]; then
    echo "Error: No genome file found for task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Extract the base name of the genome file (without path and extension)
GENOME_BASE=$(basename "$GENOME_FILE" | sed 's/\.[^.]*$//')

# Create a directory for this genome's results
GENOME_OUTPUT_DIR="${OUTPUT_DIR}/${GENOME_BASE}"
mkdir -p "$GENOME_OUTPUT_DIR"

# Log the start of processing
echo "Starting annotation of $GENOME_FILE at $(date)"
echo "Output directory: $GENOME_OUTPUT_DIR"
echo "Annotation tool: $ANNOTATION_TOOL"

# Function to run Prokka annotation
run_prokka() {
    local genome_file="$1"
    local output_dir="$2"
    
    # Load Prokka module if available
    module load prokka/1.14 2>/dev/null || true
    
    # Check if Prokka is installed
    if ! command -v prokka &> /dev/null; then
        echo "Error: Prokka not found. Please install Prokka or load the appropriate module."
        return 1
    fi
    
    # Run Prokka
    prokka --outdir "$output_dir" \
           --prefix "$GENOME_BASE" \
           --cpus "$SLURM_CPUS_PER_TASK" \
           --force \
           "$genome_file"
    
    return $?
}

# Function to run Bakta annotation
run_bakta() {
    local genome_file="$1"
    local output_dir="$2"
    
    # Load Bakta module if available
    module load bakta/1.5 2>/dev/null || true
    
    # Check if Bakta is installed
    if ! command -v bakta &> /dev/null; then
        echo "Error: Bakta not found. Please install Bakta or load the appropriate module."
        return 1
    fi
    
    # Run Bakta
    bakta --output "$output_dir" \
          --prefix "$GENOME_BASE" \
          --threads "$SLURM_CPUS_PER_TASK" \
          --force \
          "$genome_file"
    
    return $?
}

# Function to run eggNOG-mapper annotation
run_eggnog() {
    local genome_file="$1"
    local output_dir="$2"
    
    # Load eggNOG-mapper module if available
    module load eggnog-mapper/2.1.7 2>/dev/null || true
    
    # Check if eggNOG-mapper is installed
    if ! command -v emapper.py &> /dev/null; then
        echo "Error: eggNOG-mapper not found. Please install eggNOG-mapper or load the appropriate module."
        return 1
    fi
    
    # If input is a protein FASTA file
    if [[ "$genome_file" == *"_protein.faa"* || "$genome_file" == *".faa"* ]]; then
        # Run eggNOG-mapper with protein input
        emapper.py -i "$genome_file" \
                  --output "$GENOME_BASE" \
                  --output_dir "$output_dir" \
                  --cpu "$SLURM_CPUS_PER_TASK" \
                  --usemem \
                  --pfam \
                  --go_evidence \
                  --target_orthologs all \
                  --seed_ortholog_evalue 0.001 \
                  --override
    else
        # For nucleotide input, first predict genes with Prodigal
        local protein_file="${output_dir}/${GENOME_BASE}_proteins.faa"
        
        # Run Prodigal for gene prediction
        prodigal -i "$genome_file" -a "$protein_file" -o "${output_dir}/${GENOME_BASE}_genes.gbk" -p meta
        
        # Then run eggNOG-mapper with the predicted proteins
        emapper.py -i "$protein_file" \
                  --output "$GENOME_BASE" \
                  --output_dir "$output_dir" \
                  --cpu "$SLURM_CPUS_PER_TASK" \
                  --usemem \
                  --pfam \
                  --go_evidence \
                  --target_orthologs all \
                  --seed_ortholog_evalue 0.001 \
                  --override
    fi
    
    return $?
}

# Function to run InterProScan annotation
run_interproscan() {
    local genome_file="$1"
    local output_dir="$2"
    
    # Load InterProScan module if available
    module load interproscan/5.59-91.0 2>/dev/null || true
    
    # Check if InterProScan is installed
    if ! command -v interproscan.sh &> /dev/null; then
        echo "Error: InterProScan not found. Please install InterProScan or load the appropriate module."
        return 1
    fi
    
    # If input is a protein FASTA file
    if [[ "$genome_file" == *"_protein.faa"* || "$genome_file" == *".faa"* ]]; then
        local protein_file="$genome_file"
    else
        # For nucleotide input, first predict genes with Prodigal
        local protein_file="${output_dir}/${GENOME_BASE}_proteins.faa"
        
        # Run Prodigal for gene prediction
        prodigal -i "$genome_file" -a "$protein_file" -o "${output_dir}/${GENOME_BASE}_genes.gbk" -p meta
    fi
    
    # Run InterProScan with the protein file
    interproscan.sh -i "$protein_file" \
                   -d "$output_dir" \
                   -f TSV,GFF3,JSON \
                   -goterms \
                   -pa \
                   -cpu "$SLURM_CPUS_PER_TASK"
    
    return $?
}

# Function to run custom Python annotation pipeline
run_custom_pipeline() {
    local genome_file="$1"
    local output_dir="$2"
    local config_file="$3"
    
    # Check if the custom annotation script exists
    if [ ! -f "custom_annotation.py" ]; then
        echo "Error: custom_annotation.py not found."
        return 1
    fi
    
    # Run the custom annotation pipeline
    python3 custom_annotation.py \
            --input "$genome_file" \
            --output-dir "$output_dir" \
            --config "$config_file" \
            --threads "$SLURM_CPUS_PER_TASK" \
            --gpu
    
    return $?
}

# Decompress the genome file if it's compressed
if [[ "$GENOME_FILE" == *.gz ]]; then
    UNCOMPRESSED_FILE="${GENOME_OUTPUT_DIR}/$(basename "$GENOME_FILE" .gz)"
    echo "Decompressing $GENOME_FILE to $UNCOMPRESSED_FILE"
    gunzip -c "$GENOME_FILE" > "$UNCOMPRESSED_FILE"
    GENOME_FILE="$UNCOMPRESSED_FILE"
fi

# Run the appropriate annotation tool
case "$ANNOTATION_TOOL" in
    "prokka")
        run_prokka "$GENOME_FILE" "$GENOME_OUTPUT_DIR"
        ;;
    "bakta")
        run_bakta "$GENOME_FILE" "$GENOME_OUTPUT_DIR"
        ;;
    "eggnog")
        run_eggnog "$GENOME_FILE" "$GENOME_OUTPUT_DIR"
        ;;
    "interproscan")
        run_interproscan "$GENOME_FILE" "$GENOME_OUTPUT_DIR"
        ;;
    "custom")
        run_custom_pipeline "$GENOME_FILE" "$GENOME_OUTPUT_DIR" "$CONFIG_FILE"
        ;;
    *)
        echo "Error: Unknown annotation tool: $ANNOTATION_TOOL"
        echo "Supported tools: prokka, bakta, eggnog, interproscan, custom"
        exit 1
        ;;
esac

# Check if annotation was successful
if [ $? -eq 0 ]; then
    echo "Annotation of $GENOME_FILE completed successfully at $(date)"
else
    echo "Error: Annotation of $GENOME_FILE failed at $(date)"
    exit 1
fi

# Create a flag file to indicate completion
touch "${GENOME_OUTPUT_DIR}/.completed"

# Exit successfully
exit 0
