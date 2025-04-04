# Predicting Microbial Traits from Genome Annotations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

Master's Thesis Project: Machine learning pipelines for predicting microbial phenotypes from genomic annotations using comprehensive BacDive data.

## Introduction

This repository contains code and analysis for predicting microbial traits from genome annotations, developed as part of a master's thesis project. The project leverages machine learning techniques to predict measurable microbial traits (cell morphology, gram status, oxygen tolerance, nutrient requirements, etc.) based on functional genome annotations from databases like KEGG, COG, and Gene Ontology, with enhanced capabilities for processing BacDive microbial data.

Key objectives:
- Develop automated pipelines for microbial trait prediction
- Identify which traits can be reliably predicted from genomic data
- Determine the most relevant annotation features for different traits
- Compare performance of various machine learning approaches
- Integrate and analyze comprehensive BacDive microbial data
- Scale annotation and analysis using high-performance computing resources

## Repository Structure

```markdown
📦 Repository Structure
├── 📂 Data_processing
│   ├── 📜 data_processing.py
│   ├── 📜 improved_data_processing.py
│   ├── 📜 Bacdive_Data_merge.py
├── 📂 Datasets
│   ├── 📂 assembledDataset.zip
│   ├── 📂 reducedDataset.zip
│   ├── 📂 terms_COG.zip
│   ├── 📂 terms_GO.zip
│   ├── 📂 terms_KO.zip
│   ├── 📂 tsv.zip
│   ├── 📂 merged_bacdive_data.csv
├── 📂 Genome_Annotation
│   ├── 📜 Genome_download.py
│   ├── 📜 improved_genome_download.py
│   ├── 📜 genome_annotation_slurm.sh
│   ├── 📜 custom_annotation.py
├── 📂 Pipelines
│   ├── 📜 Multilabel_pipline.ipynb
│   ├── 📜 Single_feature_pipeline.ipynb
├── 📂 README.md
├── 📂 Supplementary_scripts
│   ├── 📜 API_cred.py
│   ├── 📜 K_func.py
│   ├── 📜 Object_oriented_dataprocess.py
│   ├── 📜 REMOVER.ipynb
│   ├── 📜 bacdive_filtered_metadata.json
```

## Key Components

### Data Processing
- [**data_processing.py**](Data_processing/data_processing.py): Original data processing module
- [**improved_data_processing.py**](Data_processing/improved_data_processing.py): Enhanced module with support for BacDive data
- [**Bacdive_Data_merge.py**](Data_processing/Bacdive_Data_merge.py): Specialized script for merging BacDive TSV files

The improved data processing module includes:
- Specialized processors for different annotation types (KO, GO, COG)
- Dedicated BacDiveProcessor class for handling BacDive-specific data
- Enhanced feature extraction and trait preprocessing
- Robust error handling and logging
- Support for merging multiple data sources

### Genome Annotation
- [**Genome_download.py**](Genome_Annotation/Genome_download.py): Original genome download script
- [**improved_genome_download.py**](Genome_Annotation/improved_genome_download.py): Enhanced parallel genome download script
- [**genome_annotation_slurm.sh**](Genome_Annotation/genome_annotation_slurm.sh): SLURM script for GPU cluster annotation
- [**custom_annotation.py**](Genome_Annotation/custom_annotation.py): Python-based annotation pipeline

The enhanced genome annotation tools provide:
- Parallel genome downloading with robust error handling
- GPU-accelerated annotation on HPC clusters using SLURM
- Support for multiple annotation tools (Prokka, Bakta, eggNOG-mapper, InterProScan)
- Comprehensive feature extraction from annotation results

### Pipelines
- **Multilabel Pipeline**: For predicting multiple traits simultaneously
- **Single Feature Pipeline**: For focused prediction of individual traits
- Includes feature selection and model training steps

### Datasets
- **Annotation Terms**: Processed COG, GO, and KO term matrices
- **Reduced Dataset**: Subsampled data for quick testing
- **Assembled Dataset**: Complete dataset for final analysis
- **BacDive Data**: Comprehensive microbial trait data from BacDive database
  - Multiple TSV files with various trait information
  - Merged into a single comprehensive CSV for analysis

## New Features

### BacDive Data Integration
This project now includes comprehensive support for BacDive microbial data:
- Specialized TSV merging script for handling BacDive's complex data structure
- Feature extraction from various BacDive trait categories
- Mapping between BacDive strain identifiers and genome accession numbers
- Support for multiple trait types and annotation formats

### High-Performance Computing Support
The project has been enhanced with tools for large-scale analysis:
- SLURM scripts for GPU-accelerated genome annotation
- Parallel genome downloading and processing
- Optimized data processing for large datasets
- Support for distributed computing environments

### Enhanced Data Processing
The improved data processing module offers:
- Better handling of missing and inconsistent data
- Support for multiple annotation formats
- Specialized processors for different data types
- Comprehensive logging and error handling
- Flexible feature extraction and selection

## Usage

### Data Preparation
1. Merge BacDive TSV files into a comprehensive dataset:
   ```bash
   python improved_tsv_merge.py
   ```

2. Download genome sequences based on accession numbers:
   ```bash
   python improved_genome_download.py --input merged_bacdive_data.csv --output-dir genomes
   ```

3. Annotate genomes on a GPU cluster:
   ```bash
   sbatch genome_annotation_slurm.sh genome_list.txt annotation_results prokka
   ```

### Data Processing
1. Process the annotated genomes and trait data:
   ```python
   from improved_data_processing import BacDiveProcessor
   
   processor = BacDiveProcessor(bacdive_data_path='merged_bacdive_data.csv')
   dataset = processor.prepare_ml_dataset(trait_column='gramStain')
   ```

### Machine Learning
1. Run either:
   - `Pipelines/Multilabel_pipeline/` for multi-trait prediction
   - `Pipelines/Single_feature_pipeline/` for individual trait analysis

## References

1. Merkesvik, J. (2022). Towards genotype—phenotype association: Leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome. *NTNU Master's Thesis*.
2. Weimann et al. (2016). From genomes to phenotypes: Traitar, the microbial trait analyzer. *mSystems*. https://doi.org/10.1128/msystems.00101-16
3. Davis et al. (2020). The PATRIC Bioinformatics Resource Center. *NAR*. https://doi.org/10.1093/nar/gkz943
4. Gene Ontology Consortium. (2000). *Nature Genetics*. https://doi.org/10.1038/75556
5. Tatusov et al. (2000). The COG database. *NAR*. https://doi.org/10.1093/nar/28.1.33
6. Reimer, L.C., Söhngen, C., Vetcininova, A. et al. (2022). BacDive – the Bacterial Diversity Metadatabase in 2022. *Nucleic Acids Research*. https://doi.org/10.1093/nar/gkab961
7. Hudgins, E.M. *Predicting Microbial Traits from Genome Annotations* (Current Thesis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
