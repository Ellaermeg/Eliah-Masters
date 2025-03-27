# Predicting Microbial Traits from Genome Annotations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

Master's Thesis Project: Machine learning pipelines for predicting microbial phenotypes from genomic annotations.

## Introduction
This repository contains code and analysis for predicting microbial traits from genome annotations, developed as part of a master's thesis project. The project aims to leverage machine learning techniques to predict measurable microbial traits (cell morphology, gram status, oxygen tolerance, nutrient requirements, etc.) based on functional genome annotations from databases like KEGG, COG, and Gene Ontology.

Key objectives:
- Develop automated pipelines for microbial trait prediction
- Identify which traits can be reliably predicted from genomic data
- Determine the most relevant annotation features for different traits
- Compare performance of various machine learning approaches

## Repository Structure
<details>
<summary>Repository Structure</summary>
├── Data_processing/
│ └── data_processing.py
├── Datasets/
│ ├── Terms_COG/
│ ├── Terms_GO/
│ ├── Terms_KO/
│ ├── reduced_dataset/
│ └── assembled_dataset/
├── Genome_annotation/
│ ├── Bacdive_metadata/
│ └── Genome_download/
└── Pipelines/
├── Multilabel_pipeline/
└── Single_feature_pipeline/

Copy
</details>


## Key Components

### [Data Processing](Data_processing/data_processing.py)
- Handles data cleaning and transformation
- Merges phenotypic data with genomic annotations
- Prepares datasets for machine learning

### [Pipelines](Pipelines)
- **Multilabel Pipeline**: For predicting multiple traits simultaneously
- **Single Feature Pipeline**: For focused prediction of individual traits
- Includes feature selection and model training steps

### [Datasets](Datasets)
- **Annotation Terms**: Processed COG, GO, and KO term matrices
- **Reduced Dataset**: Subsampled data for quick testing
- **Assembled Dataset**: Complete dataset for final analysis

### [Genome Annotation](Genome_annotation)
- **BacDive Metadata**: Curated phenotypic trait data
- **Genome Download**: Scripts for retrieving genomic data

## Usage
1. Prepare data using `Data_processing/data_processing.py`
2. Run either:
   - `Pipelines/Multilabel_pipeline/` for multi-trait prediction
   - `Pipelines/Single_feature_pipeline/` for individual trait analysis

## References
1. Merkesvik, J. (2022). Towards genotype—phenotype association: Leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome. *NTNU Master's Thesis*.
2. Weimann et al. (2016). From genomes to phenotypes: Traitar, the microbial trait analyzer. *mSystems*. https://doi.org/10.1128/msystems.00101-16
3. Davis et al. (2020). The PATRIC Bioinformatics Resource Center. *NAR*. https://doi.org/10.1093/nar/gkz943
4. Gene Ontology Consortium. (2000). *Nature Genetics*. https://doi.org/10.1038/75556
5. Tatusov et al. (2000). The COG database. *NAR*. https://doi.org/10.1093/nar/28.1.33
6. Hudgins, E.M. *Predicting Microbial Traits from Genome Annotations* (Current Thesis)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
