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

## Data Preparation
### [Data Processing](Data_processing)
- **Annotation Processing**: Scripts for handling KEGG, COG, and Gene Ontology annotations
- **Trait Data Integration**: Integration of phenotypic data from PATRIC database
- **Data Normalization**: Tools for handling imbalanced classes and data standardization

### [Datasets](Datasets)
- Curated collection of microbial genomes with associated phenotypic traits
- Preprocessed annotation matrices (KO, COG, GO terms)
- Sample datasets for quick testing of pipelines

## Feature Selection
### [Pipelines/feature_selection](Pipelines/feature_selection)
- Statistical filtering using ANOVA and χ² tests
- Tree-based feature importance analysis
- Regularization techniques (L1/L2 normalization)
- Dimensionality reduction with PCA and t-SNE

## Model Training
### [Pipelines/model_training](Pipelines/model_training)
- Implementation of various ML algorithms:
  - Random Forest
  - Gradient Boosting Machines (XGBoost)
  - Support Vector Machines
  - Logistic Regression
- Hyperparameter optimization with GridSearchCV
- Cross-validation strategies for small datasets
- Model interpretation using SHAP values

## Future Work
- Expansion to uncultured microbiome trait prediction
- Integration of deep learning approaches
- Multi-task learning for correlated traits
- Development of web-based prediction tool
- Incorporation of metabolic pathway information

## Repository Structure

├── Pipelines/ # Machine learning workflows
│ ├── feature_selection # Feature selection scripts
│ └── model_training # Model training implementations
├── Data_processing/ # Data cleaning and preprocessing
└── Datasets/ # Sample data and annotation files




## References
1. Merkesvik, J. (2022). Towards genotype—phenotype association: Leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome. *NTNU Master's Thesis*.
2. Weimann et al. (2016). From genomes to phenotypes: Traitar, the microbial trait analyzer. *mSystems*. https://doi.org/10.1128/msystems.00101-16
3. Davis et al. (2020). The PATRIC Bioinformatics Resource Center. *NAR*. https://doi.org/10.1093/nar/gkz943
4. Gene Ontology Consortium. (2000). *Nature Genetics*. https://doi.org/10.1038/75556
5. Tatusov et al. (2000). The COG database. *NAR*. https://doi.org/10.1093/nar/28.1.33
6. Hudgins, E.M. *Predicting Microbial Traits from Genome Annotations* (Current Thesis)

## Contributing
Contributions are welcome! Please open an issue first to discuss proposed changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
