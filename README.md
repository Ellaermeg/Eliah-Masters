## README: Predicting Microbial Traits from Genome Annotations #
# Overview
This project leverages next-generation sequencing (NGS) and machine learning techniques to predict microbial traits from genome annotations. By analyzing the links between genotype and phenotype, this pipeline aims to enhance our understanding of microbial traits, particularly for microbes that are challenging to cultivate and study in traditional laboratory settings.

# Background
Next-generation sequencing (NGS) has enabled the assembly of genomes for thousands of microbial species. NGS is also helping to identify the species present in environmental samples and to shed a light on the so-called ”great plate count anomaly”, showing that we are still unable to cultivate a vast majority of microbes. Understanding the link from genotype to phenotype is therefore limited by the availability of phenotypic data. The cellular phenotype includes a variety of measurable traits such as cell morphology, gram status, nutrient requirements, oxygen tolerance, byproduct formation, etc. In a previous work, a database of microbial traits was created and curated through the integration of specialized databases. This database was the used to show that phenotypic traits, such as gram status, could be predicted from genome annotation features such as gene ontology terms (GOs) and gene ortholog groups (COGs, KOs).

Previous work has established a database of microbial traits by integrating data from specialized databases, demonstrating that phenotypic traits can be predicted from genome annotation features, such as gene ontology terms (GOs) and gene ortholog groups (COGs, KOs). Building on this, our project explores various machine learning methods to predict these traits more reliably and understand which features of genome annotations are most predictive.

# Project Goals
Data Preprocessing: Apply feature selection and extraction methods to genome annotation data.
Model Comparison: Test and compare different supervised and unsupervised machine learning methods, including Naive Bayes, Logistic Regression, Support Vector Machines, and Random Forests.
Trait Prediction: Determine which microbial traits can be most accurately predicted.
Feature Relevance: Identify which genome annotation features have the highest predictive power.
Biological Relevance: Analyze the biological significance of the associations between gene features and microbial traits.

# Methodology
The pipeline involves several key steps:

- Data Preparation:

Genome Annotation Features: Import and preprocess features from the KO (KEGG Orthology) database, applying variance thresholds to remove constant features.
Trait Data: Align microbial trait data with genome annotation features for model training.


- Machine Learning Models:

-----

- Evaluation:

Cross-Validation: Use stratified k-fold cross-validation to evaluate model performance across different k-values for feature selection.
Metrics: Assess models using F1 scores and Matthews Correlation Coefficient (MCC) to compare effectiveness across different classifiers.

- Visualization:

Feature Importance: Visualize the top KO terms and their corresponding pathways to understand the biological implications of the predictions.
Heatmaps and Networks: Generate heatmaps and network graphs to explore the relationships between KO terms and predicted pathways.

# Challenges Encountered
During development, several issues were addressed:

Handling Multilabel Data: Transitioning from binary to multilabel classifiers required adjusting how data was fed into models, ensuring compatibility with scikit-learn's pipeline and evaluation methods.

Optimization: Strived to reduce computational overhead by minimizing redundant operations, particularly in the feature selection and model evaluation processes.

# Key Features
- Interactive Visualizations: Using PyVis and Seaborn, the project provides interactive visualizations that allow users to explore the relationships between genome annotations and microbial traits.
- Customizable Analysis: The pipeline is designed to be flexible, enabling users to select specific trophic levels or classifiers for detailed analysis.
- Reproducible Research: All code is structured to support easy replication and extension, making it suitable for further research or educational purposes.

# Future Directions
The pipeline offers a robust foundation for predicting microbial traits, with potential extensions including:

Integration with New Databases: Incorporating additional genomic and phenotypic databases to improve prediction accuracy.
Enhanced Biological Interpretation: Further analysis of the biological significance of identified features and pathways, potentially linking them to ecological functions or environmental adaptations.
# References:

Merkesvik, J. (2022). Towards genotype—phenotype association: leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome. NTNU.

Weimann, A., et al. (2016). From Genomes to Phenotypes: Traitar, the Microbial Trait Analyzer. mSystems.

Davis, J.J., et al. (2020). The PATRIC Bioinformatics Resource Center: expanding data and analysis capabilities. Nucleic Acids Research.
