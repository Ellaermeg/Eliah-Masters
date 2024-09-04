# Predicting Microbial Traits from Genome Annotations

This repository contains the code and analysis for predicting microbial traits from genome annotations, as part of a master's thesis project. The goal of this project is to leverage machine learning techniques to predict traits/phenotypes of microorganisms based on their genome annotations.

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Feature Importance and Pathway Mapping](#feature-importance-and-pathway-mapping)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [Object-Oriented Data Processing](#object-oriented-data-processing)
- [References](#references)

## Introduction

Understanding the traits of microorganisms is crucial for various applications in biotechnology, ecology, and medicine. This project aims to predict microbial traits from genome annotations.  The cellular phenotype includes a variety of measurable traits such as but not limited too; cell morphology, gram status, nutrient requirements, oxygen tolerance, byproduct formation, trophy, etc.. By applying different machine learning methods, this project aims to create a pipeline for predicting microbial traits from genome annotations and understand which kind of traits can be more reliably predicted and which annotation features are the most relevant.



## Data Preparation

### Genome Annotation Data
The dataset consists of genome annotations. These annotation terms serve as features for the machine learning models.

### Trait Data
The traits of interest, an example being trophic levels (e.g., photo, chemo, litho, hetero, organo, auto), were extracted and binarized to serve as target labels for the models.

### Data Alignment
The feature matrix (X) and the target matrix (Y) were aligned based on common indices to ensure that the data used for training and evaluation were consistent.

```python
# Align X (features) and Y (labels) based on common keys
common_keys = X_filtered_df.index.intersection(y.index)
X_aligned = X_filtered_df.loc[common_keys]
Y_aligned = y.loc[common_keys]
```
## Feature selection
Feature selection was performed using various techniques such as Variance Thresholding and SelectKBest to reduce the dimensionality of the dataset and improve model performance.

```python
# Apply VarianceThreshold to remove constant features
selector = VarianceThreshold(threshold=0.05)
X_filtered = selector.fit_transform(X_terms)
```

## Model Training and Evaluation

### MultiOutput classifiers
Due to the multilabel nature of the problem (where a microorganism can exhibit multiple traits, without them being mutually exclusive), MultiOutput Classifiers were used.

### Cross-validation
Cross-validation was employed to evaluate the performance of different classifiers, including RandomForest, SVC, LogisticRegression, and BernoulliNB, across various feature subsets.

```python
# Perform cross-validation for F1-score
f1_scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1)
results[name]['f1'].append(f1_scores.mean())
```
### Preformance Metrics
The performance of the models was assessed using F1 Score and Matthews Correlation Coefficient (MCC).

## Feature Importance and Pathway Mapping
### RandomForest and LogisitcRegression
Feature importance was analyzed using RandomForest and LogisticRegression models. The most important features (annotation terms) were identified and mapped to their respective pathways using different databases (KEGG, COGS, eGGnoGGmapper).

### Pathway Mapping
Selected features were mapped to biological pathways, and an interactive network visualization was created to explore the relationships between the annotation terms and pathways.

```python
# KEGG Pathway Mapping
def map_ko_to_pathways(ko_terms):
    pathways = {}
    for ko in ko_terms:
        try:
            gene_links = kegg.link("pathway", ko)
            if gene_links:
                for entry in gene_links.strip().split("\n"):
                    split_entry = entry.split("\t")
                    if len(split_entry) >= 2:
                        ko_id, pathway_id = split_entry[0], split_entry[1]
                        if pathway_id not in pathways:
                            pathways[pathway_id] = set()
                        pathways[pathway_id].add(ko)
        except Exception as e:
            print(f"Error processing {ko}: {e}")
    return pathways
```

## Results
The results section highlights the key findings from the analysis, including the most predictive annotation terms for each trophic level and the associated biological pathways (In this example).

## Troubleshooting
This section provides solutions to common issues encountered during the analysis, such as handling multioutput classifiers, resolving warnings, and improving model performance.

## Future Work
Potential future directions for this project include expanding the dataset, exploring additional microbial traits, and refining the machine learning models for better accuracy and interpretability.

## Object-Oriented Data Processing

An object-oriented approach was implemented for processing and managing the data used in this project. The script `Object_oriented_dataprocess.py` provides a modular and scalable framework that enhances the maintainability and flexibility of the code. This approach allows for:

- **Modular Design**: Encapsulation of data processing tasks into classes and methods, making the code more organized and reusable.
- **Scalability**: Easy integration of additional features and data processing steps without disrupting the existing codebase.
- **Maintainability**: Simplified debugging and testing due to the clear structure and separation of concerns within the code.

### Key Components

1. **Data Loading and Preprocessing**: 
   - The script handles the loading of genome annotations and trait data, ensuring that both are properly aligned for subsequent analysis.
   
2. **Feature Selection**: 
   - Integrated feature selection methods such as Variance Thresholding and SelectKBest within the object-oriented framework to streamline the process.
   
3. **Model Training and Evaluation**: 
   - The script includes methods for training and evaluating machine learning models using cross-validation techniques.

4. **Pathway Mapping**: 
   - An extension that allows for mapping selected features to biological pathways, leveraging the modular structure to add this functionality without affecting other parts of the code.

You can explore the implementation details by accessing the script [here](https://github.com/Ellaermeg/Eliah-Masters/blob/main/Data_Feature/Object_oriented_dataprocess.py).


## References
1. Merkesvik, J. (2022). *Towards genotype—phenotype association: Leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome* (Master's thesis). Norwegian University of Science and Technology (NTNU).

2. Weimann, A., Mooren, K., Frank, J., Pope, P. B., Bremges, A., & McHardy, A. C. (2016). From genomes to phenotypes: Traitar, the microbial trait analyzer. *mSystems, 1*(6), e00101-16. https://doi.org/10.1128/msystems.00101-16

3. Davis, J. J., Wattam, A. R., Aziz, R. K., Brettin, T., Butler, R., Butler, R. M., ... & Stevens, R. (2020). The PATRIC Bioinformatics Resource Center: Expanding data and analysis capabilities. *Nucleic Acids Research, 48*(D1), D606-D612. https://doi.org/10.1093/nar/gkz943

4. The Gene Ontology Consortium. (2000). Gene Ontology: Tool for the unification of biology. *Nature Genetics, 25*(1), 25-29. https://doi.org/10.1038/75556

5. Tatusov, R. L., Galperin, M. Y., Natale, D. A., & Koonin, E. V. (2000). The COG database: A tool for genome-scale analysis of protein functions and evolution. *Nucleic Acids Research, 28*(1), 33-36. https://doi.org/10.1093/nar/28.1.33

6. Eliah Mathias Hudgins. *Predicting Microbial Traits from Genome Annotations* (Master's Thesis).

7. Kyoto Encyclopedia of Genes and Genomes (KEGG). *KEGG Orthology (KO) database*. Retrieved from https://www.genome.jp/kegg/ko.html

8. Scikit-learn developers. (n.d.). *Scikit-learn: Machine Learning in Python*. Retrieved from https://scikit-learn.org/stable/






