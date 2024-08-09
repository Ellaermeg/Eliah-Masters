# Predicting Microbial Traits from Genome Annotations

This repository contains the code and analysis for predicting microbial traits from genome annotations, as part of a master's thesis project. The goal of this project is to leverage machine learning techniques to predict the trophic levels and other traits of microorganisms based on their genomic data.

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Feature Importance and Pathway Mapping](#feature-importance-and-pathway-mapping)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [References](#references)

## Introduction

Understanding the traits of microorganisms is crucial for various applications in biotechnology, ecology, and medicine. This project aims to predict microbial traits, particularly trophic levels, using genomic data. By applying different machine learning models, we aim to classify these traits and identify the key genomic features that contribute to these classifications.

## Data Preparation

### Genome Annotation Data
The dataset consists of genome annotations that were processed to extract KEGG Orthology (KO) terms. These KO terms serve as features for the machine learning models.

### Trait Data
The traits of interest, particularly trophic levels (e.g., photo, chemo, litho, hetero, organo, auto), were extracted and binarized to serve as target labels for the models.

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
Due to the multilabel nature of the problem (where a microorganism can exhibit multiple traits), MultiOutput Classifiers were used.

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
Feature importance was analyzed using RandomForest and LogisticRegression models. The most important features (KO terms) were identified and mapped to their respective pathways using the KEGG database.

### Pathway Mapping
Selected features were mapped to biological pathways, and an interactive network visualization was created to explore the relationships between the KO terms and pathways.

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
The results section highlights the key findings from the analysis, including the most predictive KO terms for each trophic level and the associated biological pathways.

## Troubleshooting
This section provides solutions to common issues encountered during the analysis, such as handling multioutput classifiers, resolving warnings, and improving model performance.

## Future Work
Potential future directions for this project include expanding the dataset, exploring additional microbial traits, and refining the machine learning models for better accuracy and interpretability.

## References
1. Merkesvik, J. (2022). *Towards genotype—phenotype association: Leveraging multiple-source microbial data and genome annotations to infer trait attributes for the uncultured microbiome* (Master's thesis). Norwegian University of Science and Technology (NTNU).

2. Weimann, A., Mooren, K., Frank, J., Pope, P. B., Bremges, A., & McHardy, A. C. (2016). From genomes to phenotypes: Traitar, the microbial trait analyzer. *mSystems, 1*(6), e00101-16. https://doi.org/10.1128/msystems.00101-16

3. Davis, J. J., Wattam, A. R., Aziz, R. K., Brettin, T., Butler, R., Butler, R. M., ... & Stevens, R. (2020). The PATRIC Bioinformatics Resource Center: Expanding data and analysis capabilities. *Nucleic Acids Research, 48*(D1), D606-D612. https://doi.org/10.1093/nar/gkz943

4. The Gene Ontology Consortium. (2000). Gene Ontology: Tool for the unification of biology. *Nature Genetics, 25*(1), 25-29. https://doi.org/10.1038/75556

5. Tatusov, R. L., Galperin, M. Y., Natale, D. A., & Koonin, E. V. (2000). The COG database: A tool for genome-scale analysis of protein functions and evolution. *Nucleic Acids Research, 28*(1), 33-36. https://doi.org/10.1093/nar/28.1.33

6. Eliah Mathias Hudgins. *Predicting Microbial Traits from Genome Annotations* (Master's Thesis).

7. Kyoto Encyclopedia of Genes and Genomes (KEGG). *KEGG Orthology (KO) database*. Retrieved from https://www.genome.jp/kegg/ko.html

8. Scikit-learn developers. (n.d.). *Scikit-learn: Machine Learning in Python*. Retrieved from https://scikit-learn.org/stable/






