CattleAssigner is a framework that deploys two Machine Learning (ML) models/classifiers for assigning animal to one of four types of cattle lineages  and  their populations within the prospective lineage usning the minimum number of Single nucleotide polymorphisms (SNPs) determined by different feature reduction and feature selection techniques. The lineage types include African lineage, Indicine lineage, European lineage, and Admixed lineage.  The ML classification models include:

Random forest (RF) 

XGBoost

The RF model  have been implemented in python using Scikit-learn library and XGBoost ML model has been implemented using Python XGBoost package. 

To run RF classification model, please run clf_RF.py

To run XGBoost classification model, please run clf_XGBoost.py

The minimum number of Single nucleotide polymorphisms (SNPs) are determined one of four methods:

Principal Component Analysis(PCA) for feature extraction

The RF with GINI feature selection method.

The RF with Mean Decrease in Accuracy (MDA) feature selection methods.

The fixation index (FST) is a measure of population differentiation due to genetic structure.

To run PCA for feature extraction, please run pca_select_features.py

To run RF with GINI feature selection method, please run RF_gini.py

To run RF with MDA feature selection method, please run RF_mda.py

The most informative SNPs determined by fixation index (FST) were provided by the first author of this paper from external data source.


