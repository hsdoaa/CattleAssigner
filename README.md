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

The most informative SNPs determined by fixation index (FST) were provided by the first author of CattleAssigner paper from external public data source.

To allow user practicing with  CattleAssigner ML models clf_XGBoost.py or clf_RF.py, we provide a sample dataset (192_slected_SNPs_RF_mda.csv) with the most informative SNPs selected by RF_mda feature selection method. The user can generate similar datasets with most informative SNPs using the CattleAssigner feature selection methods pca_select_features.py or RF_gini.py or RF_mda.py.  We also provide dataset annotation file (dataset_annotation.csv) that is needed to annotate/label each sample in 192_slected_SNPs_RF_mda.csv for any of the four lineages (African or European or Indicine lineage or admixed lineage).




