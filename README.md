# Geo-PLA
## Codes and data of "Local-global Structure-aware Geometric Equivariant Graph Representation Learning for Predicting Protein-Ligand Binding Affinity"

Predicting protein-ligand binding affinities is a critical problem in drug discovery and design. A majority of existing methods fail to accurately characterize and exploit the geometrically invariant structures of protein-ligand complexes for predicting binding affinities. In this study, we introduce Geo-PLA, a framework for learning local-global structure-aware geometric equivariant graph representation to predict binding affinity by capturing the geometric information of protein-ligand complexes. Specifically, the local structural information of 3D protein-ligand complexes is extracted by using an equivariant graph neural network, which iteratively updates node representations while preserving the equivariance of coordinate transformations. Meanwhile, a graph transformer is utilized to capture long-range interactions among atoms, offering a global view that adaptively focus on complex regions with a significant impact on binding affinities. Furthermore, the multi-scale information from the two channels is integrated to enhance the predictive capability of the model. Extensive experimental studies on two benchmark datasets confirm the superior performance of Geo-PLA. Moreover, the visual interpretation of the learned protein-ligand complexes further indicates that our model could offer valuable biological insights for virtual screening and drug repositioning.

Contact: haichengyi@gmail.com