# rbm-review
This repository accompanies the review paper, ______________, providing sample python code __________ the model architectures _____ in the paper.

How to run the training:
python .\scripts\run_train.py --config configs/default.yaml

Data and File structure
The configs directory contains .yaml files that can are used as an argument for specifying parameters and hyperparameters for training, used like:
    python .\scripts\run_train.py --config configs\default.yaml

In the 
Code
1 - 

A short summary of the related research investigation
An overview of files and folder structure
A key to file and column labels, variables, data codes, and units of measurement
Details of processing and analysis, such as software
Links to other data sources, if applicable

Access this dataset on Dryad DOI: 10.5061/dryad.hqbzkh1nh

Description of the data and file structure
Each script starts with a brief description of what each does. The main scripts are noted from 1 to 7.

Phylogenetic trees (spe.tree.txt and phylogenetic_tree.tre) are also available. The phylogenetic_tree.tre file is a combined tree of 2 *BEAST chains after convergence was verified, and spe.tree.txt is the filtered tree (i.e. species in the tree but for which we don't have 2D morphometrics). All the filtering is detailed in script 4.

The folder Photos_of_Gesneriaceae_flowers.zip contains raw images used for geometric morphometrics.

The Landmarks folder contains the raw data of individual floral morphology of species of Antillean Gesneriaceae and their pollination syndromes used in downstream analyses:

data.csv provides the metadata relevant to identifying each image in the photos zip folder, the pollination strategy, and a column "Confirmed", which defines if the pollination strategy has been confirmed by observation (yes) or if it is inferred from floral traits (no). 
matrix_semilandmarks.csv is used to identify the order of landmarks and sliding semi-landmarks.
numerisation_YYY.MM.DD.txt corresponds to the 2D geometric morphometric quantification of corolla shape, their corresponding file names numerisation_YYY.MM.DD_names.txt identifies the specimens numerised, and landmarks_YYY.MM.JJ.txt is the resulting file using the two precedent ones and the script landmarks_protocole_script.R, which is a shorter and generic version of the landmarks.R file. It transforms the landmark file obtained from ImageJ and pointpicker into a TPS-like file for further geometric morphometrics analyses.
Files_to_remove.csv identifies the files that were discarded.
Code
1 - ImageJ logs to TPS-like.R
This script is to prepare the landmark files we created using ImageJ/Fiji in the same format as if data were retrieved from the software TPS.

2 - TPS and cleaning.R
This script is to use the TPS files and create a landmarks.rds file to use in downstream morphometric analyses.
Additionally, the script landmarks_protocole_script.R is provided in the Landmarks folder as a generic and shorter version of the landmarks.R file. It transforms the landmark file obtained from ImageJ and PointPicker into a TPS-like file for further geometric morphometrics analyses.

3 - 2D_morphometrics on specialists and generalists.R
Read landmarks.
Filter landmarks data per syndrome and remove species without enough photos.
Procrustes analysis.
Procrustes ANOVA.
Create the objects land.gpa and data.sorted_hummingbird.mixed.
PCA on individual coordinates, and scree plot.
Create TPS figures of the mean floral shape and the morphology variation along the PCA axes.
Same thing with the mean PC coordinates (shape) of each syndrome.
Visualisation of the PCA and PCA on species means, as well as interactive PCA.
From this script, you will produce the objects data.sorted_hummingbird.mixed and land.pca used in the JIVE analysis.

4 - Analysis of inter intra specific trait evolution.R
This is the hierarchical Bayesian Integrative model of Trait evolution analysis (JIVE).
First, we retrieve the phylogenetic tree and trim it to fit with the morphometric data.
Prepare the trait matrices used in the joint inter- and intraspecific evolution analysis.
Stochastic mapping of the pollination syndromes along the phylogenetic tree (SIMMAP), and testing of the two models of evolution used for the SIMMAPs.
Preparation for the JIVE analysis.
The actual analyses.
The ESS estimations.
The calcul of marginal likelihoods using stepping stone and the Bayes Factors, as well as figures related to BF.
The plots of the estimated theta values for specialists and generalists and finally, the exploration of the SIMMAPs in favor of each model or neither.

5 - Preparation for calcul on server.R
This is an example of a JIVE analysis for only a subset of MCMC and one PC axis of the floral morphospace.
The jive objects are created and tuned for the 3 PCs and 100 SIMMAPs, but the MCMCs are run for a subset of 10 SIMMAPs and only one PC.
Each script then accounts for 10 MCMC each. We then have 30 subsets representing 600 MCMC that were run remotely on the Calcul server.

6 - PGLS.R
From the Procrustes aligned landmark data saved in script 3, calcul of variances, and PGLS analysis on variances.
Figures of residuals and barplot of variances that fit the phylogeny.

7 - mvMORPH.R
Calcul of the log variation for each PC axis.
Multivariate analysis with mvOU and mvOUM on the variances used in the PGLS, the log-variance of PC1, PC2, PC3 independently, and PC1+PC2+PC3 jointly, for all SIMMAPs.
Plot of the theta values resulting from the analyses.
Calcul of the AICc weights to compare the fit of the models and plot the AICc weights.

utility_functions.R
Additional utility script used for geometric morphometrics for thin-plate spline transformation grids.

outlierKD.R
Additional utility script used for geometric morphometrics to check for outliers in landmarks data.