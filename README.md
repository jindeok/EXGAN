# EXGAN

Description : a prototype version 


----- ARGUMENT HELPER ----------------------------------------

- n_epochs : number of epochs of training for base set generator
- n_epochs_dual : number of epochs of dual D - training for support set generator

- XAI_method : (String type) type of XAI method to use. (e.g. IntegratedGradients, DeepLift, LRP, ...)"

- shots : few shot numbers for XAI
- alpha : balancing reals and samples when draw common heatmap
- beta : balancing reals and masked when computing total loss .
 
- MNISTlabel : select a label in MNIST imgs 
