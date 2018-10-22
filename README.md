# Pytorch_ehr
***************** 
* Predictive analytics of heart failture onset risk on EHR cerner data using Pytorch library;
* Models built: Vanilla RNN, RNN with GRU, RNN with LSTM, Bidirectional RNN, Dialated RNN, Logistic Regression with embedding dimension 1 and 2;
* The framework has modularized components for models, data loading and processing, and training, validation and test, main file for parsing arguments;
* Dataloader: a separate function to allow you to utilize pytorch dataloader child object to load preprocessed data for testing our models;
* Bayesian Optimization implemented for hyperparameters search for models, both locally using open source BayesianOptimization package and SigOpt software; 
* Other experiments include visualizations, larger datasets with separate tests for longer and shorter visits;
* Data used: Cerner, with 15815 unique medical codes. 

## Prerequisites

* Pytorch library, installation instructions could be found at <http://pytorch.org/> 
* Bayesian Optimization package at github repo: <https://github.com/fmfn/BayesianOptimization>
* SigOpt software: <https://sigopt.com/> 

## Authors

See the list of [contributors]( https://github.com/ZhiGroup/pytorch_ehr/graphs/contributors) who participated in this project.



