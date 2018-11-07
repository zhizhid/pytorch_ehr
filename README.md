# Pytorch_ehr
***************** 

**Overview**
* Predictive analytics of risk onset on EHR cerner data using Pytorch library;
* Data description: Cerner, with 15815 unique medical codes. Full cohort with >1,000,000 records.
format: 
code types & what they stand for: 
some visuals of the what the data looks like: 
* Models built: Vanilla RNN, RNN with GRU, RNN with LSTM, Bidirectional RNN, Dialated RNN, QRNN, T-LSTM, GRU-Logistic Regression, plain LR, LR with embedding, Random Forest;

**Folder Organization**
* ehr_pytorch: main folder with modularized components for all models, data loading and processing, and training, validation and test of models, main file for parsing arguments, and a EHRDataloader;
* 1. EHRDataloader: a separate function to allow for utilizing pytorch dataloader child object to create preprocessed data batch for testing our models;
* data: sample processed (pickled) data from Cerner, can be directly utilized for dataloader, and then models
* Tutorials: jupyter notebooks with examples on how to utilize our dataloader and run our models with visuals
* Test: coming up soon. Shell commands to quickly test on our package functionalities
* Sphinx build documentations
* Sample results:(? keep or discard? prob discard) 
* Heart Failure   &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Readmission
  <p float= "left">
       <img src="SampleResults/HF.png" alt="Heart Failure" alt="List screen" title="List screen" height = "330" width="270" />
       <img src="SampleResults/Readm.png" alt="Readmission" height = "330" width="270" />
  </p>
  <p float="left">
        <img src="SampleResults/comparision.png" alt="Comparision" width="540" />
  </p>
* The [paper]() upon which this repo was built. (include paper link)

## Prerequisites

* Pytorch library, <http://pytorch.org/> 


## Tests


* To try our dataloader, use:
<pre>
data = EHRdataFromPickles(root_dir = '/data/projects/py_ehr_2/Data/', 
                                      file = 'hf50_cl2_h143_ref_t1.train')
loader =  ataLoader(data, batch_size=10, shuffle=False, collate_fn=my_collate)
#if you want to shuffle batches before using them 
iterbatchloader(loader = loader)
#otherwise 
iterloader = iter(loader)
iterloader.__next__()
</pre>

* To run our models, use:
<pre>
python main.py -seq_file ... -label_file ...
</pre>


## Authors

See the list of [contributors]( https://github.com/ZhiGroup/pytorch_ehr/graphs/contributors)


## Acknowledgements
Hat-tip to:
* [DRNN github](https://github.com/zalandoresearch/pt-dilate-rnn)
* [QRNN github](https://github.com/salesforce/pytorch-qrnn)
* [T-LSTM paper](http://biometrics.cse.msu.edu/Publications/MachineLearning/Baytasetal_PatientSubtypingViaTimeAwareLSTMNetworks.pdf)



