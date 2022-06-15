# Motley: Benchmarking Heterogeneity and Personalization in Federated Learning

This repo contains **cross-silo** experiments for the **ADNI** dataset.

## ADNI dataset*


ADNI data were obtained from the [Alzheimer’s Disease Neuroimaging Initiative database](http://adni.loni.usc.edu).


* Please follow the [instructions](https://adni.loni.usc.edu/data-samples/access-data/) to apply for access to the data
* Task: Predict Standardized Uptake Value Ratio (SUVR) from PET scans of human brains
* We treat each scanner vendor as a silo. There are 9 silos in total.
* Dataset: From the ADNI database, we specifically take a subset of PET scans (with AV45 and preprocessing step ```Coreg, Avg, Std Img and Vox Siz, Uniform Resolution```) that have existing labels obtained from UC Berkeley study (also part of the database). The label information can be found on the website under 
```study_data/imaging/PET_Image_Analysis/UC_Berkeley-AV45-Analysis-*```.
 Meta data specifying the silo attribute would be automatically included in the downloaded zip files, together with images in DCM format. 
 
#### Preprocessing

```
bash preprocess.sh  # replace the variables in preprocess.sh with the correct paths
```

## Directory Structure

* `data/adni` folder contains everything related to data (preprocessing scripts, raw data, preprocessed data, etc)
* `flearn/models` defines the CNN regression model and sets up clients
* `flearn/trainers_global` implements FedAvg, FedAvg with server-side momentum, and FedAvg (or FedAvg w/ momentum) + fine-tuning
* `flearn/trainers_persinalization` implements the clustering (with warm start) method, Ditto, ensembling, and local training
* `utils/model_utils.py` builds part of the data pipeline
* `main.py` is the driver script with flags


### Single run

For example, local training on adni can be run as

```bash
bash run_adni.sh local
```
where the training method can be chosen from ```['fedavg', 'local', 'clustering', 'ensemble', 'fedavgM', 'ditto']```.

### Hyperparameter values

See our paper for details.

## *Acknowledgement

A complete listing of ADNI investigatorscan be found [here](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf). The investigators within the ADNI contributed to the design and implementation of ADNI and/orprovided data but did not participate in analysis or writing of this paper.

