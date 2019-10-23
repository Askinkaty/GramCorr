# GramCorr
Experiments with Grammatical Error Correction

## Data
Experiments are performed using KoKo German L1 Learner Corpus (version 3):
* [Link where KoKo corpus can be downloaded](https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/12?download=https%3A%2F%2Fclarin.eurac.edu%2Frepository%2Fxmlui%2Fbitstream%2Fhandle%2F20.500.12124%2F12%2Fmmax-v3.zip%3Fsequence%3D12%26dtoken%3Da0ec2a899953dd350b90b3f4dd4a47ef)

As a source of additional data, we used eScape datasets for Automatic Post-Editing (APE) task:
* [eSCAPE](http://hltshare.fbk.eu/QT21/eSCAPE.html)


# Dev and Use

## Reproducible environment
### Initial set-up
The `environment.yml` file gives an overview of the dependencies to use this project.  The easiest way to get a working environment is to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) in the following way:

`conda env create -f gramcorr/environment.yml`

## Update
The `environment.yml` file may change (occasionally). To get your local environment in sync with the update use:

`conda env update -f environment.yml --prune`

## More information
* https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
