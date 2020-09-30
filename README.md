# GramCorr
Experiments with Grammatical Error Correction (**In progress**)

## Data
Experiments are performed using KoKo German L1 Learner Corpus (version 3):
* [Link where KoKo corpus can be downloaded](https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/12?download=https%3A%2F%2Fclarin.eurac.edu%2Frepository%2Fxmlui%2Fbitstream%2Fhandle%2F20.500.12124%2F12%2Fmmax-v3.zip%3Fsequence%3D12%26dtoken%3Da0ec2a899953dd350b90b3f4dd4a47ef)

As a source of additional data, we used eScape datasets for Automatic Post-Editing (APE) task:
* [eSCAPE](http://hltshare.fbk.eu/QT21/eSCAPE.html)

# Submodules

Initialize submodules by

`git submodule update --init --recursive`

### Repository with experiments on Low Resource GEC for German:
https://github.com/adrianeboyd/boyd-wnut2018

The repo contains a submodule `errant` with the extensions to ERRANT for German.
Install spacy and download the German models:

```
pip install -U spacy==2.0.0
python -m spacy download de
```

Install [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) under `errant/resources/tree-tagger-3.2`.
German parameter file `german.par` is [here](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german.par.gz).
If you save the parameter file in `errant/` submodule, you should specify the path to it when initializing TreeTagger, e.g.
in `parallel_to_m2.py` specify TAGPARFILE.    

Install treetaggerwrapper:

```
pip install treetaggerwrapper
```

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

