# README

## Experiment with KoKo data

### Pre-processing
- Take the (manual) orthographic error corrections, 
- train different MT systems (e.g. word n-gramms, character n-gramms) 

### Experimental setup
Try to outsmart the individual algorithms with an ensemble learning setup:
- Take the individual predictions of the MT systems,
- test different ML algorithms on the combined predictions,
- use the best resulting model to make overall predications.

## Notes

### Input data representation

Generally, the data encodes (possibly multiple) suggestions from different MT
systems for individual errors, originally marked and annotated in the KoKo
files.  The MT systems try to correct the original text into the corrected
version.  The data comes in three csv (folds) files, where each file has an
identical structure.  

- **errorId** the error id for which multiple suggestions were generated	
- **errorType** not needed
- **correction** not needed
- **corr_1_incorr_0** TRUE class of the suggested correction
  - *1* correct (identical to the manually annotated data)
  - *0* incorrect
Pairs of values for each system
- **ALGO_1_suggested_-1_other_0_nothing** nominal label
  - *1*  this is the (best) suggested correction for this system
  - *0*  the system had NO suggestion for this 'errorID' (i.e. all values for
         this system and this errorID are 0)
  - *-1* this correction was suggested by the system but not as the best
- **ALGO**  confidence value dependent on the nominal label
  - *if 1*  confidence with which the system suggested this correction
  - *if 0*  0.0
  - *if -1* the confidence for the best suggestion
