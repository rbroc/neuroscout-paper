# neuroscout-paper

This repository contains the analysis code for the analyses in the manuscript: "Neuroscout: a unified platform for generalizable and reproducible fMRI research"

Analyses in the paper required running models at the individual dataset / task level, following by a meta-analysis across all datasets and tasks. Code related to single task models is found in `analyses/` while meta-analysis code is found in `/meta`. The Neuroscout model IDs for each analysis can be found under `/analyses/models/`. 

The analyses are organized by Figure in the manuscript. Most analyses requires first running single dataset results and then performing a meta-analysis for the final figure. In the top level directory, you can find the notebooks to create and 

Figure 1)
analyses/0 - figure 1 methods.ipynb 

Figure 3-4)
analyses/1 - single predictor models.ipynb 
meta/1 - single predictor models.ipynb 

Figure 5)
analyses/2 - face features.ipynb
meta/2 - face features.ipynb

Figure 6)
analyses/3 - language features.ipynb
meta/3 - language features.ipynb

