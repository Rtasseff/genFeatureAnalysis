genFeatureAnalysis
====================

Pipeline code and scripts to run a general work flow for the 
analysis of data features with a focus on pairwise 
univariate statistics and random forest prediction.

Its become a bit messy, but implimenting the main
workflows is still pretty stright forward from the
command line.

Dependencies and Programs
------------------
Cloud Forest (https://github.com/ryanbressler/CloudForest)
will be used to implement the random forest.

statUtils from genTools (https://github.com/Rtasseff/genTools) 

We will be using python2.7 + numpy, scipy, matplotlib
for tasks, including pre and post processing
and implementation of the pairwise analysis.

Installation
-------------------
From scratch:

1. Install GO (http://golang.org/doc/install)
2. Install CloudForest and optional utilities 
(https://github.com/ryanbressler/CloudForest)
3. Install Python 2.7.x (https://www.python.org/downloads/;
Personally I prefer using mac ports at http://www.macports.org/)
  1. You also need Numpy v-1.8.1 
  2. and scipy v-0.14.0
  3. and matplotlib v-1.3.1 (all of which can be 
downloaded and installed using macports).
4. Download my genTools module (https://github.com/Rtasseff/genTools)
all you realy need is statsUtil.py in genTools
5. Download this repo.
  

Usage
------------------
The main purpose here was to construct a single 
work-flow that implemented a standard pairwise analysis 
followed by random forest prediction. However, it soon seemed
more valuable to split up the separate pieces into 
individual command line tools. Each with multiple options 
to customize the work-flow.  Use -h for any to get 
a better listing of the options (eg. $python pairwise.py -h)

A list of command-line work-flow tools:

pairwise.py 
All by all, Pairwise only workflows. Usage: 
$ python2.7 pairwise.py testData.afm -outDir test -log log.out -v

randforest.py 
One target vs all features, randomforest only workflows. Usage:
$ python2.7 pairwise.py testData.afm N:PRIME:division -outDir test -log log.out -v

runGFA.py
COmplete general feature analysis implimenting all-by-all using
pariwise.py and randforest.py. Usage:
$ python2.7 runGFA.py testData.afm -outDir test -log log.out -v

Formats 
------------------
We adopt the same input format as in Cloud Forest.
An annotated feature matrix (.afm) file is a tab 
delineated file with column and row headers. 
By default columns represent cases and rows 
represent features.

A row header / feature id includes a prefix to specify the data type.
We except N:,C:,B: prefixes to indicate numerical (ordered continuous or
discrete), categorical and binary. It also includes a term to indicate
the feature type. That is features can have groupings, which requires
some prior knowledge. Here we focus on PRIME (of primary scientific interest) 
and BATCH (measures of confounding factors like experimental batch).
This is used mostly to tease out batch effect from interesting measurements.

We assum 'NaN', 'NAN' and 'nan' to stand for missing data.
 


Work-Flows
------------------
We have work-flows for individual and combined analysis:

###1) All by all pairwise analysis (pairwise.py): 
Here we attempt to identify strong univariate relationships.
The required input is data in AFM format.
Tests are determined based on the features being compared:
N-N = Spearman rank, B-B = Fisher Exact, C-C = ChiSq,
N-B = Ranks Sum, N-C = Kruskal Wallis.
The test also determines what is reported in r (correlation, 
effect size, separability), which, like p, is symmetric.
N-N = spearman rho correlation coefficient (bounded by -1,1),
B-B = phi coefficient which is the person coefficient analogue 
C-C = cramer's V (generalization of phi), which is related to 
correlation for the chi sq test. N-C = as Kruskal Wallis has no simple 
single metric for effect size,we will use a measure of separability 
similar to multiclass LDA sqrt(variance of class means / variance of samples) 
(note this is non-robust) which ranges from zero with no separability to 1
N-B = effect size determined by z/sqrt(N), which is related to 
t-test correlation coeff

Output includes P-values, false discover rate corrected 
q-values and correlation coefficients that reported in 3 tsv matrices.
A summary output is all made identifying the most significant 
assosiations by q-value.

###2) Single target Random Forest (RF) analysis (randforest.py):
This workflow is organized in python, but calls out
to the shell to launch Cloud Forest, see above for 
repository info. The workflow is designed to do a set 
of random forest runs to assess the predictive value of 
all features on a particular target (chosen out of 
the complete feature matrix by use of the header id).
There are three main parts. 

One main objective is determining relationships 
by considering predictive power of features as 
measured by errors. Internal out of bag error 
can be reported and used as a very rough estimate 
of predictive power. However this will be biased low 
due to over training as the oob samples are 
not used in a particular tree, but will be seen 
in other trees of the forest. Cross-validation 
typically has much less bias (although much higher
variability) and is used here as prediction error.
A very rough estimate of variability is given by 
the std dev over the folds.  In both cases the error
metric depends on the data type, for N we use mean
squared error and for C and B we use balanced 
classification error rate.


(A) Cross-Validation folds are created via stratified,
balanced, sampling.   
Alternatively, you can specify preexisting folds using
-preFoldsDir.  The folds are used to run cross-validation.
Both the final cross validation error and the training 
out of bag (oob) error as a function of forest size are recorded.
The oob gives some indication on convergence.

The output is the results from each CV fold,
a vector holding the oob error (mean and var) a
vector holding the cv error (mean and var) and 
a plot of oob error vs forest size.
The folds are also saved for later use.

(B) Permutation, or shuffling, analysis is used 
to estimate a comparable baseline prediction error.
Shuffling breaks the joint distribution between 
features and the target while maintaining the 
individual distribution and properties such as 
number of features ect. The cv is calculated over
multiple permutations and a mean and var is reported.
Here we only shuffle features indicated as ':PRIME:'
which allows relationships between the target and other 
features, say ':BATCH:' to persist. This provides
a baseline relative to batch (confounding) effects. If
cv error < perm error it indicates that there is predictive
power in your features beyond dependencies produced by
confounding factors, which indicates a non trivial 
association.

The output is the perm error (mean and var).


(C) Feature selection can be valuable if you have 
predictive power indicated in A and B.
Feature selection is done on the full feature matrix
(not the CV folds). Here the decrease 
in impurity (error) by splitting on a feature is calculated
per tree and averaged over the forest. This 
metric provides an indication of 'importance'.
A p-value is estimated by comparing to shuffled 
versions of the feature. See: Tuv's "Feature Selection with 
Ensembles, Artificial Variables, and Redundancy Elimination."

The output is the run and importance results (importance scores 
and p-values).

A summary file is also created to record mean cv error +- std dev,
mean perm error +- std dev, and a list of significant features.
 
###3) Full General Feature Analysis (runGFA.py):
This workflow first runs (1) then it runs (2) over each 
feature.

More functionality is coming to better integrate the two 
workflows and to produce summary results.

ToDos
----------------
By priority:
1. Summary results for random forest all by all in runGFA including 
  1. using imp scores to estimate a metric that 
is comparable over different targets/forests,
comparable concept to effect size.  
It will probably be a fraction of the COD.
  2. Gathering the above, the p-vaule and the 
q-value into matrices comparable to pairwise.
  3. Creating a single barplot showing the 
cv errors vs perm errors for each target feature
  4. A combined summary.
2. Creating heatmaps/p-colors for each save matrix.
3. Add blacklisting
4. Adding option to remove strong univariate (pairwise)
features from random forest. 


License
-------------------

Copyright (C) 2003-2014 Institute for Systems Biology
		     Seattle, Washington, USA.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
