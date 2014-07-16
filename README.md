genFeatureAnalysis
====================

Pipeline code and scripts to run a general work flow for the 
analysis of data features with a focus on pairwise 
univariate statistics and random forest prediction.

Dependencies and Programs
------------------
Cloud Forest (https://github.com/ryanbressler/CloudForest)
will be used to implement the random forest.

We will be using python2.7 + numpy, scipy
to most additional tasks, including pre and post processing
and implementation of the pairwise analysis.


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


Formats 
------------------
We adopt the same input format as in Cloud Forest.
An annotated feature matrix (.afm) file is a tab 
delineated file with column and row headers. 
By default columns represent cases and rows 
represent features.

A row header / feature id includes a prefix to specify the feature type.
We except N:,C:,B: prefixes to indicate numerical (ordered continuous or
discrete), categorical and binary. 



Work-Flows
------------------
We have work-flows for individual and combined analysis:

1) Pair-wise analysis (pairwise.py): 
Here we attempt to identify strong univariate relationships.
The required input is data in AFM format.
Tests are determined based on the features being compared:
N-N = Spearman rank, B-B = Fisher Exact, C-C = ChiSq,
N-B = Ranks Sum, N-C = Kruskal Wallis.
The test also determines what is reported in r (correlation, effect size, separability), 
which, like p, is symmetric.
N-N = spearman rho correlation coefficient (bounded by -1,1),
B-B = phi coefficient which is the person coefficient analogue 
C-C = cramer's V (generalization of phi), which is related to correlation for the chi sq test.
N-C = as Kruskal Wallis has no simple single metric for effect size,
we will use a measure of separability similar to multiclass LDA
sqrt(variance of class means / variance of samples) (note this is non-robust)
which ranges from zero with no separability to 1
N-B = effect size determined by z/sqrt(N), which is related to t-test correlation coeff
P-values, false discover rate corrected q-values and correlation coefficients are 
reported in 3 text matrices.



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
