"""
Python2.7 script/module to run workflows for 
general feature analysis.

Part of the genFeatureAnalysis package, and
impliments sub workflows from pairwise.py
and randforest.py.

Input:
Annotated feature matrix file (see package readme for info)
Currently, we assume all data can be loaded into memory.

Output:
all by all pairwise outputs
individual rf outputs
all by all summary of rf

Creates a dir structure that is used to organize results
and temporary files.

Dependencies:
statUtils from genTools (https://github.com/Rtasseff/genTools) 
Cloud Forest (https://github.com/ryanbressler/CloudForest) 

created on 20140717 by rtasseff@systemsbiology.org
"""


import sys
import numpy as np
import argparse
import os
import subprocess
import matplotlib.pyplot as plt
import pairwise
import randforest
import logging



disc="General Feature Analysis workflow"
version=1.0










def make_outDir(outDir):
	if outDir==".":
		logging.warning("Using current directory for output, previous files may be overwritten.")
	elif os.path.exists(outDir):
		logging.warning("Using an existing directory for output, previous files may be overwritten: {}".format(outDir))
	else:
		os.makedirs(outDir)
		logging.info("Making new directory for output: {}".format(outDir))






def parse_CmdArgs(parser):
	"""Get the command line parameters to be used."""
	parser.add_argument("fm",help="input feature matrix path")
	parser.add_argument("-v","--verbose", help="increase output verbosity",
		action="store_true")
	parser.add_argument("-outDir",help="specify new output directory",default=".")
	parser.add_argument("-pOutFile",help="specify alternate file name for output p matrix",
		default="p.dat")
	parser.add_argument("-rOutFile",help="specify alternate file name for output r matrix",
		default="r.dat")	
	parser.add_argument("-qOutFile",help="specify alternate file name for output q matrix",
		default="q.dat")	
	parser.add_argument("-obsMinWarn",
		help="minimum number of feature observations before warning is issued",
		type=int,default=5)
	parser.add_argument("-obsMinError",
		help="minimum number of feature observations before error assumed and nan is issued",
		type=int,default=1)
	parser.add_argument("-qMax",
		help="maximum q-value to be included in results summary.",
		type=float,default=0.001)

	parser.add_argument("-nFolds",help="the number of cross validation folds to be used",type=int,default="10")
	parser.add_argument("-nTrees",help="the number of trees in each forest",type=int,default="1000")
	parser.add_argument("-nCores",help="the number of cores used in growing trees",type=int,default="1")
	parser.add_argument("-mTry",help="the number of features to try when splitting",type=int,default="0")
	parser.add_argument("-nPerms",help="the number of permutations to do for random control",type=int,default="10")
	parser.add_argument("-nAce",help="the number of artificial contrast permutations for importance est",
		type=int,default="100")
	parser.add_argument("-pMax",help="maximum p-value cutoff to include in summary output",
		type=float,default="0.001")
	parser.add_argument("-preFoldsDir",help="if specified, assumes CV fold fms are here",default="")
	parser.add_argument("-permRE",
		help="regular expression to indicate  primary variables (will be shuffled in permutation)",
		default=".:PRIME:.*")
	parser.add_argument("-log",help="print info to specified log file",default="")



	return(parser.parse_args())




def run_fullWorkFlow(args):
	"""Run full pairwise analysis + run
	rf all by all and summarise.
	"""
	
	# --Do all by all pairwise:
	if args.verbose:
		print "Doing all by all pairwise analysis..."
	
	pairwiseOutDir = args.outDir+'/pairwiseResults'
	make_outDir(pairwiseOutDir)

	logging.info("Running {}, version {}, in {}.".format(pairwise.disc,pairwise.version,pairwiseOutDir))
	pairwise.run_mainWorkFlow(args,pairwiseOutDir)
	logging.info("Completed the all by all pairwise analysis")

	# --Do all by all rf
	data,header,dataType = pairwise.load_parse_afm(args.fm)
	rfOutDir = args.outDir+'/randforestResults'
	make_outDir(rfOutDir)

	if args.verbose:
		print "Doing all by all random forest analysis..."

	logging.info("Running {}, version {}, in {}.".format(randforest.disc,randforest.version,rfOutDir))
	for label in header:
		if args.verbose:
			print "Starting RF on "+label

		tmpLabel = label.replace(':','-')
		tmpOutDir = rfOutDir+'/'+tmpLabel
		make_outDir(tmpOutDir)
		logging.info("Starting feature "+label+" in "+tmpOutDir+".")
		try:
			randforest.run_mainWorkFlow(args,tmpOutDir,label)
		except:
			# need to fix this up a bit!!
			logging.warning("Error while running RF on {}, results my be incomplete.".format(label))
			


		

	

def main():
	# --get the input arguments
	parser = argparse.ArgumentParser(description="run work-flows for General Feature Analysis")
	args = parse_CmdArgs(parser)

	# --set up working dir:
	make_outDir(args.outDir)

	# --setup logger 
	# if file for log specified, set and decrease level
	if args.log!="":
		logging.getLogger('').handlers = []
		logging.basicConfig(filename=args.outDir+'/'+args.log, level=logging.INFO, format='%(asctime)s %(message)s')

	# --record some basic information
	logging.info("Running {}, {}, version={}.".format(sys.argv[0],disc,version))
	
	run_fullWorkFlow(args)
	
	if args.verbose:
		print "done!"
	logging.info("run complete")
		


	#-- dump all options to file
	logging.info("Reporting all options for completed run:\n"+str(vars(args)))



if __name__ == '__main__':
	main()
