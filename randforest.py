#!/opt/local/bin/python2.7
"""
python2.7 script/module to run work-flows using the command line
random forest utility, Cloud Forest.  The goal is to use metrics 
of random forest prediction to estimate information content 
of features regarding a particular target. In particular 
we are interested in an all by all comparison of moderate sized
feature matrices to identify multivariate non-linear relationships 
between all features.

NOTE: This mostly involves calls out to the command-line,
but rather than shell scripting I have chose to use python 
as I am familiar with all the options and functionalities.

Input:
Annotated feature matrix file (see package readme for info)
Currently, we assume all data can be loaded into memory.

Dependencies:
Cloud Forest (https://github.com/ryanbressler/CloudForest) 

created on 20140716 by rtasseff@systemsbiology.org
"""


import sys
import numpy as np
import argparse
import logging
import os
import subprocess
import matplotlib.pyplot as plt

disc="Random (Cloud) Forest implementation script"
version=1.0



def run_growforest(fmPath,targetID,testFMPath="",outInfoPath=os.devnull,outForestPath="",progress=False,
	mTry=0,nTrees=100,nCores=1,blacklistPath="",permRE="",ace=10,impPath=""):

	call = "growforest -train "+fmPath+" -target "+targetID
	if testFMPath !="":call=call+" -test "+testFMPath
	if outForestPath !="":call=call+" -rfpred "+outForestPath

	if progress:call = call + " -progress"
	if mTry!=0: call = call + " -mTry "+str(mTry)
	if nTrees!=100: call = call + " -nTrees "+str(nTrees)
	if nCores!=1: call = call + " -nCores "+str(nCores)
	if permRE!="":call = call + " -shuffleRE "+permRE
	if impPath!="": call = call + " -ace "+str(ace)+" -importance "+impPath

	with open(outInfoPath,'w') as stdout:
		subprocess.check_call(call,shell=True,stdout=stdout)


def make_CVFolds(fmPath,targetID,folds=10,outDir='.',trainName='train',testName='test'):
	"""Create the CV folds used for this feature,
	these folds will not be removed, and can therefore 
	be used again in the future.
	Assumes outDir exists.
	names do not include directory prefix nor the ext suffix.
	Currently we assume that folds are stored in 
	<outDir>/folds/..., which may be created here.
	"""
	foldsDir = outDir+'/folds'

	# make the folder
	make_outDir(foldsDir)

	call = "nfold -fm "+fmPath+" -target "+targetID+" -folds "+str(folds)+" -test "+foldsDir+"/"+testName+"_%v.fm -train "+foldsDir+"/"+trainName+"_%v.fm"

	# redirecting output to an info file
	with open(foldsDir+"/foldInfo.dat",'w') as stdout:
		subprocess.check_call(call,shell=True,stdout=stdout)


	
def run_CV(targetID,folds=10,outDir='.',trainName='train',testName='test',outName='cvResults',
	progress=False,mTry=0,nTrees=100,nCores=1,blacklistPath="",altDir="",v=False,permRE=""):
	"""Runs a cross validation loop assuming 
	that the training and testing matrices exist.
	outName will be appended so that the output path
	for data will be:
	<outDir>/<outName>_fold_<fold index>.dat

	names was added if later version want to make changes;
	however, for now we assume they remain defult
	
	if altDir specified then it will be assumed that
	the cv fold FMs are in that directory, if not
	we assumes they are in <outDir>/folds/...
	In both cases the output results still go to outDir.
	a bit ad hoc because use of existing folds from 
	a different output directory was added after 
	the original work-flow was done.
	"""
	if altDir!="":foldsDir=altDir
	else:foldsDir=outDir+'/folds'
	

	for i in range(folds):
		if v: print "doing fold "+str(i)
		trainPath = foldsDir+'/'+trainName+'_'+str(i)+'.fm'
		testPath = foldsDir+'/'+testName+'_'+str(i)+'.fm'
		outInfoPath = outDir+'/'+outName+'_fold_'+str(i)+'.dat'

		run_growforest(trainPath,targetID,testFMPath=testPath,outInfoPath=outInfoPath,
			outForestPath="",progress=progress,mTry=mTry,nTrees=nTrees,nCores=nCores,
			blacklistPath=blacklistPath,permRE=permRE)

def collect_CVProgress(folds=10,outDir='.',outName='cvResults',nTrees=100):
	"""Collect the oob progress results for each 
	cv fold and return the average (row 0) and 
	variance row (1), assumes results are in:
	<outDir>/<outName>_fold_<fold index>.dat.
	Since its avalible, we also capture
	the mean (elem 0) and var (elem 1) of CV error 
	and return it in an array.
	"""
	progErrSum = np.zeros((2,nTrees))
	cvErrSum = np.zeros(2)
	for i in range(folds):
		resultsPath = outDir+'/'+outName+'_fold_'+str(i)+'.dat'
		progErr,cvErr = parse_progressResults(resultsPath,nTrees=nTrees)
		progErrSum[0] += progErr
		progErrSum[1] += progErr**2
		cvErrSum[0] += cvErr
		cvErrSum[1] += cvErr**2



	progErrSum[0] = progErrSum[0]/folds
	progErrSum[1] = progErrSum[1]/folds - progErrSum[0]**2

	cvErrSum[0] = cvErrSum[0]/folds
	cvErrSum[1] = cvErrSum[1]/folds - cvErrSum[0]**2

	return(progErrSum,cvErrSum)

def plot_CVProgress(progErr,outDir='.'):
	"""Plot and save a png image
	of the progress of the forest 
	averaged over cv runs with st. dev.
	used as error bars.
	"""
	n,m = progErr.shape
	plt.errorbar(range(m),progErr[0],yerr=np.sqrt(progErr[1]))
	plt.xlabel('number of trees')
	plt.ylabel('out of bag error')
	plt.savefig(outDir+'/errorProgress.pdf',format='pdf')
	np.savetxt(outDir+'/cvOOBProgErr.dat',progErr,delimiter='\t',fmt='%5.4E')
	plt.clf()
	plt.close()


def run_perms_getErr(targetID,folds=10,perms=10,permRE='.*',outDir='.',outName='tmpPermResults',mTry=0,
	nTrees=100,nCores=1,blacklistPath="",altDir="",v=False):
	# all we really want here is the permutation error for later comparision
	permErrSum = np.zeros(2) 
	# loop through permutations
	for i in range(perms):
		if v: print "doing perm "+str(i)
		# rerun the CV analysis 		
		run_CV(targetID,folds=folds,outDir=outDir,outName=outName,mTry=mTry,nTrees=nTrees,
			nCores=nCores,blacklistPath=blacklistPath,altDir=altDir,v=v,permRE=permRE)
		# get all of these results
		progErr,cvErr=collect_CVProgress(folds=folds,outDir=outDir,outName=outName,nTrees=nTrees)
		permErrSum[0] += cvErr[0]
		permErrSum[1] += cvErr[0]**2
		# get rid of those tmp files
		call = 'rm '+outDir+'/'+outName+'*'
		subprocess.check_call(call,shell=True)

	permErrSum[0] = permErrSum[0]/perms
	permErrSum[1] = permErrSum[1]/perms - permErrSum[0]**2

	return (permErrSum)




	
def parse_progressResults(resultsPath,nTrees=100):
	"""Parse the results of a cloud forest run 
	that output progress on oob error as a function 
	of forest size.
	Return the vector of results
	Assumes all oob errors in file  
	are ordered by forest size.
	To avoid resizing the array many times,
	we will assume that the exact number in the file 
	is equivalent to nTrees (should be).
	Since we can easily get the final cv error 
	we will look for and return that too.
	"""
	oob = np.zeros(nTrees)+np.nan
	err = np.nan
	count = 0
	with open(resultsPath) as fin:
		for line in fin:
			# look for indication of proper line
			if line.find("Model oob error after tree") >= 0:
				oob[count] = float(line.strip().split(':')[1])
				count+=1
			elif line.find("Error") >= 0:
				err = float(line.strip().split(':')[1])
	

	return(oob,err)

		

def write_summary(cvErr,permErr,permErrBatch,permRE,impPath,target,pMax,fout):
	fout.write('Summary of results for Random (Cloud) Forest analysis of '+target+'.\n')
	fout.write("We found a cv error of {} +/- {},\ncompared to a full randomized baseline of {} +/- {},\nand a semi randomized baseline of {} +/- {} (shuffling only {} features to identify batch effects).\n".format(cvErr[0],np.sqrt(cvErr[1]),permErr[0],np.sqrt(permErr[1]),permErrBatch[0],np.sqrt(permErrBatch[1]),permRE))
	fout.write("Features with significant importance scores:\n")
	fout.write("Feature ID\tImportance\tp-value\n")
	# lets get the importance scores
	data = np.loadtxt(impPath,dtype=str,delimiter='\t')
	p = np.array(data[:,2],dtype=float)
	ind = np.argsort(p)
	p = p[ind]
	data = data[ind]
	imp = np.array(data[:,3],dtype=float)
	for i in range(len(p)):
		if p[i]<pMax:
			fout.write("{}\t{}\t{}\n".format(data[i,1],imp[i],p[i]))

	del data
	del imp
	del p
	

def run_mainWorkFlow(args,outDir,target):
		
	# -- (1) create cross validation folds, if needed
	if args.preFoldsDir=="":
		if args.verbose:
			print "Creating feature matrices for cross validation" 
		logging.info("Creating feature matrices for {} fold cross validation. \n\tUsing feature matrix: {}.\n\tStoring results in {}/folds.".format(args.nFolds,args.fm,outDir))
		make_CVFolds(args.fm,target,folds=args.nFolds,outDir=outDir)
	else:
		logging.info("Here we will assume that feature matrices for {} fold cross validation derived from {} already exist at {}".format(args.nFolds,args.fm,outDir))


	# -- (2-A) run random forest to obtain CV error and convergence results 
	if args.verbose:
		print "Running initial cross validation {} trees for each".format(args.nTrees)	
	run_CV(target,folds=args.nFolds,outDir=outDir,progress=True,mTry=args.mTry,
		nTrees=args.nTrees,nCores=args.nCores,altDir=args.preFoldsDir,v=args.verbose)
	logging.info("Finished initial cross validation, results in {}/cvResults_fold_i.dat.".format(outDir))

	# -- (2-B) collect output, with focus on progress
	if args.verbose:
		print "Collecting progress data from CV runs"
	progErr,cvErr = collect_CVProgress(folds=args.nFolds,outDir=outDir,nTrees=args.nTrees)
	plot_CVProgress(progErr,outDir=outDir)

	# gonna save the other stuff
	np.savetxt(outDir+'/cvErr.dat',cvErr)
	if args.verbose:
		print "Saved oob progress figure."
		print "CV error at "+str(cvErr[0])

	logging.info("A figure on oob error progress and convergence was saved to {}/errorProgress.pdf.".format(outDir))



	# -- (3 A) run permutation analysis for comparison
	if args.verbose:
		print "Running permutation analysis by shuffeling all features"
	permErr = run_perms_getErr(target,folds=args.nFolds,perms=args.nPerms,permRE='.*',
		outDir=outDir,mTry=args.mTry,nTrees=args.nTrees,nCores=args.nCores,
		altDir=args.preFoldsDir,v=args.verbose)
	np.savetxt(outDir+'/permErr.dat',permErr)

	if args.verbose:
		print "permutation random comparison error at "+str(permErr[0])
	logging.info("Finished finished {} rounds of permutation analysis on all variables.".format(args.nPerms))


	# -- (3B) run permutation analysis for comparison
	if args.verbose:
		print "Running permutation analysis only on "+args.permRE+" features."
	permErrBatch = run_perms_getErr(target,folds=args.nFolds,perms=args.nPerms,permRE=args.permRE,
		outDir=outDir,mTry=args.mTry,nTrees=args.nTrees,nCores=args.nCores,
		altDir=args.preFoldsDir,v=args.verbose)
	np.savetxt(outDir+'/permErrBatch.dat',permErrBatch)

	if args.verbose:
		print "permutation of "+args.permRE+" random comparison error at "+str(permErrBatch[0])
	logging.info("Finished finished {} rounds of permutation analysis by shuffeling {} features".format(args.nPerms,args.permRE))

	# -- (4) feature selection 
	if args.verbose:
		print "Running Feature Selection."
	
	run_growforest(args.fm,target,outInfoPath=outDir+'/fullACERunResults.dat',mTry=args.mTry,
		nTrees=args.nTrees,nCores=args.nCores,ace=args.nAce,impPath=outDir+'/fullACEImpResults.dat')

	# -- Record summary
	fout = open(outDir+'/summary.txt','w')
	write_summary(cvErr,permErr,permErrBatch,args.permRE,outDir+'/fullACEImpResults.dat',target,args.pMax,fout)
	fout.close()


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
	parser.add_argument("-v","--verbose", help="increase output verbosity",
		action="store_true")
	parser.add_argument("fm",help="input feature matrix path")
	parser.add_argument("target",help="feature ID of target")
	parser.add_argument("-nFolds",help="the number of cross validation folds to be used",type=int,default="10")
	parser.add_argument("-nTrees",help="the number of trees in each forest",type=int,default="1000")
	parser.add_argument("-nCores",help="the number of cores used in growing trees",type=int,default="1")
	parser.add_argument("-mTry",help="the number of features to try when splitting",type=int,default="0")
	parser.add_argument("-nPerms",help="the number of permutations to do for random control",type=int,default="10")
	parser.add_argument("-nAce",help="the number of artificial contrast permutations for importance est",
		type=int,default="100")
	parser.add_argument("-pMax",help="maximum p-value cutoff to include in summary output",
		type=float,default="0.001")
	parser.add_argument("-outDir",help="directory used for temporary files and final output",default=".")
	parser.add_argument("-preFoldsDir",help="if specified, assumes CV fold fms are here",default="")
	parser.add_argument("-permRE",
		help="regular expression to indicate  primary variables (will be shuffled in permutation)",
		default=".:PRIME:.*")
	parser.add_argument("-log",help="print info to specified log file",default="")


	return(parser.parse_args())



def main():
	# --get the input arguments
	parser = argparse.ArgumentParser(description="run random forest analysis of all features vs specified target")
	args = parse_CmdArgs(parser)

	# --set up working dir:
	make_outDir(args.outDir)

	# --setup logger 
	# if file for log specified, set and decrease level
	if args.log!="":
		logging.getLogger('').handlers = []
		logging.basicConfig(filename=args.outDir+'/'+args.log, level=logging.INFO, format='%(asctime)s %(message)s')

	# --record some basic information
	logging.info("Running {}, {}, version={} on target {}.".format(sys.argv[0],disc,version,args.target))


	#-- run the main workflow	
	run_mainWorkFlow(args,args.outDir,args.target)

	if args.verbose:
		print "done!"
	logging.info("run complete")
		


	#-- dump all options to file
	logging.info("Reporting all options for completed run:\n"+str(vars(args)))


if __name__ == '__main__':
	main()
