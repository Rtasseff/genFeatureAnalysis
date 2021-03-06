#!/opt/local/bin/python2.7
"""
python2.7 script/module to run a standard pairwise analysis on 
input data of potentially different types. Robust or nonparametric 
statistical measures/tests are employed when easy-to-implement 
standards are available (known to the author). 

Tests are determined based on the features being compared:
N-N = Spearman rank, B-B = Fisher Exact, C-C = ChiSq,
N-B = Ranks Sum, N-C = Kruskal Wallis.

Part of the genFeatureAnalysis package

Input:
Annotated feature matrix file (see package readme for info)
Currently, we assume all data can be loaded into memory.

Output:
text files for the p-value matrix, the FDR corrected q-value matrix,
the correlation coefficient matrix and a log of the run details.
Note all outputs are symmetric except for the q-values,
here the rows are the target variables and the cols
show the relation to other variables.

Dependencies:
statUtils from genTools (https://github.com/Rtasseff/genTools) 

created on 20140715 by rtasseff@systemsbiology.org

"""

import sys
import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt

from genTools import statsUtil

disc="Pairwise analysis script"
version=1.0

def load_parse_afm(finPath):
	"""Load data from file and, assuming afm format,
	parse the data.
	returns a data matrix of values (np str matrix) and 
	an array of header entries and a list of parsed data types.
	"""
	# load data
	fin = np.loadtxt(finPath,dtype=str,delimiter='\t')
	# get header
	header = fin[:,0]
	data = fin[:,1:]
	# parse header (also ensure proper format)
	n = len(header)
	dataType=[]
	for i in range(n):
		dataType.append(header[i].split(':')[0])

	return(data,header,np.array(dataType,dtype=str))

	
def run_all_pariwise(data,dataType,labels,obsMinWarn=5,obsMinError=1,v=False):
	"""Runs the all by all pairwise analysis 
	returning several matrix for different 
	measures/metrics/indicators: pM (p-values),
	rM (r-values).
	"""
	n,m = data.shape
	pM = np.zeros((n,n))
	rM = np.zeros((n,n))
	
	for i in range(n):
		if v: print "Comparing all to feature "+str(i)+", "+labels[i]
		for j in range(i+1,n):
			rM[i,j],pM[i,j],warn = statsUtil.runPairwise(data[i],data[j],dataType[i], dataType[j], 
				obsMinWarn=obsMinWarn, obsMinError=obsMinError)
			if warn:
				logging.warning("Potentially limited observations in run_all_pariwise comparing feature {} and {}".format(labels[i],labels[j]))
			
	# all symmetric matrices
	pM += pM.T
	rM += rM.T 
	rM += np.diag(np.ones(n))

	return(rM,pM)

def get_qFDR(p):
	"""Calculate the q-values using FDR correction"""
	n = len(p) 
	q = np.zeros((n,n))
	for i in range(n):
		tmp = p[i]
		# hide the element refereing to i,i comparision
		tmp[i]=np.nan
		_,qTmp,_ = statsUtil.fdr_bh(tmp)
		q[i] = qTmp
	
	return(q)


def run_filter(data,label):
	"""do some data filtering.
	Remove features with no variation.
	"""
	n = len(data)
	keep = np.array(np.ones(n,dtype=int),dtype=bool)
	for i in range(n):
		tmp = data[i]
		if type(tmp[0])==np.string_:isnan=tmp=='nan'
		else:isnan=np.isnan(tmp)
		if len(set(tmp[~isnan])) < 2:
			keep[i]=False
			logging.warning("No varriaiton found in {}, excluding from analysis.".format(label[i]))
	return(keep)


def save_outputMats(mats,labels,keep=[],outDir='.',names=['r.dat','p.dat','q.dat']):
	"""save data to text file and create/save pcolor plot of data"""

	m = len(mats)

	nKeep = len(keep)

	for i in range(m):
		# if keep specified we may have to fill in blanks with nan
		# note that if features were filtered keep, based on original size, should be bigger 
		if nKeep>len(mats[i]):
			tmp = np.ones((len(keep),len(mats[i])))+np.nan
			tmp[keep]=mats[i]
			tmp2 = np.ones((len(keep),len(keep)))+np.nan
			tmp2[:,keep]=tmp
		else: tmp2 = mats[i]
		np.savetxt(outDir+'/'+names[i],tmp2,delimiter='\t',fmt='%5.4E')
		# create p-color
		plt.imshow(tmp2,interpolation='none')
		#print len(tmp2)	
		#print len(labels)
		plt.yticks(np.arange(len(labels)),labels)
		#plt.xticks(np.arange(len(labels))+.5,labels)
		plt.colorbar()
		name = names[i].split('.')[0]
		plt.savefig(outDir+'/pcolor_'+name+'.png',format='png')
		plt.clf()
		plt.close()
		

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
	parser.add_argument("-log",help="print info to specified log file",default="")

	return(parser.parse_args())


def write_summary(r,q,qMax,labels,keep,fout):
	fout.write("Summary of results for all by all pairwise analysis.\n")
	
	if len(labels) > len(r):
		fout.write("Some variables were ignored due to no variation:\n")
		tmp = labels[~keep]
		for label in tmp:
			fout.write(label+'\n')
	names = labels[keep]
	fout.write('\n')
	fout.write("Feature by feature list of relationships:\n")
	for i in range(len(names)):
		# probably should report what is nan somewhere
		# would help with error checking,
		# need to ignore i,i
		qTmp = q[i].copy()
		qTmp[np.isnan(qTmp)]=np.inf 
		ind = qTmp<qMax
		if np.sum(ind)>0:
			qTmp = qTmp[ind]
			rTmp = r[i,ind]
			namesTmp = names[ind]
			ind = np.argsort(qTmp)
			qTmp = qTmp[ind]
			rTmp = rTmp[ind]
			namesTmp = namesTmp[ind]
			fout.write('----'+names[i]+'----\n')
			fout.write('feature ID\tcorr\tq-value\n')
			for j in range(len(qTmp)):
				fout.write("{}\t{}\t{}\n".format(namesTmp[j],rTmp[j],qTmp[j]))

def run_mainWorkFlow(args,outDir):
	
	# --get data 
	# report
	if args.verbose:
		print "Loading input data..."
	logging.info("Using the feature matrix at {}.".format(args.fm))
	
	data,header,dataType = load_parse_afm(args.fm)

	# --pre-process data
	# currently we are only looking for nonchanging features
	keep = run_filter(data,header)


	# --run all by all pairwise
	# report
	if args.verbose:
		print "Running all by all pairwise analysis..."
	logging.info("Running all-by-all pairwise analysis.")

	r,p = run_all_pariwise(data[keep],dataType[keep],header[keep],obsMinWarn=args.obsMinWarn,
		obsMinError=args.obsMinError,v=args.verbose)
	q = get_qFDR(p)

	# --save output

	if args.verbose:
		print "saving output..."
	save_outputMats([r,p,q],header,keep=keep,outDir=outDir,
		names=[args.rOutFile,args.pOutFile,args.qOutFile])
	logging.info("Output files saved to dir at {}\n\tp-values saved as {}\n\tr-values saved as {}\n\tq-values saved as {}".format(args.outDir,args.pOutFile,args.rOutFile,args.qOutFile))


	fout = open(outDir+'/summary.txt','w')
	write_summary(r,q,args.qMax,header,keep,fout)
	fout.close()

	

def main():
	# --get the input arguments
	parser = argparse.ArgumentParser(description="run all by all pairwise analysis on feature matrix")
	args = parse_CmdArgs(parser)

	make_outDir(args.outDir)

	# --setup logger 
	# if file for log specified, set and decrease level
	if args.log!="":
		logging.getLogger('').handlers = []
		logging.basicConfig(filename=args.outDir+'/'+args.log, level=logging.INFO, format='%(asctime)s %(message)s')
	
	# --record some basic information
	logging.info("Running {}, {}, version={}...".format(sys.argv[0],disc,version))

	run_mainWorkFlow(args,args.outDir)

	if args.verbose:
		print "done!"
	logging.info("run complete")
		
	# dump all options to file
	logging.info("Reporting all options for completed run:\n"+str(vars(args)))
	


if __name__ == '__main__':
	main()
