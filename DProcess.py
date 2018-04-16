import string

import re
import pandas as pd
import numpy as np

import keras.utils.np_utils as kutils

def convertSampleToProbMatr(sampleSeq3DArr): #changed add one column for '1'
    """
    Convertd the raw data to probability matrix
    
    PARAMETER
    ---------
    sampleSeq3DArr: 3D numpy array
       X denoted the unknow amino acid.
    
    
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """
    
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["-"] =20 ##add -
    AACategoryLen = 21 ##add -
    
    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
    
    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
    
        AANo	 = 0
        for AA in sequence:
            
            if not AA in letterDict:
                probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 1.0/AACategoryLen)
            
            else:
                index = letterDict[AA]
                probMatr[sampleNo][0][AANo][index] = 1
                
            AANo += 1
        sampleNo += 1
    
    return probMatr
    


def convertSampleToIndex(sampleSeq3DArr):
	"""
	Convertd the raw data to probability matrix
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {}
	letterDict["A"] = 1
	letterDict["C"] = 2
	letterDict["D"] = 3
	letterDict["E"] = 4
	letterDict["F"] = 5
	letterDict["G"] = 6
	letterDict["H"] = 7
	letterDict["I"] = 8
	letterDict["K"] = 9
	letterDict["L"] = 10
	letterDict["M"] = 11
	letterDict["N"] = 12
	letterDict["P"] = 13
	letterDict["Q"] = 14
	letterDict["R"] = 15
	letterDict["S"] = 16
	letterDict["T"] = 17
	letterDict["V"] = 18
	letterDict["W"] = 19
	letterDict["Y"] = 20
	letterDict["-"] = 21
	letterDict["X"] = 0
	probMatr = np.zeros((len(sampleSeq3DArr),len(sampleSeq3DArr[0])))
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
		AANo	 = 0
		for AA in sequence:
			probMatr[sampleNo][AANo]= letterDict[AA]
			AANo += 1
		sampleNo += 1
	
	return probMatr
	

	
def convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName):
	"""
	Convertd the raw data to probability matrix
	PARAMETER
	---------
	sampleSeq3DArr: 3D List
		List -  numpy matrix(3D)
	Sample List: List (nb_windows, nb_samples, SEQLen/nb_windows , 100)
	"""
	
	rawDataFrame = pd.read_table(refMatrFileName, sep='\t', header=None)
	
	raw_data_seq_index_df = pd.DataFrame({'seq' : rawDataFrame[0] , 'indexing':rawDataFrame.index})
	raw_data_seq_df_index_dict = raw_data_seq_index_df.set_index('seq')['indexing'].to_dict()

	
	nb_raw_data_frame_column = len(rawDataFrame.columns)
	
	nb_sample = sampleSeq3DArr.shape[0]
	len_seq = len(sampleSeq3DArr[1]) 
	re_statement =  ".{%d}" % (nb_windows)
	
	
	probMatr_list = []
	for tmp_idx in range(nb_windows):
		probMatr_list.append( np.zeros((nb_sample, int((len_seq - tmp_idx)/nb_windows) , 100)) )

	
	for sample_index, sample_sequence in enumerate(sampleSeq3DArr):
		
		if sample_index%10000 == 0:
			print( "%d / %d " % (sample_index, nb_sample))
		
		#start_time = time.time()
		seq = "".join(sample_sequence)
		
		for begin_idx in range(nb_windows):
			
			# Get sub-sequence
			sub_seq_list = re.findall(re_statement, seq[begin_idx:])
			
			sub_seq_indexing_list = []
			for sub_seq in sub_seq_list:
				if sub_seq in raw_data_seq_df_index_dict:
					sub_seq_indexing_list.append( raw_data_seq_df_index_dict[sub_seq] )
				else:
					sub_seq_indexing_list.append( raw_data_seq_df_index_dict['<unk>'] )

			matrix_arr = rawDataFrame.loc[ sub_seq_indexing_list ][ range(1, nb_raw_data_frame_column)].as_matrix()
			for idx_i in range(matrix_arr.shape[0]):
				for idx_j in range(matrix_arr.shape[1]):
					probMatr_list[begin_idx][sample_index][idx_i][idx_j] = matrix_arr[idx_i][idx_j]

		#print("2. --- %s seconds ---" % (time.time() - start_time))
		

	return probMatr_list

def convertSampleToPhysicsVector(sampleSeq3DArr): #from wulihuaxueshuxing.txt 
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} #hydrophobicty, hydrophilicity, side-chain mass, pK1 (alpha-COOH), pK2 (NH3), PI, Average volume of buried residue, Molecular weight, Side chain volume, Mean polarity
	letterDict["A"] = [0.62,	-0.5,	15,	2.35,	9.87,	6.11,	91.5,	89.09,	27.5,	-0.06]
	letterDict["C"] = [0.2900,	-1.0000,	47.0000,    1.7100,   10.7800,    5.0200,	117.7,	121.15,	44.6,	1.36]
	letterDict["D"] = [-0.9000,    3.0000,   59.0000,    1.8800,    9.6000,    2.9800,	124.5,	133.1,	40,	-0.8]
	letterDict["E"] = [-0.7400,    3.0000,   73.0000,    2.1900,    9.6700,    3.0800,	155.1,	147.13,	62,	-0.77]
	letterDict["F"] = [1.1900,   -2.5000,   91.0000,    2.5800,    9.2400,    5.9100,	203.4,	165.19,	115.5,	1.27]
	letterDict["G"] = [0.4800,         0,    1.0000,    2.3400,    9.6000,    6.0600,	66.4,	75.07,	0,	-0.41]
	letterDict["H"] = [-0.4000,   -0.5000,   82.0000,    1.7800,    8.9700,    7.6400,	167.3,	155.16,	79,	0.49]
	letterDict["I"] = [1.3800,   -1.8000,   57.0000,    2.3200,    9.7600,    6.0400,	168.8,	131.17,	93.5,	1.31]
	letterDict["K"] = [-1.5000,    3.0000,   73.0000,    2.2000,    8.9000,    9.4700,	171.3,	146.19,	100,	-1.18]
	letterDict["L"] = [1.0600,   -1.8000,   57.0000,    2.3600,    9.6000,    6.0400,	167.9,	131.17,	93.5,	1.21]
	letterDict["M"] = [0.6400,   -1.3000,   75.0000,    2.2800,    9.2100,    5.7400,	170.8,	149.21,	94.1,	1.27]
	letterDict["N"] = [-0.7800,    0.2000,   58.0000,    2.1800,    9.0900,   10.7600,	135.2,	132.12,	58.7,	-0.48]
	letterDict["P"] = [0.1200,         0,   42.0000,    1.9900,   10.6000,    6.3000,	129.3,	115.13,	41.9,	0]
	letterDict["Q"] = [-0.8500,    0.2000,   72.0000,    2.1700,    9.1300,    5.6500,	161.1,	146.15,	80.7,	-0.73]
	letterDict["R"] = [-2.5300,    3.0000,  101.0000,    2.1800,    9.0900,   10.7600,	202,	174.2,	105,	-0.84]
	letterDict["S"] = [-0.1800,    0.3000,   31.0000,    2.2100,    9.1500,    5.6800,	99.1,	105.09,	29.3,	-0.5]
	letterDict["T"] = [-0.0500,   -0.4000,   45.0000,    2.1500,    9.1200,    5.6000,	122.1,	119.12,	51.3,	-0.27]	
	letterDict["V"] = [1.0800,   -1.5000,   43.0000,    2.2900,    9.7400,    6.0200,	141.7,	117.15,	71.5,	1.09]
	letterDict["W"] = [0.8100,   -3.4000,  130.0000,    2.3800,    9.3900,    5.8800,	237.6,	204.24,	145.5,	0.88]
	letterDict["Y"] = [0.2600,   -2.3000,  107.0000,    2.2000,    9.1100,    5.6300,	203.6,	181.19,	117.3,	0.33]
	AACategoryLen = 10
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr

def convertSampleToPhysicsVector_3(sampleSeq3DArr):
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} 
	letterDict["A"] = [0.8056266,0.453125,0.1085271,0.73563218,0.51595745,0.40231362,0.1466121,0.1085391,0.1890034,0.4409449,0]
	letterDict["C"] = [0.7212276,0.375000,0.3565891,0.00000000,1.00000000,0.26221080,0.2996495,0.3567392,0.3065292,1.0000000,0]
	letterDict["D"] = [0.4168798,1.000000,0.4496124,0.19540230,0.37234043,0.00000000,0.3393692,0.4492529,0.2749141,0.1496063,0]
	letterDict["E"] = [0.4578005,1.000000,0.5581395,0.55172414,0.40957447,0.01285347,0.5181075,0.5578695,0.4261168,0.1614173,0]
	letterDict["F"] = [0.9514066,0.140625,0.6976744,1.00000000,0.18085106,0.37660668,0.8002336,0.6976852,0.7938144,0.9645669,0]
	letterDict["G"] = [0.7698210,0.531250,0.0000000,0.72413793,0.37234043,0.39588689,0.0000000,0.0000000,0.0000000,0.3031496,0]
	letterDict["H"] = [0.5447570,0.453125,0.6279070,0.08045977,0.03723404,0.59897172,0.5893692,0.6200356,0.5429553,0.6574803,0]
	letterDict["I"] = [1.0000000,0.250000,0.4341085,0.70114943,0.45744681,0.39331620,0.5981308,0.4343114,0.6426117,0.9803150,0]
	letterDict["K"] = [0.2634271,1.000000,0.5581395,0.56321839,0.00000000,0.83419023,0.6127336,0.5505922,0.6872852,0.0000000,0]
	letterDict["L"] = [0.9181586,0.250000,0.4341085,0.74712644,0.37234043,0.39331620,0.5928738,0.4343114,0.6426117,0.9409449,0]
	letterDict["M"] = [0.8107417,0.328125,0.5736434,0.65517241,0.16489362,0.35475578,0.6098131,0.5739723,0.6467354,0.9645669,0]
	letterDict["N"] = [0.4475703,0.562500,0.4418605,0.54022989,0.10106383,1.00000000,0.4018692,0.4416660,0.4034364,0.2755906,0]
	letterDict["P"] = [0.6777494,0.531250,0.3178295,0.32183908,0.90425532,0.42673522,0.3674065,0.3101339,0.2879725,0.4645669,0]
	letterDict["Q"] = [0.4296675,0.562500,0.5503876,0.52873563,0.12234043,0.34318766,0.5531542,0.5502826,0.5546392,0.1771654,0]
	letterDict["R"] = [0.0000000,1.000000,0.7751938,0.54022989,0.10106383,1.00000000,0.7920561,0.7674383,0.7216495,0.1338583,0]
	letterDict["S"] = [0.6010230,0.578125,0.2325581,0.57471264,0.13297872,0.34704370,0.1910047,0.2324069,0.2013746,0.2677165,0]
	letterDict["T"] = [0.6342711,0.468750,0.3410853,0.50574713,0.11702128,0.33676093,0.3253505,0.3410235,0.3525773,0.3582677,0]
	letterDict["V"] = [0.9232737,0.296875,0.3255814,0.66666667,0.44680851,0.39074550,0.4398364,0.3257722,0.4914089,0.8937008,0]
	letterDict["W"] = [0.8542199,0.000000,1.0000000,0.77011494,0.26063830,0.37275064,1.0000000,1.0000000,1.0000000,0.8110236,0]
	letterDict["Y"] = [0.7135550,0.171875,0.8217054,0.56321839,0.11170213,0.34061697,0.8014019,0.8215530,0.8061856,0.5944882,0]
	letterDict["X"] = [0,0,0,0,0,0,0,0,0,0,0]
	letterDict["-"] = [0,0,0,0,0,0,0,0,0,0,1]
	AACategoryLen = 11
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr

def convertSampleToAAindexforUbiVector(sampleSeq3DArr):
    letterDict = {} # Total 13*20
    letterDict["A"] = [0,6,154.33,0.892,27.8,0,-0.31,15,0.3,7.62,50.76,0.984,-1.895,0]
    letterDict["R"] = [10,10.76,341.01,0.901,94.7,1,1.3,	67,	-1.4,6.81,48.66,1.008,-1.475,0]
    letterDict["N"] = [1.3,5.41,207.9,0.93,60.1,0,0.49,	49,	-0.5,	6.17,	45.8,1.048,-1.56,0]
    letterDict["D"] = [1.9,2.77,194.91,0.932,60.6,0,0.58,	50,	-0.6,	6.18,	43.17,	1.068,	-1.518,0]
    letterDict["C"] = [0.17,5.05,219.79,0.925,15.5,0,-0.87,	5,	0.9,	10.93,	58.74,	0.906,	-2.035,0]
    letterDict["Q"] = [1.9,5.65,235.51,0.885,68.7,0,0.7,	56,	-0.7,	6.67,	46.09,	1.037,	-1.521,0]
    letterDict["E"] = [3,3.22,223.16,0.933,68.2,0,0.68,55,	-0.7,	6.38,	43.48,	1.094,	-1.535,0]
    letterDict["G"] = [0,5.97,127.9,0.923,24.5,0,-0.33,10,	0.3,	7.31,	50.27,	1.031,	-1.898,0]
    letterDict["H"] = [0.99,7.59,242.54,0.894,50.7,1,0.13,	34,	-0.1,	7.85,	49.33,	0.95,	-1.755,0]
    letterDict["I"] = [1.2,6.02,233.21,0.872,22.8,0,-0.66,	13,	0.7,	9.99,	57.3,	0.927,	-1.951,0]
    letterDict["L"] = [1,5.98,232.3,0.921,7.6,0,-0.53,16,	0.5,	9.37,	53.89,	0.935,	-1.966,0]
    letterDict["K"] = [5.7,9.74,300.46,1.057,103,1,1.79,85,	-1.8,	5.72,	42.92,	1.102,	-1.374,0]
    letterDict["M"] = [1.9,5.74,202.65,0.804,33.5,0,-0.38,20,	0.4,	9.83,	52.75,	0.952,	-1.963,0]
    letterDict["F"] = [1.1,5.48,204.74,0.914,25.5,0,-0.45,10,	0.5,	8.99,	53.45,	0.915,	-1.864,0]
    letterDict["P"] = [0.18,6.3,179.93,0.932,51.5,0,0.34,45,-0.3,	6.64,	45.39,	1.049,	-1.699,0]
    letterDict["S"] = [0.73,5.68,174.06,0.923,42,0,0.1,32,-0.1,6.93,47.24,1.046,	-1.753,0]
    letterDict["T"] = [1.5,5.66,205.8,0.934,45,0,0.21,32,-0.2,7.08,49.26,0.997,-1.767,0]
    letterDict["W"] = [1.6,5.89,237.01,0.803,34.7,0,-0.27,17,0.3,8.41,53.59,0.904,-1.869,0]
    letterDict["Y"] = [1.8,5.66,229.15,0.837,55.2,0,0.4,41,-0.4,8.53,51.79,0.929,-1.686,0]
    letterDict["V"] = [0.48,5.96,207.6,0.913,23.7,0,-0.62,14,0.6,10.38,56.12,0.931,-1.981,0]
    letterDict["X"] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    letterDict["-"] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    AACategoryLen = 14
    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
    
    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
    
    	AANo	 = 0
    	for AA in sequence:
    		
    		if not AA in letterDict:
    			probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
    		
    		else:
    			probMatr[sampleNo][0][AANo]= letterDict[AA]
    			
    		AANo += 1
    	sampleNo += 1
    
    return probMatr

def convertSampleToPhysicsVector_pca(sampleSeq3DArr):
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} 
	letterDict["A"] = [0.008,0.134,-0.475,-0.039,0.181,0]
	letterDict["R"] = [0.171,-0.361,0.107,-0.258,-0.364,0]
	letterDict["N"] = [0.255,0.038,0.117,0.118,-0.055,0]
	letterDict["D"] = [0.303,-0.057,-0.014,0.225,0.156,0]
	letterDict["C"] = [-0.132,0.174,0.070,0.565,-0.374,0]
	letterDict["Q"] = [0.149,-0.184,-0.030,0.035,-0.112,0]
	letterDict["E"] = [0.221,-0.280,-0.315,0.157,0.303,0]
	letterDict["G"] = [0.218,0.562,-0.024,0.018,0.106,0]
	letterDict["H"] = [0.023,-0.177,0.041,0.280,-0.021,0]
	letterDict["I"] = [-0.353,0.071,-0.088,-0.195,-0.107,0]
	letterDict["L"] = [-0.267,0.018,-0.265,-0.274,0.206,0]
	letterDict["K"] = [0.243,-0.339,-0.044,-0.325,-0.027,0]
	letterDict["M"] = [-0.239,-0.141,-0.155,0.321,0.077,0]
	letterDict["F"] = [-0.329,-0.023,0.072,-0.002,0.208,0]
	letterDict["P"] = [0.173,0.286,0.407,-0.215,0.384,0]
	letterDict["S"] = [0.199,0.238,-0.015,-0.068,-0.196,0]
	letterDict["T"] = [0.068,0.147,-0.015,-0.132,-0.274,0]
	letterDict["W"] = [-0.296,-0.186,0.389,0.083,0.297,0]
	letterDict["Y"] = [-0.141,-0.057,0.425,-0.096,-0.091,0]
	letterDict["V"] = [-0.274,0.136,-0.187,-0.196,-0.299,0]
	letterDict["X"] = [0,0,0,0,0,0]
	letterDict["-"] = [0,0,0,0,0,1]
	AACategoryLen = 6
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr

def convertSampleToPhysicsVector_2(sampleSeq3DArr):
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} 
	letterDict["A"] = [-0.591, -1.302, -0.733, 1.570,-0.146]
	letterDict["C"] = [ -1.343, 0.465, -0.862, -1.020, -0.255]
	letterDict["D"] = [1.050, 0.302, -3.656, -0.259, -3.242]
	letterDict["E"] = [1.357, -1.453, 1.477, 0.113, -0.837]
	letterDict["F"] = [-1.006, -0.590, 1.891, -0.397, 0.412]
	letterDict["G"] = [-0.384, 1.652, 1.330, 1.045, 2.064]
	letterDict["H"] = [0.336, -0.417, -1.673, -1.474, -0.078]
	letterDict["I"] = [-1.239, -0.547, 2.131, 0.393, 0.816]
	letterDict["K"] = [1.831, -0.561, 0.533, -0.277, 1.648]
	letterDict["L"] = [-1.019, -0.987, -1.505, 1.266, -0.912]
	letterDict["M"] = [-0.663, -1.524, 2.219, -1.005, 1.212]
	letterDict["N"] = [0.945, 0.828, 1.299, -0.169, 0.933]
	letterDict["P"] = [0.189, 2.081, -1.628, 0.421, -1.392]
	letterDict["Q"] = [0.931, -0.179, -3.005, -0.503, -1.853]
	letterDict["R"] = [1.538, -0.055, 1.502, 0.440, 2.897]
	letterDict["S"] = [-0.228, 1.399, -4.760, 0.670, -2.647]
	letterDict["T"] = [-0.032, 0.326, 2.213, 0.908, 1.313]	
	letterDict["V"] = [-1.337, -0.279, -0.544, 1.242, -1.262]
	letterDict["W"] = [-0.595, 0.009, 0.672, -2.128, -0.184]
	letterDict["Y"] = [0.260, 0.830, 3.097, -0.838, 1.512]
	AACategoryLen = 5
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr
	
#def convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName):
#	"""
#	Convertd the raw data to probability matrix
#	PARAMETER
#	---------
#	sampleSeq3DArr: 3D List
#		List -  numpy matrix(3D)
#	Sample List: List (nb_windows, nb_samples, SEQLen/nb_windows , 100)
#	"""
#	rawDataFrame = pd.read_table(refMatrFileName, sep='\t', header=None)
#	raw_data_seq_index_df = pd.DataFrame({'seq' : rawDataFrame[0] , 'indexing':rawDataFrame.index})
#	raw_data_seq_df_index_dict = raw_data_seq_index_df.set_index('seq')['indexing'].to_dict()
#	nb_raw_data_frame_column = len(rawDataFrame.columns)
#	nb_sample = sampleSeq3DArr.shape[0]
#	len_seq = len(sampleSeq3DArr[1]) 
#	
#	
#	probMatr_list = []
#	for tmp_idx in range(nb_windows):
#		probMatr_list.append( np.zeros((nb_sample, int((len_seq - tmp_idx)/nb_windows) , 100)) )
#	for sample_index, sample_sequence in enumerate(sampleSeq3DArr):
#		
#		if sample_index%10000 == 0:
#			print( "%d / %d " % (sample_index, nb_sample))
#		
#		#start_time = time.time()
#		for begin_idx in range(nb_windows):
#			seq_len=int((len_seq - begin_idx)/nb_windows)
#			sub_seq_index = -1
#			for tmp_idx in range(seq_len):
#				sub_seq="".join(sample_sequence[(begin_idx+nb_windows*tmp_idx):(begin_idx+nb_windows*tmp_idx+nb_windows)])
#				if sub_seq in raw_data_seq_df_index_dict:
#					sub_seq_index=raw_data_seq_df_index_dict[sub_seq] 
#				else:
#					sub_seq_index=raw_data_seq_df_index_dict['<unk>']
#				probMatr_list[begin_idx][sample_index][tmp_idx]=rawDataFrame.loc[sub_seq_index][ range(1, nb_raw_data_frame_column)]
#
#	return probMatr_list

def convertSampleToDoubleVec(sampleSeq3DArr, nb_neibor):
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    
    
    double_letter_dict = {}
    for key_row in letterDict:
        for key_col in letterDict:
            idx_row = letterDict[key_row]
            idx_col = letterDict[key_col]
            
            final_key = key_row    + key_col
            final_idx = idx_row*20 + idx_col
            
            double_letter_dict[final_key] = final_idx
    
    
    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0])-nb_neibor, len(double_letter_dict)))

    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
    
        nb_sub_AA   = 0
        sequence = sequence.tolist()
        for idx in range(len(sequence)-nb_neibor):
            
            sub_AA = ("").join( sequence[idx:idx+nb_neibor+1] )
            
            if sub_AA in double_letter_dict:
                index = double_letter_dict[sub_AA]
                probMatr[sampleNo][0][nb_sub_AA][index] = 1
            print(sub_AA)
            break
            nb_sub_AA += 1
        break
        sampleNo += 1

    
    return probMatr
    
    
    

def convertRawToXY(rawDataFrame, refMatrFileName="", nb_windows=3, codingMode=0):#rawDataFrame is numpy.ndarray
    """
    convertd the raw data to probability matrix and target array 
    
    
    #Output:
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    targetArr: Target. Shape (nb_samples)
    """
    
    
    #rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
    
    targetList = rawDataFrame[:, 0]
    targetArr = kutils.to_categorical(targetList)
    
    sampleSeq3DArr = rawDataFrame[:, 1:]
    
    if codingMode == 0:
        probMatr = convertSampleToProbMatr(sampleSeq3DArr)
    elif codingMode == 1:
        probMatr = convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName)
    elif codingMode == 2:
        probMatr = convertSampleToDoubleVec(sampleSeq3DArr, 1)
    elif codingMode == 3:
        probMatr = convertSampleToPhysicsVector(sampleSeq3DArr)
    elif codingMode == 4:
        probMatr = convertSampleToPhysicsVector_2(sampleSeq3DArr)
    elif codingMode == 41:
        probMatr = convertSampleToPhysicsVector_3(sampleSeq3DArr)
    elif codingMode == 42:
        probMatr = convertSampleToPhysicsVector_pca(sampleSeq3DArr)
    elif codingMode==43:
        probMatr = convertSampleToAAindexforUbiVector(sampleSeq3DArr)
    
    return probMatr, targetArr
     


def convertRawToIndex(rawDataFrame):
	#rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
	
	targetList = rawDataFrame[:, 0]
	targetArr = kutils.to_categorical(targetList)
	
	sampleSeq3DArr = rawDataFrame[:, 1:]
	
	index = convertSampleToIndex(sampleSeq3DArr)
	
	
	return index, targetArr
	


def convertRawToX(fileName, refMatrFileName="", nb_windows=3, codingMode=0):
	"""
	convertd the raw data to probability matrix
	
	
	#Output:
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	
	rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
	
	sampleSeq3DArr = rawDataFrame[:, 0:]
	
	if codingMode == 0:
		probMatr = convertSampleToProbMatr(sampleSeq3DArr)
	elif codingMode == 1:
		probMatr = DProcess.convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName)
	
	
	return probMatr
