import sys
import pandas as pd
import numpy as np
import argparse
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-input',  dest='inputfile', type=str, help='Protein sequences to be predicted in fasta format.', required=True)
    parser.add_argument('-output',  dest='outputfile', type=str, help='prefix of the prediction results.', required=True)
    parser.add_argument('-model-prefix',  dest='modelprefix', type=str, help='prefix of custom model used for prediciton. If donnot have one, please run train_general.py to train a custom general PTM model or run train_kinase.py to train a custom kinase-specific PTM model.', required=False,default=None)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that to be predicted. For multiple residues, seperate each with \',\'',required=False,default="S,T,Y")
    
    args = parser.parse_args()
    
    inputfile=args.inputfile;
    outputfile=args.outputfile;
    residues=args.residues.split(",")
    modelprefix=args.modelprefix;
    
    if modelprefix is None:
       print "Please specify the prefix for an existing custom model by -model-prefix!\n\
       It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
       If you don't have such files, please run train_models.py to get the custom model first!\n"
       exit()
    else: #custom prediction
      model=modelprefix+str("_HDF5model")
      parameter=modelprefix+str("_parameters")
      try:
          f=open(parameter,'r')
      except IOError:
          print 'cannot open '+ parameter+" ! check if the model exists. please run train_general.py or train_kinase.py to get the custom model first!\n"
      else:
           f= open(parameter, 'r')
           parameters=f.read()
           f.close()
      from DProcess import convertRawToXY
      from EXtractfragment_sort import extractFragforPredict
      from capsulenet import Capsnet_main
      nclass=int(parameters.split("\t")[0])
      window=int(parameters.split("\t")[1])
      residues=parameters.split("\t")[2]
      residues=residues.split(",")
      codemode=int(parameters.split("\t")[4])
      modeltype=str(parameters.split("\t")[5])
      nb_classes=int(parameters.split("\t")[6])
      #print "nclass="+str(nclass)+"codemode="+str(codemode)+"\n";
      testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=residues)
      
      testX,testY = convertRawToXY(testfrag.as_matrix(),codingMode=codemode) 
      if len(testX.shape)>3:
          testX.shape = (testX.shape[0],testX.shape[2],testX.shape[3])
      
      predictproba=np.zeros((testX.shape[0],2))
      models=Capsnet_main(testX,testY,nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype=modeltype,nb_classes=nb_classes,predict=True)# only to get config
      
      nclass_ini=1;
      for bt in range(nclass):
             models[0].load_weights(model+"_class"+str(bt))
             predictproba+=models[1].predict(testX)[0]
      
      predictproba=predictproba/(nclass*nclass_ini);
      poses=poses+1;
      results=np.column_stack((ids,poses,focuses,predictproba[:,1]))
      result=pd.DataFrame(results)
      result.to_csv(outputfile+".txt", index=False, header=None, sep='\t',quoting=csv.QUOTE_NONNUMERIC)
      print "Successfully predicted from custom models !\n";
    
    
    


if __name__ == "__main__":
    main()         
   
