#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/python3
import os
import sys
from datetime import datetime
from sklearn import metrics

def evalTags(predfile, goldfile, outfile, text='\n'):
    '''Writes accuracy, P,R,F1 to outfile.'''
    print("predfile::: ",predfile)
    print("goldfile::::" ,goldfile)
    print("outputfile:::: ", outfile)
    goldlines = [line.strip() for line in open(goldfile)]
    predlines = [line.strip() for line in open(predfile)]
    total_pred = len(predlines)
    
    try:
        check = predlines[-1]
    except IndexError:
        print("No predictions in file")
    
    # get performance metrics
    sklearn_prf1 = metrics.classification_report(goldlines, predlines, zero_division=0)
    #confusions = metrics.multilabel_confusion_matrix(skgold, skpred)
    
    text += sklearn_prf1
                                                           
    with open(outfile, "w") as R:     
        R.write(text)


def main(directory, experiment):
    
    TEXTDIR = directory
    
    testout = TEXTDIR + 'test.' + experiment  + '.output'
    guesspath = TEXTDIR + 'preds/' + experiment + '.guess'
    resultspath = TEXTDIR + 'scores/' + experiment + '.score'
    
    message = experiment + datetime.now().isoformat(timespec='minutes') + '\n'
    print("in main evaluate binary::::** ",testout," &&&&",guesspath, "&&&&&&&&&&&&& ", resultspath)
    evalTags(guesspath, testout, resultspath, text=message)


if __name__== '__main__':    
    '''Command: python pathtothisfile pathtodatadir 
    e.g. python evaluate_binary.py /path to folder/
    evaluates with sklearn (need python module)'''
    
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&", sys.argv[1])
    folders=[f for f in os.listdir(sys.argv[1])]
    print("folders%%%%%%%%%%%%%%%%%%%% ",folders)
    
    for folder in folders:
        path=os.path.join(sys.argv[1],folder)
        path=path+'/'
        print("path***************************b ",path)
        files = [f for f in os.listdir(path) if f.endswith('input') and f.startswith('train')]
        print ('files&&&&&&&&&&&&&&&&&&&&&&&  ', files)
        
        for filename in files:
            filenameparts = filename.split('.')
            split = filenameparts[0]
            experimentname = filenameparts[1]
            print("split 888888 ",split," 99999 ", experimentname)
            if split == 'train': # one experiment per train/test in/output file
                print("files 7777777777777 ",files)
                print("experiment name _eval_tag: ", experimentname,sys.argv[1])
                main(path, experimentname)        
                print('\n########### EVALUATED ', experimentname, ' ###########\n')

