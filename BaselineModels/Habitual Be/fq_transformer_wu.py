#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
from datetime import datetime


def main(datadir, experiment):
    print ('experiment here ************************************************'
           , experiment)
    print('Datadir************************************* ',datadir)
    PREDDIR = datadir + '/preds/'
    FROMDIR = datadir + '/data-bin/'
    SAVEDIR = datadir + '/checkpoints/' + experiment + '-models/'
    SCOREDIR = datadir + '/scores/'

    MODELPATH = SAVEDIR + 'checkpoint_best.pt'

    SRC = experiment + '.input'
    TGT = experiment + '.output'
    TESTIN = datadir + '/test.' + experiment + '.input'

    # GUESSPATH = PREDDIR + experiment + '.testpredict'

    GUESSPATH = PREDDIR + experiment + '.guess'
    U_CONFIDENCE = PREDDIR + experiment + '.confidence'
    U_PREDICTIONS = PREDDIR + experiment + '.predict'

    # create folders

    subprocess.run(['mkdir', '-p', SAVEDIR])
    subprocess.run(['mkdir', '-p', PREDDIR])
    subprocess.run(['mkdir', '-p', SCOREDIR])

    # clear checkpoint and model folders from previous runs

    subprocess.run(['rm', '-rf', SAVEDIR])
    subprocess.run(['rm', '-rf', datadir + '/data-bin/'])

    # preprocess

    subprocess.call([  # '--testpref='+TEXTDIR+'test',
        'fairseq-preprocess',
        '--source-lang=' + SRC,
        '--target-lang=' + TGT,
        '--trainpref=' + datadir + 'train',
        '--validpref=' + datadir + 'dev',
        '--destdir=' + FROMDIR,
        ])

    print ('##################### PREPROCESSED', experiment,
           '#####################')

    # train
    # Transformer Parameters from Wu 2021 et al.: ACTIVATION_DROPOUT=0.3, ACTIVATION_FN='RELU', ADAM_BETAS='(0.9, 0.98)', ADAM_EPS=1E-08, ADAPTIVE_INPUT=FALSE, ADAPTIVE_SOFTMAX_CUTOFF=NONE, ADAPTIVE_SOFTMAX_DROPOUT=0, CROSS_SELF_ATTENTION=FALSE, DECODER_ATTENTION_HEADS=4, DECODER_EMBED_DIM=256, DECODER_EMBED_PATH=NONE, DECODER_FFN_EMBED_DIM=1024, DECODER_INPUT_DIM=256, DECODER_LAYERDROP=0, DECODER_LAYERS=4, DECODER_LEARNED_POS=FALSE, DECODER_NORMALIZE_BEFORE=TRUE, DECODER_OUTPUT_DIM=256, DISTRIBUTED_WRAPPER='DDP', DROPOUT=0.3, ENCODER_ATTENTION_HEADS=4, ENCODER_EMBED_DIM=256, ENCODER_FFN_EMBED_DIM=1024, ENCODER_LAYERDROP=0, ENCODER_LAYERS=4, ENCODER_LEARNED_POS=FALSE, ENCODER_NORMALIZE_BEFORE=TRUE, EVAL_BLEU=FALSE, EVAL_BLEU_ARGS=NONE, EVAL_BLEU_DETOK='SPACE', EVAL_BLEU_DETOK_ARGS=NONE, EVAL_BLEU_PRINT_SAMPLES=FALSE, EVAL_BLEU_REMOVE_BPE=NONE, EVAL_TOKENIZED_BLEU=FALSE, FIXED_VALIDATION_SEED=NONE, IGNORE_PREFIX_SIZE=0, LABEL_SMOOTHING=0.1, LAYERNORM_EMBEDDING=FALSE, LEFT_PAD_SOURCE='TRUE', LEFT_PAD_TARGET='FALSE', LOAD_ALIGNMENTS=FALSE, MAX_SOURCE_POSITIONS=1024, MAX_TARGET_POSITIONS=1024, NO_CROSS_ATTENTION=FALSE, NO_SCALE_EMBEDDING=FALSE, NO_TOKEN_POSITIONAL_EMBEDDINGS=FALSE, NUM_BATCH_BUCKETS=0,
    # pipeline_balance=None, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_encoder_balance=None,
    # QUANT_NOISE_PQ=0, QUANT_NOISE_PQ_BLOCK_SIZE=8, QUANT_NOISE_SCALAR=0, REPORT_ACCURACY=FALSE, SHARE_ALL_EMBEDDINGS=FALSE, SHARE_DECODER_INPUT_OUTPUT_EMBED=TRUE, TIE_ADAPTIVE_WEIGHTS=FALSE, TRUNCATE_SOURCE=FALSE, UPSAMPLE_PRIMARY=1, USE_OLD_ADAM=FALSE=, WARMUP_INIT_LR=-1, WARMUP_UPDATES=1000, WEIGHT_DECAY=0.0)

    # subprocess.call(['./train.sh', dataset])

    subprocess.call([  # '--restore-file=' + MODELPATH
                       # '--pipeline-encoder-balance=4',
                       # '--pipeline-decoder-balance=4',
                       # "--ddp-backend=c10d",
                       # '--seed=212'
                       # early stopping
        'fairseq-train',
        FROMDIR,
        '--save-dir=' + SAVEDIR,
        '--source-lang=' + SRC,
        '--target-lang=' + TGT,
        '--arch=transformer',
        '--batch-size=400',
        '--batch-size-valid=400',
        '--clip-norm=1.0',
        '--criterion=label_smoothed_cross_entropy',
        '--ddp-backend=legacy_ddp',
        '--lr=[0.001]',
        '--lr-scheduler=inverse_sqrt',
        '--max-update=6000',
        '--optimizer=adam',
        '--save-interval-updates=10',
        '--patience=10',
        ])

                       

    print ('##################### TRAINED', experiment,
           '#####################')

    # test

    testing = subprocess.Popen([
        'fairseq-interactive',
        FROMDIR,
        '--path',
        MODELPATH,
        '--source-lang=' + SRC,
        '--target-lang=' + TGT,
        ], stdin=open(TESTIN), stdout=subprocess.PIPE)

    # Write test predictions to file

    with open(GUESSPATH, 'w') as predout:
        writeline = []
        testout = testing.stdout.readlines()
        for line in testout:
            line = line.decode('utf-8')
            if line[0] == 'H':
                writeline.append(line.split('\t')[2].strip())
        predout.write('\n'.join(writeline))

    print ('##################### TESTED', experiment,
           '#####################')

    print ('##################### PREDICTED', experiment,
           '#####################')

    predicting = subprocess.Popen([
        'fairseq-interactive',
        FROMDIR,
        '--path',
        MODELPATH,
        '--source-lang=' + SRC,
        '--target-lang=' + TGT,
        ], stdin=open(TESTIN), stdout=subprocess.PIPE)

    with open(U_CONFIDENCE, 'w') as predwconfidence:
        with open(U_PREDICTIONS, 'w') as predict:
            writeline = []
            all_writeline = []
            predictions = predicting.stdout.readlines()
            predictions = [str(line.decode('utf-8').strip())
                           for line in predictions]
            for lineU in predictions:
                if lineU[0] == 'H':
                    writeline.append(lineU.split('\t')[2].strip())  # no confidence
                    all_writeline.append(str(lineU.strip()))  # w confidence
            predict.write('\n'.join(predictions))
            predwconfidence.write('\n'.join(all_writeline))

    print ('##################### PREDICTED', experiment,
           '#####################')
    

    


if __name__ == '__main__':
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
            print("filenameparts ***************************** ",filenameparts)
            if filenameparts[0] == 'train':
                experimentname = filenameparts[1]
                print("Experiment Name:", experimentname)
                main(path, experimentname)
                cmd1 = f'rm -rf {path}/checkpoints/{experimentname}-models/checkpoint_[1,2,3,4,5,6,7,8,9]*'
                cmd2 = f'rm -rf {path}/checkpoints/{experimentname}-models/checkpoint[1,2,3,4,5,6,7,8,9]*'
                subprocess.run(cmd1,shell=True)
                subprocess.run(cmd2,shell=True)
                print ('\n########### FINISHED ', experimentname, ' ###########\n')
         

        # if experiment == sys.argv[2]:
        #    main(sys.argv[1], experiment)
        #    print('\n########### FINISHED ', experiment, ' ###########\n')
        # else:
        #    print('\n########### CANNOT FIND', experiment, ' ###########\n')
