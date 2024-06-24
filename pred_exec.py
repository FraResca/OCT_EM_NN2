from pred_models import *
from pred_trainer import *
from pred_eval import *
from pred_0 import *
import sys
import random
import json

if __name__ == '__main__':
    with open (f'runs/jobs/{sys.argv[1]}.json', 'r') as f:
        params = json.load(f)
    
    modelname = params['modelname']
    modeltype = params['modeltype']
    n_epochs = params['n_epochs']
    overs = params['overs']
    batch_size = params['batch_size']
    steps_per_epoch = params['steps_per_epoch']
    stopping = params['stopping']

    print(f'\nModel name: {modelname}')
    print(f'Model type: {modeltype}')
    print(f'Number of epochs: {n_epochs}')
    print(f'Oversampling: {overs}')
    print(f'Batch size: {batch_size}')
    print(f'Steps per epoch: {steps_per_epoch}')
    print(f'Early stopping: {stopping}')
    
    if stopping == True and 'patience' in params:
        patience = params['patience']
    elif stopping == True and 'patience' not in params:
        patience = 10
    elif stopping == False:
        patience = 0
    
    print(f'Patience: {patience}')

    if 'random_state' in params:
        random_state = params['random_state']
    else:
        random_state = random.randint(0, 2**32 - 1)
    print(f'Random state: {random_state}')

    if 'shuffle_state' in params:
        shuffle_state = params['shuffle_state']
    else:
        shuffle_state = random.randint(0, 2**32 - 1)   
    print(f'Shuffle state: {shuffle_state}\n')


    #download_vgg16()

    if modeltype in range(1, 6):
        n_epochs = train_pred(modelname, modeltype, n_epochs, overs, batch_size, steps_per_epoch, False, stopping, patience, random_state, shuffle_state)
        eval_general(modelname, False, random_state, shuffle_state, n_epochs)
        visualize_attention_general(modelname, False, random_state, shuffle_state)
        
        train_pred(modelname, modeltype, n_epochs, overs, batch_size, steps_per_epoch, True, stopping, patience, random_state, shuffle_state)
        eval_general(modelname, True, random_state, shuffle_state, n_epochs)
        visualize_attention_general(modelname, True, random_state, shuffle_state)
    elif modeltype == 0:
        epochs = train_pred_single(f'{modelname}_PEC', n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, 1)
        eval_single(f'{modelname}_PEC', 1, random_state, shuffle_state, epochs)
        visualize_attention_single(f'{modelname}_PEC', False, random_state, shuffle_state)        

        epochs = train_pred_single(f'{modelname}_EZD', n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, 2)
        eval_single(f'{modelname}_EZD', 2, random_state, shuffle_state, epochs)
        visualize_attention_single(f'{modelname}_EZD', False, random_state, shuffle_state)

        epochs = train_pred_single(f'{modelname}_CST', n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, 3)
        eval_single(f'{modelname}_CST', 3, random_state, shuffle_state, epochs)
        visualize_attention_single(f'{modelname}_CST', False, random_state, shuffle_state)

        epochs = train_pred_single(f'{modelname}_CT', n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, 4)
        eval_single(f'{modelname}_CT', 4, random_state, shuffle_state, epochs)
        visualize_attention_single(f'{modelname}_CT', False, random_state, shuffle_state)

        epochs = train_pred_single(f'{modelname}_Vis', n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, 5)
        eval_single(f'{modelname}_Vis', 5, random_state, shuffle_state, epochs)
        visualize_attention_single(f'{modelname}_Vis', False, random_state, shuffle_state)
        