from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model, load_model
from sklearn.utils import shuffle
from scipy.ndimage import zoom
from predata import *
from pred_trainer import class_weighter, custom_loss_imbalance, custom_loss_imbalance_visus
import matplotlib.pyplot as plt
import numpy as np
import json


def test_data_generator(X, images, Y, batch_size, random_state, shuffle_state):
    _, X_test, _, images_test, _, Y_test = splits(X, images, Y, shuffle_state)
    num_samples = len(images)
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield [images_test[start:end], X_test[start:end]], Y_test[start:end]


def eval_general(modelname, vis, random_state, shuffle_state, n_epochs):
    if vis == True:
        print(f'\nEvaluating model {modelname}_visus...\n')
    else:
        print(f'\nEvaluating model {modelname}...\n')

    global class_weights

    X, images, Y, scaler = data_prep_general(True, vis)
    class_weights = class_weighter(Y)
    
    if vis == True:
        model = load_model(f'saved_models/{modelname}_visus.keras', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus}, safe_mode=False)
    else:
        model = load_model(f'saved_models/{modelname}.keras', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus}, safe_mode=False)

    batch_size = 1
    test_generator = test_data_generator(X, images, Y, batch_size, random_state, shuffle_state)
    _, X_prova, _, _, _, _ = splits(X, images, Y, shuffle_state)
    print
    num_test_samples = len(X_prova)

    results = []

    y_tests = [[], [], [], []]
    preds = [[], [], [], []]
    if vis == True:
        y_tests.append([])
        preds.append([])

    for i in range(num_test_samples):
        [image_test, X_test], y_test = next(test_generator)

        predictions = model.predict([image_test, X_test], steps=1)

        numerical_cols = [2, 3, 9, 10, 17, 18]
        if vis == True:
            target = 5
            numerical_cols.append(19)
        else:
            target = 4

        for j in range(target):
            y_tests[j].append(y_test[0, j])
            if j < 2:
                preds[j].append(predictions[0, j] > 0.5) 
            else:
                preds[j].append(predictions[0, j])

        

        numerical_cols = [x - 1 for x in numerical_cols]
        print(f'\nPrediction: {i+1}')
        '''
        print(f'Edema\tPredicted: {predictions[0, 0]:.2f}\tTrue: {y_test[0, 0]}')
        print(f'Ellip\tPredicted: {predictions[0, 1]:.2f}\tTrue: {y_test[0, 1]}')
        print(f'CST\tPredicted: {predictions[0, 2]:.2f}\tTrue: {y_test[0, 2]}')
        print(f'CT\tPredicted: {predictions[0, 3]:.2f}\tTrue: {y_test[0, 3]}')
        
        if vis == True:
            print(f'Vis\tPredicted: {predictions[0, 4]:.2f}\tTrue: {y_test[0, 4]}')
        '''
        new_data = np.concatenate((X_test, predictions), axis=1)
        old_data = np.concatenate((X_test, y_test), axis=1)

        new_data[:, numerical_cols] = scaler.inverse_transform(new_data[:, numerical_cols])
        old_data[:, numerical_cols] = scaler.inverse_transform(old_data[:, numerical_cols])
        np.set_printoptions(precision=2, suppress=True)

        print(f'\nPredicted Data:')
        print(new_data[0, -target:])

        print(f'\nTrue Data:')
        print(old_data[0, -target:])

        # results.append({'Predicted': new_data[0, -target:].tolist(), 'True': old_data[0, -target:].tolist()})

    # Convert lists to numpy arrays
    for j in range(target):
        y_tests[j] = np.array(y_tests[j])
        preds[j] = np.array(preds[j])

    # Calculate the metric for each target
    metrics = []
    for j in range(target):
        if j < 2:
            metrics.append([recall_score(y_tests[j], preds[j]), precision_score(y_tests[j], preds[j]), roc_auc_score(y_tests[j], preds[j])])
        else:
            metrics.append([mean_absolute_error(y_tests[j], preds[j]), np.sqrt(mean_squared_error(y_tests[j], preds[j])), r2_score(y_tests[j], preds[j])])

    # Add the metrics to your results
    if vis == True:
        results.append({'PEC': metrics[0], 'EZD': metrics[1], 'CST': metrics[2], 'CT': metrics[3], 'Vis': metrics[4]})    
    else:
        results.append({'PEC': metrics[0], 'EZD': metrics[1], 'CST': metrics[2], 'CT': metrics[3]})

    results.append({'Shuffle state': shuffle_state, 'Random state': random_state})

    results.append({'Number of epochs': n_epochs - 10})

    print('\nSaving results to json file...\n')

    if vis == True:
        with open(f'results/results_{modelname}_vis.json', 'w') as f:
            json.dump(results, f)
    else:
        with open(f'results/results_{modelname}.json', 'w') as f:
            json.dump(results, f)


def visualize_attention_general(modelname, vis, random_state, shuffle_state):
    if vis == True:
        print(f'\nVisualizing attention maps for model {modelname}_visus...\n')
    else:
        print(f'\nVisualizing attention maps for model {modelname}...\n')

    X, images, Y, _ = data_prep_general(True, vis)
    # X, images, Y = data_aug(X, images, Y)
    _, X_test, _, images_test, _, _ = splits(X, images, Y, shuffle_state)

    if vis == True:
        model = load_model(f'saved_models/{modelname}_visus.keras', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus}, safe_mode=False)
    else:
        model = load_model(f'saved_models/{modelname}.keras', custom_objects={'custom_loss_imbalance': custom_loss_imbalance, 'custom_loss_imbalance_visus': custom_loss_imbalance_visus}, safe_mode=False)

    multiply_layer_indexes = [i for i, layer in enumerate(model.layers) if 'multiply' in layer.name]

    multiply_layer_outputs = [model.layers[i].output for i in multiply_layer_indexes]
    attention_model = Model(inputs=model.input, outputs=multiply_layer_outputs)

    num_chunks = len(images_test)
    attention_map = []
    for i in range(num_chunks):
        start = i
        end = start + 1
        attention_map_chunk = attention_model.predict([images_test[start:end], X_test[start:end]])
        attention_map_chunk = [np.squeeze(output, axis=0) if output.shape[0] == 1 else output for output in attention_map_chunk]
        attention_map.append(attention_map_chunk)

    attention_map = np.stack(attention_map, axis=0)

    attention_map -= attention_map.min()
    attention_map /= attention_map.max()

    attention_map_2d = np.sum(attention_map, axis=-1)

    # attention_map_resized = np.array([zoom(img, (images_test[0].shape[0]/img.shape[0], images_test[0].shape[1]/img.shape[1])) for img in attention_map_2d])
    attention_map_resized = np.array([[zoom(img, (images_test[0].shape[0]/img.shape[0], images_test[0].shape[1]/img.shape[1])) for img in img_set] for img_set in attention_map_2d])
    attention_map_resized = attention_map_resized.astype('float32') / attention_map_resized.max()

    num_images = len(images_test)

    for i in range(num_images):
        fig, ax = plt.subplots(1, len(multiply_layer_indexes), figsize=(10, 5))
        if len(multiply_layer_indexes) == 1:
            ax = [ax]
        for j in range(len(multiply_layer_indexes)):
            ax[j].imshow(images_test[i], cmap='gray')
            ax[j].imshow(attention_map_resized[i][j], cmap='jet', alpha=0.5)

        if vis == True:
            plt.savefig(f'attention_maps/attention_{modelname}_vis_{i}.png')
        else:
            plt.savefig(f'attention_maps/attention_{modelname}_{i}.png')

        plt.close(fig)