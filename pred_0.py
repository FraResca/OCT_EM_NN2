from pred_models import *
from pred_trainer import *
from pred_eval import *
import sys
import random
import json
import keras

class_weight = None
variable = None

def class_weighter_single(y_true):
    one_count = tf.math.count_nonzero(y_true == 1)
    return one_count/len(y_true)

@keras.saving.register_keras_serializable()
def custom_loss_imbalance_single(y_true, y_prvis):

    if variable == 0 or variable == 1:
        weight = tf.cast(class_weight, tf.float32)
        loss = 2*BinaryCrossentropy()(y_true, y_prvis) * tf.where(y_true == 1, weight, 1 - weight)
    else:
        loss = MeanSquaredError()(y_true, y_prvis)

    return loss


def train_data_generator_single(X, images, Y, batch_size, oversample, random_state, shuffle_state):
    
    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y, shuffle_state)
    X_train, images_train, Y_train = noise_injection_training(X_train, images_train, Y_train, random_state)
    
    if oversample == True:
        X_train, images_train, Y_train = oversample_training(X_train, images_train, Y_train, random_state)

    num_samples = len(images_train)

    while True:
        X_train, images_train, Y_train = shuffle(X_train, images_train, Y_train)
        
        if Y_train.ndim == 2:
            Y_train = Y_train[:, variable]
    
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            print(f'batch: {start} - {end}')
            yield [images_train[start:end], X_train[start:end]], Y_train[start:end]

def train_pred_single(modelname, n_epochs, overs, batch_size, steps_per_epoch, stopping, patience, random_state, shuffle_state, variable_choice):
    lr = 0.001
    print(f'\nTraining model {modelname}\n')

    global class_weight
    global variable

    variable = variable_choice - 1

    if variable == 4:
        X, images, Y, scaler = data_prep_general(True, True)
    else:
        X, images, Y, scaler = data_prep_general(True, False)

    train_generator = train_data_generator_single(X, images, Y, batch_size, overs, random_state, shuffle_state)
    class_weight = class_weighter_single(Y)

    custom_object = {'custom_loss_imbalance_single': custom_loss_imbalance_single}

    model = net_0()
    model.compile(optimizer=Adam(lr), loss=custom_loss_imbalance_single, metrics=['mse', 'mae'])

    if stopping == True:
        early_stopping = EarlyStopping(monitor='mse', patience=patience, verbose=1, mode='min', restore_best_weights=True)
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, callbacks=[early_stopping])
    else:
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)

    num_epochs = len(history.history['loss'])

    model.save(f'saved_models/{modelname}.keras')

    return num_epochs

def eval_single(modelname, variable_choice, random_state, shuffle_state, num_epochs):
    variable = variable_choice - 1

    if variable == 4:
        X, images, Y, scaler = data_prep_general(True, True)
    else:
        X, images, Y, scaler = data_prep_general(True, False)
    Y = Y[:, variable]

    class_weight = class_weighter_single(Y)

    model = load_model(f'saved_models/{modelname}.keras', custom_objects={'custom_loss_imbalance_single': custom_loss_imbalance_single}, safe_mode=False)

    batch_size = 1

    test_generator = test_data_generator(X, images, Y, batch_size, random_state, shuffle_state)

    _, X_prova, _, _, _, _ = splits(X, images, Y, shuffle_state)
    num_test_samples = len(X_prova)

    results = []

    y_tests = []

    preds = []

    for i in range(num_test_samples):
        [image_test, X_test], y_test = next(test_generator)

        predictions = model.predict([image_test, X_test], batch_size=1, steps=1)

        #print(predictions)

        y_tests.append(y_test)
        if variable == 0 or variable == 1:
            preds.append(predictions[0] > 0.5)
        else:
            preds.append(predictions[0])

    #results.append({'y_tests': y_tests, 'preds': preds})

    y_tests = np.array(y_tests)
    preds = np.array(preds)

    metrics = []

    if variable == 0 or variable == 1:
        metrics.append([recall_score(y_tests, preds), precision_score(y_tests, preds), roc_auc_score(y_tests, preds)])
    else:
        metrics.append([mean_absolute_error(y_tests, preds), np.sqrt(mean_squared_error(y_tests, preds)),r2_score(y_tests, preds)])

    results.append({'Metric': metrics})

    results.append({'Shuffle state': shuffle_state, 'Random state': random_state})

    results.append({'Number of epochs': num_epochs - 10})

    with open(f'results/results_{modelname}.json', 'w') as f:
        json.dump(results, f)

    
def visualize_attention_single(modelname, vis, random_state, shuffle_state):
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