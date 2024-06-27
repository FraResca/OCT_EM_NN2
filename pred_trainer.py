from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from pred_models import net_select, net_vis
from predata import *
from sklearn.utils import shuffle
import tensorflow as tf

class_weights = None

def class_weighter(y_true):
    one_count_1 = tf.math.count_nonzero(y_true[:, 0] == 1)
    one_count_2 = tf.math.count_nonzero(y_true[:, 1] == 1)

    print(one_count_1, one_count_2)

    return one_count_1/len(y_true), one_count_2/len(y_true)

def custom_loss_imbalance(y_true, y_prvis):
    weight1, weight2 = class_weights

    weight1 = tf.cast(weight1, tf.float32)
    weight2 = tf.cast(weight2, tf.float32)

    binary_loss_1 = 2*BinaryCrossentropy()(y_true[:,0], y_prvis[:,0]) * tf.where(y_true[:,0] == 1, weight1, 1 - weight1)
    binary_loss_2 = 2*BinaryCrossentropy()(y_true[:,1], y_prvis[:,1]) * tf.where(y_true[:,1] == 1, weight2, 1 - weight2)
    
    binary_loss = binary_loss_1 + binary_loss_2

    mse_loss = MeanSquaredError()(y_true[:,2:], y_prvis[:,2:])

    return binary_loss + mse_loss

def custom_loss_imbalance_visus(y_true, y_prvis):

    weight1, weight2 = class_weights

    weight1 = tf.cast(weight1, tf.float32)
    weight2 = tf.cast(weight2, tf.float32)

    binary_loss_1 = 2*BinaryCrossentropy()(y_true[:,0], y_prvis[:,0]) * tf.where(y_true[:,0] == 1, weight1, 1 - weight1)
    binary_loss_2 = 2*BinaryCrossentropy()(y_true[:,1], y_prvis[:,1]) * tf.where(y_true[:,1] == 1, weight2, 1 - weight2)
    
    binary_loss = binary_loss_1 + binary_loss_2

    mse_loss = MeanSquaredError()(y_true[:,3:], y_prvis[:,3:])

    return binary_loss + mse_loss

def download_vgg16():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    model_noruler = VGG16(weights='imagenet', include_top=False, input_shape=(444, 444, 3))
    model.save('saved_models/vgg16.keras')
    model_noruler.save('saved_models/vgg16_noruler.keras')

def train_data_generator(X, images, Y, batch_size, oversample, random_state, shuffle_state):
    
    X_train, _, images_train, _, Y_train, _ = splits(X, images, Y, shuffle_state)
    X_train, images_train, Y_train = noise_injection_training(X_train, images_train, Y_train, random_state)
    
    if oversample == True:
        X_train, images_train, Y_train = oversample_training(X_train, images_train, Y_train, random_state)

    num_samples = len(images_train)

    while True:
        X_train, images_train, Y_train = shuffle(X_train, images_train, Y_train)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            print(f'batch: {start} - {end}')
            yield [images_train[start:end], X_train[start:end]], Y_train[start:end]

def train_pred(modelname, modeltype, n_epochs, overs, batch_size, steps_per_epoch, visus, stopping, patience, random_state, shuffle_state):
    lr = 0.001

    print(f'\nTraining model {modelname}\n')

    global class_weights

    X, images, Y, scaler = data_prep_general(True, visus)
    train_generator = train_data_generator(X, images, Y, batch_size, overs, random_state, shuffle_state)

    class_weights = class_weighter(Y)

    if visus == True:
        custom_object = {'custom_loss_imbalance': custom_loss_imbalance_visus}

        model = net_vis(modelname, modeltype)

        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss_imbalance_visus, metrics=['mse', 'mae'])
        
    else:
        custom_object = {'custom_loss_imbalance': custom_loss_imbalance}

        model = net_select(modeltype)

        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss_imbalance, metrics=['mse', 'mae'])

    model.summary()

    if stopping == True:
        early_stopping = EarlyStopping(monitor='mse', patience=patience, verbose=1, mode='min', restore_best_weights=True)
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, callbacks=[early_stopping])
    else:
        history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)

    num_epochs = len(history.history['loss'])

    if visus == True:
        model.save(f'saved_models/{modelname}_visus.keras')
    else:
        model.save(f'saved_models/{modelname}.keras')

    return num_epochs

