import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate, Conv2DTranspose, Reshape, BatchNormalization, Dropout, Multiply, GlobalAveragePooling2D, Lambda
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.regularizers import l1
from predata import data_prep_gan
from itertools import cycle


# Define the Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

#def one_hot_encode(binary_variables):
def one_hot_encode(binary_variables):    
    binary_integer = int(''.join(map(str, binary_variables)), 2)
    one_hot_encoded_array = np.zeros(2**12)
    one_hot_encoded_array[binary_integer] = 1

    return one_hot_encoded_array

def one_hot_decode_train(one_hot_encoded_array):
    binary_integer = np.argmax(one_hot_encoded_array)
    binary_variables = [int(x) for x in format(binary_integer, '012b')]
    return np.array(binary_variables)

# [2,3,4,5,6,7,10,11,12,13,14,15]

def reconstruct_dato_train(dato, classe, vis):
    if vis:
        array = np.zeros(0)
    else:
        array = np.zeros(0)

    for i in range(2):
        array = np.concatenate([array, np.reshape(dato[i], [-1])], axis=0)
    for i in range(6):
        array = np.concatenate([array, np.reshape(classe[i], [-1])], axis=0)
    for i in range(2, 4):
        array = np.concatenate([array, np.reshape(dato[i], [-1])], axis=0)
    for i in range(6, 12):
        array = np.concatenate([array, np.reshape(classe[i], [-1])], axis=0)
    for i in range(4, 6):
        array = np.concatenate([array, np.reshape(dato[i], [-1])], axis=0)
        
    if vis:
        array = np.concatenate([array, np.reshape(dato[6], [-1])], axis=0)

    return array

def decimal_to_binary_tensor(tensor, num_bits):
    binary_tensor = []
    for i in range(num_bits):
        binary_tensor.append(tf.math.floordiv(tensor, 2**i) % 2)
    return tf.stack(binary_tensor, axis=-1)

# [2,3,4,5,6,7,10,11,12,13,14,15]

def reconstruct_dato_build(dato, classe, vis):
    classe = tf.argmax(classe, axis=-1)
    classe = decimal_to_binary_tensor(classe, 12)
    classe = classe[:, -12:]

    array = tf.zeros([1, 0])

    for i in range(2):
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(dato[:,i], [-1]), tf.float32), axis=1)], axis=1)
    for i in range(6):
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(classe[:,i], [-1]), tf.float32), axis=1)], axis=1)
    for i in range(2, 4):
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(dato[:,i], [-1]), tf.float32), axis=1)], axis=1)
    for i in range(6, 12):
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(classe[:,i], [-1]), tf.float32), axis=1)], axis=1)
    for i in range(4, 6):
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(dato[:,i], [-1]), tf.float32), axis=1)], axis=1)

    if vis:
        array = tf.concat([array, tf.expand_dims(tf.cast(tf.reshape(dato[:,6], [-1]), tf.float32), axis=1)], axis=1)

    # array = array[:, 1:]  # Remove the first column of zeros

    return array



def build_generator(vis):
    input_class = Input(shape=(2**12,), name='input_class')
    input_noise = Input(shape=(100,), name='input_noise')

    concatenated = Concatenate()([input_class, input_noise])

    dense_img = Dense(111*111*1)(concatenated)
    dense_img = LeakyReLU(alpha=0.2)(dense_img)
    reshaped = Reshape((111, 111, 1))(dense_img)
    up = Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='same')(reshaped)
    up = LeakyReLU(alpha=0.2)(up)
    up = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(up)
    up = LeakyReLU(alpha=0.2)(up)
    up = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(up)
    up = LeakyReLU(alpha=0.2)(up)
    img = Conv2DTranspose(1, (3, 3), padding='same', activation='tanh')(up)
    output_img = tf.image.grayscale_to_rgb(img)

    dense = Dense(256)(concatenated)
    dense = LeakyReLU(alpha=0.2)(dense)
    dense = Dense(128)(dense)
    dense = LeakyReLU(alpha=0.2)(dense)
    dense = Dense(64)(dense)
    dense = LeakyReLU(alpha=0.2)(dense)
    dense = Dense(32)(dense)
    dense = LeakyReLU(alpha=0.2)(dense)
    dense = Dense(16)(dense)
    dense = LeakyReLU(alpha=0.2)(dense)

    if vis:
        output_cln = Dense(7, activation='sigmoid')(dense)
    else:
        output_cln = Dense(6, activation='sigmoid')(dense)

    model = Model(inputs=[input_class, input_noise], outputs=[output_img, output_cln])

    return model

from keras.layers import LeakyReLU

def build_discriminator(vis):
    if vis:
        input_cln = Input(shape=(19,), name='input_cln')
    else:
        input_cln = Input(shape=(18,), name='input_cln')

    input_img = Input(shape=(444, 444, 3), name='input_img')

    dense = Dense(20, name='dense_cln_1')(input_cln)
    dense = LeakyReLU(alpha=0.2)(dense)
    dense = Dense(16, name='dense_cln_2')(dense)
    dense = LeakyReLU(alpha=0.2)(dense)
    denseput = Dense(12, activation='sigmoid', name='dense_cln_3')(dense)

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)

    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model(input_img)

    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)
    conv = Conv2D(256, (3, 3), padding='same', name='conv_img_1')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(64, (3, 3), padding='same', name='conv_img_2')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(16, (3, 3), padding='same', name='conv_img_3')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv2D(512, (3, 3), padding='same', name='conv_img_4')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    
    postcnn = conv
    
    conv = Multiply(name='multiply')([conv, prevgg16_output])
    conv = GlobalAveragePooling2D(name='global_avg_pool_1')(conv)
    
    y = GlobalAveragePooling2D(name='global_avg_pool_2')(postcnn)
    
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide')([conv, y])
    convput = Flatten(name='flatten')(conv)

    concat = Concatenate(name='concat')([convput, denseput])
    dropout = Dropout(0.5, name='dropout')(concat)
    dense_fin = Dense(1024, name='dense_fin', activity_regularizer=l1(0.001))(dropout)
    dense_fin = LeakyReLU(alpha=0.2)(dense_fin)
    hid = Dense(512, name='hid')(dense_fin)
    hid = LeakyReLU(alpha=0.2)(hid)

    output = Dense(1, activation='sigmoid', name='output')(hid)

    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_1_model')
    model.compile(optimizer=RMSprop(learning_rate=0.00005), loss=wasserstein_loss)

    return model

def class_counter(vis):
    data, images, Y = data_prep_gan(True, vis)

    class_dict = {}
    for i, dato in enumerate(data):
        binary_vars = [int(x) for x in dato[[2,3,4,5,6,7,10,11,12,13,14,15]]]
        classe = tuple(one_hot_encode(binary_vars))
        if classe not in class_dict:
            class_dict[classe] = []
        class_dict[classe].append((dato, images[i], Y[i]))

    return class_dict

def gan_data_generator(vis):
    classi = class_counter(vis)

    # Create a cycle iterator for each class
    class_cycles = {classe: cycle(dati) for classe, dati in classi.items()}

    while True:
        for classe, dati_cycle in class_cycles.items():
            dato, image, y = next(dati_cycle)
            yield dato, image, y, classe


def build_gan(generator, discriminator, vis):
    gan_noise_input = Input(shape=(100,))
    gan_class_input = Input(shape=(2**12,))

    [fake_img, fake_cln] = generator([gan_class_input, gan_noise_input])

    fake_cln = reconstruct_dato_build(fake_cln, gan_class_input, vis)

    discriminator.trainable = False  # Freeze the discriminator's weights

    gan_output = discriminator([fake_img, fake_cln])
    gan = Model([gan_class_input, gan_noise_input], gan_output)

    # Compile the combined model
    gan.compile(optimizer=RMSprop(learning_rate=0.00005), loss=wasserstein_loss)

    return gan


def train_gan(n_epochs, vis):
    gan_data_gen = gan_data_generator(vis)
    generator = build_generator(vis)
    discriminator = build_discriminator(vis)
    gan = build_gan(generator, discriminator, vis)

    for epoch in range(n_epochs):
        batch_size = 1

        print(f'Epoch {epoch+1}/{n_epochs}')

        # Get the real data
        real_data, real_images, _, classe = next(gan_data_gen)
        classe = np.array(classe)
        classe = np.tile(classe, (batch_size, 1))
        real_data = np.tile(real_data, (batch_size, 1))
        real_images = np.tile(real_images, (batch_size, 1, 1, 1))

        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, 100))
        [fake_img, fake_cln] = generator.predict([classe, noise])
        fake_cln = reconstruct_dato_train(fake_cln[0], one_hot_decode_train(classe), vis)
        fake_cln = np.tile(fake_cln, (batch_size, 1))

        for i in range(batch_size):
            normalized_fake_img = (fake_img[i] - np.min(fake_img[i])) / (np.max(fake_img[i]) - np.min(fake_img[i]))
            plt.imsave(f'gan_imgs/image_{epoch}_{i}.png', normalized_fake_img)

        real_labels = np.ones((len(real_data), 1)) * 0.9 + 0.05 * np.random.random((len(real_data), 1))
        fake_labels = -(np.ones((len(real_data), 1)) * 0.9 + 0.05 * np.random.random((len(real_data), 1)))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([real_images, real_data], real_labels)
        d_loss_fake = discriminator.train_on_batch([fake_img, fake_cln], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        for i in range(5):
            noise = np.random.normal(0, 1, (len(real_data), 100))
            g_loss = gan.train_on_batch([classe, noise], np.ones((len(real_data), 1)))

        print(f'Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

train_gan(1000, False)