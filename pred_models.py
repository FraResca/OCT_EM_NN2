import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Multiply, BatchNormalization, Conv2D, GlobalAveragePooling2D, Lambda, Dense, Concatenate
from tensorflow.keras.regularizers import l1
from keras.layers import Input, Dense, Flatten, Concatenate, Dropout, Conv2D
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

def custom_loss_imbalance_visus(y_true, y_prvis):
    global class_weights

    weight1, weight2 = class_weights

    weight1 = tf.cast(weight1, tf.float32)
    weight2 = tf.cast(weight2, tf.float32)

    binary_loss_1 = 2*BinaryCrossentropy()(y_true[:,0], y_prvis[:,0]) * tf.where(y_true[:,0] == 1, weight1, 1 - weight1)
    binary_loss_2 = 2*BinaryCrossentropy()(y_true[:,1], y_prvis[:,1]) * tf.where(y_true[:,1] == 1, weight2, 1 - weight2)
    
    binary_loss = binary_loss_1 + binary_loss_2

    mse_loss = MeanSquaredError()(y_true[:,3:], y_prvis[:,3:])

    return binary_loss + mse_loss

def net_0():
    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    dense = Dense(20, activation='relu', name='dense_cln_1')(input_cln)
    dense = Dense(16, activation='relu', name='dense_cln_2')(dense)
    denseput = Dense(12, activation='sigmoid', name='dense_cln_3')(dense)

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)

    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model(input_img)

    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1')(conv)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2')(conv)
    conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3')(conv)
    conv = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4')(conv)
    
    postcnn = conv
    
    conv = Multiply(name='multiply')([conv, prevgg16_output])
    conv = GlobalAveragePooling2D(name='global_avg_pool_1')(conv)
    
    y = GlobalAveragePooling2D(name='global_avg_pool_2')(postcnn)
    
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide')([conv, y])
    convput = Flatten(name='flatten')(conv)

    concat = Concatenate(name='concat')([convput, denseput])
    dropout = Dropout(0.5, name='dropout')(concat)
    dense_fin = Dense(1024, activation='relu', name='dense_fin', activity_regularizer=l1(0.001))(dropout)
    hid = Dense(512, activation='relu', name='hid')(dense_fin)

    output = Dense(1, activation='sigmoid', name='output')(hid)
    
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_0_model')

    return model

# Design di rete n°1
# Gli output sono tutti collegati a un'unico strato precedente
def net_1():
    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    dense = Dense(20, activation='relu', name='dense_cln_1')(input_cln)
    dense = Dense(16, activation='relu', name='dense_cln_2')(dense)
    denseput = Dense(12, activation='sigmoid', name='dense_cln_3')(dense)

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)

    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model(input_img)

    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1')(conv)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2')(conv)
    conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3')(conv)
    conv = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4')(conv)
    
    postcnn = conv
    
    conv = Multiply(name='multiply')([conv, prevgg16_output])
    conv = GlobalAveragePooling2D(name='global_avg_pool_1')(conv)
    
    y = GlobalAveragePooling2D(name='global_avg_pool_2')(postcnn)
    
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide')([conv, y])
    convput = Flatten(name='flatten')(conv)

    concat = Concatenate(name='concat')([convput, denseput])
    dropout = Dropout(0.5, name='dropout')(concat)
    dense_fin = Dense(1024, activation='relu', name='dense_fin', activity_regularizer=l1(0.001))(dropout)
    hid = Dense(512, activation='relu', name='hid')(dense_fin)

    edema_output = Dense(1, activation='sigmoid', name='edema_output')(hid)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(hid)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(hid)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(hid)

    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_1_model')

    return model


# Design di rete n°2
# Gli output sono collegati a strati separati
def net_2():
    input_cln = Input(shape=(14, ), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    dense = Dense(20, activation='relu', name='dense_cln_1')(input_cln)
    dense = Dense(16, activation='relu', name='dense_cln_2')(dense)
    denseput = Dense(12, activation='sigmoid', name='dense_cln_3')(dense)
    
    # Create the input tensor
    input_img = Input(shape=(444, 444, 3))

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)


    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model(input_img)

    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)
    conv = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1')(conv)
    conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2')(conv)
    conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3')(conv)
    conv = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4')(conv)
    
    postcnn = conv
    
    conv = Multiply(name='multiply')([conv, prevgg16_output])
    conv = GlobalAveragePooling2D(name='global_avg_pool_1')(conv)
    
    y = GlobalAveragePooling2D(name='global_avg_pool_2')(postcnn)
    
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide')([conv, y])
    convput = Flatten(name='flatten')(conv)

    concat = Concatenate(name='concat')([convput, denseput])
    dropout = Dropout(0.5, name='dropout')(concat)
    dense_fin = Dense(1024, activation='relu', name='dense_fin', activity_regularizer=l1(0.001))(dropout)
    
    edema_hid = Dense(512, activation='relu', name='edema_hid')(dense_fin)
    ellip_hid = Dense(512, activation='relu', name='ellip_hid')(dense_fin)
    cst_hid = Dense(512, activation='relu', name='cst_hid')(dense_fin)
    ct_hid = Dense(512, activation='relu', name='ct_hid')(dense_fin)

    edema_output = Dense(1, activation='sigmoid', name='edema_output')(edema_hid)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(ellip_hid)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(cst_hid)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(ct_hid)


    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_2_model')

    return model


# Design di rete n°3
# 4 reti separate con lo stesso input
def net_3():

    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)


    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model.output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)

    # Rete per Edema
    dense_ed = Dense(20, activation='relu', name='dense_cln_1_ed')(input_cln)
    dense_ed = Dense(16, activation='relu', name='dense_cln_2_ed')(dense_ed)
    denseput_ed = Dense(12, activation='sigmoid', name='dense_cln_3_ed')(dense_ed)

    prevgg16_output = vgg16_output

    conv_ed = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_ed')(conv)
    conv_ed = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_ed')(conv_ed)
    conv_ed = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_ed')(conv_ed)
    conv_ed = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_ed')(conv_ed)
    
    postcnn_ed = conv_ed
    
    conv_ed = Multiply(name='multiply_ed')([conv_ed, prevgg16_output])
    conv_ed = GlobalAveragePooling2D(name='global_avg_pool_1_ed')(conv_ed)
    
    y_ed = GlobalAveragePooling2D(name='global_avg_pool_2_ed')(postcnn_ed)
    
    convput_ed = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_ed')([conv_ed, y_ed])

    concat_ed = Concatenate(name='concat_ed')([convput_ed, denseput_ed])
    dropout = Dropout(0.5, name='dropout_ed')(concat_ed)
    dense_fin_ed = Dense(1024, activation='relu', name='dense_fin_ed', activity_regularizer=l1(0.001))(dropout)
    hid_ed = Dense(512, activation='relu', name='hid_ed')(dense_fin_ed)

    # Rete per Ellissoide
    dense_ell = Dense(20, activation='relu', name='dense_cln_1_ell')(input_cln)
    dense_ell = Dense(16, activation='relu', name='dense_cln_2_ell')(dense_ell)
    denseput_ell = Dense(12, activation='sigmoid', name='dense_cln_3_ell')(dense_ell)

    conv_ell = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_ell')(conv)
    conv_ell = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_ell')(conv_ell)
    conv_ell = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_ell')(conv_ell)
    conv_ell = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_ell')(conv_ell)

    postcnn_ell = conv_ell

    conv_ell = Multiply(name='multiply_ell')([conv_ell, prevgg16_output])
    conv_ell = GlobalAveragePooling2D(name='global_avg_pool_1_ell')(conv_ell)

    y_ell = GlobalAveragePooling2D(name='global_avg_pool_2_ell')(postcnn_ell)

    convput_ell = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_ell')([conv_ell, y_ell])

    concat_ell = Concatenate(name='concat_ell')([convput_ell, denseput_ell])
    dropout = Dropout(0.5, name='dropout_ell')(concat_ell)
    dense_fin_ell = Dense(1024, activation='relu', name='dense_fin_ell', activity_regularizer=l1(0.001))(dropout)
    hid_ell = Dense(512, activation='relu', name='hid_ell')(dense_fin_ell)

    # Rete per CST
    dense_cst = Dense(20, activation='relu', name='dense_cln_1_cst')(input_cln)
    dense_cst = Dense(16, activation='relu', name='dense_cln_2_cst')(dense_cst)
    denseput_cst = Dense(12, activation='sigmoid', name='dense_cln_3_cst')(dense_cst)

    conv_cst = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_cst')(conv)
    conv_cst = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_cst')(conv_cst)
    conv_cst = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_cst')(conv_cst)
    conv_cst = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_cst')(conv_cst)

    postcnn_cst = conv_cst

    conv_cst = Multiply(name='multiply_cst')([conv_cst, prevgg16_output])
    conv_cst = GlobalAveragePooling2D(name='global_avg_pool_1_cst')(conv_cst)

    y_cst = GlobalAveragePooling2D(name='global_avg_pool_2_cst')(postcnn_cst)

    convput_cst = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_cst')([conv_cst, y_cst])

    concat_cst = Concatenate(name='concat_cst')([convput_cst, denseput_cst])
    dropout = Dropout(0.5, name='dropout_cst')(concat_cst)
    dense_fin_cst = Dense(1024, activation='relu', name='dense_fin_cst', activity_regularizer=l1(0.001))(dropout)
    hid_cst = Dense(512, activation='relu', name='hid_cst')(dense_fin_cst)

    # Rete per CT
    dense_ct = Dense(20, activation='relu', name='dense_cln_1_ct')(input_cln)
    dense_ct = Dense(16, activation='relu', name='dense_cln_2_ct')(dense_ct)
    denseput_ct = Dense(12, activation='sigmoid', name='dense_cln_3_ct')(dense_ct)

    conv_ct = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_ct')(conv)
    conv_ct = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_ct')(conv_ct)
    conv_ct = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_ct')(conv_ct)
    conv_ct = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_ct')(conv_ct)

    postcnn_ct = conv_ct

    conv_ct = Multiply(name='multiply_ct')([conv_ct, prevgg16_output])
    conv_ct = GlobalAveragePooling2D(name='global_avg_pool_1_ct')(conv_ct)

    y_ct = GlobalAveragePooling2D(name='global_avg_pool_2_ct')(postcnn_ct)

    convput_ct = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_ct')([conv_ct, y_ct])
    
    concat_ct = Concatenate(name='concat_ct')([convput_ct, denseput_ct])
    dropout = Dropout(0.5, name='dropout_ct')(concat_ct)
    dense_fin_ct = Dense(1024, activation='relu', name='dense_fin_ct', activity_regularizer=l1(0.001))(dropout)
    hid_ct = Dense(512, activation='relu', name='hid_ct')(dense_fin_ct)

    # Output
    edema_output = Dense(1, activation='sigmoid', name='edema_output')(hid_ed)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(hid_ell)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(hid_cst)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(hid_ct)

    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_3_model')

    return model

# Design di rete n°4
# 3 reti separate con lo stesso input
# Output numerici(ct/cst) uniti
def net_4():

    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)


    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model.output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)

    # Rete per Edema
    dense_ed = Dense(20, activation='relu', name='dense_cln_1_ed')(input_cln)
    dense_ed = Dense(16, activation='relu', name='dense_cln_2_ed')(dense_ed)
    denseput_ed = Dense(12, activation='sigmoid', name='dense_cln_3_ed')(dense_ed)

    prevgg16_output = vgg16_output

    conv_ed = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_ed')(conv)
    conv_ed = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_ed')(conv_ed)
    conv_ed = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_ed')(conv_ed)
    conv_ed = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_ed')(conv_ed)
    
    postcnn_ed = conv_ed
    
    conv_ed = Multiply(name='multiply_ed')([conv_ed, prevgg16_output])
    conv_ed = GlobalAveragePooling2D(name='global_avg_pool_1_ed')(conv_ed)
    
    y_ed = GlobalAveragePooling2D(name='global_avg_pool_2_ed')(postcnn_ed)
    
    convput_ed = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_ed')([conv_ed, y_ed])

    concat_ed = Concatenate(name='concat_ed')([convput_ed, denseput_ed])
    dropout = Dropout(0.5, name='dropout_ed')(concat_ed)
    dense_fin_ed = Dense(1024, activation='relu', name='dense_fin_ed', activity_regularizer=l1(0.001))(dropout)
    hid_ed = Dense(512, activation='relu', name='hid_ed')(dense_fin_ed)

    # Rete per Ellissoide
    dense_ell = Dense(20, activation='relu', name='dense_cln_1_ell')(input_cln)
    dense_ell = Dense(16, activation='relu', name='dense_cln_2_ell')(dense_ell)
    denseput_ell = Dense(12, activation='sigmoid', name='dense_cln_3_ell')(dense_ell)

    conv_ell = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_ell')(conv)
    conv_ell = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_ell')(conv_ell)
    conv_ell = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_ell')(conv_ell)
    conv_ell = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_ell')(conv_ell)

    postcnn_ell = conv_ell

    conv_ell = Multiply(name='multiply_ell')([conv_ell, prevgg16_output])
    conv_ell = GlobalAveragePooling2D(name='global_avg_pool_1_ell')(conv_ell)

    y_ell = GlobalAveragePooling2D(name='global_avg_pool_2_ell')(postcnn_ell)

    convput_ell = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_ell')([conv_ell, y_ell])

    concat_ell = Concatenate(name='concat_ell')([convput_ell, denseput_ell])
    dropout = Dropout(0.5, name='dropout_ell')(concat_ell)
    dense_fin_ell = Dense(1024, activation='relu', name='dense_fin_ell', activity_regularizer=l1(0.001))(dropout)
    hid_ell = Dense(512, activation='relu', name='hid_ell')(dense_fin_ell)

    # Rete per output numerici
    dense_c = Dense(20, activation='relu', name='dense_cln_1_c')(input_cln)
    dense_c = Dense(16, activation='relu', name='dense_cln_2_c')(dense_c)
    denseput_c = Dense(12, activation='sigmoid', name='dense_cln_3_c')(dense_c)

    conv_c = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_c')(conv)
    conv_c = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_c')(conv_c)
    conv_c = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_c')(conv_c)
    conv_c = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_c')(conv_c)

    postcnn_c = conv_c

    conv_c = Multiply(name='multiply_c')([conv_c, prevgg16_output])
    conv_c = GlobalAveragePooling2D(name='global_avg_pool_1_c')(conv_c)

    y_c = GlobalAveragePooling2D(name='global_avg_pool_2_c')(postcnn_c)

    convput_c = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_c')([conv_c, y_c])

    concat_c = Concatenate(name='concat_c')([convput_c, denseput_c])
    dropout = Dropout(0.5, name='dropout_c')(concat_c)
    dense_fin_c = Dense(1024, activation='relu', name='dense_fin_c', activity_regularizer=l1(0.001))(dropout)
    hid_c = Dense(512, activation='relu', name='hid_c')(dense_fin_c)

    # Output
    edema_output = Dense(1, activation='sigmoid', name='edema_output')(hid_ed)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(hid_ell)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(hid_c)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(hid_c)

    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_4_model')

    return model

# Design di rete n°5
# 2 reti separate con lo stesso input
# Output numerici uniti
def net_5():

    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    # Create the VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)


    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model.output
    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)
    
    # Rete per output binari
    dense_b = Dense(20, activation='relu', name='dense_cln_1_b')(input_cln)
    dense_b = Dense(16, activation='relu', name='dense_cln_2_b')(dense_b)
    denseput_b = Dense(12, activation='sigmoid', name='dense_cln_3_b')(dense_b)

    conv_b = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_b')(conv)
    conv_b = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_b')(conv_b)
    conv_b = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_b')(conv_b)
    conv_b = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_b')(conv_b)

    postcnn_b = conv_b

    conv_b = Multiply(name='multiply_b')([conv_b, prevgg16_output])
    conv_b = GlobalAveragePooling2D(name='global_avg_pool_1_b')(conv_b)

    y_b = GlobalAveragePooling2D(name='global_avg_pool_2_b')(postcnn_b)

    convput_b = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_b')([conv_b, y_b])

    concat_b = Concatenate(name='concat_b')([convput_b, denseput_b])
    dropout = Dropout(0.5, name='dropout_b')(concat_b)
    dense_fin_b = Dense(1024, activation='relu', name='dense_fin_b', activity_regularizer=l1(0.001))(dropout)
    hid_b = Dense(512, activation='relu', name='hid_b')(dense_fin_b)

    # Rete per output numerici
    dense_c = Dense(20, activation='relu', name='dense_cln_1_c')(input_cln)
    dense_c = Dense(16, activation='relu', name='dense_cln_2_c')(dense_c)
    denseput_c = Dense(12, activation='sigmoid', name='dense_cln_3_c')(dense_c)

    conv_c = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_c')(conv)
    conv_c = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_c')(conv_c)
    conv_c = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_c')(conv_c)
    conv_c = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_c')(conv_c)

    postcnn_c = conv_c

    conv_c = Multiply(name='multiply_c')([conv_c, prevgg16_output])
    conv_c = GlobalAveragePooling2D(name='global_avg_pool_1_c')(conv_c)

    y_c = GlobalAveragePooling2D(name='global_avg_pool_2_c')(postcnn_c)

    convput_c = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_c')([conv_c, y_c])

    concat_c = Concatenate(name='concat_c')([convput_c, denseput_c])
    dropout = Dropout(0.5, name='dropout_c')(concat_c)
    dense_fin_c = Dense(1024, activation='relu', name='dense_fin_c', activity_regularizer=l1(0.001))(dropout)
    hid_c = Dense(512, activation='relu', name='hid_c')(dense_fin_c)

    # Output
    edema_output = Dense(1, activation='sigmoid', name='edema_output')(hid_b)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(hid_b)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(hid_c)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(hid_c)

    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_5_model')

    return model

def net_select(modeltype):
    if modeltype == 1:
        return net_1()
    elif modeltype == 2:
        return net_2()
    elif modeltype == 3:
        return net_3()
    elif modeltype == 4:
        return net_4()
    elif modeltype == 5:
        return net_5()


def net_vis(modelname, modeltype):

    custom_object = {'custom_loss_imbalance': custom_loss_imbalance_visus}
    model = load_model(f'saved_models/{modelname}.keras', custom_objects=custom_object, safe_mode=False)

    if modeltype == 1:
        hid = model.get_layer('hid').output
        vis_output = Dense(1, activation='sigmoid', name='vis_output')(hid)
    elif modeltype == 2:
        dense_fin = model.get_layer('dense_fin').output
        vis_hid = Dense(512, activation='relu', name='vis_hid')(dense_fin)
        vis_output = Dense(1, activation='sigmoid', name='vis_output')(vis_hid)
    elif modeltype == 3 or modeltype == 4 or modeltype == 5:
        input_cln = model.inputs[1]
        if 'block5_pool' in [layer.name for layer in model.layers]:
            prevgg16_output = model.get_layer('block5_pool').output
        elif 'vgg16' in [layer.name for layer in model.layers]:
            prevgg16_output = model.get_layer('vgg16').output
        else:
            raise ValueError("Neither 'block5_pool' nor 'vgg16' layers found in the model.")
        
        conv = model.get_layer('batch_norm').output

        # Rete per Visus
        dense_vis = Dense(20, activation='relu', name='dense_cln_1_vis')(input_cln)
        dense_vis = Dense(16, activation='relu', name='dense_cln_2_vis')(dense_vis)
        denseput_vis = Dense(12, activation='sigmoid', name='dense_cln_3_vis')(dense_vis)
        
        conv_vis = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_img_1_vis')(conv)
        conv_vis = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_img_2_vis')(conv_vis)
        conv_vis = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_img_3_vis')(conv_vis)
        conv_vis = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_img_4_vis')(conv_vis)
        
        postcnn_vis = conv_vis
        
        conv_vis = Multiply(name='multiply_vis')([conv_vis, prevgg16_output])
        conv_vis = GlobalAveragePooling2D(name='global_avg_pool_1_vis')(conv_vis)
        
        y_vis = GlobalAveragePooling2D(name='global_avg_pool_2_vis')(postcnn_vis)
        
        convput_vis = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide_vis')([conv_vis, y_vis])

        concat_vis = Concatenate(name='concat_vis')([convput_vis, denseput_vis])
        dropout = Dropout(0.5, name='dropout_vis')(concat_vis)
        dense_fin_vis = Dense(1024, activation='relu', name='dense_fin_vis', activity_regularizer=l1(0.001))(dropout)
        vis_hid = Dense(512, activation='relu', name='vis_hid')(dense_fin_vis)

        vis_output = Dense(1, activation='sigmoid', name='vis_output')(vis_hid)

    output = Concatenate(name='output_vis')([model.output, vis_output])

    vis_model = Model(inputs=model.inputs, outputs=output)
    
    return vis_model

def unify_convs(layers, i):
    if i==0:
        input_shape = layers[0].input_shape[1:]
    else:
        input_shape = layers[0].input_shape[1:-1] + (layers[0].input_shape[-1]*4,)
    input_layer = Input(shape=input_shape)
    print(input_layer.shape)


    
    conv_layers = []
    for layer in layers:
        new_conv = Conv2D(layer.filters, layer.kernel_size, activation=layer.activation, padding=layer.padding)(input_layer)
        conv_layers.append(new_conv)

    output = Concatenate(axis=-1)(conv_layers)
    return Model(inputs=input_layer, outputs=output)


def unify_nets(models):
    input_cln = Input(shape=(14,), name='input_cln')
    input_img = Input(shape=(444, 444, 3), name='input_img')

    dense = models[0].get_layer('dense_cln_1')(input_cln)
    dense = models[0].get_layer('dense_cln_2')(dense)
    denseput = models[0].get_layer('dense_cln_3')(dense)

    vgg16_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_img)
    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model(input_img)

    prevgg16_output = vgg16_output

    conv = BatchNormalization(name='batch_norm')(vgg16_output)

    for i in range(4):
        layers = []
        for j in range(len(models)):
            layers.append(models[j].get_layer(f'conv_img_{i+1}'))
            conv = unify_convs(layers, i)(conv)

    postcnn = conv
    
    conv = Multiply(name='multiply')([conv, prevgg16_output])
    conv = GlobalAveragePooling2D(name='global_avg_pool_1')(conv)
    
    y = GlobalAveragePooling2D(name='global_avg_pool_2')(postcnn)
    
    conv = Lambda(lambda tensors: tensors[0] / (tensors[1] + 1e-7), name='divide')([conv, y])
    convput = Flatten(name='flatten')(conv)

    concat = Concatenate(name='concat')([convput, denseput])
    dropout = Dropout(0.5, name='dropout')(concat)
    dense_fin = Dense(1024, activation='relu', name='dense_fin', activity_regularizer=l1(0.001))(dropout)
    hid = Dense(512, activation='relu', name='hid')(dense_fin)

    edema_output = Dense(1, activation='sigmoid', name='edema_output')(hid)
    ellip_output = Dense(1, activation='sigmoid', name='ellip_output')(hid)
    cst_output = Dense(1, activation='sigmoid', name='cst_output')(hid)
    ct_output = Dense(1, activation='sigmoid', name='ct_output')(hid)

    output = Concatenate(name='output')([edema_output, ellip_output, cst_output, ct_output])
    model = Model(inputs=[input_img, input_cln], outputs=output, name='net_1_model')

    return model