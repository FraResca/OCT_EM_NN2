from pred_models import *
from pred_trainer import *
from pred_eval import *
import sys
import random
import json

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

    binary_loss_1 = BinaryCrossentropy()(y_true[:,0], y_prvis[:,0]) * tf.where(y_true[:,0] == 1, weight1, 1 - weight1)
    binary_loss_2 = BinaryCrossentropy()(y_true[:,1], y_prvis[:,1]) * tf.where(y_true[:,1] == 1, weight2, 1 - weight2)
    
    binary_loss = binary_loss_1 + binary_loss_2

    mse_loss = MeanSquaredError()(y_true[:,3:], y_prvis[:,3:])

    return binary_loss + mse_loss

r1 = random.randint(0, 100000)

shuffle_state = 2963220438
''

models = []
for i in ['A1', 'A2', 'A3', 'A4']:
    model = load_model(f'saved_models/{i}_visus.keras', custom_objects={'custom_loss_imbalance_visus': custom_loss_imbalance_visus}, safe_mode=False)
    models.append(model)

concat_model = unify_nets(models)
concat_model.summary()
concat_model.save('saved_models/concat_model_visus.keras')

eval_general('concat_model', True, r1, shuffle_state, 0)