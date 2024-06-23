from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import pprint

def crop_images(directory, output_directory):
    print('Cropping images...')
    
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename))
        width, height = img.size
        print(f'Original size: {width}x{height}')
        exc1 = ['16_0.tif','36_0.tif','63_0.tif', '70_0.tif','89_0.tif','89_3.tif','101_0.tif','105_0.tif',
                '113_0.tif','116_1.tif','123_0.tif','124_0.tif','129_1.tif','159_0.tif','192_0.tif','194_0.tif',
                '231_0.tif','253_0.tif','276_0.tif','345_0.tif','347_0.tif','271_0.tif','212_0.tif']
        exc2 = []
        exc3 = ['86_0.tif','112_0.tif','350_0.tif','337_0.tif','324_0.tif','307_0.tif','301_0.tif','292_0.tif',
                '288_0.tif','257_0.tif','245_0.tif','242_0.tif','199_0.tif','172_0.tif','151_0.tif','146_0.tif',
                '131_0.tif',]
        if filename in exc1:
            w_ratio = 1/2
            h_ratio = 0
        elif filename in exc2:
            w_ratio = 1/3
            h_ratio = 0
        elif filename in exc3:
            w_ratio = 0
            h_ratio = 0
            img = img.convert('L')
            img = img.convert('RGB')
        else:
            w_ratio = 3/5
            h_ratio = 1/6
        cropped_img = img.crop((width*w_ratio, height*h_ratio, width, height*(1-h_ratio)))
        print(f'Cropped size: {cropped_img.size[0]}x{cropped_img.size[1]}')
        cropped_img = cropped_img.resize((512, 512))
        print(f'Resized to: {cropped_img.size[0]}x{cropped_img.size[1]}')
        cropped_img.save(os.path.join(output_directory, filename))
        print(f'{filename} cropped')

def remove_ruler(directory, output_directory):
    print('Removing ruler...')

    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename))
        width, height = img.size
        cropped_img = img.crop((width/15, height/15, 14*width/15, 14*height/15))
        print(f'Final size: {cropped_img.size[0]}x{cropped_img.size[1]}')
        cropped_img.save(os.path.join(output_directory, filename))
        print(f'{filename} cropped')


def corr_mat(X, Y):
    df = pd.DataFrame(np.concatenate([X, Y], axis=1))
    print('\nDataFrame originale:')
    print(df)
    print('\nMatrice di correlazione:')
    print(df.corr())

def fixer(vis):
    df = pd.read_excel('RAW.xlsx', engine='openpyxl')

    '''
    labenc = LabelEncoder()
    df.iloc[:, 1] = labenc.fit_transform(df.iloc[:, 1])
    '''

    '''
    print("\nOriginale:\n")
    print(df)
    print(df.shape)
    '''

    # Drop rows where id refers to an image that doesn't exist in 'OCT_noruler'
    df = df[df.iloc[:, 0].apply(lambda id: os.path.isfile(f'OCT_noruler/{str(id).zfill(2)}_0.tif'))]
    
    '''
    print("\nDropped rows with non-existing image ids:\n")
    print(df)
    print(df.shape)
    '''

    if vis == True:
        df.dropna(subset=df.columns[-5:], how='any', inplace=True)
    else:
        df.drop(df.columns[-1], axis=1, inplace=True)
        '''
        print("\nDropped last column:\n")
        print(df)
        print(df.shape)
        '''

        df.dropna(subset=df.columns[-4:], how='any', inplace=True)
    '''
    print("\nDropped NaN:\n")
    print(df)
    print(df.shape)
    '''

    mice_imputer = IterativeImputer(max_iter=50)

    df_imputed = mice_imputer.fit_transform(df)

    # Convert numpy array back to DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    df_imputed.iloc[:, 2] = df_imputed.iloc[:, 2] * 100
    if vis == True:
        df_imputed.iloc[:, -1] = df_imputed.iloc[:, -1] * 100

    df_imputed = df_imputed.astype(int)

    '''
    print("\nImputed:\n")
    print(df_imputed)
    print(df_imputed.shape)
    '''
    
    return df_imputed

def load_data(noruler, vis):
    print(f'Loading data...')
    
    if noruler == True:
        dirname = 'OCT_noruler'
        dim = 444
    else:
        dirname = 'OCT_crops'
        dim = 512

    df = fixer(vis)

    numerical_cols = [1, 2, 9, 10, 17, 18]
    if vis == True:
        n_target = 5
        numerical_cols.append(19)
    else:
        n_target = 4
        

    data = df.to_numpy()

    ids = data[:, 0]  # first column

    images = []

    for id in ids:
        if id < 10:
            image_file = os.path.join(f'{dirname}/0{id}_0.tif')
        else:
            image_file = os.path.join(f'{dirname}/{id}_0.tif')

        img = image.load_img(image_file, target_size=(dim,dim))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        images.append(img_array)

    images = np.concatenate(images)

    # print("Raw clinical data:")
    # print(data)

    return data, images, n_target, numerical_cols

def normalize_clinical(data, numerical_cols):
    print('Normalizing data...')

    '''
    labenc = LabelEncoder()
    data[:, 1] = labenc.fit_transform(data[:, 1])
    '''

    data[:, 11] = data[:, 11] - 1

    ohenc = OneHotEncoder()

    multiEnc = ohenc.fit_transform(data[:, 3].reshape(-1, 1)).toarray()    
    data_before = data[:, :3]
    data_after = data[:, 4:]
    data = np.hstack((data_before, multiEnc, data_after))

    # print(data)
    # print(data[:, numerical_cols])

    data = data.astype(float)

    scaler = MinMaxScaler()

    data[:, numerical_cols] = scaler.fit_transform(data[:, numerical_cols])

    '''
    print('Data normalized')
    print(data)
    '''

    return data, scaler

def denormalize_clinical(data, numerical_cols, scaler):
    data[:, numerical_cols] = scaler.inverse_transform(data[:, numerical_cols])

    return data

def oversample_training(X_train, images_train, Y_train, random_state):
    count = [0, 0, 0, 0]
    classes = [[], [], [], []]
    random.seed(random_state)

    for i in range(len(Y_train)):
        if Y_train[i][0] == 0 and Y_train[i][1] == 0:
            classes[0].append([X_train[i], images_train[i], Y_train[i]])
        elif Y_train[i][0] == 0 and Y_train[i][1] == 1:
            classes[1].append([X_train[i], images_train[i], Y_train[i]])
        elif Y_train[i][0] == 1 and Y_train[i][1] == 0:
            classes[2].append([X_train[i], images_train[i], Y_train[i]])
        else:
            classes[3].append([X_train[i], images_train[i], Y_train[i]])
    
    count = list(map(len, classes))

    print(f'Count pre-oversampling: {count} - {sum(count)} samples')

    max_count = max(map(len, classes))

    while not all(len(cls) == max_count for cls in classes):
        for i in range(4):
            if len(classes[i]) < max_count:
                classes[i] += [random.choice(classes[i]) for _ in range(max_count - len(classes[i]))]
    
    count = list(map(len, classes))

    print(f'Count post-oversampling: {count} - {sum(count)} samples')

    combined = classes[0] + classes[1] + classes[2] + classes[3]

    random.shuffle(combined)

    X_train, images_train, Y_train = zip(*combined)

    X_train = np.array(X_train)
    images_train = np.array(images_train)
    Y_train = np.array(Y_train)

    return X_train, images_train, Y_train


def oversample(data, images, random_state):
    count = [0, 0, 0, 0]
    classes = [[], [], [], []]
    random.seed(random_state)

    if len(data) == 19:
        edema = 4
        ellip = 3
    else:
        edema = 5
        ellip = 4

    for i in range(len(data)):
        if data[i][-edema] == 0 and data[i][-ellip] == 0:
            classes[0].append([data[i], images[i]])
        elif data[i][-edema] == 0 and data[i][-ellip] == 1:
            classes[1].append([data[i], images[i]])
        elif data[i][-edema] == 1 and data[i][-ellip] == 0:
            classes[2].append([data[i], images[i]])
        else:
            classes[3].append([data[i], images[i]])

    count = list(map(len, classes))
    print(f'Count pre-oversampling: {count}')

    print('Oversampling...')

    # Get the maximum count
    max_count = max(map(len, classes))

    # While not all classes have the same count...
    while not all(len(cls) == max_count for cls in classes):
        # For each class...
        for i in range(4):
            # If this class has less than the maximum count...
            if len(classes[i]) < max_count:
                # Oversample this class
                classes[i] += [random.choice(classes[i]) for _ in range(max_count - len(classes[i]))]

    count = list(map(len, classes))
    print(f'Count post-oversampling: {count}')

    # Concatenate the lists
    combined = classes[0] + classes[1] + classes[2] + classes[3]

    # Shuffle the combined list
    random.shuffle(combined)

    # Separate the data and images
    data, images = zip(*combined)

    data = np.array(data)
    images = np.array(images)

    return data, images

def data_prep_general(noruler, vis):

    data, images, n_target, numerical_cols = load_data(noruler, vis)
    data, scaler = normalize_clinical(data, numerical_cols)
    # data, images = oversample(data, images, random_state)

    X = data[:, 1:-n_target]  # all columns except first and last n_target
    Y = data[:, -n_target:]  # last n_target columns

    return X, images, Y, scaler

def splits(X_norm, images, Y_norm, shuffle_state):
    X_norm, images, Y = shuffle(X_norm, images,  Y_norm, random_state=shuffle_state)
    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_norm, test_size=0.15, random_state=shuffle_state)
    images_train, images_test, _, _ = train_test_split(images, Y_norm, test_size=0.15, random_state=shuffle_state)

    return X_train, X_test, images_train, images_test, Y_train, Y_test

def data_prep_gan(noruler, vis):
    data, images, _, numerical_cols = load_data(noruler, vis)
    data, _ = normalize_clinical(data, numerical_cols)
    data, images = oversample(data, images, 42)

    data = data[:, 1:]  # all columns except first

    '''
    # Generate a random sample of indices
    indices = np.random.choice(data.shape[0], n_samples, replace=False)

    # Select the random samples
    data = data[indices]
    images = images[indices]
    
    Y = np.ones((n_samples, 1))
    '''

    #Y = np.ones((data.shape[0], 1))

    Y = np.full((data.shape[0], 1), 0.9)

    return data, images, Y

def missing_percentage():
    df = pd.read_excel('START.xlsx', engine='openpyxl')
    total = df.shape[0]*df.shape[1]
    missing = df.isnull().sum().sum()
    perc = (missing/total)*100
    print(f'Missing values: {missing} ({perc:.2f}%)')
    
# should count the number of values for each combination of edema and ellipsoid
def target_distribution():
    X, images, Y, _ = data_prep_general(True, True)
    count = [0, 0, 0, 0]
    for y in Y:
        if y[0] == 0 and y[1] == 0:
            count[0] += 1
        elif y[0] == 0 and y[1] == 1:
            count[1] += 1
        elif y[0] == 1 and y[1] == 0:
            count[2] += 1
        else:
            count[3] += 1
    print(count)

def binary_array_to_number(binary_array):
    number = 0
    for index, value in enumerate(reversed(binary_array)):
        number += value * (2 ** index)
    return number

def noise_injection_training(X_train, images_train, Y_train, random_state):
    keys = []
    random.seed(random_state)
    boolean_cols = [2, 3, 4, 5, 6 ,7, 10, 11]
    numerical_cols = [0, 1, 8, 9]

    np.set_printoptions(threshold=np.inf)

    for i in range(len(X_train)):
        key = int(binary_array_to_number(X_train[i, boolean_cols]))
        keys.append(key)

    classes = {key: [] for key in keys}

    for i in range(len(X_train)):
        key = int(binary_array_to_number(X_train[i, boolean_cols]))
        classes[key].append([X_train[i], images_train[i], Y_train[i]])
    
    for key in classes.keys():
        print(f'Key: {key} - Count: {len(classes[key])}')
        for i in classes[key]:
            Xi, imagei, Yi = i
            print(f'X: {Xi} - Y: {Yi}')
    
    max = 0
    for key in classes.keys():
        if len(classes[key]) > max:
            max = len(classes[key])

    col_means = {col: np.mean(X_train[:, col]) for col in numerical_cols}
    print(f'col_means: {col_means}')

    col_stds = {col: np.std(X_train[:, col]) for col in numerical_cols}
    print(f'col_stds: {col_stds}')

    while not all(len(classes[key]) == max for key in classes.keys()):
        for key in classes.keys():
            if len(classes[key]) < max:
                X_key, image_key, Y_key = random.choice(classes[key]).copy()
                
                # Add noise to numerical columns of X_key
                for col in numerical_cols:
                    noise = np.random.normal(0, col_stds[col] * 0.05)
                    print(f'Noise: {noise}')
                    X_key[col] += noise
                
                classes[key].append([X_key, image_key, Y_key])

    for key in classes.keys():
        print(f'Key: {key} - Count: {len(classes[key])}')
        for i in classes[key]:
            Xi, imagei, Yi = i
            print(f'X: {Xi} - Y: {Yi}')

    combined = []

    for key in classes.keys():
        combined += classes[key]

    X_train, images_train, Y_train = zip(*combined)

    X_train = np.array(X_train)
    images_train = np.array(images_train)
    Y_train = np.array(Y_train)
    
    return X_train, images_train, Y_train