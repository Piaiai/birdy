import os
import numpy as np 
import pandas as pd
import librosa

import tensorflow as tf
import tensorflow.keras.backend as K 
import tensorflow.keras.layers as L 
import tensorflow_io as tfio
import tensorflow_addons as tfa
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.models import Sequential


convlayer=ResNet101V2(input_shape=(224,224,3),weights='imagenet',include_top=False)
model_resnet=Sequential()
model_resnet.add(convlayer)
model_resnet.add(Dropout(0.5))
model_resnet.add(Flatten())
model_resnet.add(BatchNormalization())
model_resnet.add(Dense(2048,kernel_initializer='he_uniform'))
model_resnet.add(BatchNormalization())
model_resnet.add(Activation('relu'))
model_resnet.add(Dropout(0.5))
model_resnet.add(Dense(1024,kernel_initializer='he_uniform'))
model_resnet.add(BatchNormalization())
model_resnet.add(Activation('relu'))
model_resnet.add(Dropout(0.5))
model_resnet.add(Dense(230,activation='softmax'))
opt=tf.keras.optimizers.Adam(lr=0.001)
model_resnet.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=opt)
model_resnet.load_weights('./training_1/cp.ckpt')

NUM_CLASSES = 225 
IMG_SHAPE = (112, 112, 3)
DENSE_BLOCKS_NUM = 3
GROWTH_RATE = 12
COMPRESSION_FACTOR = 0.5
DEPTH = 120
TARGET_SIZE = IMG_SHAPE[:2]
EPOCHS = 60
OUTPUT = './'
BATCH_SIZE = 16

def create_dense_block(filters, kernel_size, prev_layer, padding='same',
                      kernel_initializer='he_normal'):
    x = BatchNormalization()(prev_layer)
    x = Activation('relu')(x)
    return Conv2D(filters=filters, kernel_size=kernel_size, 
                  padding=padding, kernel_initializer=kernel_initializer)(x)
    


def create_densenet_bc(shape, num_classes, dense_blocks_num, 
                      depth, growth_rate, compression_factor):
    num_bottleneck_layers = (depth - 4) // (2 * dense_blocks_num)
    num_filters_before_dense_block = 2 * growth_rate
    
    inputs = Input(shape=shape)
    x = create_dense_block(num_filters_before_dense_block, 3, inputs)
    x = concatenate([inputs, x])
    
    for i in range(dense_blocks_num):
        for j in range(num_bottleneck_layers):
            y = create_dense_block(4*growth_rate, 1, x)
            y = Dropout(0.2)(y)
            y = create_dense_block(growth_rate, 3, y)
            y = Dropout(0.2)(y)
            x = concatenate([x, y])
            
        if i == dense_blocks_num - 1:
            continue
            
        num_filters_before_dense_block += num_bottleneck_layers * growth_rate
        num_filters_before_dense_block = int(num_filters_before_dense_block * compression_factor)
        
        y = BatchNormalization()(x)
        y = Conv2D(num_filters_before_dense_block, 1, padding='same',
                   kernel_initializer='he_normal')(y)
        x = AveragePooling2D()(y)
        
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, kernel_initializer='he_normal',
                    activation='softmax')(y)
    
    return Model(inputs=inputs, outputs=outputs)

image_model = create_densenet_bc(IMG_SHAPE, NUM_CLASSES, DENSE_BLOCKS_NUM, 
                           DEPTH, GROWTH_RATE, COMPRESSION_FACTOR)

image_model.load_weights("image_model.hdf5")


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

def F1(y_true, y_pred):
    TPFN = possible_positives(y_true, y_pred)
    TPFP = predicted_positives(y_true, y_pred)
    TP = true_positives(y_true, y_pred)
    return  (TP * 2) / (TPFN + TPFP + K.epsilon())

BIRD_CODE = {'Alder Flycatcher': 0,
 'American Avocet': 1,
 'American Bittern': 2,
 'American Crow': 3,
 'American Goldfinch': 4,
 'American Kestrel': 5,
 'Buff-bellied Pipit': 6,
 'American Redstart': 7,
 'American Robin': 8,
 'American Wigeon': 9,
 'American Woodcock': 10,
 'American Tree Sparrow': 11,
 "Anna's Hummingbird": 12,
 'Ash-throated Flycatcher': 13,
 "Baird's Sandpiper": 14,
 'Bald Eagle': 15,
 'Baltimore Oriole': 16,
 'Sand Martin': 17,
 'Barn Swallow': 18,
 'Black-and-white Warbler': 19,
 'Belted Kingfisher': 20,
 "Bell's Sparrow": 21,
 "Bewick's Wren": 22,
 'Black-billed Cuckoo': 23,
 'Black-billed Magpie': 24,
 'Blackburnian Warbler': 25,
 'Black-capped Chickadee': 26,
 'Black-chinned Hummingbird': 27,
 'Black-headed Grosbeak': 28,
 'Blackpoll Warbler': 29,
 'Black-throated Sparrow': 30,
 'Black Phoebe': 31,
 'Blue Grosbeak': 32,
 'Blue Jay': 33,
 'Brown-headed Cowbird': 34,
 'Bobolink': 35,
 "Bonaparte's Gull": 36,
 'Barred Owl': 37,
 "Brewer's Blackbird": 38,
 "Brewer's Sparrow": 39,
 'Brown Creeper': 40,
 'Brown Thrasher': 41,
 'Broad-tailed Hummingbird': 42,
 'Broad-winged Hawk': 43,
 'Black-throated Blue Warbler': 44,
 'Black-throated Green Warbler': 45,
 'Black-throated Grey Warbler': 46,
 'Bufflehead': 47,
 'Blue-grey Gnatcatcher': 48,
 'Blue-headed Vireo': 49,
 "Bullock's Oriole": 50,
 'American Bushtit': 51,
 'Blue-winged Teal': 52,
 'Blue-winged Warbler': 53,
 'Cactus Wren': 54,
 'California Gull': 55,
 'California Quail': 56,
 'Cape May Warbler': 57,
 'Canada Goose': 58,
 'Canada Warbler': 59,
 'Canyon Wren': 60,
 'Carolina Wren': 61,
 "Cassin's Finch": 62,
 'Caspian Tern': 63,
 "Cassin's Vireo": 64,
 'Cedar Waxwing': 65,
 'Chipping Sparrow': 66,
 'Chimney Swift': 67,
 'Chestnut-sided Warbler': 68,
 'Chukar Partridge': 69,
 "Clark's Nutcracker": 70,
 'American Cliff Swallow': 71,
 'Common Goldeneye': 72,
 'Common Grackle': 73,
 'Common Loon': 74,
 'Common Merganser': 75,
 'Common Nighthawk': 76,
 'Northern Raven': 77,
 'Common Redpoll': 78,
 'Common Tern': 79,
 'Common Yellowthroat': 80,
 "Cooper's Hawk": 81,
 "Costa's Hummingbird": 82,
 'California Scrub Jay': 83,
 'Dark-eyed Junco': 84,
 'Double-crested Cormorant': 85,
 'Downy Woodpecker': 86,
 'American Dusky Flycatcher': 87,
 'Black-necked Grebe': 88,
 'Eastern Bluebird': 89,
 'Eastern Kingbird': 90,
 'Eastern Meadowlark': 91,
 'Eastern Phoebe': 92,
 'Eastern Towhee': 93,
 'Eastern Wood Pewee': 94,
 'Eurasian Collared Dove': 95,
 'Common Starling': 96,
 'Evening Grosbeak': 97,
 'Field Sparrow': 98,
 'Fish Crow': 99,
 'Red Fox Sparrow': 100,
 'Gadwall': 101,
 'Grey-crowned Rosy Finch': 102,
 'Green-tailed Towhee': 103,
 'Eurasian Teal': 104,
 'Golden-crowned Kinglet': 105,
 'Golden-crowned Sparrow': 106,
 'Golden Eagle': 107,
 'Great Blue Heron': 108,
 'Great Crested Flycatcher': 109,
 'Great Egret': 110,
 'Greater Roadrunner': 111,
 'Greater Yellowlegs': 112,
 'Great Horned Owl': 113,
 'Green Heron': 114,
 'Great-tailed Grackle': 115,
 'Grey Catbird': 116,
 'American Grey Flycatcher': 117,
 'Hairy Woodpecker': 118,
 "Hammond's Flycatcher": 119,
 'European Herring Gull': 120,
 'Hermit Thrush': 121,
 'Hooded Merganser': 122,
 'Hooded Warbler': 123,
 'Horned Grebe': 124,
 'Horned Lark': 125,
 'House Finch': 126,
 'House Sparrow': 127,
 'House Wren': 128,
 'Indigo Bunting': 129,
 'Juniper Titmouse': 130,
 'Killdeer': 131,
 'Ladder-backed Woodpecker': 132,
 'Lark Sparrow': 133,
 'Lazuli Bunting': 134,
 'Least Bittern': 135,
 'Least Flycatcher': 136,
 'Least Sandpiper': 137,
 "LeConte's Thrasher": 138,
 'Lesser Goldfinch': 139,
 'Lesser Nighthawk': 140,
 'Lesser Yellowlegs': 141,
 "Lewis's Woodpecker": 142,
 "Lincoln's Sparrow": 143,
 'Long-billed Curlew': 144,
 'Long-billed Dowitcher': 145,
 'Loggerhead Shrike': 146,
 'Long-tailed Duck': 147,
 'Louisiana Waterthrush': 148,
 "MacGillivray's Warbler": 149,
 'Magnolia Warbler': 150,
 'Mallard': 151,
 'Marsh Wren': 152,
 'Merlin': 153,
 'Mountain Bluebird': 154,
 'Mountain Chickadee': 155,
 'Mourning Dove': 156,
 'Northern Cardinal': 157,
 'Northern Flicker': 158,
 'Northern Harrier': 159,
 'Northern Mockingbird': 160,
 'Northern Parula': 161,
 'Northern Pintail': 162,
 'Northern Shoveler': 163,
 'Northern Waterthrush': 164,
 'Northern Rough-winged Swallow': 165,
 "Nuttall's Woodpecker": 166,
 'Olive-sided Flycatcher': 167,
 'Orange-crowned Warbler': 168,
 'Western Osprey': 169,
 'Ovenbird': 170,
 'Palm Warbler': 171,
 'Pacific-slope Flycatcher': 172,
 'Pectoral Sandpiper': 173,
 'Peregrine Falcon': 174,
 'Phainopepla': 175,
 'Pied-billed Grebe': 176,
 'Pileated Woodpecker': 177,
 'Pine Grosbeak': 178,
 'Pinyon Jay': 179,
 'Pine Siskin': 180,
 'Pine Warbler': 181,
 'Plumbeous Vireo': 182,
 'Prairie Warbler': 183,
 'Purple Finch': 184,
 'Pygmy Nuthatch': 185,
 'Red-breasted Merganser': 186,
 'Red-breasted Nuthatch': 187,
 'Red-breasted Sapsucker': 188,
 'Red-bellied Woodpecker': 189,
 'Red Crossbill': 190,
 'Redhead': 191,
 'Red-eyed Vireo': 192,
 'Red-necked Phalarope': 193,
 'Red-shouldered Hawk': 194,
 'Red-tailed Hawk': 195,
 'Red-winged Blackbird': 196,
 'Ring-billed Gull': 197,
 'Ring-necked Duck': 198,
 'Rose-breasted Grosbeak': 199,
 'Rock Dove': 200,
 'Rock Wren': 201,
 'Ruby-throated Hummingbird': 202,
 'Ruby-crowned Kinglet': 203,
 'Ruddy Duck': 204,
 'Ruffed Grouse': 205,
 'Rufous Hummingbird': 206,
 'Rusty Blackbird': 207,
 'Sagebrush Sparrow': 208,
 'Sage Thrasher': 209,
 'Savannah Sparrow': 210,
 "Say's Phoebe": 211,
 'Scarlet Tanager': 212,
 "Scott's Oriole": 213,
 'Semipalmated Plover': 214,
 'Semipalmated Sandpiper': 215,
 'Short-eared Owl': 216,
 'Sharp-shinned Hawk': 217,
 'Snow Bunting': 218,
 'Snow Goose': 219,
 'Solitary Sandpiper': 220,
 'Song Sparrow': 221,
 'Sora': 222,
 'Spotted Sandpiper': 223,
 'Spotted Towhee': 224,
 "Steller's Jay": 225,
 "Swainson's Hawk": 226,
 'Swamp Sparrow': 227,
 "Swainson's Thrush": 228,
 'Tree Swallow': 229,
 'Trumpeter Swan': 230,
 'Tufted Titmouse': 231,
 'Tundra Swan': 232,
 'Veery': 233,
 'Vesper Sparrow': 234,
 'Violet-green Swallow': 235,
 'Warbling Vireo': 236,
 'Western Bluebird': 237,
 'Western Grebe': 238,
 'Western Kingbird': 239,
 'Western Meadowlark': 240,
 'Western Sandpiper': 241,
 'Western Tanager': 242,
 'Western Wood Pewee': 243,
 'White-breasted Nuthatch': 244,
 'White-crowned Sparrow': 245,
 'White-faced Ibis': 246,
 'White-throated Sparrow': 247,
 'White-throated Swift': 248,
 'Willow Flycatcher': 249,
 "Wilson's Snipe": 250,
 'Wild Turkey': 251,
 'Winter Wren': 252,
 "Wilson's Warbler": 253,
 'Wood Duck': 254,
 "Woodhouse's Scrub Jay": 255,
 'Wood Thrush': 256,
 'American Coot': 257,
 'Yellow-bellied Flycatcher': 258,
 'Yellow-bellied Sapsucker': 259,
 'Yellow-headed Blackbird': 260,
 'Mangrove Warbler': 261,
 'Myrtle Warbler': 262,
 'Yellow-throated Vireo': 263}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


SAMPLE_RATE=32000
NUM_CLASSES_SOUND = 264
PERIOD = 5

base_model = ResNet50(include_top=False, weights=None)
x = base_model.output
x = tf.reduce_mean(x, axis=2)
x1 = L.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
x2 = L.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
x = x1 + x2 
x = L.Dropout(0.5)(x)
x = L.Dense(1024, activation='relu')(x)
x = L.Dropout(0.5)(x)

norm_att = L.Conv1D(filters=NUM_CLASSES_SOUND, kernel_size=1, padding='same')(x)
norm_att = tf.keras.activations.tanh(norm_att/10)*10
norm_att = tf.keras.activations.softmax(norm_att, axis=-2)
segmentwise_output = L.Conv1D(filters=NUM_CLASSES_SOUND, kernel_size=1, padding='same', activation='sigmoid', name='segmentwise_output')(x)
clipwise_output = tf.math.reduce_sum(norm_att * segmentwise_output, axis=1)
clipwise_output = L.Lambda(lambda x: x, name="clipwise_output")(clipwise_output)
output = [segmentwise_output, clipwise_output]

sound_model = Model(inputs=base_model.input, outputs=output)
optimizer= tfa.optimizers.RectifiedAdam(
    lr=1e-3,
    total_steps=10000,
    warmup_proportion=0.1,
    min_lr=1e-8,
)

sound_model.compile(optimizer, loss=[None, "binary_crossentropy"],loss_weights=[0,1], metrics=[[],["accuracy", F1,true_positives,possible_positives,predicted_positives]])
sound_model.load_weights('sound_model.h5')

def audio_to_mel_spectrogram(audio):
    spectrogram = tfio.experimental.audio.spectrogram(audio, nfft=2048, window=2048, stride=320)
    mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=SAMPLE_RATE, mels=500, fmin=50, fmax=14000)
    dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
    return dbscale_mel_spectrogram

def mono_to_color(audio, eps=1e-6, img_size=224):
    X = audio
    X = tf.stack([X, X, X], axis=-1)

    mean = tf.math.reduce_mean(X)
    X = X - mean
    std = tf.math.reduce_std(X)
    Xstd = X / (std + eps)
    _min, _max = tf.math.reduce_min(Xstd), tf.math.reduce_max(Xstd)
    norm_max = _max
    norm_min = _min
    if (_max - _min) > eps:
        V = Xstd
        V = 255 * (V - norm_min) / (norm_max - norm_min)
    else:
        V = tf.zeros_like(Xstd)

    image = tf.image.resize(V, (img_size,img_size))
    return preprocess_input(image)

def upsample(x, ratio=72):
    (time_steps, classes_num) = x.shape
    upsampled = np.repeat(x, ratio, axis=0)
    upsampled = upsampled[2:-2]
    return upsampled

def prediction_for_clip(clip: np.ndarray, 
                        model: tf.keras.models.Model,
                        threshold=0.5, 
                        PERIOD = 5,
                        INTERVAL_RATE = 0.5,
                        OFFSET_LEGNTH = 0.01):
    audios = []
    LENGTH_THRESHOLD = 0.1
    y = clip.astype(np.float32)
    len_y = len(y)
    start = 0
    end = PERIOD * SAMPLE_RATE
    while True:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != PERIOD * SAMPLE_RATE:
            y_pad = np.zeros(PERIOD * SAMPLE_RATE, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
            break
        start = end - int(PERIOD * (1.0-INTERVAL_RATE) * SAMPLE_RATE)
        end = start + PERIOD * SAMPLE_RATE
        audios.append(y_batch)
        
    array = np.asarray(audios)

    estimated_event_list = []
    global_time = 0.0

    for audio in array:
        melspec = audio_to_mel_spectrogram(audio)
        image = mono_to_color(melspec)
        image = tf.expand_dims(image, axis=0)

        framewise_outputs, _ = model.predict(image)
        framewise_outputs = upsample(framewise_outputs[0])
        
        thresholded = framewise_outputs >= threshold
        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                pass
            else:
                detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                head_idx = 0
                tail_idx = 0
                while True:
                    if (tail_idx + 1 == len(detected)) or (
                            detected[tail_idx + 1] - 
                            detected[tail_idx] != 1):
                                
                        onset = OFFSET_LEGNTH * detected[head_idx] + global_time
                        offset = OFFSET_LEGNTH * detected[tail_idx] + global_time
                        onset_idx = detected[head_idx]
                        offset_idx = detected[tail_idx]
                        max_confidence = framewise_outputs[onset_idx:offset_idx, target_idx].max()
                        mean_confidence = framewise_outputs[onset_idx:offset_idx, target_idx].mean()
                                                    
                        estimated_event = {
                            "ebird_code": INV_BIRD_CODE[target_idx],
                            "onset": onset,
                            "offset": offset,
                            "max_confidence": max_confidence,
                            "mean_confidence": mean_confidence
                        }
                        if offset-onset > LENGTH_THRESHOLD or max_confidence > threshold * 1.5:
                            estimated_event_list.append(estimated_event)
                        else:
                            None
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1
        global_time += PERIOD*INTERVAL_RATE
        
    prediction_df = pd.DataFrame(estimated_event_list)
    print(prediction_df)
    if prediction_df.empty:
        return "Couldn't recognize the bird"
    return prediction_df['ebird_code'].value_counts().index[0]

bird_dict = {0: 'African crowned crane',
 1: 'African firefinch',
 2: 'Albatross',
 3: 'Alexandrine parakeet',
 4: 'American avocet',
 5: 'American bittern',
 6: 'American coot',
 7: 'American goldfinch',
 8: 'American kestrel',
 9: 'American pipit',
 10: 'American redstart',
 11: 'Anhinga',
 12: 'Annas hummingbird',
 13: 'Antbird',
 14: 'Araripe manakin',
 15: 'Asian crested ibis',
 16: 'Bald eagle',
 17: 'Bali starling',
 18: 'Baltimore oriole',
 19: 'Bananaquit',
 20: 'Bar-tailed godwit',
 21: 'Barn owl',
 22: 'Barn swallow',
 23: 'Barred puffbird',
 24: 'Bay-breasted warbler',
 25: 'Bearded barbet',
 26: 'Belted kingfisher',
 27: 'Bird of paradise',
 28: 'Black francolin',
 29: 'Black skimmer',
 30: 'Black swan',
 31: 'Black throated warbler',
 32: 'Black vulture',
 33: 'Black-capped chickadee',
 34: 'Black-necked grebe',
 35: 'Black-throated sparrow',
 36: 'Blackburniam warbler',
 37: 'Blue grouse',
 38: 'Blue heron',
 39: 'Bobolink',
 40: 'Brown noody',
 41: 'Brown thrasher',
 42: 'Cactus wren',
 43: 'California condor',
 44: 'California gull',
 45: 'California quail',
 46: 'Canary',
 47: 'Cape may warbler',
 48: 'Capuchinbird',
 49: 'Carmine bee-eater',
 50: 'Caspian tern',
 51: 'Cassowary',
 52: 'Chara de collar',
 53: 'Chipping sparrow',
 54: 'Chukar partridge',
 55: 'Cinnamon teal',
 56: 'Cock of the  rock',
 57: 'Cockatoo',
 58: 'Common grackle',
 59: 'Common house martin',
 60: 'Common loon',
 61: 'Common poorwill',
 62: 'Common starling',
 63: 'Couchs kingbird',
 64: 'Crested auklet',
 65: 'Crested caracara',
 66: 'Crow',
 67: 'Crowned pigeon',
 68: 'Cuban tody',
 69: 'Curl crested aracuri',
 70: 'D-arnauds barbet',
 71: 'Dark eyed junco',
 72: 'Downy woodpecker',
 73: 'Eastern bluebird',
 74: 'Eastern meadowlark',
 75: 'Eastern rosella',
 76: 'Eastern towee',
 77: 'Elegant trogon',
 78: 'Elliots  pheasant',
 79: 'Emperor penguin',
 80: 'Emu',
 81: 'Eurasian magpie',
 82: 'Evening grosbeak',
 83: 'Flame tanager',
 84: 'Flamingo',
 85: 'Frigate',
 86: 'Gambels quail',
 87: 'Gila woodpecker',
 88: 'Gilded flicker',
 89: 'Glossy ibis',
 90: 'Gold wing warbler',
 91: 'Golden cheeked warbler',
 92: 'Golden chlorophonia',
 93: 'Golden eagle',
 94: 'Golden pheasant',
 95: 'Golden pipit',
 96: 'Gouldian finch',
 97: 'Gray catbird',
 98: 'Gray partridge',
 99: 'Green jay',
 100: 'Grey plover',
 101: 'Guinea turaco',
 102: 'Guineafowl',
 103: 'Gyrfalcon',
 104: 'Harpy eagle',
 105: 'Hawaiian goose',
 106: 'Hooded merganser',
 107: 'Hoopoes',
 108: 'Hornbill',
 109: 'Horned guan',
 110: 'Horned sungem',
 111: 'House finch',
 112: 'House sparrow',
 113: 'Imperial shaq',
 114: 'Inca tern',
 115: 'Indian bustard',
 116: 'Indigo bunting',
 117: 'Jabiru',
 118: 'Javan magpie',
 119: 'Kakapo',
 120: 'Killdear',
 121: 'King vulture',
 122: 'Kiwi',
 123: 'Kookaburra',
 124: 'Lark bunting',
 125: 'Lears macaw',
 126: 'Lilac roller',
 127: 'Long-eared owl',
 128: 'Malabar hornbill',
 129: 'Malachite kingfisher',
 130: 'Maleo',
 131: 'Mallard duck',
 132: 'Mandrin duck',
 133: 'Marabou stork',
 134: 'Masked booby',
 135: 'Masked lapwing',
 136: 'Mikado  pheasant',
 137: 'Mourning dove',
 138: 'Myna',
 139: 'Nicobar pigeon',
 140: 'Northern bald ibis',
 141: 'Northern cardinal',
 142: 'Northern flicker',
 143: 'Northern gannet',
 144: 'Northern goshawk',
 145: 'Northern jacana',
 146: 'Northern mockingbird',
 147: 'Northern parula',
 148: 'Northern red bishop',
 149: 'Ocellated turkey',
 150: 'Okinawa rail',
 151: 'Osprey',
 152: 'Ostrich',
 153: 'Painted buntig',
 154: 'Palila',
 155: 'Paradise tanager',
 156: 'Parus major',
 157: 'Peacock',
 158: 'Pelican',
 159: 'Peregrine falcon',
 160: 'Philippine eagle',
 161: 'Pink robin',
 162: 'Puffin',
 163: 'Purple finch',
 164: 'Purple gallinule',
 165: 'Purple martin',
 166: 'Purple swamphen',
 167: 'Quetzal',
 168: 'Rainbow lorikeet',
 169: 'Razorbill',
 170: 'Red bellied pitta',
 171: 'Red faced cormorant',
 172: 'Red faced warbler',
 173: 'Red headed duck',
 174: 'Red headed woodpecker',
 175: 'Red honey creeper',
 176: 'Red throated bee eater',
 177: 'Red winged blackbird',
 178: 'Red wiskered bulbul',
 179: 'Ring-necked pheasant',
 180: 'Roadrunner',
 181: 'Robin',
 182: 'Rock dove',
 183: 'Rosy faced lovebird',
 184: 'Rough leg buzzard',
 185: 'Ruby throated hummingbird',
 186: 'Rufous kingfisher',
 187: 'Rufuos motmot',
 188: 'Sand martin',
 189: 'Scarlet ibis',
 190: 'Scarlet macaw',
 191: 'Shoebill',
 192: 'Smiths longspur',
 193: 'Snowy egret',
 194: 'Snowy owl',
 195: 'Sora',
 196: 'Spangled cotinga',
 197: 'Splendid wren',
 198: 'Spoon biled sandpiper',
 199: 'Spoonbill',
 200: 'Steamer duck',
 201: 'Stork billed kingfisher',
 202: 'Strawberry finch',
 203: 'Stripped swallow',
 204: 'Superb starling',
 205: 'Taiwan magpie',
 206: 'Takahe',
 207: 'Tasmanian hen',
 208: 'Teal duck',
 209: 'Tit mouse',
 210: 'Touchan',
 211: 'Townsends warbler',
 212: 'Tree swallow',
 213: 'Trumpter swan',
 214: 'Turkey vulture',
 215: 'Turquoise motmot',
 216: 'Varied thrush',
 217: 'Venezuelian troupial',
 218: 'Vermilion flycather',
 219: 'Violet green swallow',
 220: 'Wattled curassow',
 221: 'Whimbrel',
 222: 'White cheeked turaco',
 223: 'White necked raven',
 224: 'White tailed tropic',
 225: 'Wild turkey',
 226: 'Wilsons bird of paradise',
 227: 'Wood duck',
 228: 'Yellow cacique',
 229: 'Yellow headed blackbird'}

