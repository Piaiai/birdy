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

BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

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

