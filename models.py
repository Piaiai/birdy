import os
import numpy as np 

import tensorflow as tf
import tensorflow.keras.backend as K 
import tensorflow.layers as L 
import tensorfkiw_io as tfio  
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

base_model = ResNet50(include_top=False, weights=None)
x = base_model.output
x = tf.reduce_mean(x, axis=2)
x1 = L.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
x2 = L.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
x = x1 + x2 
x = L.Dropout(0.5)(x)
x = L.Dense(1024, activation='relu')(x)
x = L.Dropout(0.5)(x)

norm_att = L.Conv1D(filters=NUM_CLASSES, kernel_size=1, padding='same')(x)
norm_att = tf.keras.activations.tanh(norm_att/10)*10
norm_att = tf.keras.activations.softmax(norm_att, axis=-2)
segmentwise_output = L.Conv1D(filters=NUM_CLASSES, kernel_size=1, padding='same', activation='sigmoid', name='segmentwise_output')(x)
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

bird_dict = {0: 'AFRICAN FIREFINCH', 1: 'ALBATROSS', 2: 'ALEXANDRINE PARAKEET', 3: 'AMERICAN AVOCET', 4: 'AMERICAN BITTERN', 5: 'AMERICAN COOT',
             6: 'AMERICAN GOLDFINCH', 7: 'AMERICAN KESTREL', 8: 'AMERICAN PIPIT', 9: 'AMERICAN REDSTART', 10: 'ANHINGA', 11: 'ANNAS HUMMINGBIRD',
            12: 'ANTBIRD', 13: 'ARARIPE MANAKIN', 14: 'ASIAN CRESTED IBIS',
 15: 'BALD EAGLE',
 16: 'BALI STARLING',
 17: 'BALTIMORE ORIOLE',
 18: 'BANANAQUIT',
 19: 'BAR-TAILED GODWIT',
 20: 'BARN OWL',
 21: 'BARN SWALLOW',
 22: 'BARRED PUFFBIRD',
 23: 'BAY-BREASTED WARBLER',
 24: 'BEARDED BARBET',
 25: 'BELTED KINGFISHER',
 26: 'BIRD OF PARADISE',
 27: 'BLACK FRANCOLIN',
 28: 'BLACK SKIMMER',
 29: 'BLACK SWAN',
 30: 'BLACK THROATED WARBLER',
 31: 'BLACK VULTURE',
 32: 'BLACK-CAPPED CHICKADEE',
 33: 'BLACK-NECKED GREBE',
 34: 'BLACK-THROATED SPARROW',
 35: 'BLACKBURNIAM WARBLER',
 36: 'BLUE GROUSE',
 37: 'BLUE HERON',
 38: 'BOBOLINK',
 39: 'BROWN NOODY',
 40: 'BROWN THRASHER',
 41: 'CACTUS WREN',
 42: 'CALIFORNIA CONDOR',
 43: 'CALIFORNIA GULL',
 44: 'CALIFORNIA QUAIL',
 45: 'CANARY',
 46: 'CAPE MAY WARBLER',
 47: 'CAPUCHINBIRD',
 48: 'CARMINE BEE-EATER',
 49: 'CASPIAN TERN',
 50: 'CASSOWARY',
 51: 'CHARA DE COLLAR',
 52: 'CHIPPING SPARROW',
 53: 'CHUKAR PARTRIDGE',
 54: 'CINNAMON TEAL',
 55: 'COCK OF THE  ROCK',
 56: 'COCKATOO',
 57: 'COMMON GRACKLE',
 58: 'COMMON HOUSE MARTIN',
 59: 'COMMON LOON',
 60: 'COMMON POORWILL',
 61: 'COMMON STARLING',
 62: 'COUCHS KINGBIRD',
 63: 'CRESTED AUKLET',
 64: 'CRESTED CARACARA',
 65: 'CROW',
 66: 'CROWNED PIGEON',
 67: 'CUBAN TODY',
 68: 'CURL CRESTED ARACURI',
 69: 'D-ARNAUDS BARBET',
 70: 'DARK EYED JUNCO',
 71: 'DOWNY WOODPECKER',
 72: 'EASTERN BLUEBIRD',
 73: 'EASTERN MEADOWLARK',
 74: 'EASTERN ROSELLA',
 75: 'EASTERN TOWEE',
 76: 'ELEGANT TROGON',
 77: 'ELLIOTS  PHEASANT',
 78: 'EMPEROR PENGUIN',
 79: 'EMU',
 80: 'EURASIAN MAGPIE',
 81: 'EVENING GROSBEAK',
 82: 'FLAME TANAGER',
 83: 'FLAMINGO',
 84: 'FRIGATE',
 85: 'GAMBELS QUAIL',
 86: 'GILA WOODPECKER',
 87: 'GILDED FLICKER',
 88: 'GLOSSY IBIS',
 89: 'GOLD WING WARBLER',
 90: 'GOLDEN CHEEKED WARBLER',
 91: 'GOLDEN CHLOROPHONIA',
 92: 'GOLDEN EAGLE',
 93: 'GOLDEN PHEASANT',
 94: 'GOLDEN PIPIT',
 95: 'GOULDIAN FINCH',
 96: 'GRAY CATBIRD',
 97: 'GRAY PARTRIDGE',
 98: 'GREEN JAY',
 99: 'GREY PLOVER',
 100: 'GUINEAFOWL',
 101: 'GYRFALCON',
 102: 'HARPY EAGLE',
 103: 'HAWAIIAN GOOSE',
 104: 'HOODED MERGANSER',
 105: 'HOOPOES',
 106: 'HORNBILL',
 107: 'HORNED GUAN',
 108: 'HORNED SUNGEM',
 109: 'HOUSE FINCH',
 110: 'HOUSE SPARROW',
 111: 'IMPERIAL SHAQ',
 112: 'INCA TERN',
 113: 'INDIAN BUSTARD',
 114: 'INDIGO BUNTING',
 115: 'JABIRU',
 116: 'JAVAN MAGPIE',
 117: 'KAKAPO',
 118: 'KILLDEAR',
 119: 'KING VULTURE',
 120: 'KIWI',
 121: 'KOOKABURRA',
 122: 'LARK BUNTING',
 123: 'LEARS MACAW',
 124: 'LILAC ROLLER',
 125: 'LONG-EARED OWL',
 126: 'MALABAR HORNBILL',
 127: 'MALACHITE KINGFISHER',
 128: 'MALEO',
 129: 'MALLARD DUCK',
 130: 'MANDRIN DUCK',
 131: 'MARABOU STORK',
 132: 'MASKED BOOBY',
 133: 'MIKADO  PHEASANT',
 134: 'MOURNING DOVE',
 135: 'MYNA',
 136: 'NICOBAR PIGEON',
 137: 'NORTHERN CARDINAL',
 138: 'NORTHERN FLICKER',
 139: 'NORTHERN GANNET',
 140: 'NORTHERN GOSHAWK',
 141: 'NORTHERN JACANA',
 142: 'NORTHERN MOCKINGBIRD',
 143: 'NORTHERN PARULA',
 144: 'NORTHERN RED BISHOP',
 145: 'OCELLATED TURKEY',
 146: 'OKINAWA RAIL',
 147: 'OSPREY',
 148: 'OSTRICH',
 149: 'PAINTED BUNTIG',
 150: 'PALILA',
 151: 'PARADISE TANAGER',
 152: 'PARUS MAJOR',
 153: 'PEACOCK',
 154: 'PELICAN',
 155: 'PEREGRINE FALCON',
 156: 'PHILIPPINE EAGLE',
 157: 'PINK ROBIN',
 158: 'PUFFIN',
 159: 'PURPLE FINCH',
 160: 'PURPLE GALLINULE',
 161: 'PURPLE MARTIN',
 162: 'PURPLE SWAMPHEN',
 163: 'QUETZAL',
 164: 'RAINBOW LORIKEET',
 165: 'RAZORBILL',
 166: 'RED FACED CORMORANT',
 167: 'RED FACED WARBLER',
 168: 'RED HEADED DUCK',
 169: 'RED HEADED WOODPECKER',
 170: 'RED HONEY CREEPER',
 171: 'RED THROATED BEE EATER',
 172: 'RED WINGED BLACKBIRD',
 173: 'RED WISKERED BULBUL',
 174: 'RING-NECKED PHEASANT',
 175: 'ROADRUNNER',
 176: 'ROBIN',
 177: 'ROCK DOVE',
 178: 'ROSY FACED LOVEBIRD',
 179: 'ROUGH LEG BUZZARD',
 180: 'RUBY THROATED HUMMINGBIRD',
 181: 'RUFOUS KINGFISHER',
 182: 'RUFUOS MOTMOT',
 183: 'SAND MARTIN',
 184: 'SCARLET IBIS',
 185: 'SCARLET MACAW',
 186: 'SHOEBILL',
 187: 'SMITHS LONGSPUR',
 188: 'SNOWY EGRET',
 189: 'SNOWY OWL',
 190: 'SORA',
 191: 'SPANGLED COTINGA',
 192: 'SPLENDID WREN',
 193: 'SPOON BILED SANDPIPER',
 194: 'SPOONBILL',
 195: 'STEAMER DUCK',
 196: 'STORK BILLED KINGFISHER',
 197: 'STRAWBERRY FINCH',
 198: 'STRIPPED SWALLOW',
 199: 'SUPERB STARLING',
 200: 'TAIWAN MAGPIE',
 201: 'TAKAHE',
 202: 'TASMANIAN HEN',
 203: 'TEAL DUCK',
 204: 'TIT MOUSE',
 205: 'TOUCHAN',
 206: 'TOWNSENDS WARBLER',
 207: 'TREE SWALLOW',
 208: 'TRUMPTER SWAN',
 209: 'TURKEY VULTURE',
 210: 'TURQUOISE MOTMOT',
 211: 'VARIED THRUSH',
 212: 'VENEZUELIAN TROUPIAL',
 213: 'VERMILION FLYCATHER',
 214: 'VIOLET GREEN SWALLOW',
 215: 'WATTLED CURASSOW',
 216: 'WHIMBREL',
 217: 'WHITE CHEEKED TURACO',
 218: 'WHITE NECKED RAVEN',
 219: 'WHITE TAILED TROPIC',
 220: 'WILD TURKEY',
 221: 'WILSONS BIRD OF PARADISE',
 222: 'WOOD DUCK',
 223: 'YELLOW CACIQUE',
 224: 'YELLOW HEADED BLACKBIRD'}

