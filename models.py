import os
import numpy as np 
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
    return prediction_df['ebird_code'].value_counts().index[0]

bird_dict = {
 'AFRICAN CROWNED CRANE': 0,
 'AFRICAN FIREFINCH': 1,
 'ALBATROSS': 2,
 'ALEXANDRINE PARAKEET': 3,
 'AMERICAN AVOCET': 4,
 'AMERICAN BITTERN': 5,
 'AMERICAN COOT': 6,
 'AMERICAN GOLDFINCH': 7,
 'AMERICAN KESTREL': 8,
 'AMERICAN PIPIT': 9,
 'AMERICAN REDSTART': 10,
 'ANHINGA': 11,
 'ANNAS HUMMINGBIRD': 12,
 'ANTBIRD': 13,
 'ARARIPE MANAKIN': 14,
 'ASIAN CRESTED IBIS': 15,
 'BALD EAGLE': 16,
 'BALI STARLING': 17,
 'BALTIMORE ORIOLE': 18,
 'BANANAQUIT': 19,
 'BAR-TAILED GODWIT': 20,
 'BARN OWL': 21,
 'BARN SWALLOW': 22,
 'BARRED PUFFBIRD': 23,
 'BAY-BREASTED WARBLER': 24,
 'BEARDED BARBET': 25,
 'BELTED KINGFISHER': 26,
 'BIRD OF PARADISE': 27,
 'BLACK FRANCOLIN': 28,
 'BLACK SKIMMER': 29,
 'BLACK SWAN': 30,
 'BLACK THROATED WARBLER': 31,
 'BLACK VULTURE': 32,
 'BLACK-CAPPED CHICKADEE': 33,
 'BLACK-NECKED GREBE': 34,
 'BLACK-THROATED SPARROW': 35,
 'BLACKBURNIAM WARBLER': 36,
 'BLUE GROUSE': 37,
 'BLUE HERON': 38,
 'BOBOLINK': 39,
 'BROWN NOODY': 40,
 'BROWN THRASHER': 41,
 'CACTUS WREN': 42,
 'CALIFORNIA CONDOR': 43,
 'CALIFORNIA GULL': 44,
 'CALIFORNIA QUAIL': 45,
 'CANARY': 46,
 'CAPE MAY WARBLER': 47,
 'CAPUCHINBIRD': 48,
 'CARMINE BEE-EATER': 49,
 'CASPIAN TERN': 50,
 'CASSOWARY': 51,
 'CHARA DE COLLAR': 52,
 'CHIPPING SPARROW': 53,
 'CHUKAR PARTRIDGE': 54,
 'CINNAMON TEAL': 55,
 'COCK OF THE  ROCK': 56,
 'COCKATOO': 57,
 'COMMON GRACKLE': 58,
 'COMMON HOUSE MARTIN': 59,
 'COMMON LOON': 60,
 'COMMON POORWILL': 61,
 'COMMON STARLING': 62,
 'COUCHS KINGBIRD': 63,
 'CRESTED AUKLET': 64,
 'CRESTED CARACARA': 65,
 'CROW': 66,
 'CROWNED PIGEON': 67,
 'CUBAN TODY': 68,
 'CURL CRESTED ARACURI': 69,
 'D-ARNAUDS BARBET': 70,
 'DARK EYED JUNCO': 71,
 'DOWNY WOODPECKER': 72,
 'EASTERN BLUEBIRD': 73,
 'EASTERN MEADOWLARK': 74,
 'EASTERN ROSELLA': 75,
 'EASTERN TOWEE': 76,
 'ELEGANT TROGON': 77,
 'ELLIOTS  PHEASANT': 78,
 'EMPEROR PENGUIN': 79,
 'EMU': 80,
 'EURASIAN MAGPIE': 81,
 'EVENING GROSBEAK': 82,
 'FLAME TANAGER': 83,
 'FLAMINGO': 84,
 'FRIGATE': 85,
 'GAMBELS QUAIL': 86,
 'GILA WOODPECKER': 87,
 'GILDED FLICKER': 88,
 'GLOSSY IBIS': 89,
 'GOLD WING WARBLER': 90,
 'GOLDEN CHEEKED WARBLER': 91,
 'GOLDEN CHLOROPHONIA': 92,
 'GOLDEN EAGLE': 93,
 'GOLDEN PHEASANT': 94,
 'GOLDEN PIPIT': 95,
 'GOULDIAN FINCH': 96,
 'GRAY CATBIRD': 97,
 'GRAY PARTRIDGE': 98,
 'GREEN JAY': 99,
 'GREY PLOVER': 100,
 'GUINEA TURACO': 101,
 'GUINEAFOWL': 102,
 'GYRFALCON': 103,
 'HARPY EAGLE': 104,
 'HAWAIIAN GOOSE': 105,
 'HOODED MERGANSER': 106,
 'HOOPOES': 107,
 'HORNBILL': 108,
 'HORNED GUAN': 109,
 'HORNED SUNGEM': 110,
 'HOUSE FINCH': 111,
 'HOUSE SPARROW': 112,
 'IMPERIAL SHAQ': 113,
 'INCA TERN': 114,
 'INDIAN BUSTARD': 115,
 'INDIGO BUNTING': 116,
 'JABIRU': 117,
 'JAVAN MAGPIE': 118,
 'KAKAPO': 119,
 'KILLDEAR': 120,
 'KING VULTURE': 121,
 'KIWI': 122,
 'KOOKABURRA': 123,
 'LARK BUNTING': 124,
 'LEARS MACAW': 125,
 'LILAC ROLLER': 126,
 'LONG-EARED OWL': 127,
 'MALABAR HORNBILL': 128,
 'MALACHITE KINGFISHER': 129,
 'MALEO': 130,
 'MALLARD DUCK': 131,
 'MANDRIN DUCK': 132,
 'MARABOU STORK': 133,
 'MASKED BOOBY': 134,
 'MASKED LAPWING': 135,
 'MIKADO  PHEASANT': 136,
 'MOURNING DOVE': 137,
 'MYNA': 138,
 'NICOBAR PIGEON': 139,
 'NORTHERN BALD IBIS': 140,
 'NORTHERN CARDINAL': 141,
 'NORTHERN FLICKER': 142,
 'NORTHERN GANNET': 143,
 'NORTHERN GOSHAWK': 144,
 'NORTHERN JACANA': 145,
 'NORTHERN MOCKINGBIRD': 146,
 'NORTHERN PARULA': 147,
 'NORTHERN RED BISHOP': 148,
 'OCELLATED TURKEY': 149,
 'OKINAWA RAIL': 150,
 'OSPREY': 151,
 'OSTRICH': 152,
 'PAINTED BUNTIG': 153,
 'PALILA': 154,
 'PARADISE TANAGER': 155,
 'PARUS MAJOR': 156,
 'PEACOCK': 157,
 'PELICAN': 158,
 'PEREGRINE FALCON': 159,
 'PHILIPPINE EAGLE': 160,
 'PINK ROBIN': 161,
 'PUFFIN': 162,
 'PURPLE FINCH': 163,
 'PURPLE GALLINULE': 164,
 'PURPLE MARTIN': 165,
 'PURPLE SWAMPHEN': 166,
 'QUETZAL': 167,
 'RAINBOW LORIKEET': 168,
 'RAZORBILL': 169,
 'RED BELLIED PITTA': 170,
 'RED FACED CORMORANT': 171,
 'RED FACED WARBLER': 172,
 'RED HEADED DUCK': 173,
 'RED HEADED WOODPECKER': 174,
 'RED HONEY CREEPER': 175,
 'RED THROATED BEE EATER': 176,
 'RED WINGED BLACKBIRD': 177,
 'RED WISKERED BULBUL': 178,
 'RING-NECKED PHEASANT': 179,
 'ROADRUNNER': 180,
 'ROBIN': 181,
 'ROCK DOVE': 182,
 'ROSY FACED LOVEBIRD': 183,
 'ROUGH LEG BUZZARD': 184,
 'RUBY THROATED HUMMINGBIRD': 185,
 'RUFOUS KINGFISHER': 186,
 'RUFUOS MOTMOT': 187,
 'SAND MARTIN': 188,
 'SCARLET IBIS': 189,
 'SCARLET MACAW': 190,
 'SHOEBILL': 191,
 'SMITHS LONGSPUR': 192,
 'SNOWY EGRET': 193,
 'SNOWY OWL': 194,
 'SORA': 195,
 'SPANGLED COTINGA': 196,
 'SPLENDID WREN': 197,
 'SPOON BILED SANDPIPER': 198,
 'SPOONBILL': 199,
 'STEAMER DUCK': 200,
 'STORK BILLED KINGFISHER': 201,
 'STRAWBERRY FINCH': 202,
 'STRIPPED SWALLOW': 203,
 'SUPERB STARLING': 204,
 'TAIWAN MAGPIE': 205,
 'TAKAHE': 206,
 'TASMANIAN HEN': 207,
 'TEAL DUCK': 208,
 'TIT MOUSE': 209,
 'TOUCHAN': 210,
 'TOWNSENDS WARBLER': 211,
 'TREE SWALLOW': 212,
 'TRUMPTER SWAN': 213,
 'TURKEY VULTURE': 214,
 'TURQUOISE MOTMOT': 215,
 'VARIED THRUSH': 216,
 'VENEZUELIAN TROUPIAL': 217,
 'VERMILION FLYCATHER': 218,
 'VIOLET GREEN SWALLOW': 219,
 'WATTLED CURASSOW': 220,
 'WHIMBREL': 221,
 'WHITE CHEEKED TURACO': 222,
 'WHITE NECKED RAVEN': 223,
 'WHITE TAILED TROPIC': 224,
 'WILD TURKEY': 225,
 'WILSONS BIRD OF PARADISE': 226,
 'WOOD DUCK': 227,
 'YELLOW CACIQUE': 228,
 'YELLOW HEADED BLACKBIRD': 229}

