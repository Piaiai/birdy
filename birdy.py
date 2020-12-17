import cv2
import librosa 
import dima_pb2
import numpy as np

from dima_pb2_grpc import *
from dima_pb2 import *
from concurrent import futures 
from models import image_model, sound_model, model_resnet
from models import bird_dict
from models import prediction_for_clip
from pydub import AudioSegment


class MyModelEndpointServicer(ModelEndpointServicer):
    def __init__(self, image_model, sound_model):
        super().__init__()
        self.image_model = image_model
        self.sound_model = sound_model

    def RecognizeBirdByPhoto(self, request, context):
        print('Initializing bird recognition by image')
        #print(request.data)
        file = open("bird.jpg", "wb")
        file.write(request.data)
        file.close()
        src = cv2.imread("bird.jpg")
        image = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
        print("hi")
        resp = dima_pb2.RecognizeBirdResponse(name=f"{bird_dict[self.predict_class(image)]}")
        print(resp)
        return resp


    def predict_class(self, image):
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image.astype('float32') / 255, axis=0)
        return np.argmax(self.image_model.predict(image))

    def is_bird_from_dataset(self, image):
        pass

    def RecognizeBirdBySound(self, request, context, SAMPLE_RATE=32000):
        print('Initializing bird recognition by sound')
        file = open("bird.mp4", "wb")
        file.write(request.data)
        file.close()
        sound = AudioSegment.from_file("bird.mp4", "mp4")
        sound.export("bird.wav", format="wav")
        print("before librosa")
        sound = np.zeros((1,1))
        try:
            sound, sr = librosa.load('bird.wav')
            print(sr)
            print(sound.shape)
        except RuntimeError as e: 
            print("runtime error")
        else:
            print("hello")
            #, sr=SAMPLE_RATE, mono=True, res_type="kaiser_fast")
        finally:
            print("hello.1")
        print("after librosa")
        response = prediction_for_clip(clip=sound, model=self.sound_model, PERIOD=2)
        print(response)
        print("hi")
        print(len(response))
        return dima_pb2.RecognizeBirdResponse(name=response)



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ModelEndpointServicer_to_server(
        MyModelEndpointServicer(model_resnet, sound_model), server)
    server.add_insecure_port('0.0.0.0:1488')
    print("Server is starting")
    server.start()
    server.wait_for_termination()

serve()