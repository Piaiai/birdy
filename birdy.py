import cv2
import librosa 
import dima_pb2
import numpy as np

from dima_pb2_grpc import *
from dima_pb2 import *
from concurrent import futures 
from models import image_model, sound_model
from models import bird_dict
from models import prediction_for_clip


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
        image = cv2.imread('bird.jpg')
        print("hi")
        return dima_pb2.RecognizeBirdResponse(name=f"{bird_dict[self.predict_class(image)]}")


    def predict_class(self, image):
        image = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image.astype('float32') / 255, axis=0)
        return np.argmax(self.image_model.predict(image))

    def is_bird_from_dataset(self, image):
        pass

    def RecognizeBirdBySound(self, request, context, SAMPLE_RATE=32000):
        print('Initializing bird recognition by sound')
        file = open("bird.mp3", "wb")
        file.write(request.data)
        file.close()
        sound, _ = librosa.load('bird.mp3', sr=SAMPLE_RATE, mono=True, res_type="kaiser_fast")
        response = prediction_for_clip(clip=sound, model=self.sound_model)
        return dima_pb2.RecognizeBirdResponse(response)



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ModelEndpointServicer_to_server(
        MyModelEndpointServicer(image_model, sound_model), server)
    server.add_insecure_port('0.0.0.0:1488')
    server.start()
    server.wait_for_termination()

serve()