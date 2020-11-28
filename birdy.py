import dima_pb2
from dima_pb2_grpc import *
from dima_pb2 import *
from concurrent import futures 
from models import image_model
from models import bird_dict
import cv2
import numpy as np 

class MyModelEndpointServicer(ModelEndpointServicer):
    def __init__(self, image_model, sound_model):
        super().__init__()
        self.image_model = image_model
        self.sound_model = sound_model

    def RecognizeBirdByPhoto(self, request, context):
        print('Initializing bird recognition by image')
        # print(request.data)
        image = cv2.imread(request.data)
        return dima_pb2.RecognizeBirdResponse(name=f"{bird_dict[self.predict_class(image)]}")


    def predict_class(self, image):
        image = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        image = np.expand_dims(image.astype('float32') / 255, axis=0)
        return np.argmax(self.image_model.predict(image))

    def is_bird_from_dataset(self, image):
        pass

    def RecognizeBirdBySound(self, request, context):
        print('Initializing bird recognition by sound')
        print(request.data)
        return dima_pb2.RecognizeBirdResponse(name='Pizda')



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ModelEndpointServicer_to_server(
        MyModelEndpointServicer(image_model, image_model), server)
    server.add_insecure_port('0.0.0.0:1488')
    server.start()
    server.wait_for_termination()

serve()