import dima_pb2
from dima_pb2_grpc import *
from dima_pb2 import *
from concurrent import futures 
from models import image_model
from keras.preprocessing.image import load_img

class MyModelEndpointServicer(ModelEndpointServicer):
    def __init__(self, image_model, sound_model):
        super().__init__()
        self.image_model = image_model
        self.sound_model = sound_model

    def RecognizeBirdByPhoto(self, request, context):
        print('Initializing bird recognition by image')
        # print(request.data)
        image = load_img(request.data, target_size=(112, 112))
        return dima_pb2.RecognizeBirdResponse(name=f"{self.predict_class(image)}")


    def predict_class(self, image):
        return self.image_model.predict(image)

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