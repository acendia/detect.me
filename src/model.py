from ultralytics import YOLO
import torch


class Model:
    """
    This class is used to fine tune any (maybe?/else V8) YOLO model for object detection.
    """
    def __init__(self, model_name="../model/yolov8n.pt", data="../dataset/data.yaml", epochs=5):
        self.model_name = model_name
        self.data = data
        self.epochs = epochs

    def get_pretrained_model(self,):
        """
        This function is used to get the pretrained model from the ultralytics repo.
        """
        self.model = YOLO(self.model_name)
        

    def fine_tune_model(self,):
        """
        This function is used to fine tune the model.
        """
        self.model.train(data=self.data, epochs=self.epochs)
    
    def freeze(self, freeze = 10):
        """
        This function is used to freeze the model.
        """
        freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False
