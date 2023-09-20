from ultralytics import YOLO
import torch


class ObjectDetectionModel:
    """
    This class is used to fine tune any (maybe?/else V8) YOLO model for object detection.
    """
    def __init__(self, model_name="yolov8n.pt", data="data.yaml", epochs=3):
        self.model_name = model_name
        self.data = "Dataset/" + data
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
        

def simplified():
    """
    This function is used to simplify the code.
    """
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=3)

def main():
    obm = ObjectDetectionModel()
    obm.get_pretrained_model()
    obm.fine_tune_model()



if __name__ == "__main__":
    main()