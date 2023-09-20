from ultralytics import YOLO
import torch
from model import Model

def main():
    # run = 0
    # obm = ObjectDetectionModel(model_name=f"runs/train{run}/weights/best.pt") # train fine tuned model.

    obm = Model() 
    obm.get_pretrained_model()  # get the pretrained model.
    obm.freeze()    # freeze the 10 first layers (backbone).
    obm.fine_tune_model()   # fine tune the model.


if __name__ == "__main__":
    main()