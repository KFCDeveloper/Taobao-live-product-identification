# coding=utf-8
from Yolo_v3.usage_util import Read_Img_2_Tensor, Load_DeepFashion2_Yolov3, Draw_Bounding_Box
from cloth_detection import Detect_Clothes


def picture_deal(img_path):
    img = Read_Img_2_Tensor(img_path)
    model = Load_DeepFashion2_Yolov3()
    list_obj = Detect_Clothes(img, model)
    img_with_boxes = Draw_Bounding_Box(img, list_obj)
