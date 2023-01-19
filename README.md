# Smartathon
This is the code implemantation of the team "The Boys"

[training_yolov5_potholes](training_yolov5_potholes)

    * In this approach we have taken a pre-trained yolov5 mdoel and trained the modelto detect
    the potholes form the 2D scene and build boundingboxes around it for further analysis

[training_yolov5_potholes_removing_background](training_yolov5_potholes_removing_background)
    
    * To have better accruacy than previous model used and remove confusion during model learning we
    used a pretrained model to remove background objects and then trained on potholes

[LaneDetector](LaneDetector)

    * For further increment in robustness of the model we used a road detction model and only taken
    the bounding boxes that appear on the road to remove confusion during inferencing

[EdgeDetection](EdgeDetection)

    * We have used three type of edge detector namely Laplacian, OTST, and Canny to detect the boundary
    of the potholes and there will be fine edges around a given pothole in the inference video