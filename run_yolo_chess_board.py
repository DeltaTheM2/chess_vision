import cv2
import time
import numpy as np
# from skimage.transform import resize
import torch
from PIL import Image

from picamera2 import Picamera2

from myutils import get_reference_corners, calibrate_image, predict_yolo
        
corners_ref = get_reference_corners()

shape_ref = [480, 480] 

model_path = 'models/yolo5n_chess_pieces_rg.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

font = cv2.FONT_HERSHEY_SIMPLEX 
prev_frame_time = 0
new_frame_time = 0

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1.0, (480,480))

while True:
    new_frame_time = time.time() 
    img = picam2.capture_array()
    image_test = img[:,:,:3]

    image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    height, width = image_test.shape[:2]

    ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    
    ret, output_image = calibrate_image(image_test_rgb, corners_ref, shape_ref, height, width)
    
    if ret:
        predictions_bboxes, new_centers = predict_yolo(output_image, model, shape_ref)   
    
        for bbox in predictions_bboxes:
            if bbox[4] == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
        
            output_image = cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = str(int(fps))
    
    cv2.putText(output_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    
    cv2.imshow('img',output_image)
    # out.write(output_image)
    
    if cv2.waitKey(1) == ord('q'):
        break
 
# out.release()
cv2.destroyAllWindows()