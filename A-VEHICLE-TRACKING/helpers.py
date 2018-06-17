import cv2
import matplotlib.pyplot as plt

def convert(frame, src_model = "rgb", dest_model = "hls"):
    
    if src_model == "rgb" and dest_model == "hsv": 
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    elif src_model == "rgb" and dest_model == "hls":
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    elif src_model == "hsv" and dest_model == "rgb":
      frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    elif src_model == "hls" and dest_model == "rgb":
      frame = cv2.cvtColor(frame, cv2.COLOR_HLS2RGB)
    else: 
      raise Exception('ERROR:', 'src_model or dest_model not implemented')

    return frame


def show_images(imgs, per_row = 3, per_col = 2, W = 10, H = 5, tdpi = 80):
      
  fig, ax = plt.subplots(per_col, per_row, figsize = (W, H), dpi = tdpi)
  ax = ax.ravel()
  
  for i in range(len(imgs)):
    img = imgs[i]
    ax[i].imshow(img)
  
  for i in range(per_row * per_col):
    ax[i].axis('off')


def box_boundaries(box):
  x1, y1 = box[0], box[1]
  x2, y2 = box[0] + box[2], box[1] + box[2]  
  return x1, y1, x2, y2

def put_boxes(frame, boxes, color = (255, 0, 0), thickness = 10):
    
  out_img = frame.copy()

  for box in boxes:
    x1, y1, x2, y2 = box_boundaries(box)    
    cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)
    
  return out_img