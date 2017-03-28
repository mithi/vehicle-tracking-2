from scipy.ndimage.measurements import label
from helpers import box_boundaries
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class HeatMap:

  def __init__(self, frame, memory, thresh):
    
    self.blank = np.zeros_like(frame[:, :, 0]).astype(np.float)
    self.map = np.copy(self.blank)
    self.thresholded_map = None
    self.labeled_map = None
    self.samples_found = 0
    self.thresh = thresh
    self.memory = memory
    self.history = []

  def reset(self):
    self.map = np.copy(self.blank)
    self.history = []

  def do_threshold(self):
    self.thresholded_map = np.copy(self.map)
    self.thresholded_map[self.map < self.thresh] = 0
        
  def get(self):
    self.do_threshold()
    self.label()
    return self.map, self.thresholded_map, self.labeled_map
      
  def remove(self, boxes):
    for box in boxes: 
      x1, y1, x2, y2 = box_boundaries(box)    
      self.map[y1: y2, x1: x2] -= 1
      
  def add(self, boxes): 
    for box in boxes: 
      x1, y1, x2, y2 = box_boundaries(box)
      self.map[y1: y2, x1: x2] += 1

  def update(self, boxes):
    
    if len(self.history) == self.memory:
      self.remove(self.history[0])
      self.history = self.history[1:]
    
    self.add(boxes)
    self.history.append(boxes)

  def label(self):
    labeled = label(self.thresholded_map)
    self.samples_found = labeled[1]
    self.labeled_map = labeled[0]

  def draw(self, frame, color = (0, 225, 0), thickness = 10):
    
    this_frame = frame.copy()
    _, _, this_map = self.get()
    
    for n in range(1, self.samples_found + 1):
      coords =  (this_map == n).nonzero()
      xs, ys = np.array(coords[1]), np.array(coords[0])
      p1 = (np.min(xs), np.min(ys))
      p2 = (np.max(xs), np.max(ys))
      cv2.rectangle(this_frame, p1, p2, color, thickness)
    
    return this_frame

  def show(self, frame, tdpi = 80):
      
    mp, tmp, lmp = self.get()
    labeled_img = self.draw(frame)
    
    fig, ax = plt.subplots(1, 4, figsize = (15, 8), dpi = tdpi)
    ax = ax.ravel()

    ax[0].imshow(np.clip( mp, 0, 255), cmap = 'hot')
    ax[1].imshow(np.clip(tmp, 0, 255), cmap = 'hot')
    ax[2].imshow(np.clip(lmp, 0, 255), cmap = 'gray')
    ax[3].imshow(labeled_img)

    for i in range(4):
      ax[i].axis('off')