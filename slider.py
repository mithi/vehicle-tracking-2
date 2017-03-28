import numpy as np 
import cv2
from featuresourcer import FeatureSourcer
from binaryclassifier import BinaryClassifier

class Slider:
  
  def __init__(self, sourcer, classifier, increment):
    self.sourcer = sourcer
    self.classifier = classifier
    self.i = increment
    self.h = sourcer.s
    self.current_strip = None 
    
  def prepare(self, frame, y, ws):
    
    scaler = ws / self. h
    y_end = y + ws 
    w = np.int(frame.shape[1] / scaler)
    
    strip = frame[y: y_end, :, :]
    strip = cv2.resize(strip, (w, self.h))
    self.current_strip = strip 
    
    return scaler, strip

  def strip(self):
    return self.current_strip

  def locate(self, frame, window_size, window_position):
    
    y, ws = window_position, window_size 
    scaler, strip = self.prepare(frame, y, ws)
    
    boxes = []
    self.sourcer.new_frame(strip)
    
    x_end = (strip.shape[1] // self.h - 1) * self.h
        
    for resized_x in range(0, x_end, self.i):

      features = self.sourcer.slice(resized_x, 0, self.h, self.h)
    
      if self.classifier.predict(features): 
        x = np.int(scaler * resized_x)
        boxes.append((x, y, ws))
        
    return boxes