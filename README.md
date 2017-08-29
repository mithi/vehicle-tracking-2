------

Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Optimized and evaluated the model on video data from a automotive camera taken during highway driving.

------
[MEDIUM: VEHICLE TRACKING ](https://medium.com/@mithi/vehicles-tracking-with-hog-and-linear-svm-c9f27eaf521a)
------

**Version 2 uses the following parameters for feature extraction**

```
feature_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             # 64 pixels x 64 pixel image
  'number_of_orientations': 11,        # 6 - 12
  'pixels_per_cell': 16,               # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}
# [3 x 3 block positions] x [2 x 2 cells per block] x [11 orientations] x [3 channels] = 1,188 features
```

**Version 1 uses the following parameters for feature extraction**
```
feature_params = {
  'color_model': 'hls',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             # 64 pixels x 64 pixel image
  'number_of_orientations': 12,        # 6 - 12
  'pixels_per_cell': 8,                # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}

# [7 x 7 block positions] x [2 x 2 cells per block] x [12 orientations] x [3 channels] = 7,056 features
```


## SAMPLE USAGE

To run any notebook properly, copy the jupyter notebooks from the `/ipynb` folder to the root directory.
This is so that each notebook sees relevant files, the most relevant files being the python classes.

- classifier_training.ipynb
- feature_sourcer_test.ipynb
- classifier_test.ipynb
- slider_test.ipynb
- heatmap_test.ipynb
- pipeline.ipynb

## CLASSES
- featuresourcer.py
- binaryclassifier.py
- slider.py
- heatmap.py

## VERSION 1
- https://github.com/mithi/vehicle-tracking/
- https://github.com/mithi/vehicle-tracking/blob/master/WRITEUP.pdf
