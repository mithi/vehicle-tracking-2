## PROJECT 5: VEHICLE TRACKING
For this project I have written a software pipeline to detect and track vehicles in a video. 

- Performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
- Normalized these features and randomized a selection for training and testing a classifier
- The classifier trained and used for prediction is a Linear SVM classifier 
- Implemented a sliding-window algorithm and used my trained classifier to search for vehicles in images
- Ran my pipeline on a video stream 
- Created a heatmap of recurring detections frame by frame to reject outliers and follow detected images
- Estimated a bounding box for vehicles detected 


## WRITEUP
- WRITEUP.pdf

## TESTS 
- feature_sourcer_test.ipynb
- classifier_test.ipynb
- slider_test2.ipynb
- heatmap_test.ipynb

## CLASSES 
- featuresourcer.py
- binaryclassifier.py
- slider.py
- heatmap.py

## PIPELINE 
- classifier_training.ipynb
- pipeline.ipynb
- project_video_output.mp4
- project_video_output_conser.mp4

## PROBLEMS ENCOUNTERED 
I had three major headaches in this project. The first is that It was hard to estimate the position of the “strip” (y_start) of the window I will perform the sliding-window technique as well as the size of the window. A large window like 240 pixels by 240 pixels would perform consistent false positives. The 2nd major problem was tweaking the “memory” (buffer of frames) and “threshold” of the HeatMap in order to reject false positives but not reject true positives. The biggest pain was that extracting the HOG features, it was really time consuming, so it took me a long time to see the results of tweaking the parameters. Even up to now, it can be tweaked further for improvements.
