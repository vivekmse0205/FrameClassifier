import os
import csv
import cv2
from sklearn import metrics
from video_classifier import VideoFrameClassifier
if __name__ == '__main__':
    mapper_dict = { 'codewalk' : 0,
                    'handwritten':1,
                    'misc':2,
                    'slides':3
                    }
    frame_type_mapper = {
        '0': 'codewalk',
        '1': 'handwritten',
        '2': 'misc',
        '3': 'slides'
    }
    model_path = 'models/epoch_18_224.pth'
    frame_classifier = VideoFrameClassifier(model_path)
    test_set_path =''
    with open(test_set_path,'r') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
    prediction_list =[]
    groundtruth_list = []
    for each_file in data:
        file_path = each_file[0]
        groundtruth = each_file[1]
        image = cv2.imread(file_path)
        frame_type = frame_classifier.classify_frame(image, ground_truth=None)
        prediction_list.append(frame_type)
        groundtruth_list.append(frame_type_mapper[str(groundtruth)])

    # Print the confusion matrix
    print(metrics.confusion_matrix(groundtruth_list, prediction_list))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(groundtruth_list, prediction_list, digits=3))