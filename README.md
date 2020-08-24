# Frame Classifier

Automatic video frame classification using Computer Vision.

## Algorithm
   * Data Exploration - https://colab.research.google.com/drive/1-Nas6RX-8xK92VyS6exPeDfEnM4XziWl?usp=sharing
   * Deep learning  algorithm to classify the input image type.


## Usage

```python
 * Run python server.py
 * Open postman and create a post request
 * Goto localhost:5000/api/v1/getframetype
 * Send a image file in Body of the request
```

## Curl Command
```CURL Command
curl --location --request POST 'localhost:5000/api/v1/getframetype' 
--form 'file=<image file location>'

# Example
curl --location --request POST "localhost:5000/api/v1/getframetype" 
-F "file=@test/cgqt3xirYSM_key_frame_0_0.png"
*The above command tested in windows 8.1 machine with all dependencies installed.
```
## Results
 ```
 https://github.com/vivekmse0205/FrameClassifier/blob/master/frame_classifier_validation.ipynb
 ```
