# Gemini_Google_Hackathon_Repo
- a repo for submission of the project for the google ai hackathon.

## How to run:
First clone the repo or download it here, then you will need the following files:
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4.cfg](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg)
- and [efficientdet_d1_coco17_tpu-32](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md?plain=1) make sure the name matches.

Then open the repo in visual studio code or whatever coding IDE you want and run this pip statement:
```
pip install opencv-python numpy tensorflow google-generativeai scikit-learn deepface SpeechRecognition pydub requests spotipy elevenlabs
```
Finally run the main.py script and watch magic happen. Any questions or issues please let me know in the issues [here](https://github.com/TDWolff/Gemini_Google_Hackathon_Repo/issues)
