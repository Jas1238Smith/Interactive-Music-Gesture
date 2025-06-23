# Interactive-Music-Gesture
 
This application captures gestures in Python and maps them to musical output in Pure Data.

First, it uses the OpenCV and Mediapipe Python Libraries to recognize and store hand position data.

Next, it sends that data via Open Sound Control (OSC) to audio filter effects in a Pure Data patch.

Download Pure Data: [https://puredata.info/downloads/pure-data]