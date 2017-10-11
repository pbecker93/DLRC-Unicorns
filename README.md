## DLRC for Volkswagen
For this challenge we used a *Lego Mindstorm*, *Nvidia Jetson TX2* and a *PS3EYE* to sort *LEGO* bricks by color. We had to detect the bricks, pick them up and put them in the correct boxes.

1. Python 3 Packages
* scipy
* opencv
* opencv-contrib
* tensorflow

2. Additional Ubuntu Packages
* Chillytag
We used a C++ library to generate easily recognizable tags. [Tutorial https://github.com/pbecker93/PyChiliTagDetect].
* qv4l2
This application allows you to control the driver settings of the PS3EYE.
* Mosquito
Using Mosquito we were able to communicate from the jetson to the ev3.

### SRC
We can use the *._camera_calibration.py* scripts to make sure that the camera is positioned correctly on the robot. Executing run.py starts the model initialization and launches the policy. The *shut_down.py* allows you to reset the robot to the inital postion tso that you do not need to recalibrate all the time. The src consists of the following parts:

#### *camera*
This path contains the class that has control over the camera. It runs in a seperate thread when initialized and can be accessed from any of the other classes to request the current frame.

#### *color_cluster*
This class contains three different clustering algorithms, that can all be called in the */policy/detector.py* to determine the color of the lego bricks that are detected. 
* *mean_cluster.py* clusters bricks by just taking the average of the pixels in the crop (after some image preprocessing). This is completetely unsupervised but showed to be very sensitive to different lighting conditions.
* *shay_cluster.py* uses a histogram based knn-method to classify. The model was pretrained on a dataset we made ourselves. Unfortunately this method was not very robust but might improve if the dataset corresponded better to the experiment.
* **
#### *fcn*

#### *mqtt_master*

#### *policy*
#### *tracking*
#### *vae_clustering*
