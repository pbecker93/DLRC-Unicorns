## DLRC for Volkswagen
For this challenge we used a *Lego Mindstorm*, *Nvidia Jetson TX2* and a *PS3EYE* to sort *LEGO* bricks by color. We had to detect the bricks, pick them up and put them in the correct boxes.

1. Python 3 Packages
* scipy
* opencv
* opencv-contrib
* tensorflow

2. Additional Ubuntu Packages
* Chillytag
We used a C++ library to generate easily recognizable tags. [Tutorial](https://github.com/pbecker93/PyChiliTagDetect).
* qv4l2
This application allows you to control the driver settings of the PS3EYE.
* Mosquito
Using Mosquito we were able to communicate from the jetson to the ev3.
* VGG16 Weights
Download the VGG16 weights [here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM).

### SRC
We can use the *._camera_calibration.py* scripts to make sure that the camera is positioned correctly on the robot. Executing run.py starts the model initialization and launches the policy. The *shut_down.py* allows you to reset the robot to the inital postion tso that you do not need to recalibrate all the time. The src consists of the following parts:

#### *camera*
This path contains the class that has control over the camera. It runs in a seperate thread when initialized and can be accessed from any of the other classes to request the current frame.

#### *color_cluster*
This class contains three different clustering algorithms, that can all be called in the */policy/detector.py* to determine the color of the lego bricks that are detected. 
* *mean_cluster.py* clusters bricks by just taking the average of the pixels in the crop (after some image preprocessing). This is completetely unsupervised but showed to be very sensitive to different lighting conditions.
* *unsupervisedClustering.py* tries three different clustering methods (Birch, MiniBatchKMeans, Agglomerative clustering) and then trains classifiers(Decision tree and KNN) on the boundaries of these clustering results. The dataset was made by ourselves. Unfortunately this method was not very robust but might improve if the dataset corresponded better to the experiment.
* *siamese_cluster.py* is the final method we used. It uses a siamese network to create a latent space from a dataset of brick and non-bricks. We then use simply check to which cluster the new brick is nearest to do the "unsupervised" clustering.

#### *fcn*
The vision system we implemented uses the first 4 layers of VGG16. By freezing these layers and adding 4 fresh layers, we enable our network to specialise for a specific task: finding Lego. By retrieving the activations of the final layers (filters) we can extract heatmaps displaying the lego and non-lego parts of an image. We can then use blob detection and tracking to make sure that we stay on the target. The nice thing about using vgg, is that you have a build in object detector. By extracting the activations from the 4th layer we have some sort of object detection. If we subtract the Lego-heatmap from this, we can track and avoid obstacles quite well. The model was trained by creating a dataset of images with noisy backgrounds, and images with bricks in them.

#### *mqtt_master*
The mosquitto client is used to communicate between the Jetson and ev3. We send commands through the *master.py* on the Jetson
to the *slave.py* on the ev3. So the ev3 only acts as a receiver for motor commands.

#### *policy*
The first goal of the policy is to detect a brick by scanning, moving towards it (e.g. keeping it in a narrow corridor in the field of view while moving) and picking it up. Before we close the gripper we continuously scan for objects between the gripper and when a blob is detected that is large enough we create a crop and send it to the cluster algorithm. Now that we have a brick we need to bring it to a box. This is done by scanning again, and looking for the chillytag corresponding with the cluster id. When a box is detected we move towards it in a similar fashion as before. When we are close enough, we approach and drop the brick in the box. We then turn around and start scanning for new bricks. Obstacle avoidance should be build in for both the brick and box part of the policy, but is still not very robust. This is mainly due to the obstacle detection from the vgg16 being quite noisy some times.

#### *tracking*
Tracking is done in simple way, we select the nearest brick and in the next frame find the marker closest to the old one.

#### *vae_clustering*
