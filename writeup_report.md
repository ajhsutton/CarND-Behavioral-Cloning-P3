# **Behavioral Cloning** 

This report describes the process of developing a Neural Network capable of driving a simulated vehicle around a track. A Deep Neural Network was implemented, utilizing transfer learning from the VGG19 network with custom input pre-processing and output prediction layers. The model input is an image from the simulator, while the output is a steering angle prediction to steer the vehicle.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimages/CenterDriving.png "Center Driving"
[image2]: ./writeupimages/SteeringCommand.png "Steering Angle"
[image3]: ./writeupimages/MultiCamera.png "Multiple Views"
[image4]: ./writeupimages/recovery1.png "Recovery Image"


## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* SDC_Behavioual_Cloning_Data_Viz.ipynb is a notebook used to create and train the model.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Jupyter notebook file "SDC_Behavioual_Cloning_Data_Viz.ipynb" contains the code used to train the model. The notebook allows combining multiple data sets, from the sub-folder BCdata, into a single data set for training. Several stages of data visualization are performed, along with the training and saving the convolution neural network.

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a series of neural networks. For this project I decided to experiement with transfer learning, using the VGG19 Neural Network that was trained on the Imagenet data set. Keras provides VGG19 as a pre-trained network as "keras.applications.vgg19.VGG19". 

The VGG network is described in: "Very Deep Convolutional Networks for Large-Scale Image Recognition"  (https://arxiv.org/abs/1409.1556).

Pre-processing was performed using two keras lambda layers.
1. Normalization: the data input range was normalized to zero mean with a range of (-0.5,0.5).
2. Image Cropping: 40 pixels from the top (predominantly sky and background) and 30 pixels from the bottom (vehilce hood) were cropped. 

The cropped image was then passed through the VGG19 network's convolution layers. The output from the Convolutional layers were passed to a 1D Convolution (256 filters) then GlobalAveragePooling was applied. The resulting (flat) tensor was processed through a dense layer with 128 elements, and ReLu activation. Finally, a dense output layer with linear activation was used to predict the steering angle.  

#### 2. Attempts to reduce overfitting in the model

To prevent over-fitting, 20% of the data set was used as a validation set, along with a keras Checkpoint monitor to avoid overfittinf once the validation loss is minimized. Additionally, Dropout was experimented with however was not found to significantly improve performance.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used the rmsprop optimizer. Initially 10 Epochs were specified, however it was found that the model validation loss converged after 5 Epochs, hence the checkpoint-saved model for 5 epochs was used in the vidoe.

#### 4. Appropriate training data

Training data was generated using the provided simulator. Two complete laps of the track in the anti-clockwise and one lap in the clockwise direction were recorded. In addition, data describing recovery of the vehicle from the edge of the road was reocrded. For the recovery data, the car was positioned at the edge/off the road prior to recording. A record of (only) the recovery driving manouver was then saved. Likewise, additional training data was recorded to focus on the final two corners of the track, a sharp left then right turn.  

The data was augmented using the left and right camera images, where the steering angle was increased/decreased by 0.3 respectively. 

Finally, all training data was flipped left-right along with the steering angle to prevent biases towards left turns, due to the anti-clockwise track direction.

This code is implemented as part of the notebook: "SDC_Behavioual_Cloning_Data_Viz".

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I decided to investigate Transfer Learning for this task. The network architecture required development of input and output layers to match the provided data to the VGG19 pre-processing requirements, as well as conversion of the VGG19 convolutional (2D) tensor outputs to predict the steering angle.

I experimented with a number of output network sizes starting from 64 hidden units in the output dense network. Various input cropping's were also used. 

Due to the pre-trained VGG19 network providing robust spatial features, the simple output layer structure prevented significant overfitting (low degrees of freedom), so long as appropriate monitorion of the RMS validation error was used (ie. fit function callback and model saving).

When run on the track, early models were found to be unable to turn sharp corners (such as the final left and right). TO overcome this limitation, dedicated training data was gathered from 'normal' driving of those corners, along with recovery data as described previously. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
The final network structure (along with output tensor sizes) was:

| Model | Layers |
---
| Pre-Processing | lambda_6_input (None, 160, 320, 3) |
| Pre-Processing | lambda_6 (None, 160, 320, 3) |
| Pre-Processing | cropping2d_6 (None, 90, 320, 3) |
| VGG19 | block1_conv1 (None, 90, 320, 64) |
| VGG19 | block1_conv2 (None, 90, 320, 64) |
| VGG19 | block1_pool (None, 45, 160, 64) |
| VGG19 | block2_conv1 (None, 45, 160, 128) |
| VGG19 | block2_conv2 (None, 45, 160, 128) |
| VGG19 | block2_pool (None, 22, 80, 128) |
| VGG19 | block3_conv1 (None, 22, 80, 256) |
| VGG19 | block3_conv2 (None, 22, 80, 256) |
| VGG19 | block3_conv3 (None, 22, 80, 256) |
| VGG19 | block3_conv4 (None, 22, 80, 256) |
| VGG19 | block3_pool (None, 11, 40, 256) |
| VGG19 | block4_conv1 (None, 11, 40, 512) |
| VGG19 | block4_conv2 (None, 11, 40, 512) |
| VGG19 | block4_conv3 (None, 11, 40, 512) |
| VGG19 | block4_conv4 (None, 11, 40, 512) |
| VGG19 | block4_pool (None, 5, 20, 512) |
| VGG19 | block5_conv1 (None, 5, 20, 512) |
| VGG19 | block5_conv2 (None, 5, 20, 512) |
| VGG19 | block5_conv3 (None, 5, 20, 512) |
| VGG19 | block5_conv4 (None, 5, 20, 512) |
| VGG19 | block5_pool (None, 2, 10, 512) |
| Output | conv2d_1 (None, 2, 10, 256) |
| Output |global_average_pooling2d_1 (None, 256) |
| Output |dense_1 (None, 128) |
| Output |dense_2 (None, 1) |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps anti-clockwise and one lap clockwise on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

The data set also recorded images from a left- and right-hand side camera positions. An example of the Left, Center and Right camera angles is shown below. 

![alt text][image3]

These additional camera positions could be used to augment the data set. By assuming that the left/right camera position represented an error, relative to a center mounted camera, an additional steering command could be associated with each of the left- and right-camera images. For training, the steering angle for the left camera added "0.3" to the reference (center) steering angle, while the right camera image used an offset of "-0.3". The magnitude of the steering angles across all frames is shown below.

![alt text][image2]

Additional recovery images were recorded, where the recording was started with the vehicle in an undesirable/bad position, such as as the edge of the road:

![alt text][image4]

After all collection process, I had 22684 raw frames of data. With data augmentation, this number for model training was multiplied by x3, due to left + right cameras and x2 due to left-right flippling (performed in the data generator function). A validation set of 20% was used to model tuning, thus the final model was trained on approximately 108,000 images and steering positions.
