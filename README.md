# Building your own Object Detector from scratch

This notebook has **educational purposes only**.

It shows how to build a simple **Object Detector** from scratch using **TensorFlow** & **Keras**.

The model is trained over the **Labeled Mask database**.

Check this story on Medium for more details: https://medium.com/@doleron/building-your-own-object-detector-from-scratch-bfeadfaddad8

---

## Getting the Data

Before running this notebook, it is necessary to download the archive file with the training images and respective annotation files.

The file is available on Kaggle: https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-yolo-darknet

Hit the download button and copy the `archive.zip` to the `data` folder. After that, run the next cell.

The previous cell uncompresses the `archive.zip` file and stores the images & text files in the `data/obj` folder. We can inspect the contents of those files as shown below:

Basically, the data consists of image files and respective annotation files. Each annotation file has one or more lines in which we can find the class and bounding box coordinates:

The following function filters the files, generating 3 lists:

- training files (70% of images)
- validation files (20% of images)
- test files (last 10%)

This is a small dataset. We have only **904 images** to train our model!

Let's combine image and annotations in order to check if we actually understand how to deal with the data.

In addition, let's format the images for the input size of our model (**244 x 244**).

The dataset is composed of different images of **people** using or not masks. Let's create a TensorFlow dataset with the images:

The training, validation & test datasets must be set up in order to follow the TensorFlow guideline (https://www.tensorflow.org/datasets/performances):

Let's inspect our training dataset:

Based on the image above, we can realize that the training dataset is correctly set. Blue rectangles represent bounding box of **unmasked people**. A green rectangle represents a masked person.

Once the data is ready, we can start our modeling and training.

---

## The Model

The Object Detector architecture was implemented **from scratch** using the **TensorFlow Functional API**. The model uses a custom architecture based on **ResNet50** as the backbone for feature extraction.

Furthermore, to enhance feature recalibration across channels, the **Squeeze and Excitation (SE) method** was integrated into the network architecture.

The complete model features two heads: one to classify the object type (or label/class) and the other to predict the object bounding box.

It is noteworthy that each network head has a specific set of loss functions and metrics:

---

## Model Training

Our model is all set and we can start the training, as shown in the next cell:

The training performance can be checked in the charts below.

### **Classifier Accuracy Conclusions**

* Train acc = 1.00 from approximately epoch 40 onward, indicating **potential overfitting**; validation does not drop.
* Val acc is unstable at the beginning, rising and falling sharply until epochs 30–40. Typical validation signal is small, with a strong data spike or high LR.
* From epoch 60 onward, it stabilizes at 0.93–0.95, describing very good final performance.
* Validation is noisy at the beginning, but later reaches the same accuracy as with the training data.

### **Classifier Loss Conclusions**

* Train loss drops smoothly to ~0 and remains very low, meaning the model learns the train very well (risk of overfitting).
* Val loss: hypervolatile between epochs 1–45 with peaks of up to 6, then drops sharply and from epochs 55–65 it remains stable around 0.25–0.35.
* Validation was unstable at first, but the model eventually stabilizes and generalizes.

### **Regressor Head (Bounding Box) Conclusions**

* The Train Loss drops very rapidly in the first 2–3 epochs, going from 0.6 to 0.05, and then remaining almost at 0.
* The Val loss starts at approximately 0.20–0.25, with slight peaks in the first 40 epochs, but then stabilizes very low (0.01–0.02) in the final epochs.
* Both train and val converge to very low values, demonstrating an **excellent fit for the regression task** (bounding boxes), with no signs of strong overfitting or divergence.
* The regressor head is very stable and does not exhibit the same instability as the classifier. The model learned the box coordinates well, and the gap between train and val is minimal, indicating that it generalizes well in this area.

---

## Evaluation

The best way to evaluate the end model performance is by using **IoU (Intersection over Union)** metrics. The following implementation of IoU was adapted from the PyImage Search website:

The model performance is finally evaluated on the test dataset. Green boxes **indicate** correct classification in which the predicted label (masked-unmasked) matches with the actual label. A red box indicates a wrong classification.

---

## **Grad-CAM Class Activation Visualization**

The **Grad-CAM** algorithm is a technique that produces a visual "heatmap" to show what parts of an image a Convolutional Neural Network (CNN) is focusing on to make a decision. It's an **interpretability tool** that helps us understand the "why" behind a prediction.

In essence, Grad-CAM (Gradient-weighted Class Activation Mapping) allows you to see which regions of an input image were most important for the model to classify it with a specific label.

The Grad-CAM algorithm

References:
- https://keras.io/examples/vision/grad_cam/
- https://www.pinecone.io/learn/class-activation-maps/

### **Conclusions**

Based on the Grad-CAM visualizations, we can see that the model primarily focuses on the **facial region** of the images when making its predictions. The heat maps show the areas that most influence the model's decision, and these areas consistently align with the location of the person's face, whether masked or unmasked. This indicates that the model effectively uses relevant visual information to classify the images and predict bounding boxes.
