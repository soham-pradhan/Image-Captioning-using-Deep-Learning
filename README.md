# Image-Captioning-using-Deep-Learning
The aim of this project is to detect abstract objects in an image and then give description about those object as seen
in the image.
Using the Flickr 8k dataset where each image consists of 5 captions.
Used Inceptionv3 model for getting the feature vector by removing its softmax layer.
Used LSTM for generating captions.
Achieved BLEU score of 0.61.

# Flickr8k Dataset access

Send a request in the below link to download Flickr_8k_dataset https://illinois.edu/fb/sec/1713398

You will be receiving an email to download the dataset. There are two zip file

Flickr8k_Dataset.zip [Images] place all the images into data/images folder
Flickr8k_text.zip [captions] place all the captions into data/caption folder

# Where to store

Except img_cap_load.py and img_cap_text.py, store every file in htdocs.
Create a folder called BE project and save all the dataset files and the above mentioned files in it.
Image Augmentation is used, so mirrored images of 6k training images are added to the trainig set.





