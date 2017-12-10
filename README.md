## A4: Machine Learning 
#### By: Johny Rufus & Peter Russell


## KNN

## Adaboost
#### Design Choices:
The primary design decisions we needed to make for our Adaboost algorithm were: 
* What would our decision stumps be? 
* Once these decision stumps, or weak classifiers, are created, how many would we use in our prediction?

#### What would our decision stumps be?
Each image was given to us in the format of a 192 dimension vector, representing an 8x8 pixel image where each color was given 3 values for red, green and blue values. There were 36,972 training images in total. With this information, we decided to create our decision stump formulation by comparing every color value against all others in the image to find which were most telling of a particular orientation. This amounted to roughly 36,672 (=192^2 - 192) different comparisons within an image.  

Specifically, we chose to count the number of instances a particular color value of a pixel was greater than the color value of another pixel. This resulted in a binary classification and we noticed no material change in results if less than was chosen as the decision stump. We decided to do this type of comparison to leverage the optimization that comes with numpy arrays. 

First, we collected all of the training images into a large numpy array (36,972 x 192) and filtered by each of the 4 orientations (0, 90, 180, 270). Within each orientation, we sliced the array on particular columns to give us the color values of two pixels that we're interested in comparing (36,972 x 1 vs. 36,972 x 1) to see if the first is greater than the second (element wise). This would give us an array (36,972 x 1) of 1 or 0 for True or False if the condition was met. With this information, we aggregated which pixel comparisons gave the highest rate of 1 (True) for a given orientation to find our strongest classifier in each iteration of the Adaboost algorithm. Then, with this accuracy rate, weights would be updated and we compared the weighted results again and continue finding classifiers via the Adaboost algorithm. 

#### Once these decision stumps, or weak classifiers, are created, how many would we use in our prediction?
In answering how many weak classifiers to use, we noticed accuracy plateued after 4-5 weak classifiers were used. The running time of the algorithm continued to increase as a function of classifiers, so accuracy demonstrated diminishing returns. Based off a client's objective of accuracy versus speed, we would make two recommendations with satisfy both needs. If a client's goal is to have a prediction that is generated fairly quickly with reasonable accuracy, we would recommend 4 weak classifiers. In our testing, this resulted in 67.9% accuracy in 158 seconds (2m38s). Conversely, if a client wishes to have maximum accuracy, we would recommend 11 weak classifiers as this gave us 70.5% accuracy in 375 seconds (6m15s). After this point, accuracy remained roughly the same with longer run time. 

![adaboost](https://github.iu.edu/storage/user/9000/files/9705896e-dd27-11e7-8f7b-e035689b873a)

#### Results: 
For the most part, our algorithm generally seemed to exhibit uniform errors in its predictions. It was best at guessing images that had 0 rotation in fact had 0 rotation. However, this appears to be a consequence of the algorithm guessing 0 too often perhaps as it was also its most concentrated area of errors for images that were actually rotated by 90 or 270 degrees. 

![confusion](https://github.iu.edu/storage/user/9000/files/ddefca5e-dd29-11e7-9289-33fb5a323000)

With respect to the actual classifiers chosen, we were pleased to find that as the sample size varied, our Adaboost implementation fine tuned the pixel value it used to decipher a general orientation, but remained where we would expect when trying to figure this out. Namely, the classifiers that were chosen were near the edges of the picture. This aligns with our intuition as we would inspect the top of an image first and if we saw a lot of green, possibly indicating grass, this image is probably upside down. This relationship can be seen in the following images below. 

![rot0](https://github.iu.edu/storage/user/9000/files/02797c28-dd32-11e7-9766-4230167726b8)
![rot90](https://github.iu.edu/storage/user/9000/files/ca31205e-dd32-11e7-9add-28c526b2aae2)
![rot180](https://github.iu.edu/storage/user/9000/files/ca4355ee-dd32-11e7-9f82-b1472b530e04)
![rot270](https://github.iu.edu/storage/user/9000/files/ca55616c-dd32-11e7-8f97-8c54958878de)

In our testing, we saw relative stability in the prediction rate versus sample size. However, one could expect these accuracies around a smaller sample sizes to be more variable depending on the section of the training data that is chosen.

![accuracy_size](https://github.iu.edu/storage/user/9000/files/dab952d0-dd36-11e7-8fd6-d315c990dfd3)

#### Sample Images:
When analyzing some of the images that were correctly classified, it becomes clear that the Adaboost algorithm with 4 classifiers does well in photos that have distinct color disparaties near the edges of the picture. In these four examples, there is a clear difference between the top of the picture (blue) and the bottom (green). This could prove to be a helpful decision stump for a particular orientation. 

![correct1](https://github.iu.edu/storage/user/9000/files/b809dcfc-dd39-11e7-8994-f66e33337837)
![correct2](https://github.iu.edu/storage/user/9000/files/b818e8a0-dd39-11e7-8853-38c5fadcea2d)
![correct3](https://github.iu.edu/storage/user/9000/files/b82ad1fa-dd39-11e7-8574-3c7dc983fdb2)
![correct4](https://github.iu.edu/storage/user/9000/files/b84f3c48-dd39-11e7-844b-7d3a4fd47989)

Here, one of the incorrectly classified images is a landscape, which we assumed would do well with Adaboost, but on further inspection it becomes clear why this image failed. In the corners, trees are shown that block the blue sky and give the classifier the impression that whole part of the picture is green. For the remaining images, there is little color disparity throughout the corners, so Adaboost struggled to classify these correctly.  
![incorrect1](https://github.iu.edu/storage/user/9000/files/b862f882-dd39-11e7-9882-457a86755fa1)
![incorrect2](https://github.iu.edu/storage/user/9000/files/b83d8dfe-dd39-11e7-8816-15f2e791aed5)
![incorrect3](https://github.iu.edu/storage/user/9000/files/b8758c86-dd39-11e7-9125-1e510ab33043)
![incorrect4](https://github.iu.edu/storage/user/9000/files/b885f09e-dd39-11e7-88c9-1827cab42684)









## Neural Net
