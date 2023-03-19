Traffic Signals Detection

Intro
With the evolution of machine learning and computer vision, the machine has achieved the ability to detect a cat from a dog and so on. What about traffic signals(Stop, One Way Street, etc)? This project is an attempt at 
creating a model that can locate the traffic signs in the image, and categorize the sign.

Dataset
The dataset has been generated by webscraping. The webpage of choice is autoshkolla.com with hundreds of real
life and simulated images of traffic signs and traffic situations. The dataset is located in the Images folder in route directory.

The approach
Since there are two different tasks(to find and to categorize) there are to different approaches. To Find the 
traffic signs in the image we will use the histograms of oriented gradients, to create the gradients.
For the detection of the signs we will use CNN for image classification.