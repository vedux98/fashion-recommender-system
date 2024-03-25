#Exploring imagenet model 
Context: 
The experience of users in e-commerce is constantly evolving and improving as technology advances, 
One crucial aspect of this improvement is the refinement of user behavior analysis. 
understanding and examining user behavior in order to provide more personalized recommendations for products and services, 
With the increasing diversity of services and user groups, it becomes even more important to offer refined and tailored recommendations to users.
Problem: 
A fashion recommender system and presenting a learning curve while creating the system, the use cases are quite vast in terms of designing a feature related to user behaviour or providing users to explore their sense of fashion. 
To build the system i users refines techniques to build on the exist model and extracting features on created dataset.
Functions: 
Using resnet50 convolution neural network model which has 50 deep layers and can be pretrained on more than a million images from image net.
Using linear algebra module to extract features from the dataset and normalizing the results.
Stream lit module to deploy our model on the web, it is a important library to test our system accurately.
Usage:
App.py containing code for traning data and evaluating model (image 1)
Test.py containing code for feature extration the the dataset 
Frs.py contains code to test the model on the web using steamlit on by using (sample) acting as testing dataset (use the folder for testing model)
