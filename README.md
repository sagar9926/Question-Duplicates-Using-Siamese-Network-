# Siamese Network

Siamese neural network is a class of neural network architectures that contain two or more identical sub networks. identical here means they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both sub networks.It is used find the similarity of the inputs by comparing its feature vectors.

It is best to describe what a Siamese network is through an example.

![](https://lh5.googleusercontent.com/mq5rono3IYQ-TJTvtt8IFDlukh2quL90ts4CXb36Do8EmHrqVBu-nHywgLIrboUd-0LUPYLjSaPJKmP7MTcXauiaefKIYCKjkvMYdDbVe51R3VMOlEuu7v-VP3D2yHqPzwBumj9s)
Note that in the first example above, the two sentences mean the same thing but have completely different words. While in the second case, the two sentences mean completely different things but they have very similar words.
Classification: learns what makes an input what it is.
Siamese Networks: learns what makes two inputs the same
Here are a few applications of siamese networks:

 
Architecture

The model architecture of a typical siamese network could look as follows:

These two sub-networks are sister-networks which come together to produce a similarity score. Not all Siamese networks will be designed to contain LSTMs. One thing to remember is that sub-networks share identical parameters. This means that you only need to train one set of weights and not two.
The output of each sub-network is a vector. You can then run the output through a cosine similarity function to get the similarity score. In the next video, we will talk about the cost function for such a network.
