# Siamese Network | Question Duplicates | Coursera Assignment (Pytorch Implementation)

Siamese neural network is a class of neural network architectures that contain two or more identical sub networks. identical here means they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both sub networks.It is used find the similarity of the inputs by comparing its feature vectors.

It is best to describe what a Siamese network is through an example.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-PvIduqpwRi33YbyAk2G61lkPf7BVZIbU9aBcec4ZYDE1XyCvN0PtADBU1zqNqC4kcg9TJOeYIJqxj9m6lEwdeSOvMDx1tk1ye9bdw)

Note that in the first example above, the two sentences mean the same thing but have completely different words. While in the second case, the two sentences mean completely different things but they have very similar words.
Classification: learns what makes an input what it is.
Siamese Networks: learns what makes two inputs the same

Here are a few applications of siamese networks:

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-NYjZ19RZdWj5VxoG-8PW9uZkg-v7buY0Edg2WX9UBSBeyBwn9FjKZRH3AprKBRRnjweZtsipp8U0S4Dc7WnQtzQe411DcuSwUHXsM)

 
# Architecture

The model architecture of a typical siamese network could look as follows:

![](https://lh5.googleusercontent.com/mq5rono3IYQ-TJTvtt8IFDlukh2quL90ts4CXb36Do8EmHrqVBu-nHywgLIrboUd-0LUPYLjSaPJKmP7MTcXauiaefKIYCKjkvMYdDbVe51R3VMOlEuu7v-VP3D2yHqPzwBumj9s)

These two sub-networks are sister-networks which come together to produce a similarity score. Not all Siamese networks will be designed to contain LSTMs. One thing to remember is that sub-networks share identical parameters. This means that you only need to train one set of weights and not two.

The output of each sub-network is a vector. You can then run the output through a cosine similarity function to get the similarity score. In the next video, we will talk about the cost function for such a network.

# Cost Function

Let us take a close look at the following slide:

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-MvTR8IJpVV1FZu8uLu6dS-1md9DIk6qxkASMbTM4GE8dwhpjyOeo4RWsCSYTPmpLt0P-y1liTCT8OH0v5M8b9H1P4YGNgJgSBrVVA)


Note that when trying to compute the cost for a siamese network we use the triplet loss. The triplet loss usually consists of an Anchor and a Positive example. Note that the anchor and the positive example have a cosine similarity score that is very close to one. On the other hand, the anchor and the negative example have a cosine similarity score close to -1. Now we are ideally trying to optimize the following equation:

__−cos(A,P)+cos(A,N)≤0__

Note that if __cos(A,P) = 1__ and __cos(A,N) = -1__ then the equation is definitely less than 0. However, as cos(A,P) deviates from 1 and cos(A,N) deviates from -1, then you can end up getting a cost that is > 0. Here is a visualization that would help you understand what is going on. Feel free to play with different numbers.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-O2bRH27eBfjSjKVcf1nCXLskP93tAfDIblZJvxDtzUl8Ue594cjtmOpLoXV2nKemd8eZ8Ad-HYMGPRyU0yQqY0LoNlFgw0XZWz-0k)

# Triplets

We will now build on top of our previous cost function. To get the full cost function you will add a margin.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-OR-X8_XqZS6Kih_TW7IfuoSz0VdN1I6d7fgoZcXeiZvbw54M2CvpqZ3lVCTUAUl05hxuJV3K4LwZAjX2DetAJdEC9y7MLJhdp3rEs)

Note that we added an α in the equation above. This allows you to have a margin of "safety".  When computing the full cost, we take the max of that the outcome of __−cos(A,P)+cos(A,N)+α and 0__ . Note, we do not want to take a negative number as a cost.

Here is a quick summary:

* 𝜶: controls how far cos(A,P) is from cos(A,N)
* Easy negative triplet: cos(A,N) < cos(A,P)
* Semi-hard negative triplet:  cos(A,N) < cos(A,P) < cos(A,N) + 𝜶 
* Hard negative triplet: cos(A,P) < cos(A,N)

# Computing the Cost

To compute the cost, we will prepare the batches as follows:
![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-N7bJkj4ZuAaEBlxK-lvykLfGsw6aWFgGEFnUbQi5MXnAeTko7P2mLR1tJQsUoew6-xFCNh4iUmSDbuhxt7OSpQZoWjyvE62Ybm0y4)

Note that each example, has a similar example to its right, but no other example means the same thing. We will now introduce hard negative mining.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-MiDxzsaIxEZVxKQ_cy1_HQtRSXljjh9k_sPNwTaS4uNk3Vu4XV10a4knzrN0VqxozzVZNP6QHRvA61iSOYIOLEtik1jwszyFiALJE)

Each horizontal vector corresponds to the encoding of the corresponding question. Now when you multiply the two matrices and compute the cosine, you get the following:

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-Ol0JU9NbwaTP3pKT3onNktyambx0NvvJETK87U0ZZwBonv5Vgh8XuhfC0Q2A1BgAgXUwOshenTAGltCLPyNzRnZKPe_Nr6WZCtquo)

The diagonal line corresponds to scores of similar sentences, (normally they should be positive). The off-diagonals correspond to cosine scores between the anchor and the negative examples.

Now that you have the matrix with cosine similarity scores, which is the product of two matrices, we go ahead and compute the cost.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-OGP_OLQg8UfjHwcxt1StKGodkFywQjTNgZfEKQz7lINWiNhe0h7k-PsupXXyLXdj8CLUFd7GR14pmzbzbxXJECddeOJgllfesgyKo)

We now introduce two concepts, the mean_neg, which is the mean negative of all the other off diagonals in the row, and the closest_neg, which corresponds to the highest number in the off diagonals.

__Cost = max(−cos(A,P)+cos(A,N)+α,0)__

So we will have two costs now:

__Cost1 = max(−cos(A,P) + meanneg) + α , 0)__

__Cost2 = max(−cos(A,P) + closestneg + α , 0)__

The full cost is defined as: __Cost 1 + Cost 2.__

# One Shot Learning

Imagine you are working in a bank and you need to verify the signature of a check. You can either build a classifier with K possible signatures as an output or you can build a classifier that tells you whether two signatures are the same.

![](https://lh3.googleusercontent.com/keep-bbsk/AGk0z-OWkAnaBbsYGw4sgtOK690X4WRpHle_ZWABWmyB1PE7HxE13Xdji536lURjB4HlpNcyuKoF8iAyv0qKuBLs3I71uEfBwsICJQAA5rM)

Hence, we resort to one shot learning. Instead of retraining your model for every signature, you can just learn a similarity score as follows:

![https://lh3.googleusercontent.com/keep-bbsk/AGk0z-MyztOs8-fe4p0Dq8bMHPqRhKWtpfFOfPzzYa2syxKNHoLbsPBTXnJZBjVw-dz-d5XEUejHmwWWbed36fenETPxi88wnU2rFvW9kHQ] 

# Training / Testing

After preparing the batches of vectors, you can proceed to multiplying the two matrices. Here is a quick recap of the first step:

![](https://lh5.googleusercontent.com/lrZLjl-bWE7D5w72bPKBDTkM8vfxnaKo0ThJbv3Ys3AfUcsgwMLZv98dYHxd69ko31crjug-yivlk8iBom1HnFVhhJS7DiT0O0_BSRTnpcNpPTsE-xb4azOcOoc9wXcLWiu7zgfK)

The next step is to implement the siamese model as follows:

![](https://lh6.googleusercontent.com/5BJXNkYw7KuJTJJVx6fhCBKvS2GFwAozUESLS4BaUULQFf5x2XMivI1UGmh13nzNk8-N0EybPhho05hmZUcbSyEub2OVsG-GGq5H_yCJ2HxE_9-fvijlhaIMKUaTtMrYOnyvHWK-)

__Finally when testing:__
* Convert two inputs into an array of numbers
* Feed it into your model
* Compare 𝒗1,𝒗2 using cosine similarity
* Test against a threshold τ


