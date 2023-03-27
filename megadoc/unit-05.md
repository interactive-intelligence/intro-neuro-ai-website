---
title: Unit 05
parent: Megadoc
---

# Unit 5: Computer Vision 

Hello and welcome to the _Computer Vision (CV)_ section of the I2 megadoc! This section is, due to time constraints, only a very cursory glance at the foundations of CV. This is a whole subfield of ML and even if we spent 10 weeks on it, we wouldn’t scratch the surface!

Let’s start with some motivation. When you look at an image of (for example) a soda can, it does not matter where in the image the soda can is. You are able to detect it and know where it is. This detection ability is called _translational invariance_. You are able to detect an object even if it has been translated within an image. Traditional DNNs cannot do this (without being heavily overparameterized). Take a second to think about why the architecture of a DNN does not implicitly allow for translationally invariant object classification. 



* (Hint: Think about how different the vectorized images would be between the soda can image and its translated invariant. The inputs to the DNN would be drastically different and it would be hard to find any pattern!)

Convolutional Neural Networks (or CNNs) solve this problem and much more. To understand what a CNN is though, you must first understand what a convolution is!

**Task:** Watch and understand the following videos. We recommend taking notes and being able to answer the synthesis questions provided below. Send your I2 teacher/mentor/overlord the answers to the questions over Discord. **Watch up to <span style="text-decoration:underline;">13:42</span> in the video, anything after that is extra information not needed for Deep Learning.**

**Video:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA&t=773s) **(13 min)**

### `Synthesis Questions:`

* `What is the name for the smaller grid that convolves over a larger image?`
    * `Hint: Starts with a "k"`
* `What are some examples of what you can do to images if you convolve them with special matrices?`
* `How does Gaussian blur "work"?`
* `What is the name for the actual operation that occurs when the smaller grid is overlaid on the larger one?`
    * `When each element of the corresponding pixels are multiplied then summed.`
* `Give an example of a 3x3 matrix that would not do anything to the image it convolves over. Why does it not impact the image?`
    * `This is also known as the "do-nothing" matrix`


![alt_text](../assets/image6.gif)


Awesome job! Now we move onto integrating the concept of a convolution into a neural network.

**Task:** Read the following article, watch the video, and answer the synthesis questions: 

**Article:** [Comprehensive CNN Guide](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) **(15 min)**

**Video:** [Visualizing Convolutional Neural Networks \| Layer by Layer](https://www.youtube.com/watch?v=JboZfxUjLSk) **(5 min)**

### `Synthesis Questions:`


* `The architecture of a CNN is loosely based on what part of the brain?`
* `What is stride length?`
* `What is padding?`
    * `Why is padding useful?`
* `What is the objective of the convolutional layer in a CNN?`
* `What is the purpose of the pooling layer in a CNN?`
    * `What are the two ways to pool shown to you in the article?`
* `What is flattening and when is it done in a CNN?`
* `What is the purpose of the feedforward layer in a CNN?`
* `How do the convolutional layers before the feedforward layer in a CNN allow for higher accuracy?`

We have introduced you to the idea of a convolution and how convolutions are applied in CNNs. Can you begin to see how convolutions help with _translational invariance_? Think about it for a bit! Before the project, we just want to expose you to a few different types of convolutions. They aren’t all the same and serve different purposes.

**Task:** Read the following article for the following sections: 



1. **Convolution in Deep Learning**
2. **3D Convolution**
3. **Transposed Convolution/Deconvolution**

**Article:** [Comprehensive Convolution Types Guide](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215#:~:text=Convolution%20Arithmetic,Spatially%20Separable%20Convolution%2C%20Depthwise%20Convolution)) **(15 min)**

Awesome job! Feel free to move onto the project now.

For those of you more interested in CV, there are a bunch more things to do in this sphere. Here are some topics you can explore independently:



* **Transfer learning**
* **Object detection (r-cnn, yolo)**
* **Semantic segmentation (u-net, deeplab)**
* **Self-supervised learning (colorization, damage correction, noise decoding)**
* **Adversarial attacks**
* **Image generation - variational autoencoders, generative adversarial networks**

Here are some slides from a JC on Variational Autoencoders (somewhat related, but very cool!): [Variational Autoencoder JC Slides](https://docs.google.com/presentation/d/1KTb7wxnsBryuar-yB-AVrizw88Wc3Vue46iCwmN0558/edit?usp=sharing)



**Project Spec:**

The project for this “_Computer Vision_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!

A few helpful tips:



* Use GitHub, it’s really just better
* Use [Anaconda](https://www.anaconda.com/) with [Python3](https://www.python.org/downloads/) in [VSCode](https://code.visualstudio.com/).
    * If you use Anaconda, create a separate environment so you can mess with libraries and imports all day without screwing up your base environment.
* Type most of the code out yourself instead of just copying from the tutorial.
* Leave comments to cement your understanding. Link syntax to ideas.
* **Read up on what <span style="text-decoration:underline;">Fashion</span>-MNIST is (different than MNIST).**

**Clone the Git repo onto your local device if you have not already.**

Then, in your local copy of the GitHub repo, navigate to the unit-5 folder, and work on **conv-net.ipynb**. Instructions are in the Jupyter notebook. If you need help setting up your python environment, ask the TA’s!

**GH Link:** [Unit 5 Notebook](https://github.com/interactive-intelligence/intro-neuro-ai/blob/main/unit-05/conv-net.ipynb)** (1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas. 

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of Convolutional Neural networks! 

