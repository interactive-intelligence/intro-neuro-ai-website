---
title: Unit 02
parent: Megadoc
---

# Unit 2: The (deep learning) Basics 

Hello and welcome to the _Basics_ section of the I2 megadoc! We will start by throwing a few videos at you that we believe give incredibly intuitive explanations of one of the foundational building blocks of modern Deep Learning.

**Task:** Watch and understand the following videos. We recommend taking notes and being able to answer the synthesis questions provided below. Send your I2 teacher/mentor/overlord the answers to the questions over Discord.

**Video 1:** 
[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1&t=637s) **(20 min)**


### `Synthesis Questions:`



* `What is a neuron (in terms of Neural Networks) and what does its "activation" represent?`
    * `Bonus: Research and consider the correlation between a biological neuron and an artificial neuron. How are they similar/different?`
* `What is a network layer? How is it connected to other network layers?`
* `How is a picture of a digit decomposed into a network layer?`
* `What does the final layer of a neural network represent?`
* `What are weights? What are biases? Can you describe in English how information is passed from one layer to the next?`
* <code>A neural network <strong>IS/IS NOT</strong> just a very highly parameterized function (Choose one)</code>
* `What is the purpose of the sigmoid function?`

Great job! Now onto video two. Remember that all questions should be answered thoroughly and with a “because” if you can. Just to clarify some language:

**Artificial Neural Network (ANN):** A network as described in the video, but with just one hidden layer

**Deep Neural Network:** An ANN but with multiple hidden layers. When we say NN in this context, this is usually what we talk about.

**Fully Connected NN:** Specifying that every single possible connection is made between adjacent layers (this is implicit to the networks shown in the video, but is not always the case!)

**Video 2:**
[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2) **(20 min)**


### `Synthesis Questions:`


* `Why is there a need for a train/test split for a neural network? Why is it important for a NN to be able to generalize to examples it has not seen?`
* `Describe the Mean Squared Error (MSE) cost function. What does a higher value mean? What does a lower value mean? (For one training example)`
    * `Bonus: Assume you have a binary classifier neural network that outputs the vector [0.25, 0.75] and you are using the MSE Loss function to train the Network. The data label indicates that the output for this training example should have been [0, 1]. What is the MSE Loss for this training example?`
* `What is the gradient of a function? What is gradient descent?`
* `What does minimizing the loss function do to the network's performance over time?`
* `Do the hidden layers of a basic NN encode any useful information assuming you use the MSE Loss function? Why or why not?`

Great job! These last two videos definitely enter into more theoretical/difficult content, so be prepared. This is something you SHOULD have questions about, so **post at least 1 in the Discord!**

**Video 3:**
[What is backpropagation really doing? | Chapter 3, Deep learning](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3) **(15 min)**


### `Synthesis Question:`



* `Describe Gradient Descent, and how it works in principle with a Deep Neural Network.`

The last video may require understanding of **partial derivatives** (a MATH 126 concept) to fully understand. If you understand the basics of derivatives, however, this should not be too big a leap in understanding. There are no synthesis questions for this last video. Just watch and absorb!

**Video 4:**
[Backpropagation calculus | Chapter 4, Deep learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4) **(10 min)**

If you would like to jump into the math further, here is a longer lecture for JC I did last year walking through how backpropagation is done: 

**Part 1:**
[Backpropagation, Part 1 | I2 JC](https://www.youtube.com/watch?v=0HEd-ajSAZ4&t=781s)

**Part 2:**
[Backpropagation, Part 2 | I2 JC](https://www.youtube.com/watch?v=Hib0cFxJOFg)

---

# **Project Spec:**

The project for this “_Basics_” section will be following the tutorial below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!

A few helpful tips:



* Use GitHub, it’s really just better
* Use [Anaconda](https://www.anaconda.com/) with [Python3](https://www.python.org/downloads/) in [VSCode](https://code.visualstudio.com/). I personally create .py files but Jupyter Notebooks and Google Colab are also very powerful. 
    * For a simple project like this though, powerful computing is unnecessary and you can figure out the details of those other technologies next week
    * If you use Anaconda, create a separate environment so you can mess with libraries and imports all day without screwing up your base environment.
* Leave comments to cement your understanding. Link syntax to ideas.
* **Read up on what [MNIST](https://wiki.pathmind.com/mnist) is.**

**Clone the Git repo onto your local device if you have not already.**

Then, in your local copy of the GitHub repo, navigate to the unit-2 folder, and work on **mnist-dnn.ipynb**. Instructions are in the Jupyter notebook. If you need help setting up your python environment, ask the TA’s!

**GH Link:**
[Unit 2 Notebook](../../notebooks/unit-02/mnist-dnn.ipynb) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas. 

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of Deep Neural Network structure, how they learn, and how to create one using Python!
