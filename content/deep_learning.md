---
title: Deep Learning
parent: Core Content
nav_order: 3 ### Unit Number
---

# <u>Deep Learning</u>

Hello and welcome to the _Deep Learning_ section of the I2 course! Like the machine learning unit, we're going to split our content into literacy and technical understanding. 

## <u>Technical Track Content</u>

### **<u>Task 1:</u>** 

*Navigate to the relevant section of the I2 Grimoire using the link below. Read the textbook and answer all synthesis questions to the best of your ability. Be sure to save these somewhere for future reference.*

### [I2 Grimoire: Deep Learning](https://grimoire.uw-i2.org/units/Deep%20Learning.pdf) 

---

### **<u>Task 2:</u>** 

*Solve the coding challenges within the Jupyter notebook linked below (through Colab). If you encounter any issues with the notebook not functioning as described, please let us know!*

Please ask questions as you work through this project. Be sure to discuss with others in your group if you have one! Share your answers as you like, the goal is to learn and we’re not holding grades over your head.

In this project, you will be implementing a Deep Neural Network (DNN)! I would suggest to **read up on what [MNIST](https://wiki.pathmind.com/mnist) is** before starting since this is the data you will be working with.

**Colab Link:**
[Deep Learning Colab Template](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-03/mnist-dnn.ipynb) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of Deep Neural Network structure, how they learn, and how to create one using Python!

---
---
---

## <u>Literacy Track Content</u>

### **<u>Task 1:</u>** 

*Read the article below, and answer any synthesis questions placed along the way.*

Simply put, we use neural networks to make computers process information in a way similar to how the human brain processes information.  

Human brains use biological neural networks to process information. They send data from neuron to neuron in the form of electrical signals. A neural network attempts to replicate this processing using a computer. 

First, let’s look at the structure of a neural network. You may have seen an image like this before:

<div style="text-align:center">
    <img src="../assets/unit2/literacy_images/neural_network.png" alt="Diagram of a neural network" width="500"/>
</div>

The **<mark style="background-color: #d5a2fa">input layer</mark>** is where we feed in the information we want our network to process. We give it information in the form of a single column vector.

The **<mark style="background-color: #f7ee86">output layer</mark>** gives us the probabilities corresponding with our inputs, with one node corresponding to each outcome. For example, suppose we want our neural network to identify pictures of dogs. First, we’d input a picture of a dog (more on how inputs work later!). Our output layer would have two nodes: one representing the probability that the image was a dog, and one representing the probability that it wasn’t. Note that the values in the output nodes always sum up to one! If our neural network was, say 90% sure that the inputted image had a dog, the “yes dog” output node would output 90%, and the “not dog” output node would output 10%. 

Finally, the **<mark style="background-color: #f5a9e8">hidden layer</mark>** is where all the processing happens. Going back to our dog example, this is where the computer figures out whether the picture has a dog or not. It does this by identifying common features between all pictures that have dogs and distinguishing features between all pictures that don’t.

This hidden layer gets developed using **training data**. This is input data that’s been labeled with the “right answer”—in our dog analogy, it’s a bunch of images that are labeled “dog” or “not dog.” We feed one of these images into the neural network, let it make a prediction, and then tell it the right answer.

This is where the learning happens. The computer uses the right answer to “learn” from its mistakes and adjust the weights and connections between the nodes. It does this via a process called **backpropagation**, which we’ll learn more about in the rest of the homework.

This layer is called a hidden layer because often times, humans can’t understand the math involved in these computations. Remember, we aren’t telling the computer what makes a dog different from a non-dog: it makes these connections itself using math. These connections are so intricate that we humans can’t understand them, in the same way we can’t understand the fine details of how neurons transmit information in the human brain. Like the human brain, we can control the input state, or what information is provided, and we can give feedback on the output state, or what information is spit out.


Let’s look at an example. Imagine I want my computer to recognize handwritten numbers and convert them into text. How might we do that?

First of all, we need to give the computer some examples. We’re going to use the MNIST database, which contains thousands of images of handwritten digits, each one labeled with its actual number. Here’s an example ([Source](https://etzold.medium.com/mnist-dataset-of-handwritten-digits-f8cf28edafe)):

<div style="text-align:center">
    <img src="../assets/unit2/literacy_images/labeled_mnist.webp" alt="Labeled images from the MNIST database" width="500"/>
</div>

Let’s define our output layer. We want the output nodes to represent all possible outcomes. So instead of a yes-no output, like in our dog-not dog example, let’s ask the network to output the probability that each digit is represented—one node is the probability of a 0, one is the probability of a 1, and so on. This gives us 10 nodes in our output layer, each corresponding to a digit. 

Now let’s set up our input layer. As previously stated, the input has to be a single column vector of numbers. How do we convert these images to a vector?

We can assign each pixel in the image to a position in the vector. Our images are all 28x28 pixels, so positions 1-28 can represent the first row of pixels, positions 29-56 can represent the second row, and so on. Take a look at the image below for how this might work in a very simple 4x4 image. 

<div style="text-align:center">
    <img src="../assets/unit2/literacy_images/vector_flattening.png" alt="Diagram showing how a simple 4x4 image is flattened to a vector" width="500"/>
</div>

The numerical value for each pixel can correspond to the color of the pixel. All the pictures in our dataset are in grayscale, so each pixel will be assigned a number corresponding to how bright or dark it is—0 for totally white pixels, 1 for totally black pixels, and decimals in between based on how light or dark the pixel is. Then, we fill in the vector accordingly. The image below shows how this would work in the previous example—note that because we only have two colors, white and purple, we assign white pixels a 0 and purple pixels a 1. 

<div style="text-align:center">
    <img src="../assets/unit2/literacy_images/numerical_assignment.png" alt="Diagram showing how numerical values would be assigned to each pixel in the above image" width="500"/>
</div>

Now that we know how to input our images into the neural network, we can start training! The MNIST database has 60,000 designated training images. We’re going to flatten each one into a vector and feed it into the neural network. Then the neural network will give us an output back.

The first several are going to be pretty bad because the computer is still figuring out the connections between different images. But eventually it’ll start to realize that a “1” is usually a long vertical line, and an “8” is usually two ellipses on top of each other. As it starts to make these connections, the network makes better and better predictions.

Once we’ve exhausted our training data, and the computer has made all its connections, we can test our network. In addition to training data, the dataset has 10,000 designated testing images. Like the training data, these are also labeled: the difference is that the network doesn’t learn from them. We just use them to verify the accuracy of our network.

This was a good introduction. Now lets turn to some amazing videos on the topic of Neural Networks that will deepen your understanding. Watch the following videos and answer the associated questions.

### **Video 1:** [But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) **(20 min)**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/aircAruvnKk?si=RmnZFvn_yWmeKVse" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### *Synthesis Questions*
* `What is a neuron (in terms of Neural Networks) and what does its "activation" represent?`
    * `Bonus: Research and consider the correlation between a biological neuron and an artificial neuron. How are they similar/different?`
* `What is a network layer? How is it connected to other network layers?`
* `How is a picture of a digit decomposed into a network layer?`
* `What does the final layer of a neural network represent?`
* `What are weights? What are biases? Can you describe in English how information is passed from one layer to the next?`
* `A neural network **IS/IS NOT** just a very highly parameterized function (Choose one)`
* `What is the purpose of the sigmoid function?`

### **Video 2:** [Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piå) **(20 min)**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/IHZwWFHWa-w?si=Bojkhjorygxlds-6" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### *Synthesis Questions*
* `Why is there a need for a train/test split for a neural network? Why is it important for a NN to be able to generalize to examples it has not seen?`
* `Describe the Mean Squared Error (MSE) cost function. What does a higher value mean? What does a lower value mean? (For one training example)`
    * `Bonus: Assume you have a binary classifier neural network that outputs the vector [0.25, 0.75] and you are using the MSE Loss function to train the Network. The data label indicates that the output for this training example should have been [0, 1]. What is the MSE Loss for this training example?`
* `What is the gradient of a function? What is gradient descent?`
* `What does minimizing the loss function do to the network's performance over time?`
* `Do the hidden layers of a basic NN encode any useful information assuming you use the MSE Loss function? Why or why not?`

### **Video 3:** [Neural Networks and Deep Learning | Crash Course AI #3](https://youtu.be/oV3ZY6tJiA0?si=AYWscg7BYV8u7J7X) **(12 min)**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/oV3ZY6tJiA0?si=OgQblULj0RpyIT9C" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### *Synthesis Questions*
* `Why was ImageNet significant to the development of neural networks? What about AlexNet?`
* `What are some real-world applications of neural networks? What are some ethical considerations associated with them?`

---

### **<u>Task 2:</u>** 

*Complete the following writing activity.*

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* What can we learn from neuroscience to improve the efficiency and performance of artificial neural networks?
* What are the ethical implications of using insights from neuroscience to design artificial neural networks?
* How are ANNs inspired by the structure and function of neurons in the brain?
* What are some common applications of neural networks in real-world scenarios? Feel free to do some research on these!
* How do neural networks relate to the broader field of machine learning? What are their strengths and weaknesses compared to other algorithms?
* Reflecting on you learning from this unit, what is one thing you found to be most interesting about DNNs?
* What is one concept from this unit that you would like to learn more about and why?


<!-- **<mark style="background-color: lightblue">Homework Help:</mark>** if you’re having trouble with the technical homework, or just want to try a slightly easier version, try following along with this video! It references the Crash Course video from the synthesis questions, so make sure you watch that first. Reach out to a TA if you have any questions!

**[How to make an AI read your handwriting (LAB): Crash Course AI #5](https://youtu.be/6nGCGYWMObE?si=DHk1j96LDHETD-h4)** -->
<!---
# Old Course Content
Hello and welcome to the _Basics_ section of the I2 megadoc! We will start by throwing a few videos at you that we believe give incredibly intuitive explanations of one of the foundational building blocks of modern Deep Learning.

**Task:** Watch and understand the following videos. We recommend taking notes and being able to answer the synthesis questions provided below. Send your I2 teacher/mentor/overlord the answers to the questions over Discord.

**Video 1:**
[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) **(20 min)**


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
[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piå) **(20 min)**


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

If you would like to jump into the math further, here is a longer lecture for JC Varun did last year walking through how backpropagation is done:

**Part 1:**
[Backpropagation, Part 1 | I2 JC](https://www.youtube.com/watch?v=0HEd-ajSAZ4&t=781s)

**Part 2:**
[Backpropagation, Part 2 | I2 JC](https://www.youtube.com/watch?v=Hib0cFxJOFg)

---

# **Technical Project Spec:**

The project for this “_Basics_” section will **have you finish a code template through Google Colab.** Please ask questions as you work through this project. Be sure to discuss with others in your group if you have one! Share your answers as you like, the goal is to learn and we’re not holding grades over your head.

In this project, you will be implementing a Deep Neural Network (DNN)!

A few general helpful tips (if applicable):
* Be sure to appropriately make a copy of the Colab template before starting to save your progress!
* Renaming your copy to something that contains your name is a good idea, it will make it easier for us to review your submissions.
* Leave comments to cement your understanding. Link syntax to ideas.
* **Read up on what [MNIST](https://wiki.pathmind.com/mnist) is.**

Now, follow the instructions on this Jupyter notebook to implement some of the things we talked about. There is an "answers" link at the bottom of the notebook that you can use if stuck. You will need to download the '.ipynb' found in that directory and open it either locally or in a new colab project yourself. Ask around if you are unable to get it working!

**Colab Link:**
[Unit 2 Notebook](https://colab.research.google.com/drive/1R60m8LqXzZiia1vHB9nkSNsAKp7V6POJ?usp=sharing) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of Deep Neural Network structure, how they learn, and how to create one using Python!

# **Non-Technical Project Spec:**

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* What can we learn from neuroscience to improve the efficiency and performance of artificial neural networks?
* What are the ethical implications of using insights from neuroscience to design artificial neural networks?
* How are ANNs inspired by the structure and function of neurons in the brain?
* What are some common applications of neural networks in real-world scenarios? Feel free to do some research on these!
* How do neural networks relate to the broader field of machine learning? What are their strengths and weaknesses compared to other algorithms?
* Reflecting on you learning from this unit, what is one thing you found to be most interesting about DNNs?
* What is one concept from this unit that you would like to learn more about and why?

Be sure to submit your work through google drive using the submission form!
We would prefer that you upload it to your own Drive first, then use the submission form dropbox to connect that file to your submission!
--->
