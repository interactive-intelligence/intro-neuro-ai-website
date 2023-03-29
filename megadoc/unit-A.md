---
title: Unit A
parent: Megadoc
---

# Unit A: Gradient Descent Deep Dive

A deeper dive into Gradient Descent where you will be implementing backpropagation on your own! This is an involved unit that, while technically not required, will push your understanding of neural networks to the max.

**Task 1:** Watch the following video and implement _micrograd_ as specified:

[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

The templates we made for you can be found [here](https://github.com/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-A/starter_micrograd.ipynb).

Additionally, please implement the ReLU nonlinearity for the Value class. 

* (Note: if you're having a hard time with this, take a look at [this](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) code.)

Implement and train a small neural network using micrograd. The training, validation, and test data will be included in the starter code. - Try to find the best network you can!

You might want to change the learning rate, size of the network, or Note your training, validation, and test loss for your best network.


Synthesis Questions:

* `In what direction do gradients flow (with regards to loss)?`
* `How do gradients flow through addition? How do they flow through the ReLU function?`
* `What was your best loss for the test dataset?`
* `Was there something that stood out to you? Something that confused you?`
* `What's one resource that was helpful (suggested or found on your own)?`
