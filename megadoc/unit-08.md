---
title: Unit 08
parent: Megadoc
---

# Unit 8: Language Modeling 

Hello and welcome to the _Language Modeling (LM)_ section of the I2 megadoc! Language modeling is an incredibly pertinent field of deep learning that is focused on using statistical modeling to generate sequences of language tokens (I made this term very general on purpose). This can mean Q/A prompts/output like ChatGPT,  Seq2Seq models for machine translation, or sentiment analysis on text messages. Language Modeling, like Reinforcement Learning, is an incredibly large subfield that we simply cannot do justice in 1 unit. We will, however, try to introduce you to some basic language Modeling Deep Learning architectures that get more and more complex. We will also introduce you to _HuggingFace_, a service that allows you to learn about, train, load, finetune, and save DL models.

First, we will explore the concept of word embeddings. Understanding this is crucial to clearing up some FAQ about Language Modeling. So far most of the data you have worked with has been easily “castable” into n-dimensional vectors. However, when we are working with language, how can we convert a word like “King” into a vector? Performing computation directly on words is an incredibly difficult task. However, if we can “encode” the meaning of the word “King” into a vector consisting of numbers and have some way of converting between the two then the problem becomes easier.

**Task:** Read this “[Word Embeddings](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)” article and answer the following synthesis questions:

You may skip the “Recursive Neural Networks” subsection of the article if you wish


### `Synthesis Questions:`


* `What is a word embedding? How long are they usually?`
* `How does training a network to recognize the validity of 5-grams result in a Word-to-Vector "map"?`
    * `Can you think of another training method to achieve the same side-effect?`
* `Pretend you are a word embedder. Give examples (2-3 for each) of words in the same family as:`
    * `King`
    * `Button`
    * `Pain`
    * `Water Bottle`
* `What is the explanation given for the emergence of the "male-female difference vector"?`
* `What is pre training/transfer learning/multi-task learning?`

Now we go into Recurrent Neural Networks (RNN’s) and the concept of Backpropagation Through Time (BPTT) and its drawbacks. This builds off of Unit 2 (Deep learning) so feel free to revisit those if you are having trouble!

**Task:** Read this “[How RNN’s Work](https://blog.paperspace.com/recurrent-neural-networks-part-1-2/)” article and answer the following synthesis questions:

You may skip the “Word Embedding” and “Backward Pass” subsections of the article if you wish. There is a better resource below that explains the concept of a Backwards Pass in RNNs (BPTT)

**Helpful clarifications for after you read:**

* Note that what is “inside” the RNN unit (the circles on the computational graph) is modifiable, and I have attached here a simple look into the inside:

![alt_text](../assets/image7.png)

* I also have attached an optional reading about [Gated Recurrent Units](https://d2l.ai/chapter_recurrent-modern/gru.html) (GRU’s) here. “GRUs have the ability to keep memory/state from previous activations rather than replacing the entire activation like a vanilla RNN, allowing them to remember features for a long time and allowing backpropagation to happen through multiple bounded nonlinearities, which reduces the likelihood of the vanishing gradient” ([GRUs vs LSTMs](https://medium.com/paper-club/grus-vs-lstms-e9d8e2484848)).


### `Synthesis Questions:`

* `What does a language model, in general, try to predict?`
    * `Hint: Predicting X given Y. What are X and Y?`
* `What happens to the memory vector as we move through time?`
* `Describe how a RNN would deal with the sentence "How are you?" in terms of its unrolled computational graph`
    * `Basically, what happens to these words and the hidden states generated from these words?`
    * `How long would the unrolled computational graph be in terms of RNN nodes (circles)?`
* `What is the memory vector initialized to?`
* `What is &lt;\s>? What does it signify according to the article?`

**Task:** Read this “[Backpropagation Through Time](https://www.geeksforgeeks.org/ml-back-propagation-through-time/)” (BPTT) article, a small extension: “[Truncated BPTT](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)” **(read only section 2.8.6)** and answer the following synthesis questions:


### `Synthesis Questions:`

**TODO:** what are the sub tags?
* `What are W<sub>x</sub>, W<sub>y</sub>, and W<sub>s</sub>?`
* `Why does BPTT not work with a large number of timesteps?`
    * `What is this problem called?`
* `How does Truncated BPTT solve this problem?`

Finally we get into the Transformer architecture, which as of 2023 seems to have a grip on the DL market as the most generally powerful type of neural network used in all sorts of LM tasks. **This subsection is especially difficult. Please read the articles slowly and carefully. Be sure to ask for help if you have any questions or trouble understanding a statement!**

**Task:** Read the following “Intuitive Explanation of GPT Models” [Part 1](https://cswartout.com/2022/11/25/intutive-explanation-of-gpt.html) and [Part 2](https://cswartout.com/2022/12/25/intuitive-explanation-of-gpt-part-2.html) (and hopefully part 3 soon!) (Made by University of Washington Interactive Intelligence member Carter!) and answer the following synthesis questions:

**In case this article is not enough, or you find yourself struggling to fully understand transformers, take a look at this list of other links that can reinforce your knowledge:**



* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) Another good explanation of Transformers.
* [GPT in 60 lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) - explains GPT in a concise way using code!
* [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper annotated with PyTorch.
* [NanoGPT](https://github.com/karpathy/nanoGPT), simple example code for a GPT model.


### `Synthesis Questions:`

* `Why does not being able to capture long-term dependencies result in nonsensical generated paragraphs (like clicking autocomplete)?`
* `What do the probabilities that GPT outputs represent, and what is greedy decoding?`
* `What is a token?`
* `What are the big blocks that make up the GPT architecture?`
* `What are the two main blocks inside a transformer-decoder block?`
* `Describe masked self-attention in your own words (not including the vector math).`
    * `How would this help stop generated sequences from being mostly nonsensical?`
* `How are Query, Key, and Value vectors generated from each word embedding?`
* `How is the score for each word calculated using one word's query vector and all the other words' key vectors?`
* `How is the score for each word and its value vector used to create the vector for a single transformed word? (In the case of the article, the word is "He").`
* `Briefly describe multi-headed attention`

Awesome job answering those synthesis questions for Transformers! Now we move onto the project (which won't be nearly as difficult)

---

# **Project Spec:**

The project for this “_Language Modeling_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!

A few helpful tips:



* Use GitHub, it’s really just better
* Use [Anaconda](https://www.anaconda.com/) with [Python3](https://www.python.org/downloads/) in [VSCode](https://code.visualstudio.com/).
    * If you use Anaconda, create a separate environment so you can mess with libraries and imports all day without screwing up your base environment.
* Leave comments to cement your understanding. Link syntax to ideas.

**Clone the Git repo onto your local device if you have not already.**

**<span style="text-decoration:underline;">There are 2 parts (.ipynb files) to this unit. Finish both.</span>**

Now, follow the instructions on this Jupyter notebook to implement some of the things we talked about. There is an "answers" directory on the same level as this notebook that you can use if stuck. You will need to set up a Python environment to run this notebook. Ask around if you are unable to get it working!

I would recommend downloading the entire "notebooks" folder all at once. Its one folder up.

**GH Link:** [Unit 8 Notebook Part 1](https://github.com/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/hf_tutorial.ipynb) **(1 hr)**

Now navigate to the application portion of this project, where you are given a dataset and asked to train an LLM of your choice to emulate Shakespeare! Be sure to reference your Unit 8 Notebook Part 1 to figure out how to do this. The starter code is in **lm_starter_code.ipynb**

**GH Link:** [Unit 8 Notebook Part 2](https://github.com/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/lm_starter_code.ipynb) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas. 

Congratulations! You now understand the basics of Language Modeling!
