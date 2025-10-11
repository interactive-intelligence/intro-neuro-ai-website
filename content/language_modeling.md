---
title: Language Modeling
parent: Core Content
nav_order: 8 ### Unit Number
---

# <u>Language Modeling</u>

Welcome to the Language Modeling section of the I2 Course! Here you will learn 
about the core concepts underlying textual GenAI tools that took the world by
storm. Hopefully you can reason about these models better after learning more 
about how they work!

## <u>Technical Track Content</u>

### **<u>Task 1:</u>** 

*Navigate to the relevant section of the I2 Grimoire using the link below. Read the textbook and answer all synthesis questions to the best of your ability. Be sure to save these somewhere for future reference.*

### [I2 Grimoire: Language Modeling](https://grimoire.uw-i2.org/units/Language%20Modeling.pdf) 

---

### **<u>Task 2:</u>** 

*Solve the coding challenges within the Jupyter notebook linked below (through Colab). If you encounter any issues with the notebook not functioning as described, please let us know!*

Please ask questions as you work through this project. Be sure to discuss with others in your group if you have one! Share your answers as you like, the goal is to learn and we’re not holding grades over your head.

**There are 2 parts (.ipynb files) to this unit. Try to finish both.** This technical project is likely to be harder than anything you have done in this course before, so be patient with it and reach out if you need support! In the first part, you wll be learning how to work with the HuggingFace API.

**Colab Link:** [Language Modeling Colab Notebook Part 1](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/hf_tutorial.ipynb) **(1 hr)**

Now navigate to the application portion of this project (Part 2 below), where you are given a dataset and asked to train an LLM of your choice to emulate Shakespeare! Be sure to reference your the above notebook to figure out how to do this.

**Colab Link:** [Language Modeling Colab Notebook Part 2](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/lm_starter_code.ipynb) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of HuggingFace and Language Modeling!

---
---
---

## <u>Literacy Track Content</u>

### **<u>Task 1:</u>** 

*Read the article below, and answer any synthesis questions placed along the way.*

This article will cover the idea of language modeling and how computers process human language.

Put simply, the goal of language modeling is to predict the next word in a sentence using information about the definitions and grammatical rules of particular words as well as the contexts in which they appear. 

For example consider this sentence:
> I want to cook an omelet, so I went to the store to buy some ___

What comes next in this sentence? Chances are, you said “eggs.” There are plenty of “correct” answers—maybe you were out of salt and pepper—but it’s the most likely answer. Based on the sentence, we know that whatever comes next should be a noun phrase, and it’s probably related to the omelet we’re going to cook, and it’s something you can buy in a store. Given all that information, we conclude that we can fill in the blank with “eggs.” The goal of language modeling is to do something similar—that is, predict the next word in these sentences using probabilistic information about the sentence. 

There are two main types of language models: **statistical language models** and **neural language models**. 

Statistical language models use statistics and probability directly to predict the likely next word in a sentence of phrase. They generally get these statistics from a sample set of data. Based on this data, the model can identify patterns in the text and come up with predictions. 

Statistical language models usually take the form of an **n-gram model**. This model predicts the probability of a word in a sequence given the n previous words in the sequence. For example, a unigram predicts the probability of a word given the immediate previous word; a bigram predicts the probability given the two previous words; a trigram uses the three previous words, and so on. 

However, statistical language models have their limitations. For one, they will struggle with new words or phrases that don’t appear often in the original set of data (for example, if the phrase “time complexity” rarely appears in the original data, the model may struggle to predict that the word “complexity” can follow the word “time”). In addition, because these models look back at a fixed number of words, they can struggle to track and consider the long-term effects of a word on a phrase. 

This is where the other type of model, neural language models, come in. Neural language models use neural networks to predict the next word in a sequence. These models are able to handle more complex and diverse sets of training data and are better at handling context clues and long-term effects of words. 

We’ll continue to discuss neural language models in greater detail through these videos. Specifically, we’ll look at two types of neural language models: **recurrent neural networks** and **transformers**. Watch the following videos!

### **Video 1:** [Illustrated Guide to Recurrent Neural Networks: Understanding the Intuition](https://www.youtube.com/watch?v=LHXXI4-IEns) **(10 min)**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/LHXXI4-IEns?si=-ACfyBwj_xTHCpII" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### *Synthesis Questions*
* `How does an RNN interact with a feed-forward neural network? What role does the RNN play in this process?`
* `Describe the vanishing gradient problem in your own words. Does it relate to the drawbacks of statistical language models?`
* `The video describes a few solutions to the short-term memory of RNNs. What changes do they make to address the problem?`

### **Video 2:** [Transformers, explained: Understand the model behind GPT, BERT, and T5](https://www.youtube.com/watch?v=SZorAJ4I-sA) **(9 min)**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/SZorAJ4I-sA?si=wjTAV0Bpqfer93b3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

### *Synthesis Questions*
* `What are some of the limitations of previous NLP models, and how did transformers address these?`
* `Describe the ideas of positional encoding and attention in your own words.`
* `Like the “server” example at 6:50 in the video, create two sentences that can be disambiguated using self-attention.`

### **Optional:** [Natural Language Processing: Crash Course AI #7](https://www.youtube.com/watch?v=oi0JXuL19TA&list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b&index=8) **(13 min)**
**Great resource if you're still having trouble with NLP!**
<div class="center">
    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/oi0JXuL19TA?si=bF_DLRWwnYUXqdk8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

<!-- ## Unit 8 Project Specs
**<mark style="background-color: lightblue">Homework Help:</mark>** if you’re having trouble with the technical homework, try following along with this video first! It also uses Python in Google Colab and should give you some good practice. Reach out to a TA if you have any questions! 

**[Make an AI sound like a YouTuber (LAB): Crash Course AI #8](https://www.youtube.com/watch?v=kZWum5omEv4&list=PL8dPuuaLjXtO65LeD2p4_Sb5XQ51par_b&index=9)** -->

---

### **<u>Task 2:</u>** 

*Complete the following writing activity.*

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* What ethical considerations arise when developing language models that are inspired by neural processes involved in language?
* To what extent do models used in language processing reflect the actual neural networks involved with language tasks in the brain?
* How can insights from neuroscience be leveraged to enhance the design and development of language models?
* Reflecting on you learning from this unit, what is the one thing you found to be most interesting?
* What is one concept from this unit that you would like to learn more about and why?

<!---
# Old Course Content

## Intro

Language modeling has become ubiquitous in our society, used in chatbots, content moderation, translation, and more. In this unit, we will explore natural language processing (NLP), discussing common model architectures, their applications, and training methods.

## NLP Tasks

NLP encompasses various tasks, each with specific architectures and applications:

- Text Generation: Models like ChatGPT, autocomplete systems.
- Sequence-to-Sequence: Language translation models.
- Sequence-to-Vector: Hate speech detection (text classification).
- Word2Vec: Mapping words to nearby vectors.
- Word Embeddings: Identifying similarities between words.

These tasks are crucial in NLP, employing different algorithms for solutions. Our focus will be on text generation and word embeddings, but we will also touch on other tasks.

## Text Generation

Text generation involves creating new text by framing it as a classification problem: predicting the next word in a given text sequence.

**Example:**

- pass 1:
    - model input: `the` `quick` `brown` `fox`
    - model prediction: `jumps`
- pass 2:
    - model input: `the` `quick` `brown` `fox` `jumps`
    - model prediction: `over`
- pass 3:
    - model input: `the` `quick` `brown` `fox` `jumps` `over`
    - model prediction: `the`
- ...

By appending the prediction to the given sequence, we get a new base sequence for predicting the next token. This iterative process allows generating arbitrary-length sequences. Such models are autoregressive, using previous outputs as inputs for subsequent predictions. This means the model can generate as much text as desired, limited only by computational resources and the coherence of the content.

Self-supervised learning in NLP utilizes the natural structure of language, where examples for learning are implicit in the text itself. In this approach, a sentence or a passage serves as both the input and the label, with certain parts masked or predicted by the model. For instance, in the sentence "The cat sat on the ___", the blank can be used as a prediction target, teaching the model the context and structure of language. This method enables models to learn from large amounts of unlabeled text, grasping grammar, syntax, and context naturally.

## Word Embeddings and Tokenization

Tokenization and Embeddings are vital in NLP as models inherently process numerical data. Tokenization converts text into manageable units (like words or characters), which are then transformed into numbers. This allows models to interpret and process language data. Embeddings take this further by converting these tokens into vectors of real numbers, capturing semantic meanings. For example, in word embeddings, words with similar meanings are closer in the vector space, enabling the model to understand relationships and nuances in language. These techniques are crucial for handling the complexity and variability of human language in computational models.

Please read this article on [word embeddings](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) and answer the following questions:

`Synthesis questions:`
- `What is a word embedding? How long are they usually?`
- `How does training a network to recognize the validity of 5-grams result in a Word-to-Vector "map"?`
- `Pretend you are a word embedder. Give examples (2-3 for each) of words in the same family as:`
    - `King`
    - `Button`
    - `Pain`
    - `Water Bottle`
- `What is the explanation given for the emergence of the "male-female difference vector"?`
- `What is pre training/transfer learning/multi-task learning?`

## Feed-Forward Network

A simple model for text generation could be a feed-forward network. As feed-forward networks always need a constant input size, it uses a specified **context length** to determine how many previous words to consider when predicting the next one. When the input is shorter than the context length, we use padding tokens (`<PAD>`) to fill the gap.

**Example:**

Context length = 4

- "the quick brown fox"
    - Input: `the`, `quick`, `brown`, `fox`
    - Output: `jumps`
- "an apple a day keeps the doctor"
    - Input: `day`, `keeps`, `the`, `doctor`
    - Output: `away`
- "ignorance is"
    - Input: `<PAD>`, `<PAD>`, `ignorance`, `is`
    - Output: `bliss`

However, feed-forward networks have limitations:
- Fixed context length: Cannot consider words beyond a certain range.
- Fixed input size: Requires padding for shorter inputs.
- Difficulty in capturing word relationships: Not optimized for understanding the connections between words.

## Recurrent Neural Networks (RNNs)

RNNs maintain an internal hidden state updated at each token. This hidden state represents all that the model has seen so far. RNNs solve many of the problems experienced by feed-forward networks by allowing variable-length sequences and "sharing" parameters, improving efficiency.

They operate with the following weights and biases:
- $W_{xh}$: Converts a token to the hidden state.
- $W_{hh}$: Converts a previous hidden state to the next hidden state.
- $b_h$: Bias for moving to the next hidden state.
- $W_{ho}$: Predicts the next token from the current hidden state.
- $b_o$: Bias for the output prediction.

At each time step (this means for each token) the RNN does two things:
1. Update the internal hidden state from $H_{t-1}$ to $H_t$
    - This is done using the formula: $H_t = \text{tanh}(X_tW_{xh} + H_{t-1}W_{hh} + b_h)$
2. Predict the output token at time $T$
    - This is done using: $O_t = H_tW_{ho} + b_o$
    - If the task for the RNN is not text generation, this can be omitted

There always needs to be a preceding hidden state, so before any token is processed the hidden state is initialized (often to zero).

**Example:**

Let's run the RNN on a demo sentence, `i`, `love`

1. Initialize the hidden state to zeros $H_{-1} = 0$
2. Repeatedly update the hidden state and generate next-token probabilities
    0. $t=0$, `<START>` (we need a special start token to tell the model to begin)
        - Update hidden state: $H_0 = \text{tanh}(X_\text{\<START\>}W_{xh} + H_{-1}W_{hh} + b_h)$
        - Generate next-token probabilities (target is `i`): $O_1 = H_0W_{ho} + b_o$
    1. $t=1$, `i`
        - Update hidden state: $H_1 = \text{tanh}(X_\text{love}W_{xh} + H_0W_{hh} + b_h)$
        - Generate next-token probabilities (target is `love`): $O_2 = H_1W_{ho} + b_o$
    2. $t=2$, `love`
        - Update hidden state: $H_2 = \text{tanh}(X_\text{love}W_{xh} + H_1W_{hh} + b_h)$
        - Generate next-token probabilities (unknown target!): $O_3 = H_2W_{ho} + b_o$
        - If we want to continue generating, predict next word $X_3$ from $O_2$
    3. $t=3$, $X_3$ (autoregressive - we're inputting our previous outputs!)
        - Update hidden state: $H_3 = \text{tanh}(X_3W_{xh} + H_3W_{hh} + b_h)$
        - Generate next-token probabilities: $O_4 = H_3W_{ho} + b_o$
    4. ...

Advantages of RNNs:
- Efficient parameter sharing across time steps.
- Handling variable sequence lengths without padding.
- Capturing dependencies over time through the internal hidden state.
- Enhanced ability to model sequential data, reflecting the natural flow of language.

Disadvantages of RNNs:
- Tendency to forget early tokens due to continual updates of the hidden state.
- Unstable gradients, leading to training challenges.
- Difficulty in parallel processing, impacting training efficiency.

Please read (or skim) the following article on [RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and answer the synthesis questions:

`Synthesis questions:`
- `What happens to the memory vector as we move through time?`
- `What is the memory vector initialized to?`
- `Why might unstable gradients occur?`

## Transformers

Transformers represent the cutting-edge in NLP, known for their self-attention mechanisms and scalability. Unlike previous architectures, Transformers do not process text sequentially. Instead, they use self-attention to weigh the importance of each token in the context of others, regardless of their position. This allows for a more nuanced understanding of text, capturing long-range dependencies effectively. Their architecture enables parallel processing of tokens, significantly enhancing training speed.

Advantages:
- Excellent at capturing relationships between tokens, crucial for complex language understanding.
- Highly scalable and parallelizable, allowing for efficient processing of large datasets.

Disadvantages:
- Require significant computational resources, making them expensive to train.
- Complex models that are often less interpretable, making it hard to understand decision-making processes.

Take a look at some (or all) of the following resources for how transformer/GPT models work. You
might need to find external resources as well:
- [GPT in 60 lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
- [ Let's build GPT: from scratch, in code, spelled out. ](https:/youtu.be/kCc8FmEb1nY?feature=shared)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [An Intuitive Explanation of GPT Models](https://blog.cswartout.com/2022/12/25/intuitive-explanation-of-gpt-part-2.html), written by Carter Swartout of I2

Please answer the following synthesis questions:

`Synthesis Questions:`
- `What do the probabilities that GPT outputs represent, and what is greedy decoding?`
- `Describe masked self-attention in your own words (not including the vector math)`
- `How does self-attention allow tokens to have relationships with each other?`
- `What are the big blocks that make up the GPT architecture?`
- `Why are positional encodings necessary?`

---

# **Technical Project Spec:**

The project for this “_Language Modeling_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!

A few general helpful tips (if applicable):
* Be sure to appropriately make a copy of the Colab template before starting to save your progress!
* Renaming your copy to something that contains your name is a good idea, it will make it easier for us to review your submissions.
* Type most of the code out yourself instead of just copying from the tutorial.
* Leave comments to cement your understanding. Link syntax to ideas.

Now, follow the instructions on this Jupyter notebook to implement some of the things we talked about. There is an "answers" link at the bottom of the notebook that you can use if stuck. You will need to download the '.ipynb' found in that directory and open it either locally or in a new colab project yourself. Ask around if you are unable to get it working!

**<span style="text-decoration:underline;">There are 2 parts (.ipynb files) to this unit. Try to finish both.</span>**
This technical project is likely to be harder than anything you have done in this course before, so be patient with it and reach out if you need support!

**Colab Link:** [Unit 8 Notebook Part 1](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/hf_tutorial.ipynb) **(1 hr)**

Now navigate to the application portion of this project (Part 2 below), where you are given a dataset and asked to train an LLM of your choice to emulate Shakespeare! Be sure to reference your Unit 8 Notebook Part 1 to figure out how to do this.

**Colab Link:** [Unit 8 Notebook Part 2](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-08/lm_starter_code.ipynb) **(1 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Congratulations! You now understand the basics of Language Modeling!

# **Non-Technical Project Spec:**

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* What ethical considerations arise when developing language models that are inspired by neural processes involved in language?
* To what extent do models used in language processing reflect the actual neural networks involved with language tasks in the brain?
* How can insights from neuroscience be leveraged to enhance the design and development of language models?
* Reflecting on you learning from this unit, what is the one thing you found to be most interesting?
* What is one concept from this unit that you would like to learn more about and why?

Be sure to submit your work through google drive using the submission form!
We would prefer that you upload it to your own Drive first, then use the submission form dropbox to connect that file to your submission!
--->
