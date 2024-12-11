---
title: Unit 08
parent: Megadoc
---

# Unit 8: Language Modeling

#### [Unit 06 Technical Article](unit-06-tech-article.md)

&nbsp;

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

**Colab Link:** [Unit 8 Notebook Part 1](https://colab.research.google.com/drive/1KGXpdL9sxpio1Zau5LviAGlv8Y0RMRcD?usp=sharing) **(1 hr)**

Now navigate to the application portion of this project (Part 2 below), where you are given a dataset and asked to train an LLM of your choice to emulate Shakespeare! Be sure to reference your Unit 8 Notebook Part 1 to figure out how to do this.

**Colab Link:** [Unit 8 Notebook Part 2](https://colab.research.google.com/drive/1Sg6seRXQ4pd8TwO-lYkiTxygv0SPt20B?usp=sharing) **(1 hr)**

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
