---
title: Unit 09
parent: Megadoc
---

# Unit 9: Fairness and Theory


Machine learning models are interesting, but they have substantive effects on the world they are deployed in. How can we make these models fairer, safer, less biased, and/or more responsible? Is this even rigorously possible? (Some researchers suggest not!) What is the source of bias? (“Garbage In, Garbage Out” is a stunted answer, and maybe even misleading.) These are all questions which are intimately linked with _deep learning theory_, a growing field which attempts to explain how neural networks work rather than attempting to advance the SOTA in performance or a similar task. Because of the close relationship between theory and fairness research, we will be exploring them together. After going through this unit, you will be able to reason about deep learning at a very abstract level (a powerful tool for research and experimentation); identify the core theoretical essence of various models and approaches; think critically about what the concepts of ‘bias’, ‘fairness’, ‘robustness’, ‘responsibility’, and ‘fairness’ mean and how we might build models which better embody these values.

It is recommended to read the listed papers in order, and to at least skim each one.

**Theory**



1. [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). While it’s not necessary to completely understand the proof, make sure you understand at least what the theorem is stating and why it is an interesting result. 
2. Read this [introductory article](https://medium.com/analytics-vidhya/you-dont-understand-neural-networks-until-you-understand-the-universal-approximation-theorem-85b3e7677126) written by Andre (the author of this unit) on the UAT, then [this Twitter thread](https://twitter.com/ylecun/status/1409940043951742981?lang=en) of Yann Lecun blasting it. Then, read Lecun et al.’s paper [Learning in High Dimension Always Amounts to Extrapolation](https://arxiv.org/pdf/2110.09485.pdf). Lastly, read this document of [the debate on Twitter](https://gowrishankar.info/blog/deep-learning-is-not-as-impressive-as-you-think-its-mere-interpolation/). Now think about what your position in this debate is. What is interpolation? What is extrapolation? Do neural networks extrapolate? Is this a meaningful concept at all, and if not, what might be a more meaningful one? Keep thinking about these questions throughout the theory section.
3. [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/pdf/1912.02292.pdf). A ‘classic’ empirical finding which points towards a weirdness of deep learning models as opposed to less parametrized, classical models.
4. [Are Deep Neural Networks Dramatically Overfitted?](https://lilianweng.github.io/posts/2019-03-14-overfit/) A great technical blog post giving more theory on the question of overfitting.
5. [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/pdf/1611.03530.pdf). Important empirical results and speculative theoretical work.
6. [Methods for Pruning Deep Neural Networks](https://arxiv.org/pdf/2011.00241.pdf). You can skim this one, but it’s a good coverage of pruning – an empirical method whose success is surprising and is worth thinking about.
7. [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635). This is our second major theory paper. What explains the results found and answers the questions raised in 2, 3, 4, and 5? The Lottery Ticket Hypothesis is a compelling theory.
8. [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/pdf/1905.01067.pdf). A further investigation into the Lottery Ticket Hypothesis.
9. [What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/pdf/1911.13299.pdf) A fascinating result from the Lottery Ticket Hypothesis.
10. Neural Tangent Kernel. Choose at least one of these to read.
    1. [https://lilianweng.github.io/posts/2022-09-08-ntk/](https://lilianweng.github.io/posts/2022-09-08-ntk/) 
    2. [https://blog.ml.cmu.edu/2019/10/03/ultra-wide-deep-nets-and-the-neural-tangent-kernel-ntk/](https://blog.ml.cmu.edu/2019/10/03/ultra-wide-deep-nets-and-the-neural-tangent-kernel-ntk/) 
    3. [https://rajatvd.github.io/NTK/](https://rajatvd.github.io/NTK/) 
    4. [https://www.inference.vc/neural-tangent-kernels-some-intuition-for-kernel-gradient-descent/](https://www.inference.vc/neural-tangent-kernels-some-intuition-for-kernel-gradient-descent/)
11. [The Modern Mathematics of Deep Learning](https://arxiv.org/pdf/2105.04026.pdf). A landmark work in developing a mathematical theory of deep learning. Skim sections 2.3, 3.2, 3.3, and 4. Make sure you at least understand the results at a high level.
12. [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221.pdf). A theoretical method for probing language model knowledge reveals interesting epistemic structures. 
13. Bonus: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Think you know how transformers work? Think again!

**Fairness, Responsibility, Safety**



1. [A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/pdf/1908.09635.pdf). A good and comprehensive survey of general concerns and approaches to addressing bias in machine learning problems.
2. [On the (im)possibility of fairness](https://arxiv.org/pdf/1609.07236.pdf). No need to read it too in detail; skimming it and understanding the main result is fine. Argues that different mathematized components of algorithmic fairness are fundamentally incompatible with each other in the ideal.
3. [The Myth in the Methodology: Towards a Recontextualization of Fairness in Machine Learning](https://econcs.seas.harvard.edu/files/econcs/files/green_icml18.pdf). We’ve got it all wrong, philosopher Lily Hu shows us. Fairness cannot be mathematized.
4. [What’s sex got to do with fair machine learning?](https://arxiv.org/ftp/arxiv/papers/2006/2006.01770.pdf) An investigation of gender variables in machine learning models.
5. [What is ‘race’ in algorithmic discrimination on the basis of race?](https://scholar.harvard.edu/files/lilyhu/files/what_is_race.pdf) An investigation of race variables in machine learning methods.
6. [Moving beyond “algorithmic bias is a data problem”](https://www.sciencedirect.com/science/article/pii/S2666389921000611). Sara Hooker takes on the pervasive idea of GIGO (Garbage In, Garbage Out) suggests that data is the root source of bias.
7. [What Do Compressed Deep Neural Networks Forget?](https://arxiv.org/abs/1911.05248) An empirical follow-up from the previous opinion piece. How do different compression methods affect model performance?
    1. Optional: [Characterizing and Mitigating Bias in Compact Models](https://arxiv.org/pdf/2010.03058.pdf). Related work if you are interested in additional research in this direction.
8. [The Curious Case of Common Sense](https://www.amacad.org/publication/curious-case-commonsense-intelligence). Our very own professor Yejin Choi reflects on the difficulty of codifying common sense into AI models.
    2. You may be interested in Yejin Choi’s research papers, each of which address different dimensions of common sense reasoning: [https://homes.cs.washington.edu/~yejin/](https://homes.cs.washington.edu/~yejin/) 
9. [Can Machines learn Morality? The Delphi Experiment](https://arxiv.org/pdf/2110.07574.pdf). An attempt to train a language model to understand commonsense morality. You can play with the Delphi model at [https://delphi.allenai.org/](https://delphi.allenai.org/). 
10. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). An important proposal for a framework to regulate AI with AI from a set of constitutional principles.
11. [Predictability and Surprise in Large Generative Models](https://www.anthropic.com/index/predictability-and-surprise-in-large-generative-models). An empirical and theoretical investigation of the difficulties of regulating LLMs.
12. [The Hardware Lottery](https://arxiv.org/abs/2009.06489). Thinking about how AI development is constrained by the hardware available. 
13. [Socially situated artificial intelligence enables learning from human interaction](https://www.pnas.org/doi/full/10.1073/pnas.2115730119). A very interesting paper from our very own professor Ranjay Krishna on how humans can more actively engage with and shape AI models.
14. [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). A classic in AI safety. Read the introduction (section 1) and “Society” (section 5) at minimum.
15. [The Dark Side of Techno-Utopianism](https://www.newyorker.com/magazine/2019/09/30/the-dark-side-of-techno-utopianism). An accessible and thoughtful conclusion to this unit: what is our role in tech, and is it as rosy as we think it is?

**Task**

Read the [GPT-3 paper](https://arxiv.org/abs/2005.14165), the [GPT-4 model card](https://cdn.openai.com/papers/gpt-4-system-card.pdf), or the [LaMDA paper](https://arxiv.org/pdf/2201.08239.pdf) (or another recent large language model paper). Focus on the discussions on safety and fairness. Drawing upon the sources in both the theory and the fairness sections of this unit, criticize the evaluations – identify what is mistaken, what assumptions are made, what is socially problematic, what possible biases may have been unidentified or which lie latent in the design, etc.

Feel free to talk with Andre if you have any questions while reading or thinking about these topics. If you are interested in a more philosophical investigation of these topics, check in the machine subjectivity discord channel or join one of our meetings!
