---
title: Unit 06
parent: Megadoc
---

# Unit 6: Reinforcement Learning

Hello and welcome to the _Reinforcement Learning (RL)_ section of the I2 megadoc! Reinforcement learning is an incredibly powerful subfield of Machine Learning that can be used to deploy intelligent autonomous agents in both real and simulated systems. Reinforcement learning in itself is not inherently tied to AI. It is its own field that has strong ties to probability and statistics theory. 

For the purposes of simplicity, we will only cover the _very_ basics and terminology of Deep RL (RL combined with deep learning). Just as a heads up, RL is probably one of the most “deep” and currently relevant subfields of ML, followed closely by language modeling (this may also be the Dunning-Kruger effect, RL is what I have the most expertise on). Folks at OpenAI have compiled a plethora of resources for Deep RL and Deep RL research. We will show you the most relevant articles they have published but they have a wealth of knowledge to learn! Feel free to poke around on the [Spinning Up](https://spinningup.openai.com/en/latest/index.html) page in your own time.

**Task:** Read the [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#) section of the Spinning Up Intro to RL page. **This is a short, but _dense_ article. (45 min)**

You may skip the section on Diagonal Gaussian Policies.


### `Synthesis Questions:`



* `In what situations do we use reinforcement learning? What kinds of problems does it solve?`
    * `Give an example of RL used to play a game. Did it outperform humans? Does this scare you?`
* <code>What do the variables <strong>s</strong>, <strong>a</strong>, <strong>τ</strong>, represent in RL?</code>
    * <code>What do <strong>s'</strong>, <strong>a' </strong> represent?</code>
* <code>What is an action space?</code>
* <code>What is an episode?</code>
* <code>How is a neural network used to generate a probability distribution over actions given a state? </code>
    * <code>(i.e How would you construct a network if the dimensionality of the state was n<sub>1</sub> and the action space had a cardinality of m<sub>1</sub>)</code> 
    * `How is the log probability calculated from the logits of the NN output?`
    * `Note: This is called an actor NN!`
* `Describe infinite-horizon discounted return and what the discount factor is`
* `Describe what a stochastic policy is (you may have to look up what stochasticity is), and what it means for a policy to be parameterized by θ`
* `What is a good policy looking to maximize?`
    * `Think: If you set up your rewards in your environment randomly, would an RL model learn anything? No. This ties into the greater idea of reward-shaping, which is the concept of how to place rewards in an environment to encourage "good" behavior by an RL agent.`
* `What is the difference between the value function and the Q-function?`
    * `Describe the connection between the two`
* <code>Based on the idea behind Bellman Equations<strong> ("The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.") </strong>Explain how the following two equations satisfy the idea behind Bellman Equations: </code>

![alt_text](../assets/image3.png "image_tooltip")

`This may help:`

![alt_text](../assets/image4.png "image_tooltip")

* <code>What is the advantage function?</code>

This was a lot of content. Good job on getting through it all! Reinforcement learning is quite a math heavy subfield. Get used to a whole lot of probability and integration. Spelling out exactly what the equations are doing (by reading them out loud) can help quite a bit. Next we will be diving into some in-practice algorithms on a high level.

So, in general, these equations that you have learned about are ideals, not what is done in practice. What generally happens with deep learning RL algorithms is that these functions are **approximated ** by a neural network rather than calculated. This is important to keep in mind as you go on, because a lot of these equations have some recursiveness to them and often have such a huge search space that finding the perfect parameters for V<sup>*</sup> or Q<sup>*</sup> is near impossible in complex environments. 

**Task:** Read the “[Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)” section of the Spinning Up Intro to RL page. **(15 min)**

You may skip the section on What to Learn in Model-Based RL


### `Synthesis Questions:`



* `What is the difference between model-free and model-based RL?`
* `What is the difference between policy approximators and Q-learning?`
* <code>Think back to what the J([π](https://www.pisymbol.net/#:~:text=Alt%2B960%20Press%20and%20hold,enter%20960%20on%20numeric%20keypad.)) function was. How would performing gradient ascent (finding where this function is locally highest) on this function help the agent perform well in its environment?</code>
* <code>How is a policy derived (not literally a derivation) from the Q-function? Hint: argmax</code>

Great job on completing the tasks! 

The last thing I will talk about before the project is the concept of **epsilon-greediness** for Q-learning. In essence, argmaxing the Q-function can lead to being stuck in a “pocket” where the agent gets stuck taking the same few trajectories due to the deterministic selection of the action to take. Since Q-learning is usually model-free, there is no way to “see” what lies beyond what is explored. Since the model will not take the actions to explore these trajectories, possibly more efficient solutions cannot be stumbled on by chance. This is where epsilon greediness comes in. With probability [ε](http://www.unicode-symbol.com/u/03B5.html), the policy will take a random action rather than the one that maximizes the Q-function. [Ε](http://www.unicode-symbol.com/u/03B5.html) starts off large and diminishes over time, as the model is expected to have learned better by then and there is less need to explore rather than exploit the learned model for high returns. You can see how epsilon changes through episodes in the graph below.


![alt_text](../assets/image9.png)


You can read more about epsilon-greedy algorithms in [this](https://www.baeldung.com/cs/epsilon-greedy-q-learning#:~:text=The%20epsilon%2Dgreedy%20approach%20selects,what%20we%20have%20already%20learned.) article (Section 5.2). It works quite well and often gets a model “unstuck” from optimizing poorly through not exploring.

---

# **Project Spec:**

The project for this “_Reinforcement Learning_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!

A few helpful tips:



* Use GitHub, it’s really just better
* Use [Anaconda](https://www.anaconda.com/) with [Python3](https://www.python.org/downloads/) in [VSCode](https://code.visualstudio.com/).
    * If you use Anaconda, create a separate environment so you can mess with libraries and imports all day without screwing up your base environment.
* Type most of the code out yourself instead of just copying from the tutorial.
* Leave comments to cement your understanding. Link syntax to ideas.

Then, in your local copy of the GitHub repo, navigate to the unit-6 folder, and work on **rl_net.ipynb**. Instructions are in the Jupyter notebook. If you need help setting up your python environment, ask the TA’s!

**GH Link:** [Unit 6 Notebook](../../notebooks/unit-06/rl_net.ipynb) **(1.5 hr)**



* In case there is an error with the notebook not showing up on GitHub, paste the link into [this](https://kokes.github.io/nbviewer.js/viewer.html) website to view it.

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas. 

Congratulations! You now understand the (_incredibly basic)_ basics of Deep RL!