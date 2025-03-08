---
title: Unit 06
parent: Megadoc
---

# Unit 6: Reinforcement Learning
Hello and welcome to the _Basics_ section of the I2 megadoc! 

**Task 1:** Read either the <ins>literacy article</ins> "Back to Basics" or the <ins>technical article</ins> linked below to get an intuitive understanding of reinforcement learning. <span style="color:red">**This is required.**</span>

#### [Unit 06 Technical Article](unit-06-tech-article.md)

&nbsp;

**Task 2:** Go through the following videos/articles and answer the provided synthesis questions. Submit your answers to your intro course TA. 
[Link to this task](https://course.uw-i2.org/megadoc/unit-06/#unit-6-synthesis-questions)

**Task 3:** Complete either the technical project or the non-technical project. Submit your work to the intro course TA.
[Link to this task](https://course.uw-i2.org/megadoc/unit-06/#unit-6-project-specs)

## Back to Basics: Reinforcement Learning
Welcome to one of the most important units in this course—and one of the most challenging! Like all our articles, the goal of this article is to give you an intuitive understanding of reinforcement learning so that you can recognize it in daily life and apply it to technical projects.

Reinforcement learning is the study of how a free object, or **agent**, moves around and accomplishes tasks in an **environment**, either real or simulated. The very general idea involves rewarding the object for desirable actions (i.e. reinforcing that behavior) and punishing it for undesirable actions.

For example, suppose we’re teaching a computer how to play chess against a human. In this case, the agent would be the computer and the environment would be the chess game (i.e. the opponent, board, and pieces).

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/agent_environment.png" alt="Diagram of the agent and environment described above" width="500"/>
</div>

The computer starts by taking an **action**--in other words, it does something. In this case, say the computer captures one of the human opponent’s pawns. Now the environment (the chess game) looks different from before the computer took its action; the computer has changed the **state** of the environment.

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/good_action.png" alt="Diagram of the environment after a good action is taken" width="500"/>
</div>

The state of the environment is favorable—we’re glad the computer took this action, and we want to reinforce it (we want it to keep taking actions like this). So we give the computer a **reward** that’s proportional to how desirable the action was. In this case, the reward is pretty moderate—capturing a pawn is good, but it’s not one of the best moves the computer can make. If the computer had captured the opponent’s queen, for example, we’d give it more of a reward because that’s a more desirable action.

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/good_reward.png" alt="Diagram of the environment after a reward is given" width="500"/>
</div>

Suppose the computer takes a different action: instead of capturing a pawn, it knocks over the entire board. Again, our environment is in a new state as a result of this action. 

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/bad_action.png" alt="Diagram of the environment after a bad action is taken" width="500"/>
</div>

Unfortunately, this new state is very bad for the computer because there’s no way it can win the game now. So we want to punish this action and make sure the computer avoids it in the future. We can do this by giving it a negative reward to indicate that it’s an undesirable action.

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/bad_reward.png" alt="Diagram of the environment after a punishment is given" width="500"/>
</div>

The computer decides what steps to take using something called a **policy**. A computer uses a policy to decide what its next step should be. For example, maybe the computer’s policy is to maximize its rewards. Then, every single action it takes is the action that produces the highest reward. This helps it avoid undesirable actions, like knocking over the board, because they have such low rewards. 

This isn’t always the best policy, though. We said earlier that capturing the opponent’s queen is a very high-reward action. Suppose the computer is in a position where it can capture its opponent’s queen, but in doing so leaves its king vulnerable. In this case, maybe taking the highest-reward action isn’t the best way to go. We would have to use a different policy in our decision-making. 

Deep reinforcement learning combines deep learning and reinforcement learning. Its goal is to get the computer to learn to do something, and it teaches the computer using the principles of RL.

Take a look at the short video below, which tries to teach a robot to walk using deep reinforcement learning (don't worry, you'll have fewer Synthesis Questions to make up for the extra video!).

[AI Learns to Walk (deep reinforcement learning)](https://youtu.be/L_4BPjLBF4E?si=jtH5sPdoihN60MLh) (9 min)

Notice that the robot wasn’t given any directions. It was given a target, and every action it took was rewarded or punished, but it ultimately had to learn the correct sequence of actions that would allow it to walk.

If we apply this to our chess example, the target might be to win the game by capturing the opponent’s king. We won’t tell the computer how to do that, but every time it takes an action we can reward or punish it. That way, it starts to learn the correct sequence of actions that it needs to take in order to win the game. 

Also, like in the video, we can put our chess computer in different environments to force it to learn new actions. For example, we can start it out in an environment where its opponent is a three-year-old. As the computer gets better, we can put it in new environments with more and more advanced opponents to force it to learn new skills, in the same way the robot in the video became better at walking by crossing more and more difficult terrain.

<div style="text-align:center">
    <img src="../assets/unit6/literacy_images/opponents.png" alt="Diagram of how the environment gets more advanced" width="500"/>
</div>

## Unit 6 Synthesis Questions

### **Video 1:** [Reinforcement Learning: Crash Course AI #9](https://youtu.be/nIgIv4IfJ6s?si=_ppJ4srBtfy04L7x) **(12 min)**
### *Synthesis Questions*
* *When does it make sense to use reinforcement learning vs. other methods of machine learning to accomplish a task?*
* *How do the agent, action, and environment interact in reinforcement learning?*
* *Give an example of two different policies in a reinforcement learning environment that’s NOT the cookie-jar example from the video (but you can use the chess game, the walking robot, or something you come up with yourself!).* 

### **Video 2:** [Reinforcement Learning from scratch](https://www.youtube.com/watch?v=vXtfdGphr3c) **(8 min)**
### *Synthesis Questions*
* *What is the purpose of a sigmoid function, and what does its value tell us? What about an error function?*
* *Describe the idea of gradient descent and how we use it in reinforcement learning.*

## Unit 6 Project Specs
### **Non-Technical Project Spec:**

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* Can you provide examples of experimental evidence linking reinforcement learning algorithms to observed synaptic changes in the brain?
* How do human neural systems encode reward signals and how does this relate to the concept of rewards in reinforcement learning models?
* What ethical considerations should be taken into account when developing interventions based on neuroscientific findings, and how can accountability be established for the potential impacts of such interventions?
* Reflecting on you have learned from this unit, what is one thing you found to be most interesting?
* What is one concept from this unit that you would like to learn more about and why?

Be sure to submit your work through google drive using the submission form!
We would prefer that you upload it to your own Drive first, then use the submission form dropbox to connect that file to your submission!

### **Technical Project Spec:**

The project for this “_Reinforcement Learning_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!


A few general helpful tips (if applicable):
* Be sure to appropriately make a copy of the Colab template before starting to save your progress!
* Renaming your copy to something that contains your name is a good idea, it will make it easier for us to review your submissions.
* Type most of the code out yourself instead of just copying from the tutorial.
* Leave comments to cement your understanding. Link syntax to ideas.

Now, follow the instructions on this Jupyter notebook to implement some of the things we talked about. There is an "answers" link at the bottom of the notebook that you can use if stuck. You will need to download the '.ipynb' found in that directory and open it either locally or in a new colab project yourself. Ask around if you are unable to get it working!

**Colab Link:** [Unit 6 Notebook](https://colab.research.google.com/github/interactive-intelligence/intro-neuro-ai-website/blob/main/notebooks/unit-06/rl_net.ipynb) **(1.5 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the (_incredibly basic)_ basics of Deep RL!


<!---
# Old Course Content
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
* <code>Think back to what the J(π) function was. How would performing gradient ascent (finding where this function is locally highest) on this function help the agent perform well in its environment?</code>
* <code>How is a policy derived (not literally a derivation) from the Q-function? Hint: argmax</code>

Great job on completing the tasks!

The last thing I will talk about before the project is the concept of **epsilon-greediness** for Q-learning. In essence, argmaxing the Q-function can lead to being stuck in a “pocket” where the agent gets stuck taking the same few trajectories due to the deterministic selection of the action to take. Since Q-learning is usually model-free, there is no way to “see” what lies beyond what is explored. Since the model will not take the actions to explore these trajectories, possibly more efficient solutions cannot be stumbled on by chance. This is where epsilon greediness comes in. With probability ε, the policy will take a random action rather than the one that maximizes the Q-function. ε starts off large and diminishes over time, as the model is expected to have learned better by then and there is less need to explore rather than exploit the learned model for high returns. You can see how epsilon changes through episodes in the graph below.


![alt_text](../assets/image9.png)


You can read more about epsilon-greedy algorithms in [this](https://www.baeldung.com/cs/epsilon-greedy-q-learning#:~:text=The%20epsilon%2Dgreedy%20approach%20selects,what%20we%20have%20already%20learned.) article (Section 5.2). It works quite well and often gets a model “unstuck” from optimizing poorly through not exploring.

---

# **Technical Project Spec:**

The project for this “_Reinforcement Learning_” section will be following the tutorial/Jupyter Notebook below. Please ask questions in the discord as you work through this project. Be sure to discuss with others in your group!


A few general helpful tips (if applicable):
* Be sure to appropriately make a copy of the Colab template before starting to save your progress!
* Renaming your copy to something that contains your name is a good idea, it will make it easier for us to review your submissions.
* Type most of the code out yourself instead of just copying from the tutorial.
* Leave comments to cement your understanding. Link syntax to ideas.

Now, follow the instructions on this Jupyter notebook to implement some of the things we talked about. There is an "answers" link at the bottom of the notebook that you can use if stuck. You will need to download the '.ipynb' found in that directory and open it either locally or in a new colab project yourself. Ask around if you are unable to get it working!

**Colab Link:** [Unit 6 Notebook](https://colab.research.google.com/drive/1WRswrZX7KuI7xJMcHm26N3QaT2kXdqmi?usp=sharing) **(1.5 hr)**

When you are finished with your code, independently verify that it works and have fun with it! If you add any additional functionality be sure to talk about it with others and give them ideas.

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the (_incredibly basic)_ basics of Deep RL!

# **Non-Technical Project Spec:**

The non-technical project for this unit will involve some writing! **Choose 3** of the prompts below and write **at least 200** (_meaningful!_) words on each one! We will not be strictly grading you on correctness or anything like that. This is an opportunity to deeply engage with the material you have just learned about, and creatively connect it to neuroscience!

* Can you provide examples of experimental evidence linking reinforcement learning algorithms to observed synaptic changes in the brain?
* How do human neural systems encode reward signals and how does this relate to the concept of rewards in reinforcement learning models?
* What ethical considerations should be taken into account when developing interventions based on neuroscientific findings, and how can accountability be established for the potential impacts of such interventions?
* Reflecting on you have learned from this unit, what is one thing you found to be most interesting?
* What is one concept from this unit that you would like to learn more about and why?

Be sure to submit your work through google drive using the submission form!
We would prefer that you upload it to your own Drive first, then use the submission form dropbox to connect that file to your submission!
--->
