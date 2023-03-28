---
title: "Unit 01"
parent: Megadoc
---

# Unit 1: The (machine learning) Basics 

Hello and welcome to the _Basics_ section of the I2 megadoc! The items here are fundamental building blocks for Deep Learning (powerful tools that are more complex in computation, but funnily enough not as technical). A lot of the things here are statistics-heavy so be sure to pay attention! We will start off with something I am sure most of you are familiar with - linear regression.

**Task:** Read [this](https://towardsdatascience.com/the-basics-linear-regression-2fc9f5124687) article and answer the synthesis questions below! **(15 min)**


Before you begin! There are a few points in this article that I believe are more confusing than helpful, so feel free to ignore them if they do not make complete sense. Keep this in mind as you read, just take note, and don’t stress about the sections I overview below:



* In the subheader “Assumptions and caveats”, there is a bullet titled _Output variable is a linear combination of feature variables — linearity_. The author claims that linear regression can be used to find non-linear trends. He is right! The methods described check out and make sense. However, this is not what I would consider “simple linear regression” and can be considered a cool footnote. A computer will be handling most of this for you anyways. It is a good rule of thumb to apply linear regression models to linearly correlated data!
* In the subheader “Assumptions and caveats”, there is a bullet titled _Constant variance — homoscedasticity_. There is a part towards the end that begins with the words “Practically speaking, we can account for this in one model without splitting out data into three groups…”. The intuition behind the author’s solution to the problem he posed earlier makes sense, but strays from simple linear regression once again. The intuition behind the problem of heteroscedasticity is valid however, so be sure to take note of that.


### `Synthesis Questions:`



* `What is a feature in this context?`
* `What are the significance of the β terms within the modified y = mx + b equation described in the article?`
* `What is SSE?`
    * `How is it calculated?`
    * `What can it tell you about the values you chose for β?`
    * `If you modify the β<sub>1 </sub>term and the SSE goes up, was that a good modification?`
* `Write out the linear regression formula (involving β) when you wish to estimate the impact of age, height, and weight of someone regarding their marital status.`
    * `Hint: How many β terms will there be? How many features?`
* `What are the 4 assumptions described that you need to confirm before using linear regression?`
* `What is homoscedasticity? What is heteroscedasticity?`

Also skim over [this](https://medium.datadriveninvestor.com/basics-of-linear-regression-9b529aeaa0a5) article and focus on the equation provided. No questions!

Something to note about ML in general: high-dimensional stuff is simply not visualizable as-is. If you have more than two features it's near impossible to visualize the spread of dependent variables on a graph with the features as independent variables (How would you graph in 4d like we do in 2d or 3d?). Just know that equations of best fits for linear regressions define hyperplanes of the dimensions that the variables occupy. What is a hyperplane? Well, for example, if a space is 3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional, its hyperplanes are the 1-dimensional lines. Hyperplanes are one dimension less than the space they “draw” through. Think about why it's necessary to have a 2-d hyperplane for a 3d space of prediction (2 features) and a 1-d hyperplane for a 2d space of prediction (1 feature). While not visualizable, this reasoning applies to higher dimensions! Hyperplanes will become important as we move into dimensionality reduction so read up on them if you have time.

The next topic to cover will be Support Vector Machines (SVMs).

**Task:** Watch and understand the following videos. We recommend taking notes and being able to answer the synthesis questions provided below. Send your I2 teacher/mentor/overlord the answers to the questions over Discord.

This first video is intuition only. 

**Video 1:**
[Support Vector Machines: Data Science Concepts](https://www.youtube.com/watch?v=iEQ0e-WLgkQ)  **(8 min)**

### `Synthesis Questions:`



* `What are some use cases for an SVM? What does it do?`
* `What is the margin?`
* `Why does the margin need to be maximized? What does this allow for?`
* `What are the support vectors?`
* `What is the difference between a hard and soft margin SVM?`

This next video is math heavy. If you do not understand a term, look it up! Remember that the coefficients for a plane in the form (`ax + by + cz = 0)` can be found by determining a “normal” vector (a vector orthogonal to the plane). The vector will consist of numbers a, b, and c! This intuition carries over to higher dimensions (hyperplanes).

**Video 2:** [SVM (The Math): Data Science Concepts](https://www.youtube.com/watch?v=bM4_AstaBZo) **(10 min)**

### `Synthesis Questions:`



* `wx - b = -1 defines a __________ `
    * `Hint: it's like a plane`
* `What are the "w" variables in the equation?`
* `What is the equation for the decision boundary?`
* `What is the size of the margin in terms of the vector w?`
* `What needs to be minimized, and what are the constraints for finding the optimal hyperplane decision boundary? `
    * `Hint: Near the end of the video`

So far you know about Linear Regression and SVMs! Take a moment to make sure you get the general idea of these concepts. We will now be moving into the idea of dimensionality reduction.

As you may have noticed. A lot of what is done in the ML world is done in way more than three dimensions. Try as you might, you simply cannot accurately envision much above three dimensions concretely (Try and make a 4-d graph). We represent high-dimensional data in vectors, which is a nice numerical representation. But what if we want to see 100-dimensional data on a graph? This is where dimensionality reduction comes into play. We will cover a very basic dimensionality reduction algorithm. There are plenty more that have specific use cases so please** spend at least 10 minutes exploring others after learning about Principal Component Analysis (PCA)!**

**Task:** Watch and understand the following videos. We recommend taking notes and being able to answer the synthesis questions provided below. Send your I2 teacher/mentor/overlord the answers to the questions over Discord.

There is a part in the video (20 seconds) that handles some pretty complex math. Feel free to ignore it. The idea is what we want you to learn. Not really the full math.

**Video 1:** [StatQuest: PCA main ideas in only 5 minutes!!!](https://www.youtube.com/watch?v=HMOI_lkzW08) **(5 min)**

**Video 2:** [Principle Component Analysis (PCA)](https://www.youtube.com/watch?v=FD4DeN81ODY) **(5 min)**


### `Synthesis Questions:`



* `What does PCA do?`
    * `Give 3 use cases that you can think of`
* `What is a principal component?`

---

# **Project Spec:**

The project for this “_Basics_” section will **have you finish a code template on github.** Please ask questions as you work through this project. Be sure to discuss with others in your group if you have one! Share your answers as you like, the goal is to learn and we’re not holding grades over your head.

This project will be going over k-means clustering (unsupervised ML). We will be using the Scikit-Learn library.

A few general helpful tips (if applicable):



* Use GitHub, it’s really just better
* Use [Anaconda](https://www.anaconda.com/) with [Python3](https://www.python.org/downloads/) in [VSCode](https://code.visualstudio.com/). I personally create .py files but Jupyter Notebooks and Google Colab are also very powerful. 
    * For a simple project like this though, powerful computing is unnecessary and you can figure out the details of those other technologies next week
    * If you use Anaconda, create a separate environment so you can mess with libraries and imports all day without screwing up your base environment.
* Leave comments to cement your understanding. Link syntax to ideas.

Check out this handy image that gives popular sk-learn clustering algorithms and their usages:


![alt_text](../assets/image8.png)


Also this image visualizing the clustering algorithms:


![alt_text](../assets/image5.png)


Read up on k-means clustering in the provided link (Images provided above also contained here). Feel free to check out the other algorithms as well: [SK-Learn Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)

**Clone the Git repo onto your local device if you have not already.**

Then, in your local copy of the GitHub repo, navigate to the unit-1 folder, and work on **clustering-pca.ipynb**. Instructions are in the Jupyter notebook. If you need help setting up your python environment, ask the TA’s!

If you just want to download a single file, [here](https://stackoverflow.com/questions/4604663/download-single-files-from-github) is how you do it.

**GH Link:** [Unit 1 Notebook](https://github.com/interactive-intelligence/intro-neuro-ai/blob/main/unit-01/clustering-pca.ipynb) **(30 min)**

When you are finished with your code, independently verify that it works and have fun with it! You could try this method on different datasets, such as [this one for example](https://www.kaggle.com/datasets/ashwingupta3012/human-faces). If you add any additional functionality be sure to talk about it with others and give them ideas. 

Remember that this is all for your learning, so do your best and don’t stress!

Congratulations! You now understand the basics of Clustering and PCA!