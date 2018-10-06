# Week 03 Notes - Monte Carlo Methods

### Internet of Things Optimization
The Internet of things lives! More and more devices are coming prepackaged with Internet access that wouldn’t normally be. That includes everything from smart salt shakers to smart tabletops. And because these devices have a connection to the world wide web, they can communicate with the outside world and each other, sharing information and even learning from one another. In this video, I'll explain how to use a reinforcement learning technique called "Monte Carlo" to optimize electricity consumption and cooling demands for a smart home. Enjoy!

**Notes**:

- Markov Decision Process is a fundamental way of framing the Reinforcement Learning problem where an agent is interacting with an environment to maximize a reward
- Agent's goal is to learn a policy so it knows given a state the best action to take in order to maximize a reward
- In a complete Markov Decision Process where all of the environment variables are known we can use dynamic programming to learn an optimal policy
- But what if we don't know all the environment variables? This would be considered a Model Free RL, not Model Based RL.
    + Model-Based Learning: Learn the model, solve for values
    + Model-Free Learning: Solve for values directly (by Sampling)
- In Model Free RL, the we miss the transition model and the reward function. So we don't know what's going to happen after each action we take beforehand and the agent won't get an award associated with a particular state beforehand.
- Dynamic Programming won't work when these 2 Markovian variables are not available
- We use Monte Carlo Method
    + Large family of computation algorithms that rely on repeated random sampling to obtain numerical results
    + They make use of randomness to solve problems
- Reasons for using Monte Carlo vs. Dynamic Programming
    + No need for a complete Markov Decision Process: Allow for learning optimal behavior directly from interaction with the environment without needing the transition or reward function defined beforehand
    + Computationally more efficient: Focus MC methods on a small subset of the total states
    + Can be used with Stochastic Simulations
- In Monte Carlo RL, we are estimating the value function for each state based on the return of each episode. The more episodes we take into account the more accurate our estimation will be.
    + Potential problem: What if we visit the same state twice in a single episode? 
    + There are two types of Monte Carlo policy evaluations:
        * First-Visit
        * Every-Visit
- First-Visit:
    +  Only recognizes the first visited state. Every second visit does not count the return for that state visit and the return is calculated separately for each visit.
- Monte Carlo includes randomness because when it updates every episode depending on where it originated from it's a different result depending on which action we take in the same state.
- Because it contains these random elements, Monte Carlo has a high variance   


**Take Aways**:

- In Model-Free Reinforcement Learning, as opposed to Model Based, we don't know the Reward Function and the Transition Function beforehand. We have to learn them through experience
- A Model-Free Learning Technique called Monte Carlo uses Repeated Random Sampling to obtain numerical results
- In First Visit Monte Carlo, the State Value Function is defined as the average of the returns following the agents first visit to S in a set of episodes


**Learning Resources**:

- [Youtube Video](https://www.youtube.com/watch?v=kYWw6GBRjVk)
- [Code Link](https://github.com/llSourcell/Internet_of_Things_Optimization)
- [Reinforcement Learning - Monte Carlo](https://www.kth.se/social/files/58b941d5f276542843812288/RL04-Monte-Carlo.pdf)
- [Reinforcement Learning & Monte Carlo Planning](https://courses.cs.washington.edu/courses/csep573/12au/lectures/18-rl.pdf)
- [Reinforcement Learning - Monte Carlo Methods](https://stat.ethz.ch/education/semesters/ss2016/seminar/files/slides/RL_MCM_heinzer_profumo.pdf)
- [Ch 12.1: Model Free Reinforcement Learning Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4)
- [RL Lecture 5: Monte Carlo Methods](http://www-edlab.cs.umass.edu/cs689/lectures/RL%20Lecture%205.pdf)
- [Model-Free Prediction & Control with Monte Carlo](https://github.com/dennybritz/reinforcement-learning/tree/master/MC)


### Video Lesson (Exploration vs. Exploitation)
[Youtube Video](https://www.youtube.com/watch?time_continue=3&v=cEFx3wtNONE)

**Notes**:

- Fundamental concepts in Reinforcement Learning: Exploration vs. Exploitation
- Whenever an agent is faced with a series of options with unknown rewards and we are trying to maximize the reward the agent will receive, there is a need to balance exploring the options and repeating those which are shown to give rewards
- Example: Suppose you've gone to a restaurant once before and quite enjoy your meal. Looking at the menu, you will be tempted to get the same item you ordered last time, exploiting the knowledge you've previously acquired on which of the dishes is most tasty. On the other hand, you could take a more exploratory strategy by trying a new dish with the hope that this will be tastier than the last dish
- Example: Trick or treaters walking together on Halloween. Suppose that these kids want to maximize the amount of candy they can collect within the limited time span of Halloween night. One kid suggests try going to every house and see which gives the most candy. The other argues that they stick to a single block with a good payout returning continuously to the same houses.
- Which strategy is better? In generation, we get the most reward by finding a balance between exploring and exploiting.
- One way to modulate this behavior in an agent is to control the value of Epsilon, the chance of taking a random action:
    + Increase Epsilon: Epsilon Greedy - increasing our change of the agent taking a random action, we encourage our agent to explore for new rewards
    + Decrease Epsilon: Greedy - encourage our agent to exploit acquired knowledge of rewards taking the action with the highest estimated reward


### Reading Assignment (Exploration vs Exploitation and Multi-Armed Bandits)

**[Basic] Understanding concepts with math**

This article easily explains concepts of “exploration vs. exploitation trade-off” by examples like choosing a restaurant or food. It also suggests some strategies how they could play out in our lives and explains how to get money faster using multi-armed bandits, so be sure to read it and become rich.

- [Exploration vs. Exploitation](https://medium.com/@dennybritz/exploration-vs-exploitation-f46af4cf62fe)
- [Multi Arm Bandits and the Explore/Exploit trade-off](http://www.primarydigit.com/blog/multi-arm-bandits-explorationexploitation-trade-off)
- [Multi Armed Bandit Testing Can Make You More Money, Faster](https://www.searchenginepeople.com/blog/16072-multi-armed-bandits-ab-testing-makes-money.html)


**[Intermediate] Understanding deeper concepts using math**

If you understand the basic concepts, use math to study each of the strategies presented earlier. These are  well-organized lecture note useful for those who want to study deeper. If you’re not familiar with math, just skip over them without trying to understand them. Still, it is worth reading enough.

- [Deep Reinforcement Learning and Control](http://www.cs.cmu.edu/~rsalakhu/10703/Lecture_Exploration.pdf)
- [David Silver Lecture 9 - Exploration and Exploitation](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/XX.pdf)
- [RL Course by David Silver - Lecture 9: Exploration and Exploitation](https://www.youtube.com/watch?v=sGuiWX07sKw)


**[More]**

MAB is such a popular example that you will see it more often than not. 

- [Wikipedia: Multi Armed Bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Casinopedia: One Armed Bandit](https://www.casinopedia.org/terms/o/one-armed-bandit)



### Monte Carlo Coding Tutorial


### MC Control & MC Prediction


### Reading Assignment (Monte Carlo Methods)


### Q Learning for Trading


### Homework Assignment (Monte Carlo)


### Tensor Processing Units (Live Stream)