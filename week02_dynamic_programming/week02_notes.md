# Week 02 Notes - Dynamic Programming

### Sports Betting
Sports betting is a popular past-time for many and a great use-case for an important concept known as dynamic programming that I’ll introduce in this video. We'll go over concepts like value iteration, the Markov Decision Process, and the bellman optimality principle, all to help create a system that will help US optimally bet on the winning hockey team in order to maximize profits. Code, animations, theory, and yours truly. Enjoy! 

**Take Aways**:
- Dynamic Programming refers to the collection of Algorithms  that can be used to compute Optimal Policies given a perfect model of the environment
- Value Iteration is a type of Dynamic Programming Algorithm that computes the Optimal Value Function and consequently the Optimal Policy
- Dynamic Programming is useful, but limited because it requires a perfect environment model and is computationally expensive

**Learning Resources**:
- [Youtube Video](https://www.youtube.com/watch?v=mEIePvxdbkQ)
- [Code Link](https://github.com/llSourcell/sports_betting_with_reinforcement_learning)
- [Art Int - Value Iteration](https://artint.info/html/ArtInt_227.html)
- [Deep Reinforcement Learning Demystified Episode 2](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)
- [What is an intuitive explanation of value iteration in reinforcement learning?](https://www.quora.com/What-is-an-intuitive-explanation-of-value-iteration-in-reinforcement-learning-RL)
- [How is Policy Iteration different from Value Iteration?](https://www.quora.com/How-is-policy-iteration-different-from-value-iteration)
- [Markov Decision Process: Value Iteration, how does it work?](https://stackoverflow.com/questions/8337417/markov-decision-process-value-iteration-how-does-it-work)


### Bellman Advanced
[Youtube Video](https://www.youtube.com/watch?v=FsOmL4sQJxo)

**Markov Decision Process**
- We are in a State. We choose an Action. Now there are several possible S' states we could end up in based on random probability.
- Each possible State transition from an Action has an exact probability and all the probabilities add to 1.
- The Markov Property states that the probability of future transitions depends only on the present. The past doesn't matter.

**Rules for New Grid World**
- Up or down = 80% obey, 10% left, 10% right
- Left or right = 80% obey, 10% up, 10% down
- Bump into wall = stay in same square

**Algorithm**:
1. Loop through every possible next state
2. Multiply the value of that State by its probability of occurring
3. Sum them all together

We can use Dynamic Programming to calculate the values of the grid cells.


### Dynamic Programming Tutorial
- [Youtube Video](https://www.youtube.com/watch?v=aAkFtRxeP7c)
- [Tutorial Source Code](https://github.com/colinskow/move37/tree/master/dynamic_programming)
- [Dynamic Programming Explanation Video](https://www.youtube.com/watch?time_continue=1&v=DiAtV7SneRE)

Bellman Equations for both deterministic and stochastic environments.

**Dynamic Programming for Reinforcement Learning**
- Create a Lookup Table to estimate the value of each state

**Reinforcement Learning Problems**  
1. Prediction Problem: We need to calculate accurate values for each possible state  
2. Control Problem: We need to find the Optimal Policy which leads to the highest expected rewards

2 classes of Dynamic Programming Algorithms to solve this problem:  
- Policy Iteration
    - Starts out with a totally random policy and starts taking actions. It then starts estimating values for each square based on the reward received from these random actions, updating the value table and using improved values to calculate and improve policy. This continues until both the policy and value tables stabilize and stop changing.
- Value Iteration
    - Completely ignores Policy and focuses on applying the Bellman Equation to calculate values for each square. We can then calculate a perfect policy in one pass for each state by finding the action with the highest expected value.

**Value Iteration Algorithm**

We will only focus on value iteration right now because it is the simplest, we've already learned the equation for it and it converges the fastest.
- Initialize a table V of value estimates for each square of all zeroes
- Loop over every possible state S
    - From state S loop over every possible action A
        - Get a list of all (probability, reward, S') transition tuples from state S, action A
        - expected_reward = sum of all possible rewards multiplied by their probabilities
        - expected_value = lookup V[S'] for each possible S', multiply by probability, sum
        - action_value = expected_reward + GAMMA * expected_value
    - set V[s] to the best action_value found
- Repeat steps 2-3 until the largest change in V[S] between iterations is below our threshold


### Dynamic Programming Reading Assignment
[Reading Assignment](https://www.theschool.ai/wp-content/uploads/2018/09/Reinforcement-Learning-Dynamic-Programming-V1.0.pdf)

PDF Content:

**Dynamic Programming (DP) and Reinforcement Learning (RL) - Reading Assignment**

-  Many reinforcement learning (RL) algorithms utilize dynamic programming methods. Dynamic Programming methods have been researched for years to solve MDPs (Markov Decision Processes) assuming that the agent has perfect knowledge of functions that define an environment. In machine learning, the environment is usually formulated as an MDP. The main difference between the classical dynamic programming methods and RL algorithms is that RL algorithms do not assume knowledge of an exact mathematical model of the MDP. RL algorithms target large MDPs where exact methods are not possible. RL problems such as Q-learning fits a problem setting under Markov decision processes because the environment is observable and an agent can iteratively and progressively learn to improve. Agent works towards obtaining optimal discounted reward from the environment.

I suggest the following as reading assignment. Foundational material for DP in RL with examples.

- [Reinforcement Learning Reading](https://github.com/dennybritz/reinforcement-learning/tree/master/DP/)

_Additional Reads_:

1. [Good Beginner/Mid-level Summary](https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919)  
2. Comprehensive Reading:  
    - [Reinforcement Learning: An Introduction 2nd Ed In Progress](http://incompleteideas.net/book/bookdraft2018jan1.pdf)  
    - [WildML - Learning Reinforcement Learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/)  
3. [Mid-level/Advanced Summary](https://www.cs.cmu.edu/~katef/DeepRLControlCourse/lectures/lecture3_mdp_planning.pdf)  
4. [Advanced](https://djrusso.github.io/RLCourse/index)  


### Value & Policy Iteration
- [Policy & Value Iteration PDF](https://www.theschool.ai/wp-content/uploads/2018/09/policy_and_value_iteration.pdf)
- [Introduction Lecture Summary](https://blog.goodaudience.com/01-reinforcement-learning-move-37-introduction-c3449dac2d54)

See the Jupyter Notebook, **policy\_and\_value\_iteration.ipynb**.


### Dynamic Programming Quiz
See the PDF File, week2\_quiz\_v4.pdf under docs/.

1. Policy Evaluation. Choose all that apply:
    - My answers: 3 & 4 apply. 1 & 2 don't apply.

2. Policy Improvement; Policy Evaluation. Choose all that apply:
    - My answers: 1 & 2 apply. 3 & 4 don't apply.

3. Value Iteration. Choose all that apply:
    - My answers: 3 & 4 apply. 1 & 2 don't apply.

4. Choose all that apply:
    - My answers: 1, 2, 4 apply. 3 doesn't apply.
    - **Corrections:** 2 doesn't apply. 3 does apply.

5. Choose all that apply:
    - My answers: 1, 2, 3, 4 apply.


**Solutions:**
Key:
    - [o] - correct
    - [x] - wrong

_Question 1_:

1. [x] A = -1.7. v = 1 * 0.25 * (-1 + 0) + 3 * 0.25 * (-1 + -1)  
2. [x] B = -1.7. v = 1 * 0.25 * (-1 + 0) + 3 * 0.25 * (-1 + -1)  
3. [o] C = -2.0. v = 4 * 0.25 * (-1 + 1)  
4. [o] same values of -2.0

[Reference: Policy Evaluation](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)

_Question 2_:

1. [o]  
2. [o]  
3. [x] **Bellman optimality equation** is used for Policy Improvement.  
4. [x] That is the formula for value iteration.

[Reference: Policy Evaluation, Policy Iteration](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)

_Question 3_:

1. [x] A = -4. Max(-3-1, -3-1, -5-1, -5-1)  
2. [x] B = -5. Max(-4-1, -4-1, -5-1, -5-1)  
3. [o] C = -5. Max(-4-1, -4-1, -5-1, -5-1)  
4. [o] D = -6. Max(-5-1, -5-1, -5-1, -5-1)

[Reference: Value Iteration](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)

_Question 4_:

1. [o]  
2. [x] **Bellman optimality equation** is used  
3. [o]  
4. [o]

[Reference: Value Iteration](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)

_Question 5_:

1. [o]  
2. [o]   
3. [o]  
4. [o]

[Reference: Value Iteration](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf)



### Homework Assignment - Week 02
Look at hw02/ directory.


### iPhone XS Supply Chain
Apple just announced its new iPhone XS so expect the demand for it to be massive! We can use a special reinforcement learning algorithm called policy iteration to help Apple manage it's retail inventory and make sure that the demand meets supply, I’ll explain how in this video. We'll assume the role of an AI savvy retail manager for Apple in San Francisco and discuss policy iteration as a solution to our problem. Dynamic programming and real world use cases, enjoy!

**Take Aways**:
- In Dynamic Programming, Policy Iteration is a modification of Value Iteration to directly compute the Optimal Policy for a given Markov Decision Process
- Policy Iteration consists of 2 steps:
    + Policy Evaluation
    + Policy Improvement
- While Value Iteration is a simpler algorithm than Policy Iteration, its more computationally expensive

**Learning Resources**:  
- [Youtube Video](https://www.youtube.com/watch?v=XiN9Hx3Y6TA)
- [Code Link](https://github.com/llSourcell/iphone_xs_supply_chain)
- [Deep Reinforcement Learning Demystified](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)
- [Art Int - Policy Iteration](https://artint.info/html/ArtInt_228.html)
- [Planning: Policy Evaluation, Policy Iteration, Value Iteration](http://kvfrans.com/planning-policy-evaluation-policy-iteration-value-iteration/)
- [How is Policy Iteration different from Value Iteration?](https://www.quora.com/How-is-policy-iteration-different-from-value-iteration)
- [Value Iteration and Policy Iteration](http://www.inf.ed.ac.uk/teaching/courses/rl/slides15/rl08.pdf)


### Kaggle Challenge (Live Stream)

