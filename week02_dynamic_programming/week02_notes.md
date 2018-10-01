# Week 02 Notes - Dynamic Programming

### Sports Betting
[Youtube Video](https://www.youtube.com/watch?v=mEIePvxdbkQ)

- Dynamic Programming refers to the collection of Algorithms  that can be used to compute Optimal Policies given a perfect model of the environment
- Value Iteration is a type of Dynamic Programming Algorithm that computes the Optimal Value Function and consequently the Optimal Policy
- Dynamic Programming is useful, but limited because it requires a perfect environment model and is computationally expensive

### Bellman Advanced
[Youtube Video](https://www.youtube.com/watch?v=FsOmL4sQJxo)

**Markov Decision Process**
- We are in a State. We choose an Action. Now there are several possibile S' states we could end up in based on random probability.
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



### Value & Policy Iteration


### Dynamic Programming Quiz


### Homework Assignment - Week 02


### iPhone XS Supply Chain


### Kaggle Challenge (Live Stream)

