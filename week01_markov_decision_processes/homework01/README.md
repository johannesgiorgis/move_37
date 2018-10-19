# Homework Assignment 1 (OpenAI Gym Installation)
_Homework Assignment for Week 1_

Hey Wizards! For this first week, the homework assignment is to install and run a virtual game environment using the OpenAI Gym python library. You can find detailed instructions on how to install it (here)[https://gym.openai.com/docs/#installation].

After installing OpenAI Gym, have the ‘cartpole’ environment play out on your screen so that you can visualize it. Then, copy the 2 first techniques used in (this)[http://kvfrans.com/simple-algoritms-for-solving-cartpole/] blog post in your local environment (skip the last policy gradients one, we’ll cover that later on in the course). Observe and analyze how both strategies work differently. What are their pros and cons? Once you’ve run all 2, thats it for this first week, congratulations!   Siraj


# Answer

The two techniques we looked at were Random Search and Hill Climb. Random Search as its name implies involves trying random weights and picking the one that performs the best. Hill Climb starts with random initial weights and at each episode, it adds some noise to the weights and keeps the new weights if the agent improves. Hill Climb aims to gradually improve the weights instead of Random Search's approach of jumping around and hoping to find a working combination. There is a noise scaling parameter used to further control the rate of change. The author notes that if noise scaling is high enough in comparison to the weights, Hill Climb becomes the same as Random Search.

Following the blog author's approach, I ran multiple iterations (10 in this case) of both algorithms. A list was used to track the number of episodes it took for the algorithm to reach the target reward of 200. The average episode was calculated from this list.

After running through Random Search and seeing the randomness of the results, my initial thought was that Hill Climb would do a much better job. Surely, an algorithm that promises to iteratively improve on previous attempts vs. one that randomly picks would be better. To my disappointment, I was proven wrong. Random Search reached the target reward of 200 faster than Hill Climb did.



### Random Search - 10 iterations

| i  | Average Episode | Result |
| -  |  -     | - |
| 1  | 18.2 | [8, 6, 28, 3, 26, 7, 18, 26, 53, 7]
| 2  | 11.1 | [53, 6, 7, 1, 7, 6, 17, 0, 11, 3]
| 3  | 12.4 | [1, 10, 12, 1, 23, 7, 9, 34, 18, 9]
| 4  | 10.1 | [10, 4, 11, 5, 24, 1, 9, 13, 5, 19]
| 5  | 15.6 | [17, 30, 24, 17, 17, 2, 18, 22, 7, 2]
| 6  | 4.1 | [1, 8, 1, 2, 12, 2, 6, 5, 3, 1]
| 7  | 14.5 | [12, 13, 6, 1, 15, 8, 8, 41, 30, 11]
| 8  | 11.7 | [4, 47, 10, 24, 0, 4, 0, 9, 12, 7]
| 9  | 8.3 | [8, 2, 4, 10, 0, 11, 13, 13, 18, 4]
| 10 | 8.8 | [16, 2, 27, 2, 10, 9, 1, 11, 9, 1]


### Hill Climb - 10 iterations
noise_scaling = 0.1 (author's original value)

| i  | Average Episode | Result |
| -  |  -     | - |
| 1  | 5099.8 | [9999, 2, 54, 46, 9999, 9999, 881, 9999, 20, 9999]
| 2  | 6031.6 | [9999, 9999, 9999, 2, 8, 305, 9999, 7, 9999, 9999]
| 3  | 8009.0 | [9999, 9999, 9999, 9, 89, 9999, 9999, 9999, 9999, 9999]
| 4  | 4360.9 | [9999, 591, 6, 2061, 9999, 828, 9999, 121, 9999, 6]
| 5  | 5007.3 | [11, 0, 9999, 9999, 9999, 3, 9999, 9999, 2, 62]
| 6  | 7474.3 | [9999, 9999, 9999, 12, 2705, 2033, 9999, 9999, 9999, 9999]
| 7  | 7133.1 | [9999, 1204, 0, 9999, 9999, 9999, 9999, 9999, 134, 9999]
| 8  | 4039.8 | [9999, 252, 9999, 9999, 9999, 5, 0, 0, 0, 145]
| 9  | 5385.3 | [3, 9999, 3698, 9999, 105, 22, 9999, 30, 9999, 9999]
| 10 | 4025.7 | [9999, 9999, 2, 107, 0, 9, 82, 9999, 9999, 61]


### Hill Climb - 10 iterations
noise_scaling = 0.5

| i  | Average Episode | Result |
| -  |  -     | - |
| 1  | 2042.5 | [9999, 36, 27, 8, 9999, 0, 13, 336, 1, 6]
| 2  | 2046.8 | [53, 39, 16, 9, 9999, 118, 19, 9999, 4, 212]
| 3  | 120.7 | [15, 0, 25, 4, 1, 104, 1001, 12, 20, 25]
| 4  | 1019.5 | [7, 0, 9999, 55, 6, 10, 0, 40, 12, 66]
| 5  | 4033.4 | [0, 9999, 298, 5, 9999, 0, 11, 9999, 24, 9999]
| 6  | 2013.3 | [3, 9999, 38, 7, 5, 6, 59, 10, 9999, 7]
| 7  | 2048.1 | [1, 19, 13, 9999, 51, 12, 10, 183, 9999, 194]
| 8  | 2005.5 | [6, 7, 9999, 4, 0, 1, 5, 9999, 21, 13]
| 9  | 1069.4 | [15, 9999, 31, 11, 15, 7, 555, 2, 34, 25]
| 10 | 36.2 | [0, 5, 1, 11, 0, 16, 125, 33, 168, 3]


### Hill Climb - 10 iterations
noise_scaling = 0.95

| i | Average Episode | Result |
| - |  -     | - |
| 1 | 33.3 | [12, 63, 56, 8, 0, 135, 26, 1, 5, 27] | 
| 2 | 38.1 | [150, 17, 12, 22, 88, 42, 29, 15, 0, 6] | 
| 3 | 2009.1 | [37, 10, 5, 9999, 10, 4, 2, 9999, 4, 21] | 
| 4 | 10.6 | [8, 2, 11, 13, 13, 0, 23, 17, 13, 6] | 
| 5 | 488.5 | [10, 37, 44, 4716, 17, 3, 17, 25, 15, 1] | 
| 6 | 1007.9 | [0, 12, 1, 17, 9999, 22, 2, 1, 20, 5] | 
| 7 | 17.4 | [9, 1, 36, 1, 7, 1, 32, 26, 59, 2]
| 8 | 27.6 | [17, 3, 19, 67, 8, 4, 106, 8, 16, 28]
| 9 | 1075.7 | [7, 5, 9999, 22, 8, 34, 601, 5, 76, 0]
| 10 | 10.6 | [29, 9, 1, 4, 4, 17, 5, 2, 12, 23]


 



| i  | Average Episode | Result |
| -  |  -     | - |
| 1  | 
| 2  | 
| 3  | 
| 4  | 
| 5  | 
| 6  | 
| 7  | 
| 8  | 
| 9  | 
| 10 | 
