# Homework Assignment 7 (Neuroevolution)

This weeks homework assignment is to design a neuro-evolution algorithm  that will learn how to optimally play any one of the OpenAI Gym environments. [Here](https://github.com/ikergarcia1996/NeuroEvolution-Flappy-Bird) is an example notebook to get started. If your agent can learn how to play the game using both neural networks and an evolutionary algorithm, you’ll successfully complete the assignment. Good luck!


# Answer

Look at [Homework 07 Python File](sonic_winner.py).


# Report

We primarily used the Sonic AI Bot in OpenAI and NEAT Tutorial Series for this project. This tutorial series used OpenAI's Gym Retro library and the NEAT-Python library.

Gym Retro is a platform which allows for support of video game emulators. Its release increases the number of Gym compatible games from around 70 Atari games and 30 Sega games to over 1,000 games across a variety of backing emulators. The supported systems currently include Atari, NEC, Nintendo and Sega.

NEAT is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. NEAT-Python is a pure Python implementation of NEAT, with no dependencies other than the Python library.

The original repository included two files:

- tut1.py: it took the same action repeatedly at each step and rendered the game
- tut2.py: it contained an eval_genomes() function with a reward function that took into consideration the reward at the end of each game iteration


**Experiments**

We used the second file as our base and run through it to get a sense of how well Sonic was able to play through the level. We then run several different experiments involving different reward functions. These reward functions added additional parameters to the reward given at the end of each game iteration:

1. Taking into consideration how far Sonic was able to get in the level
2. Taking into consideration Sonic's momentum 
3. Taking into consideration the number of rings Sonic captured
4. Taking into consideration the number of rings Sonic captured while ensuring a lower score for subsequent rings. As rings protected Sonic when he hit an enemy or some hurtful object, we wanted to prioritize him capturing the initial several rings but make it less important to capture them after he had several in his possession


| Experiment | File Name |
| :--------: | --------- |
|     0      | sonic\_neat\_raw\_score.py |
|     1      | sonic\_neat\_raw\_score\_plus\_x\_position.py |
|     2      | sonic\_raw\_score\_and\_momentum.py |
|     3      | sonic\_winner.py |
|     4      | sonic\_neat\_raw\_score\_x\_pos\_harmonic\_rings.py |


**Results**

The best performing reward function was the one where Sonic prioritized capturing the number of rings without decreasing the importance of any. We had observed several factors contributed to the points Sonic attained during a run. Capturing more rings clearly led to higher points along with the further Sonic was able to go in the level.


# Resources

- [Youtube: Sonic AI Bot Using Open-AI and NEAT Tutorial Series](https://www.youtube.com/watch?v=pClGmU1JEsM&list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS)
- [Sonic Bot in OpenAI and NEAT Gitlab Repo](https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT)
- [OpenAI Retro Github Repo](https://github.com/openai/retro)
- [OpenAI Gym Retro Blog Post](https://blog.openai.com/gym-retro/)
- [NEAT Python Documentation](https://neat-python.readthedocs.io/en/latest/)