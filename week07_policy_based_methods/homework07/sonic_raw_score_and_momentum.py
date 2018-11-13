import retro
import numpy as np
import cv2 
import neat
import pickle
import statistics

# env = retro.make('SonicTheHedgehog2-Genesis', 'AquaticRuinZone.Act1')
env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')

imgarray = []

xpos_end = 0


def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        window_min_x = 0
        count_min_x = 0

        window_max_x = 0
        count_max_x = 0

        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        count = 0
        futility_count = 0
        while not done:
            
            env.render()
            frame += 1
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            #scaledimg = cv2.resize(scaledimg, (iny, inx)) 
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1) 

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            count += 1
            
            xpos = info['x']
            xpos_end = info['screen_x_end']

            if xpos < window_min_x:
                window_min_x = xpos
                count_min_x = count

            if xpos > window_max_x:
                window_max_x = xpos
                count_max_x = count

            # window_min_x = min(xpos, window_min_x)
            #window_max_x = max(xpos, window_max_x)

            direction = 0
            if count_max_x > count_min_x:
                direction = 1
            else:
                direction = -1
            momentum = direction * (window_max_x - window_min_x)

            if count % 100 == 0:
                bonus = int((momentum - 10) / 10)
                window_min_x = xpos
                count_min_x = count
                window_max_x = xpos
                count_max_x = count
                fitness_current += bonus
                print(f"{str(count).zfill(10):>11} bonus: {bonus}")

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            if xpos == xpos_end and xpos > 500:
                fitness_current += 100000
                done = True
            
            # fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 500:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current
                
            
            
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
    

