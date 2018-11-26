import argparse
import general_utilities
import os
import random
import simple_tag_utilities
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from maddpg import Actor, Critic
from memory import Memory
from make_env import make_env
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

# Prevent Tensorflow C++ logs from flooding the output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_expanded_list(base_string, n):
    """Expand a base string into n number of column labels, returned as a list"""
    return [f"{base_string}_{j}" for j in range(n)]


def get_stats_df(args, env):
    """Create the stats_df pandas dataframe, to hold our statistics"""
    # create all column labels for the pandas dataframe
    feature_list = ["episode", "steps"]
    compressed_features = ["reward", "loss", "collisions", "ou_theta", "ou_mu",
                           "ou_sigma", "ou_dt", "ou_x0"]
    for feature in compressed_features:
        feature_list += get_expanded_list(feature, env.n)

    # initialize the pandas dataframe to all zeros
    my_tuple = (args.episodes, len(feature_list))
    np_zeros = np.zeros(my_tuple)
    empty_df = pd.DataFrame(np_zeros, columns=feature_list)

    # cast episode and step columns to int
    empty_df[['episode', 'steps']] = empty_df[['episode', 'steps']].astype(int)

    # cast all ou_mu data types to np.ndarrays
    compressed_features_mu = ["ou_mu"]
    for mu_feature in compressed_features_mu:
        mu_list = get_expanded_list(mu_feature, env.n)
        for label in mu_list:
            empty_df[[label]] = empty_df[[label]].astype(np.ndarray)

    return empty_df


def write_stats_row(env, stats_df, episode, steps, episode_rewards, episode_losses, collision_count):
    """Write a single row of statistics into the stats_df dataframe"""
    stats_df.loc[stats_df.index[episode], 'episode'] = episode
    stats_df.loc[stats_df.index[episode], 'steps'] = steps

    collision_list = collision_count.tolist()
    mu_list = [str(actors_noise[i].mu.tolist()) for i in range(env.n)]
    for i in range(env.n):
        stats_df.loc[stats_df.index[episode], f"reward_{i}"] = episode_rewards[i]
        stats_df.loc[stats_df.index[episode], f"loss_{i}"] = episode_losses[i]
        stats_df.loc[stats_df.index[episode], f"collisions_{i}"] = collision_list[i]
        stats_df.loc[stats_df.index[episode], f"ou_theta_{i}"] = actors_noise[i].theta
        stats_df.loc[stats_df.index[episode], f"ou_mu_{i}"] = mu_list[i]
        stats_df.loc[stats_df.index[episode], f"ou_sigma_{i}"] = actors_noise[i].sigma
        stats_df.loc[stats_df.index[episode], f"ou_dt_{i}"] = actors_noise[i].dt
        stats_df.loc[stats_df.index[episode], f"ou_x0_{i}"] = actors_noise[i].x0


def play(checkpoint_interval, weights_filename_prefix, csv_filename_prefix, batch_size, stats_df):
    """Doc-string here"""
    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0

        while True:
            steps += 1

            # render
            if args.render:
                env.render()

            # act
            actions = []
            for i in range(env.n):
                action = np.clip(
                    actors[i].choose_action(states[i]) + actors_noise[i](), -2, 2)
                actions.append(action)

            # step
            states_next, rewards, done, info = env.step(actions)

            # learn
            if not args.testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)

                for i in range(env.n):
                    if done[i]:
                        rewards[i] -= 500

                    memories[i].remember(states, actions, rewards[i],
                                         states_next, done[i])

                    if memories[i].pointer > batch_size * 10:
                        s, a, r, sn, _ = memories[i].sample(batch, env.n)
                        r = np.reshape(r, (batch_size, 1))
                        loss = critics[i].learn(s, a, r, sn)
                        actors[i].learn(actors, s)
                        episode_losses[i] += loss
                    else:
                        episode_losses[i] = -1

            states = states_next
            episode_rewards += rewards
            collision_count += np.array(
                simple_tag_utilities.count_agent_collisions(env))

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_losses = episode_losses / steps

                write_stats_row(env, stats_df, episode, steps, episode_rewards, episode_losses, collision_count)

                if episode % 25 == 0:
                    print(stats_df.iloc[episode])
                break

        if episode % checkpoint_interval == 0:
            stats_file = f"{csv_filename_prefix}_{episode}.h5"
            store = pd.HDFStore(stats_file)
            store['stats_df'] = stats_df
            print(f"stats_df saved to {stats_file}")

            if not os.path.exists(weights_filename_prefix):
                os.makedirs(weights_filename_prefix)
            save_path = saver.save(session, os.path.join(
                weights_filename_prefix, "models"), global_step=episode)

    stats_file = f"{csv_filename_prefix}_{args.episodes}.h5"
    store = pd.HDFStore(stats_file)
    store['stats_df'] = stats_df
    # return statistics


def get_args():
    """Argparse stuff here"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--video_dir', default='videos/', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--good_agents', default=1, type=int)
    parser.add_argument('--adversaries', default=1, type=int)
    parser.add_argument('--collision_reward', default=500, type=int)
    parser.add_argument('--episodes', default=100000, type=int)
    parser.add_argument('--video_interval', default=1000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-maddpg',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-maddpg',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--load_weights_from_file', default='',
                        help="where to load network weights")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ou_mus', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise mus for each action for each agent")
    parser.add_argument('--ou_sigma', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise sigma for each agent")
    parser.add_argument('--ou_theta', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise theta for each agent")
    parser.add_argument('--ou_dt', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise dt for each agent")
    parser.add_argument('--ou_x0', nargs='+', type=float,
                        help="OrnsteinUhlenbeckActionNoise x0 for each agent")

    return parser.parse_args()


def get_ou_mus(args, env):
    """A doc-string"""
    if args.ou_mus is not None:
        if len(args.ou_mus) == sum([env.action_space[i].n for i in range(env.n)]):
            ou_mus = []
            prev_idx = 0
            for space in env.action_space:
                ou_mus.append(
                    np.array(args.ou_mus[prev_idx:prev_idx + space.n]))
                prev_idx = space.n
            print("Using ou_mus: {}".format(ou_mus))
        else:
            raise ValueError(
                "Must have enough ou_mus for all actions for all agents")
    else:
        ou_mus = [np.zeros(env.action_space[i].n) for i in range(env.n)]
    return ou_mus


def get_ou_list(args_key, default_val, args, env):
    """A doc-string"""
    d = vars(args)
    retval = None
    if args_key in args:
        x = getattr(args, args_key)
        if x is not None:
            if len(x) == env.n:
               retval = x
            else:
                raise ValueError(f"Must have enough {args_key} for all agents")
        else:
            retval = [default_val for _ in range(env.n)]
    else:
        raise ValueError(f"No such argument: '{args_key}'")
    return retval



if __name__ == '__main__':
    """The main function"""
    args = get_args()
    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")

    # init env
    env = make_env(args.env, args.good_agents, args.adversaries, args.collision_reward, args.benchmark)
    stats_df = get_stats_df(args, env)

    # Extract ou initialization values
    ou_mus = get_ou_mus(args, env)
    ou_sigma = get_ou_list('ou_sigma', 0.3, args, env)
    ou_theta = get_ou_list('ou_theta', 0.15, args, env)
    ou_dt = get_ou_list('ou_dt', 1e-2, args, env)
    ou_x0 = get_ou_list('ou_x0', None, args, env)

    # set random seed
    env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init actors and critics
    session = tf.Session()

    n_actions = []
    actors = []
    actors_noise = []
    memories = []
    eval_actions = []
    target_actions = []
    state_placeholders = []
    state_next_placeholders = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]
        state = tf.placeholder(tf.float32, shape=[None, state_size])
        state_next = tf.placeholder(tf.float32, shape=[None, state_size])
        speed = 0.8 if env.agents[i].adversary else 1

        actors.append(Actor('actor' + str(i), session, n_action, speed,
                            state, state_next))
        actors_noise.append(OrnsteinUhlenbeckActionNoise(
            mu=ou_mus[i],
            sigma=ou_sigma[i],
            theta=ou_theta[i],
            dt=ou_dt[i],
            x0=ou_x0[i]))
        memories.append(Memory(args.memory_size))

        n_actions.append(n_action)
        eval_actions.append(actors[i].eval_actions)
        target_actions.append(actors[i].target_actions)
        state_placeholders.append(state)
        state_next_placeholders.append(state_next)

    critics = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        reward = tf.placeholder(tf.float32, [None, 1])

        critics.append(Critic('critic' + str(i), session, n_actions,
                              eval_actions, target_actions, state_placeholders,
                              state_next_placeholders, reward))
        actors[i].add_gradients(critics[i].action_gradients[i])

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10000000)

    if args.load_weights_from_file != "":
        saver.restore(session, args.load_weights_from_file)
        print("restoring from checkpoint {}".format(
            args.load_weights_from_file))

    start_time = time.time()

    # play
    play(args.checkpoint_frequency, args.experiment_prefix + args.weights_filename_prefix,
         args.experiment_prefix + args.csv_filename_prefix, args.batch_size, stats_df)

    # bookkeeping
    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    tf.summary.FileWriter(args.experiment_prefix +
                          args.weights_filename_prefix, session.graph)
    save_path = saver.save(session, os.path.join(
        args.experiment_prefix + args.weights_filename_prefix, "models"), global_step=args.episodes)
    print("saving model to {}".format(save_path))
