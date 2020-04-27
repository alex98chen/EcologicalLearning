from gan_agent import GANAgent
from rnd_agent import RNDAgent
from generative_agent import GenerativeAgent
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import numpy as np
import pickle


def main(run_id=0, rollout=0):
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    train_method = default_config['TrainMethod']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_render = True
    model_path = 'models/{}_{}_run{}_model_{}.pt'.format(env_id, train_method, run_id, rollout)
    if train_method == 'RND':
        model_path = 'models/{}_run{}_model_{}.pt'.format(env_id, run_id, rollout)
        predictor_path = 'models/{}_run{}_pred_{}.pt'.format(env_id, run_id, rollout)
        target_path = 'models/{}_run{}_target_{}.pt'.format(env_id, run_id, rollout)
    elif train_method == 'generative':
        predictor_path = 'models/{}_{}_run{}_vae_{}.pt'.format(env_id, train_method, run_id, rollout)
    elif train_method == 'gan':
        netg_path = 'models/{}_{}_run{}_netg_{}.pt'.format(env_id, train_method, run_id, rollout)
        netd_path = 'models/{}_{}_run{}_netd_{}.pt'.format(env_id, train_method, run_id, rollout)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    hidden_dim = int(default_config['HiddenDim'])

    if train_method == 'RND':
        agent = RNDAgent
    elif train_method == 'generative':
        agent = GenerativeAgent
    elif train_method == 'GAN':
        agent = GANAgent
    else:
        raise NotImplementedError

    if default_config['EnvType'] == 'atari':
        env_type = AtariVideoEnvironment
    else:
        raise NotImplementedError

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        hidden_dim = hidden_dim
    )

    print('load model...')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        if train_method == 'RND':
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
        elif train_method == 'generative':
            agent.vae.load_state_dict(torch.load(predictor_path))
        elif train_method == 'GAN':
            agent.netG.load_state_dict(torch.load(netg_path))
            agent.netD.load_state_dict(torch.load(netd_path))
    else:
        agent.model.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        if train_method == 'RND':
            agent.rnd.predictor.load_state_dict(
                torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(
                torch.load(target_path, map_location='cpu'))
        elif train_method == 'generative':
            agent.vae.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        elif train_method == 'GAN':
            agent.netG.load_state_dict(torch.load(netg_path, map_location='cpu'))
            agent.netD.load_state_dict(torch.load(netd_path, map_location='cpu'))
    print('load finished!')

    works = []
    parent_conns = []
    child_conns = []

    name = '{}_{}_{}'.format(train_method, run_id, rollout)

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, name, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr, _ = parent_conn.recv()
            rall += r
            next_states = s.reshape([1, 4, 84, 84])
            next_obs = s[3, :, :].reshape([1, 1, 84, 84])

        # total reward = int reward + ext Reward
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]

        if rd:
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', help='run identifier', type=int, default=0)
    parser.add_argument('--rollout', help='rollout identifier', type=int, default=0)
    args = parser.parse_args()
    main(run_id=args.run_id,
         rollout=args.rollout)
