import pickle
import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from rnd_agent import RNDAgent
from generative_agent import GenerativeAgent
from envs import *
from utils import *
from config import *


def main(run_id=0, checkpoint=None, save_interval=1000):
    print({section: dict(config[section]) for section in config.sections()})

    train_method = default_config['TrainMethod'] # TrainMethod = RND/generative/GAN

    # Create environment
    env_id = default_config['EnvID']  # MontezumaRevengeNoFrameskip-v4
    env_type = default_config['EnvType'] # atari

    if env_type == 'mario':
        print('Mario environment not fully implemented - thomaseh')
        raise NotImplementedError
        env = BinarySpaceToDiscreteSpaceEnv(
            gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id) # Create environment
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4 # Box(210, 160, 3)
    output_size = env.action_space.n  # 2 # Discrete(18)

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    # Load configuration parameters
    is_load_model = checkpoint is not None
    is_render = False
    model_path = 'models/{}_{}_run{}_model'.format(env_id, train_method, run_id)
    if train_method == 'RND':
        predictor_path = 'models/{}_{}_run{}_pred'.format(env_id, train_method, run_id)
        target_path = 'models/{}_{}_run{}_target'.format(env_id, train_method, run_id)
    elif train_method == 'generative':
        predictor_path = 'models/{}_{}_run{}_vae'.format(env_id, train_method, run_id)
   

    writer = SummaryWriter(comment='_{}_{}_run{}'.format(env_id, train_method, run_id))

    use_cuda = default_config.getboolean('UseGPU') # True
    use_gae = default_config.getboolean('UseGAE') # True
    use_noisy_net = default_config.getboolean('UseNoisyNet') # False

    lam = float(default_config['Lambda']) # 0.95
    num_worker = int(default_config['NumEnv']) # 128

    num_step = int(default_config['NumStep']) # 128 length of rollout
    num_rollouts = int(default_config['NumRollouts']) # 0
    num_pretrain_rollouts = int(default_config['NumPretrainRollouts']) # 200

    ppo_eps = float(default_config['PPOEps']) # 0.1
    epoch = int(default_config['Epoch']) # 4
    mini_batch = int(default_config['MiniBatch']) # 4
    batch_size = int(num_step * num_worker / mini_batch) # 128 * 128 / 4
    learning_rate = float(default_config['LearningRate']) # 1e-4
    entropy_coef = float(default_config['Entropy']) # 0.001
    gamma = float(default_config['Gamma']) # 0.999
    int_gamma = float(default_config['IntGamma']) # 0.99
    clip_grad_norm = float(default_config['ClipGradNorm']) # 0.5
    ext_coef = float(default_config['ExtCoef']) # 2
    int_coef = float(default_config['IntCoef']) # 1

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb']) # 0.25
    life_done = default_config.getboolean('LifeDone') # False

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep']) # Number of initial steps for initializing observation normalization # 50
    discounted_reward = RewardForwardFilter(int_gamma)

    if train_method == 'RND':
        agent = RNDAgent
    elif train_method == 'generative':
        agent = GenerativeAgent
    else:
        raise NotImplementedError

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    # Initialize agent
    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma, # 0.999
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    # Load pre-existing model
    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            if train_method == 'RND':
                agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
                agent.rnd.target.load_state_dict(torch.load(target_path))
            elif train_method == 'generative':
                agent.vae.load_state_dict(torch.load(predictor_path))
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
        print('load finished!')

    # Create workers to run in environments
    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(
            env_id, is_render, idx, child_conn, sticky_action=sticky_action,
            p=action_prob, life_done=life_done,
        )
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84], dtype='float32') # [128, 4, 84, 84]

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # Initialize observation normalizers
    print('Start to initialize observation normalization parameter...')
    next_obs = np.zeros([num_worker * num_step, 1, 84, 84]) #[128 * 128, 1, 84, 84]
    for step in range(num_step * pre_obs_norm_step): # 128 * 50
        actions = np.random.randint(0, output_size, size=(num_worker,)) #[128]

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for idx, parent_conn in enumerate(parent_conns):
            s, r, d, rd, lr, _ = parent_conn.recv()#4 steps of observation, reward, force_done, done, log_reward,[self.rall, self.steps]])
            next_obs[(step % num_step) * num_worker + idx, 0, :, :] = s[3, :, :]

        if (step % num_step) == num_step - 1:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = np.zeros([num_worker * num_step, 1, 84, 84])
    print('End to initialize...') # The detail process, parallel?

    # Initialize stats dict
    stats = {
        'total_reward': [],
        'ep_length': [],
        'num_updates': [],
        'frames_seen': [],
    }

    # Main training loop
    while True:
        total_state = np.zeros([num_worker * num_step, 4, 84, 84], dtype='float32')
        total_next_obs = np.zeros([num_worker * num_step, 1, 84, 84])
        total_reward, total_done, total_next_state, total_action, \
            total_int_reward, total_ext_values, total_int_values, total_policy, \
            total_policy_np = [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout (collect data)
        for step in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(states/255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_obs = np.zeros([num_worker, 1, 84, 84])
            next_states = np.zeros([num_worker, 4, 84, 84])
            rewards, dones, real_dones, log_rewards = [], [], [], [] # what's the difference between the dones and real_dones, rewards and log_rewards?
            for idx, parent_conn in enumerate(parent_conns):
                s, r, d, rd, lr, stat = parent_conn.recv()#4 steps of observation, reward, force_done, done, log_reward,[self.rall, self.steps]])
                next_states[idx] = s # The next state of idx th worker
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs[idx, 0] = s[3, :, :]
                total_next_obs[idx * num_step + step, 0] = s[3, :, :]

                if rd:
                    stats['total_reward'].append(stat[0])
                    stats['ep_length'].append(stat[1])
                    stats['num_updates'].append(global_update)
                    stats['frames_seen'].append(global_step)

            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # Compute total reward = intrinsic reward + external reward
            next_obs -= obs_rms.mean
            next_obs /= np.sqrt(obs_rms.var)
            next_obs.clip(-5, 5, out=next_obs)
            intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            for idx, state in enumerate(states):
                total_state[idx * num_step + step] = state
            total_int_reward.append(intrinsic_reward)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        total_state /= 255.
        total_next_obs -= obs_rms.mean
        total_next_obs /= np.sqrt(obs_rms.var)
        total_next_obs.clip(-5, 5, out=total_next_obs)

        agent.train_model(total_state, ext_target, int_target, total_action,
                          total_adv, total_next_obs, total_policy)

        global_step += (num_worker * num_step)
        global_update += 1
        if global_update % save_interval == 0:
            print('Saving model at global step={}, num rollouts={}.'.format(
                global_step, global_update))
            torch.save(agent.model.state_dict(), model_path + "_{}.pt".format(global_update))
            if train_method == 'RND':
                torch.save(agent.rnd.predictor.state_dict(), predictor_path + '_{}.pt'.format(global_update))
                torch.save(agent.rnd.target.state_dict(), target_path + '_{}.pt'.format(global_update))
            elif train_method == 'generative':
                torch.save(agent.vae.state_dict(), predictor_path + '_{}.pt'.format(global_update))

            # Save stats to pickle file
            with open('models/{}_{}_run{}_stats_{}.pkl'.format(env_id, train_method, run_id, global_update),'wb') as f:
                pickle.dump(stats, f)

        if global_update == num_rollouts + num_pretrain_rollouts:
            print('Finished Training.')
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--checkpoint', help='checkpoint run identifier', type=int, default=None)
    parser.add_argument('--save_interval', help='save every ___ rollouts', type=int, default=1000)
    args = parser.parse_args()
    main(run_id=args.run_id,
         checkpoint=args.checkpoint,
         save_interval=args.save_interval)
