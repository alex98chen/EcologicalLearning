import pickle
import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from predictive_agent import PredictiveAgent
from generative_agent import GenerativeAgent
from envs import *
from utils import *
from config import *


def main(run_id=0, checkpoint=None, rec_interval=10, save_interval=100):
    print({section: dict(config[section]) for section in config.sections()})

    train_method = default_config['TrainMethod']

    # Create environment
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        print('Mario environment not fully implemented - thomaseh')
        raise NotImplementedError
        env = BinarySpaceToDiscreteSpaceEnv(
            gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    # Load configuration parameters
    is_load_model = checkpoint is not None
    is_render = False
    model_path = 'models/{}_{}_run{}_model'.format(env_id, train_method, run_id)
    vae_path = 'models/{}_{}_run{}_vae'.format(env_id, train_method, run_id)
    if train_method == 'predictive':
        predictor_path = 'models/{}_{}_run{}_predictor'.format(env_id, train_method, run_id)


    writer = SummaryWriter(logdir='runs/{}_{}_run{}'.format(env_id, train_method, run_id))

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])
    num_rollouts = int(default_config['NumRollouts'])
    num_pretrain_rollouts = int(default_config['NumPretrainRollouts'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    hidden_dim = int(default_config['HiddenDim'])

    if train_method == 'predictive':
        agent = PredictiveAgent
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
        update_proportion=1.0,
        hidden_dim=hidden_dim
    )

    # Load pre-existing model
    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.vae.load_state_dict(torch.load(vae_path))
            if train_method == 'predictive':
                agent.predictor.load_state_dict(torch.load(predictor_path))
        else:
            agent.model.load_state_dict(
                torch.load(model_path, map_location='cpu'))
            agent.vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
            if train_method == 'predictive':
                agent.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
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

    states = np.zeros([num_worker, 4, 84, 84], dtype='float32')

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # Initialize observation normalizers
    # print('Start to initialize observation normalization parameter...')
    # next_obs = np.zeros([num_worker * num_step, 1, 84, 84])
    # for step in range(num_step * pre_obs_norm_step):
    #     actions = np.random.randint(0, output_size, size=(num_worker,))

    #     for parent_conn, action in zip(parent_conns, actions):
    #         parent_conn.send(action)

    #     for idx, parent_conn in enumerate(parent_conns):
    #         s, r, d, rd, lr, _ = parent_conn.recv()
    #         next_obs[(step % num_step) * num_worker + idx, 0, :, :] = s[3, :, :]

    #     if (step % num_step) == num_step - 1:
    #         next_obs = np.stack(next_obs)
    #         obs_rms.update(next_obs)
    #         next_obs = np.zeros([num_worker * num_step, 1, 84, 84])
    # print('End to initialize...')

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

        # Step 1. n-step rollout (collect data)
        for step in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(states / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_obs = np.zeros([num_worker, 1, 84, 84])
            next_states = np.zeros([num_worker, 4, 84, 84])
            rewards, dones, real_dones, log_rewards = [], [], [], []
            for idx, parent_conn in enumerate(parent_conns):
                s, r, d, rd, lr, stat = parent_conn.recv()
                next_states[idx] = s
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
            # next_obs -= obs_rms.mean
            # next_obs /= np.sqrt(obs_rms.var)
            # next_obs.clip(-5, 5, out=next_obs)
            intrinsic_reward = agent.compute_intrinsic_reward(next_obs / 255.)
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

        writer.add_scalar('data/raw_int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/raw_int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)

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
        # obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        # random_obs_choice = np.random.randint(total_next_obs.shape[0])
        # random_obs = total_next_obs[random_obs_choice].copy()
        total_next_obs /= 255.
        # total_next_obs -= obs_rms.mean
        # total_next_obs /= np.sqrt(obs_rms.var)
        # total_next_obs.clip(-5, 5, out=total_next_obs)

        predict_losses = None
        kld_losses = None
        if global_update < num_pretrain_rollouts:
            if train_method == 'predictive':
                recon_losses = agent.train_just_vae(total_state / 255., total_next_obs)
            else:
                recon_losses, kld_losses = agent.train_just_vae(total_state / 255., total_next_obs)
        else:
            if train_method == 'predictive':
                recon_losses, predict_losses = agent.train_model(total_state / 255., ext_target, int_target, total_action,
                        total_adv, total_next_obs, total_policy)
            else:
                recon_losses, kld_losses = agent.train_model(total_state / 255., ext_target, int_target, total_action,
                        total_adv, total_next_obs, total_policy)

        writer.add_scalar('data/reconstruction_loss_per_rollout', np.mean(recon_losses), global_update)
        if kld_losses is not None:
            writer.add_scalar('data/kld_loss_per_rollout', np.mean(kld_losses), global_update)
        if predict_losses is not None:
            writer.add_scalar('data/predict_loss_per_rollout', np.mean(predict_losses), global_update)

        global_step += (num_worker * num_step)
        
        if global_update % rec_interval == 0:
            with torch.no_grad():
                # random_obs_norm = total_next_obs[random_obs_choice]
                # reconstructed_state = agent.reconstruct(random_obs_norm)

                # random_obs_norm = (random_obs_norm - random_obs_norm.min()) / (random_obs_norm.max() - random_obs_norm.min())
                # reconstructed_state = (reconstructed_state - reconstructed_state.min()) / (reconstructed_state.max() - reconstructed_state.min())

                # writer.add_image('Original', random_obs, global_update)
                # writer.add_image('Original Normalized', random_obs_norm, global_update)

                random_state = total_next_obs[np.random.randint(total_next_obs.shape[0])]
                reconstructed_state = agent.reconstruct(random_state)

                writer.add_image('Original', random_state, global_update)
                writer.add_image('Reconstructed', reconstructed_state, global_update)

        if global_update % save_interval == 0:
            print('Saving model at global step={}, num rollouts={}.'.format(
                global_step, global_update))
            torch.save(agent.model.state_dict(), model_path + "_{}.pt".format(global_update))
            torch.save(agent.vae.state_dict(), vae_path + '_{}.pt'.format(global_update))
            if train_method == 'predictive':
                torch.save(agent.predictor.state_dict(), predictor_path + '_{}.pt'.format(global_update))

            # Save stats to pickle file
            with open('models/{}_{}_run{}_stats_{}.pkl'.format(env_id, train_method, run_id, global_update),'wb') as f:
                pickle.dump(stats, f)

        global_update += 1

        if global_update == num_rollouts + num_pretrain_rollouts:
            print('Finished Training.')
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--checkpoint', help='checkpoint run identifier', type=int, default=None)
    parser.add_argument('--rec_interval', help='reconstruct every ___ rollouts', type=int, default=10)
    parser.add_argument('--save_interval', help='save every ___ rollouts', type=int, default=100)
    args = parser.parse_args()
    main(run_id=args.run_id,
         checkpoint=args.checkpoint,
         rec_interval=args.rec_interval,
         save_interval=args.save_interval)
