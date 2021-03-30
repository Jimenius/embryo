import logging
import time
import gym
from embryo.brain.central.base import compute_nstep_return
from embryo.brain.central.DQN import DQNCentral
from embryo.brain.memory.linked import LinkedMemory
from embryo.limbs.limbs import OffPolicyLimbs
from embryo.utils.log import setup
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def test_episode(
    central,
    limbs,
    global_step,
    writer,
    num_episode,
    test_fn,
):
    limbs.reset()
    central.eval()
    if test_fn:
        test_fn(0, global_step)
    result = limbs.interact(num_episode=num_episode)
    if writer is not None and global_step is not None:
        for k in result.keys():
            writer.add_scalar(
                'test_{}'.format(k),
                result[k],
                global_step=global_step
            )
    return result

def main():
    train_env = gym.vector.make('Acrobot-v1', num_envs=10)
    test_env = gym.vector.make('Acrobot-v1', num_envs=10)
    gamma = 0.95
    memory_size = 20000
    batch_size = 64
    seed = 0
    max_epoch = 10
    nstep = 3
    target_update_freq = 320
    collect_per_step = 100
    step_per_epoch = 100000
    
    def train_fn(epoch, env_step):
        if env_step <= 100000:
            central.set_explore_rate(0.5)
        elif env_step <= 500000:
            eps = 0.5 - (env_step - 100000) / 1600000
            central.set_explore_rate(eps)
        else:
            central.set_explore_rate(0.25)

    def test_fn(epoch, env_step):
        central.set_explore_rate(0.)

    save_fn = None
    resume = ''
    log_interval = 1
    test_in_train = False
    verbose = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    central = DQNCentral(network=None, target_update_frequency=target_update_freq, gamma=gamma)
    memory = LinkedMemory(max_size=memory_size)
    train_limbs = OffPolicyLimbs(env=train_env, central=central, memory=memory)
    test_limbs = OffPolicyLimbs(env=test_env, central=central)
    if resume:
        central.load(resume)
    log_path = '/Users/Minhui/Downloads/temp/exp/embryo/'
    setup(directory=log_path)
    writer = SummaryWriter(log_dir=log_path)
    train_limbs.interact(num_step=batch_size)

    # offpolicy.py
    env_step, gradient_step = 0, 0
    best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0

    start_time = time.time()
    for epoch in range(1, max_epoch + 1):
        central.train()
        while env_step < step_per_epoch * epoch:
            if train_fn:
                train_fn(epoch, env_step)
            result = train_limbs.interact(num_step=collect_per_step)
            env_step += int(result['step'])
            if writer and env_step % log_interval == 0:
                writer.add_scalar(
                    'reward', result['reward'], global_step=env_step
                )
            if test_in_train:
                pass
            for i in range(1):
                gradient_step += 1
                batch, indice = memory.sample(batch_size=batch_size)
                batch = compute_nstep_return(
                    memory=memory,
                    indice=indice,
                    target_fn=central.get_target_value,
                    n=nstep,
                    gamma=gamma,
                )
                loss = central.update(batch)
                if writer and gradient_step % log_interval == 0:
                    for k in loss:
                        writer.add_scalar(
                            k, loss[k], global_step=gradient_step,
                        )

        result = test_episode(central, test_limbs, env_step, writer, num_episode=100, test_fn=test_fn)

        if best_epoch == -1 or best_reward <= result['reward']:
            best_epoch = epoch
            best_reward = result['reward']
            best_reward_std = result['reward_std']

        if verbose:
            print('Epoch {}: Test result: {}, std: {}'.format(epoch, result['reward'], result['reward_std']))
            print('Best reward: {}, std: {}'.format(best_reward, best_reward_std))


if __name__ == '__main__':
    main()