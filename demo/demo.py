from argparse import ArgumentParser as AP
import logging

import gym
import numpy as np
import torch

from embryo.agent import AGENT_REGISTRY
from embryo.config.defaults import DEFAULT
from embryo.utils.log import setup


def parse_args():
    parser = AP()
    parser.add_argument('--config-file', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = DEFAULT.clone()
    cfg.set_new_allowed(is_new_allowed=True)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    setup(directory=output_dir)
    logging.info('Full config:\n' + str(cfg))

    env_name = cfg.ENVIRONMENT.NAME
    train_env = gym.vector.make(env_name, cfg.ENVIRONMENT.TRAIN_NUM)
    test_env = gym.vector.make(env_name, cfg.ENVIRONMENT.TEST_NUM)

    seed = cfg.SEED
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent_cls = AGENT_REGISTRY.get(cfg.AGENT.NAME)
    agent = agent_cls(
        config=cfg.AGENT,
        train_env=train_env,
        test_env=test_env,
    )
    agent.learn()
    if cfg.SAVE:
        agent.save(output_dir)
        logging.info('Saved agent information to {}.'.format(output_dir))
    agent.perform()


if __name__ == '__main__':
    main()