from argparse import ArgumentParser as AP
import logging
import gym
from embryo.agent.agent import Agent
from embryo.config.defaults import DEFAULT
from embryo.utils.log import setup

def parse_args():
    parser = AP()
    parser.add_argument('--config-file', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = DEFAULT.merge_from_file(args.config_file)
    setup(directory=cfg.OUTPUT_DIR)
    logging.info('Full config:')
    logging.info(cfg)
    env_name = cfg.ENVIRONMENT.NAME
    train_env = gym.vector.make(env_name, cfg.ENVIRONMENT.TRAIN_NUM)
    agent = Agent(cfg.AGENT)
    agent.learn(train_env)
    test_env = gym.vector.make(env_name, cfg.ENVIRONMENT.TEST_NUM)
    agent.interact(test_env)

if __name__ == '__main__':
    main()