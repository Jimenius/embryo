'''

Created by Minhui Li on April 12, 2021
'''


import logging

from embryo.agent import Agent, AGENT_REGISTRY
from embryo.agent.utils import test_episode
from embryo.brain.central.base import compute_nstep_return
from embryo.ion import Ion


@AGENT_REGISTRY.register()
class OffPolicyAgent(Agent):
    '''Off-policy agent

    An off-policy agent optimizes its policy  
    '''

    def learn(
        self,
    ):
        '''Learn from environment
        '''

        best_epoch, best_reward, best_reward_std = -1, 0., 0.

        for epoch in range(1, self.max_epoch + 1):
            start_step = self.env_step
            self.central.train()
            while self.env_step < start_step + self.step_per_epoch:
                result = self.train_limbs.interact(num_step=self.step_per_interact)
                self.env_step += int(result['step'])

                for i in range(int(self.step_per_interact * self.update_per_step)):
                    self.gradient_step += 1
                    batch, indice = self.memory.sample(batch_size=self.batch_size)
                    batch = self.return_method(
                        memory=self.memory,
                        indice=indice,
                        target_fn=self.central.get_target_value,
                        n=self.nstep,
                        gamma=self.gamma,
                    )
                    loss = self.central.update(batch)

                    # if self.writer:
                    #     pass

            result = test_episode(
                central=self.central,
                limbs=self.test_limbs,
                step=self.env_step,
                num_episode=self.test_num_episode,
            )

            if best_epoch == -1 or best_reward <= result['reward']:
                best_epoch = epoch
                best_reward = result['reward']
                best_reward_std = result['reward_std']

            # Log result
            # if self.writer:
            #     pass
            ######
            logging.info(loss)
            ######
            logging.info('Epoch {}: Test result: {}, std: {}'.format(epoch, result['reward'], result['reward_std']))
            logging.info('Best reward: {}, std: {}'.format(best_reward, best_reward_std))