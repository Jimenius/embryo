'''

Created by Minhui Li on April 13, 2021
'''


from embryo.agent.base import Agent
from embryo.agent.utils import test_episode
from embryo.brain.memory.linked import LinkedMemory
from embryo.brain.plan.plan import PlanEnv
from embryo.ion import Ion
from embryo.limbs.limbs import OffPolicyLimbs


class DynaAgent(Agent):
    '''Off-policy agent

    An off-policy agent optimizes its policy  
    '''

    def _build(
        self,
    ) -> None:
        '''
        '''

        super()._build()
        solution = self.config.SOLUTION
        self.model_train_frequency = solution.MODEL_TRAIN_FREQUENCY
        ratio = solution.ENV_SAMPLE_RATIO
        if 0. <= ratio <= 1.:
            self.env_sample_ratio = ratio
            self.env_batch_size = int(self.batch_size * ratio)
            self.model_batch_size = self.batch_size - self.env_batch_size
        else:
            raise ValueError(
                'Invalid environment sample ratio value: {}'.format(ratio),
            )

    def _train_model(
        self,
    ) -> None:
        '''
        '''

        batch, indice = self.memory.sample(batch_size=0)
        self.virtual_model.update(batch)

    def _rollout(
        self,
        size: int = 1,
    ) -> None:
        '''
        '''

        # Set rollout length
        # Set model buffer
        venv = PlanEnv(
            network=self.virtual_model,
            memory=self.memory,
            num_envs=size,
        )
        memory = LinkedMemory(
            max_size=self.size,
        )
        virtual_limbs = OffPolicyLimbs(
            env=venv,
            central=self.central,
            memory=memory,
        )
        result = virtual_limbs.interact(num_step=self.step)

    def learn(
        self,
    ):
        '''
        '''

        best_epoch, best_reward, best_reward_std = -1, 0., 0.

        for epoch in range(1, self.max_epoch + 1):
            start_step = self.env_step
            self.central.train()
            while self.env_step < start_step + self.step_per_epoch:

                if self.env_step % self.model_train_frequency == 0 \
                    and self.env_sample_ratio < 1.:
                    self._train_model()
                    self._rollout()

                result = self.train_limbs.interact(num_step=self.step_per_interact)
                self.env_step += int(result['step'])

                for i in range(int(self.step_per_interact * self.update_per_step)):
                    self.gradient_step += 1
                    batch, indice = self.memory.sample(
                        batch_size=self.env_batch_size,
                    )
                    batch = self.return_method(
                        memory=self.memory,
                        indice=indice,
                        target_fn=self.central.get_target_value,
                        n=self.nstep,
                        gamma=self.gamma,
                    )
                    batch, indice = self.model_memory.sample(
                        batch_size=self.model_batch_size,
                    )

                    loss = self.central.update(batch)

                    if self.writer:
                        pass

            result = test_episode(
                central=self.central,
                limbs=self.test_limbs,
                step=self.env_step,
                num_episode=self.test_num_episode,
                test_fn=self.test_fn,
            )

            if best_epoch == -1 or best_reward <= result['reward']:
                best_epoch = epoch
                best_reward = result['reward']
                best_reward_std = result['reward_std']

            # Log result
            if self.writer:
                pass