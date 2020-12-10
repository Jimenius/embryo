'''

Created Nov 27, 2020 by Minhui Li
'''


from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.base import Central


@CENTRAL_REGISTRY.register()
class RandomCentral(Central):
    '''Random agent as a baseline
    '''

    def learn(self, *args, **kwargs):
        '''Random agent does not need to learn.
        '''

        pass
    
    def control(self, *args, **kwargs):
        '''Randomly sample an action from the action space.
        '''
        
        return self.env.action_space.sample()