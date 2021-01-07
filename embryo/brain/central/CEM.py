'''

Created Dec 21, 2020 by Minhui Li
'''


from embryo.brain.central import CENTRAL_REGISTRY
from embryo.brain.central.base import Central


@CENTRAL_REGISTRY.register()
class CEMCentral(Central):
    '''Cross Entropy Method central
    '''

    def learn(self, *args, **kwargs):
        '''Random agent does not need to learn.
        '''

        pass
    
    def control(self, *args, **kwargs):
        '''Randomly sample an action from the action space.
        '''

        pass