# https://github.com/hummat/saliency/blob/97e44d1eb2f2c05788e02cd7b685190a20b9f0da/saliency_mask.py
class SaliencyMask(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradient = None
        self.hooks = list()

    def get_mask(self, image_tensor, target_class=None):
        raise NotImplementedError('A derived class should implemented this method')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
