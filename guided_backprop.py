import torch
from vanilla_gradient import VanillaGradient

# https://github.com/hummat/saliency/blob/97e44d1eb2f2c05788e02cd7b685190a20b9f0da/guided_backprop.py
class GuidedBackprop(VanillaGradient):
    def __init__(self, model):
        super(GuidedBackprop, self).__init__(model)
        self.relu_inputs = list()
        self.update_relus()

    def update_relus(self):
        def clip_gradient(module, grad_input, grad_output):
            relu_input = self.relu_inputs.pop()
            return (grad_output[0] * (grad_output[0] > 0.).float() * (relu_input > 0.).float(),)

        def save_input(module, input, output):
            self.relu_inputs.append(input[0])

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_forward_hook(save_input))
                self.hooks.append(module.register_backward_hook(clip_gradient))
