import torch
import sys
import numpy as np

from saliency_mask import SaliencyMask

# https://github.com/hummat/saliency/blob/97e44d1eb2f2c05788e02cd7b685190a20b9f0da/vanilla_gradient.py
class VanillaGradient(SaliencyMask):
    def __init__(self, model):
        super(VanillaGradient, self).__init__(model)

    def get_mask(self, image_tensor, target_class=None, category=None):
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits = self.model(image_tensor)
        target = torch.zeros_like(logits)

        name_order = ['Endurance_AgeAdj', 'GaitSpeed_Comp', 'Dexterity_AgeAdj', 'Strength_AgeAdj', 'PicSeq_AgeAdj', 'CardSort_AgeAdj', 'Flanker_AgeAdj', 'ReadEng_AgeAdj', 'PicVocab_AgeAdj', 'ProcSpeed_AgeAdj', 'ListSort_AgeAdj']
        category_index = name_order.index(category)

        # Specify what we want our gradients to be, which will determine which class we want to calculate derivatives for
        if category is None:
            print('ERROR: Should not be setting the gradients to the logits.')
            target[:] = logits[:]
        else:
            target[0][category_index] = 1

        self.model.zero_grad()

        # Backpropgate the output scores using the target gradients
        logits.backward(target)

        return np.moveaxis(image_tensor.grad.detach().cpu().numpy()[0], 0, -1)

    def get_smoothed_mask(self, image_tensor, target_class=None, samples=25, std=0.15, process=lambda x: x**2):
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()

        batch, channels, width, height = image_tensor.size()
        grad_sum = np.zeros((width, height, channels))
        for sample in range(samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += process(self.get_mask(noise_image, target_class))
        return grad_sum / samples

    @staticmethod
    def apply_region(mask, region):
        return mask * region[..., np.newaxis]
