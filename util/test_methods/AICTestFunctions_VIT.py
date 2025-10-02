import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# this code borrows certain functions from 
# https://github.com/eclique/RISE/blob/master/evaluation.py


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""

    # create nxn zeros
    inp = np.zeros((klen, klen))

    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1

    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k

    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class AICMetric():
    def __init__(self, model, HW, mode):
        r"""Create deletion/lerfertion metric lerftance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'ins', 'del'.
        """
        assert mode in ['ins', 'del']
        self.model = model
        self.HW = HW
        self.mode = mode

    def single_run(self, img_tensor, saliency_map, patch_mask, device, decision_flip = False, max_batch_size = 50):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            patch_mask (Tensor): A mask which indicates N patch locations in range [0, N-1].
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
            max_batch_size (int): controls the parallelization of the test.
        Return:
            n_steps (int): the number of steps used over the test.
            corrected_scores (nd.array): Array containing MAS scores at every step.
            alignment_penalty (nd.array): Array containing alignment penalty at every step.
            density_response (nd.array): Array containing the density response at every step.
            normalized_model_response (nd.array): Array containing the model response at every step.
        """

        num_patches = int(int(self.HW ** (1/2)) / self.model.patch_embed.proj.kernel_size[0])
        patch_mask_embed = torch.ones((1, 1, num_patches ** 2 + 1, num_patches ** 2 + 1))
        n_steps = num_patches ** 2
        
        batch_size = n_steps if n_steps < max_batch_size else max_batch_size
        if batch_size > n_steps:
            print("Batch size cannot be greater than number of steps: " + str(n_steps))
            return 0, 0, 0, 0, 0

        # Retrieve softmax score of the original image
        original_pred = self.model(img_tensor.to(device)).detach()
        _, index = torch.max(original_pred, 1)
        target_class = index[0]
        original_pred = 1

        model_response = np.ones(n_steps + 1)

        # set the start and stop images for each test
        # get softmax score of the substrate-applied images
        if self.mode == 'del':
            attn_mask_start = torch.zeros_like(patch_mask_embed).to(device)
            attn_mask_finish = torch.full(patch_mask_embed.shape, torch.finfo(torch.float).min).to(device)
            attn_mask_finish[:,:,:,0] = 0

            baseline_pred = self.model(img_tensor.to(device), attn_mask_finish).detach()
            _, index = torch.max(baseline_pred, 1)
            baseline_class = index[0]
            baseline_pred = (baseline_class == target_class) * 1

            model_response[0] = original_pred
        elif self.mode == 'ins':
            attn_mask_start = torch.full(patch_mask_embed.shape, torch.finfo(torch.float).min).to(device)
            attn_mask_start[:,:,:,0] = 0
            attn_mask_finish = torch.zeros_like(patch_mask_embed).to(device)

            baseline_pred = self.model(img_tensor.to(device), attn_mask_start).detach()
            _, index = torch.max(baseline_pred, 1)
            baseline_class = index[0]
            baseline_pred = (baseline_class == target_class) * 1

            model_response[0] = baseline_pred

        # patches in order of decreasing saliency
        segment_saliency = np.zeros(n_steps)
        for i in range(n_steps):
            segment = np.where(patch_mask.flatten() == i)[0]
            segment_saliency[i] = np.mean(saliency_map.reshape(self.HW)[segment])

        if self.mode == 'del' or self.mode == "ins": 
            segment_order = np.flip(np.argsort(segment_saliency, axis = 0), axis = -1)

        total_steps = 1
        num_batches = int((n_steps) / batch_size)
        leftover = (n_steps) % batch_size

        if leftover != 0:
            batches = np.full(num_batches + 1, batch_size)
            batches[-1] = leftover
        else:
            batches = np.full(num_batches, batch_size)

        for batch in batches:
            images = torch.repeat_interleave(img_tensor, batch, dim = 0)
            masks = torch.zeros((batch, attn_mask_start.shape[1], attn_mask_start.shape[2], attn_mask_start.shape[3]))
            # collect all masks at batch steps before mass prediction 
            for i in range(batch):
                segment_coords = segment_order[total_steps - 1] + 1
                # mask attn columns
                attn_mask_start[:, :, :, segment_coords] = attn_mask_finish[:, :, :, segment_coords]
                masks[i] = attn_mask_start
                total_steps += 1

            # get predictions from image batch
            output = self.model(images.to(device), masks.to(device)).detach()
            _, indicies = torch.max(output, 1)
            model_response[total_steps - batch : total_steps] = torch.eq(indicies, target_class).detach().cpu().numpy() * 1

        if decision_flip == True:
            if self.mode == "del":
                score = np.where(model_response == 0)[0][0] / len(model_response)
            elif self.mode == 'ins':    
                score = np.where(model_response == 1)[0][0] / len(model_response)

            return score, model_response

        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        # perform monotonic normalization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            normalized_pred = (normalized_model_response[i] - baseline_pred) / (abs(original_pred - baseline_pred))
            normalized_pred = np.clip(normalized_pred.cpu(), 0.0, 1.0)
            if self.mode == 'del':
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                normalized_model_response[i] = min_normalized_pred
            elif self.mode == 'ins':
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                normalized_model_response[i] = max_normalized_pred

        return n_steps + 1, normalized_model_response
