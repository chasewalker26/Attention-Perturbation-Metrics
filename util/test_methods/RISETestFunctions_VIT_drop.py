import torch
import numpy as np
from scipy.ndimage import gaussian_filter

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

class RISEMetric():
    def __init__(self, model, HW, mode):
        r"""Create deletion/lerfertion metric lerftance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'morf', 'lerf'.
        """
        assert mode in ['lerf', 'morf', 'ins', 'del']
        self.model = model
        self.HW = HW
        self.mode = mode

    def single_run(self, img_tensor, saliency_map, patch_mask, device, max_batch_size = 50, return_embeddings = False):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            patch_mask (Tensor): A mask which indicates N patch locations in range [0, N-1].
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
            max_batch_size (int): controls the parallelization of the test.
        Return:
            n_steps (int): the number of steps used over the test.
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
        orig_class = index
        target_class = index[0]
        percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
        original_pred = percentage[target_class].item()

        num_blocks = len(self.model.blocks)
        _, n_patches, embed_dim = self.model.blocks[-1].get_block_out().detach().shape

        orig_embedding = torch.empty((num_blocks, 1, 1, embed_dim))
        counter = 0
        for block in self.model.blocks:
            orig_embedding[counter] = block.get_block_out().detach()[:, 0, :]
            counter += 1
        
        if len(orig_embedding.shape) != 4:
            orig_embedding = torch.unsqueeze(orig_embedding, 1)

        model_response = np.ones(n_steps + 1)
        entropy = np.ones(n_steps + 1)

        embeddings = []
        classes = []

        # set the start and stop images for each test
        # get softmax score of the substrate-applied images
        if self.mode == 'del':
            attn_mask_start = torch.zeros_like(patch_mask_embed).to(device)
            attn_mask_finish = torch.full(patch_mask_embed.shape, torch.finfo(torch.float).min).to(device)
            attn_mask_finish[:,:,:,0] = 0

            baseline_pred = self.model(img_tensor.to(device), attn_mask_finish).detach()
            baseline_percentage = torch.nn.functional.softmax(baseline_pred, dim = 1)[0]
            baseline_pred = baseline_percentage[target_class].item()

            model_response[0] = original_pred
            entropy[0] = -torch.sum(percentage * torch.log2(percentage), dim=-1).cpu().numpy()

        elif self.mode == 'ins':
            attn_mask_start = torch.full(patch_mask_embed.shape, torch.finfo(torch.float).min).to(device)
            attn_mask_start[:,:,:,0] = 0
            attn_mask_finish = torch.zeros_like(patch_mask_embed).to(device)

            baseline_pred = self.model(img_tensor.to(device), attn_mask_start).detach()
            baseline_percentage = torch.nn.functional.softmax(baseline_pred, dim = 1)[0]
            baseline_pred = baseline_percentage[target_class].item()

            model_response[0] = baseline_pred
            entropy[0] = -torch.sum(baseline_percentage * torch.log2(baseline_percentage), dim=-1).cpu().numpy()

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
            percentage = torch.nn.functional.softmax(output, dim = 1)
            entropy[total_steps - batch : total_steps] = -torch.sum(percentage * torch.log2(percentage), dim=-1).cpu().numpy()
            model_response[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()

            # this file uses the token dropping model so we only capture embeddings of CLS token
            if return_embeddings == True:
                classes.append(torch.max(output, 1)[1])

                attn_mask_embeddings = torch.empty((num_blocks, batch, 1, embed_dim))
                counter = 0
                for block in self.model.blocks:
                    attn_mask_embeddings[counter] = block.get_block_out().detach()[:, 0, :]
                    counter += 1
                                    
                if len(attn_mask_embeddings.shape) != 4:
                    attn_mask_embeddings = torch.unsqueeze(attn_mask_embeddings, 1)
                                                                                
                embeddings.append(attn_mask_embeddings)

        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        # perform monotonic normalization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            normalized_pred = (normalized_model_response[i] - baseline_pred) / (abs(original_pred - baseline_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)

            if self.mode == 'del' or self.mode == 'morf' or self.mode == 'lerf':
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                normalized_model_response[i] = min_normalized_pred
            elif self.mode == 'ins':
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                normalized_model_response[i] = max_normalized_pred


        if return_embeddings == True:
            if self.mode == "morf" or self.mode == 'del' or self.mode == "lerf":
                embeddings.insert(0, orig_embedding)
                classes.insert(0, orig_class)
            elif self.mode == 'ins':
                classes.append(orig_class)
                embeddings.append(orig_embedding)
        
            embeddings = torch.cat(embeddings, axis = 1)
            classes = torch.cat(classes, axis = 0)

            return embeddings.cpu().numpy(), classes.cpu().numpy(), model_response, segment_order
            

        return n_steps + 1, entropy, normalized_model_response
