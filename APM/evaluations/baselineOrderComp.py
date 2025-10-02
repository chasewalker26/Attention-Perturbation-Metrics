import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from skimage.segmentation import slic
from skimage.util import img_as_float

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines
from util.test_methods import RISETestFunctions as RISE
from util.test_methods import RISETestFunctions_VIT as RISE_VIT
from util.attribution_methods.TIS import TIS
from util import model_utils
from util.visualization import attr_to_subplot
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224 
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch8_224 
    num_patches = 14

model = None

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

num_patches = 0

def run_and_save_tests(img_hw, transform, image_count, batch_size, model, explainer, model_name, device, dataset_path, num_patches):
    # make the test folder if it doesn't exist
    folder = "test_results/" + model_name + "/baseline_order_comps/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # two resizes, the first is for pixel-level attributions, the second for patch-level
    resize = transforms.Resize((img_hw, img_hw), antialias = True)

    # step size is the size of 1 patch for fairness
    step_size = int((img_hw ** 2) / (num_patches ** 2))

    # initialize the test classes that we will be using
    # insertion and deletion use black baselines for fairness
    test_a = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.ones_like)
    test_b = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
    test_c = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'del')

    # set up patch mask
    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(img_hw / num_patches), dim=0).repeat_interleave(int(img_hw / num_patches), dim=1)

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    # get the names of the classes
    labels_path = '../../util/class_maps/ImageNet/imagenet_classes.txt'
    with open(labels_path) as f:
        classes = [line.strip() for line in f.readlines()]

    # set up trackers
    images_used = 0
    zero_attr_flag = 0

    # set up temporary attribution storage
    attribution_maps = []

    torch.manual_seed(42) 

    attribution_names = np.array(["Attn", "TIS", "Grad", "IG", "Bi-Attn"])

    with tqdm(total=image_count) as pbar:
        # look at validations starting from 1
        for image in sorted(os.listdir(dataset_path))[7:]:    
            # have we used all of the images we want
            if images_used == image_count:
                break

            # check if the current image is an invalid image for testing, 0 indexed
            image_num = int((image.split("_")[2]).split(".")[0]) - 1

            # check if the current image is an invalid image for testing
            if correctly_classified[image_num] == 0:
                continue

            # get image and put it in the form needed for prediction
            image_path = dataset_path + "/" + image

            PIL_img = Image.open(image_path)
            if PIL_img.mode != "RGB":
                PIL_img = PIL_img.convert("RGB")
            trans_img = transform(PIL_img)

            tensor_img = normalize(trans_img).unsqueeze(0)

            # only rgb images can be classified
            if trans_img.shape != (3, img_hw, img_hw):
                continue

            # get the image's class
            target_class = model_utils.getClass(tensor_img, model, device)
            class_name = classes[target_class]

            tensor_img = tensor_img.to(device)

            ########  Raw Attn  ########
            attr = explainer.generate_raw_attn(tensor_img.to(device), device = device)
            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)       

            ########  TIS  ########
            saliency_method = TIS(model, batch_size=64)
            attr = saliency_method(tensor_img, class_idx=target_class).unsqueeze(0)
            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)       
                
            ######## Grad  ########
            attr = explainer.generate_grad(tensor_img.to(device), target_class, device)
            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)       

            ########  IG  ########
            _, attr, _, _, _ = explainer.generate_transition_attention_maps(tensor_img, target_class, start_layer = 0, device = device)
            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)       

            ########  Bidirectional attn  ########
            attr, _ = explainer.bidirectional(tensor_img, target_class, device = device)
            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)       

            num_attrs = len(attribution_maps)

            # check if any attribution maps are invalid
            for i in range(num_attrs):
                if np.sum(attribution_maps[i].reshape(1, 1, img_hw ** 2)) == 0:
                    print("Skipping Image due to 0 attribution in a method")
                    zero_attr_flag = 1
                    break

            # if we had any invalid attributions, try a new image
            if zero_attr_flag == 1:
                attribution_maps.clear()
                zero_attr_flag = 0
                continue

            # put the image in the form needed for attr eval
            ins_del_img = tensor_img.cpu()

            score_list_a = [0] * num_attrs
            score_list_b = [0] * num_attrs
            score_list_c = [0] * num_attrs

            # capture embeddings and scores for the attributions then perform the main test operations
            for i in range(num_attrs):
                _, _, curve_del_inp_white = test_a.single_run(ins_del_img, attribution_maps[i], device, max_batch_size = batch_size)
                _, _, curve_del_inp_black = test_b.single_run(ins_del_img, attribution_maps[i], device, max_batch_size = batch_size)
                _, _, curve_del_attn = test_c.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)

                score_list_a[i] = RISE.auc(curve_del_inp_white)
                score_list_b[i] = RISE.auc(curve_del_inp_black)
                score_list_c[i] = RISE.auc(curve_del_attn)

            # save the scores and rank the attributions for each image
            test_a_order = np.argsort(score_list_a)
            test_b_order = np.argsort(score_list_b)
            test_c_order = np.argsort(score_list_c)

            # save the attributions
            plt.rcParams.update({'font.size': 30})
            fig, axs = plt.subplots(3, num_attrs + 1, figsize = (25, 15))
            norm = "absolute"

            class_name = class_name.replace(" ", "\n")

            for i in range(num_attrs):
                attr_to_subplot(trans_img, class_name, axs[0, 0], cmap = 'jet', original_image = True)
                attr_to_subplot(np.array(attribution_maps)[test_a_order][i].reshape((img_hw, img_hw, 1)), "Rank " + str(i + 1) + "\n" + attribution_names[test_a_order][i], axs[0, i + 1], cmap = 'jet', norm = norm, blended_image=trans_img, alpha = 0.6)
                axs[0, 0].set_ylabel("IPM Del White")
            for i in range(num_attrs):
                attr_to_subplot(trans_img, class_name, axs[1, 0], cmap = 'jet', original_image = True)
                attr_to_subplot(np.array(attribution_maps)[test_b_order][i].reshape((img_hw, img_hw, 1)), attribution_names[test_b_order][i], axs[1, i + 1], cmap = 'jet', norm = norm, blended_image=trans_img, alpha = 0.6)
                axs[1, 0].set_ylabel("IPM Del Black")
            for i in range(num_attrs):
                attr_to_subplot(trans_img, class_name, axs[2, 0], cmap = 'jet', original_image = True)
                attr_to_subplot(np.array(attribution_maps)[test_c_order][i].reshape((img_hw, img_hw, 1)), attribution_names[test_c_order][i], axs[2, i + 1], cmap = 'jet', norm = norm, blended_image=trans_img, alpha = 0.6)
                axs[2, 0].set_ylabel("APM Del (ours)")

            plt.subplots_adjust(wspace=0.0, hspace = 0.40)
            plt.figure(fig)
            plt.savefig(folder + "image_" + f"{(image_num + 1):08d}" + ".png", bbox_inches='tight', facecolor="white", transparent = "False", pad_inches = .05)
            fig.clear()
            plt.close(fig)

            images_used += 1
            pbar.update(1)
            attribution_maps.clear()

    return

# main sets up model and transforms
def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform input images for model needs
    if FLAGS.model == "VIT32":
        model = vit_base_patch32_224(pretrained=True).to(device)
        num_patches = 7
        batch_size = 50
    elif FLAGS.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        num_patches = 14
        batch_size = 10
    elif FLAGS.model == "VIT8":
        model = vit_base_patch8_224(pretrained=True).to(device)
        num_patches = 28
        batch_size = 10

    # put model in eval mode and create the explainer class
    explainer = Baselines(model)

    # standard transform 
    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, transform, FLAGS.image_count, batch_size, model, explainer, FLAGS.model, device, FLAGS.dataset_path, num_patches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Consistenty check the different metrics.')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "VIT16",
                        help='Classifier to use: VIT16 or VIT32 or VIT8')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--dataset_name',
                type = str, default = "ImageNet")
    parser.add_argument('--dataset_path',
            type = str, default = "../../../ImageNet",
            help = 'The path to your dataset input')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)