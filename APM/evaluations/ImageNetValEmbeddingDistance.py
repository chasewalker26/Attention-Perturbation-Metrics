import torch
from torchvision import transforms
from PIL import Image
import os
import csv
import warnings
import argparse
import numpy as np
from tqdm import tqdm

from torch.nn.functional import pairwise_distance as euc_dist
from torch.nn.functional import cosine_similarity as cos_sim

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils

from util.test_methods import RISETestFunctions as RISE
from util.test_methods import RISETestFunctions_VIT as RISE_VIT
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch8_224

model = None

transform_normalize_VIT = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

num_patches = 0

def run_and_save_tests(img_hw, transform, image_count, batch_size, model, explainer, model_name, test_type, device, dataset_path, dataset_name, num_patches):
    # two resizes, the first is for pixel-level attributions, the second for patch-level
    resize = transforms.Resize((224, 224), antialias = True)
    resize_square = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    # step size is the size of 1 patch for fairness
    step_size = int((img_hw ** 2) / (num_patches ** 2))

    # initialize the test classes that we will be using
    # insertion and deletion use black baselines for fairness
    if "RISE" in test_type:
        if test_type == "RISE":
            test_a = RISE.RISEMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = torch.zeros_like)
            test_b = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
        elif test_type == "RISE_VIT":
            test_a = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'ins')
            test_b = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'del')

    # set up patch mask
    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(img_hw / num_patches), dim=0).repeat_interleave(int(img_hw / num_patches), dim=1)

    # normalization transform
    normalize = transform_normalize_VIT

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    # set up trackers
    images_used = 0
    zero_attr_flag = 0

    # set up temporary attribution storage
    attribution_maps = []
    attribution_maps_smooth = []

    # set up test storage
    curve_ins_total = 0
    curve_del_total = 0
    dist_ins_euclid_total = [0] * (len(model.blocks))
    dist_del_euclid_total = [0] * (len(model.blocks))
    dist_ins_cos_total = [0] * (len(model.blocks))
    dist_del_cos_total = [0] * (len(model.blocks))

    with tqdm(total=image_count) as pbar:
        # look at validations starting from 1
        for image in sorted(os.listdir(dataset_path)):    
            # have we used all of the images we want
            if images_used == image_count:
                print("method finished")
                break

            # check if the current image is an invalid image for testing, 0 indexed
            image_num = int((image.split("_")[2]).split(".")[0]) - 1
            # check if the current image is an invalid image for testing
            if correctly_classified[image_num] == 0:
                continue

            # get image and put it in the form needed for prediction
            image_path = dataset_path + "/" + image
            PIL_img = Image.open(image_path)
            trans_img = transform(PIL_img)
            tensor_img = normalize(trans_img)
            tensor_img = torch.unsqueeze(tensor_img, 0).to(device)

            # only rgb images can be classified
            if trans_img.shape != (3, img_hw, img_hw):
                continue

            # get the image's class
            target_class = model_utils.getClass(tensor_img, model, device)

            ########  Bidirectional attn  ########
            attr, _ = explainer.bidirectional(tensor_img, target_class, device = device)
            saliency_map = resize_square(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps.append(saliency_map_test)

            saliency_map = resize(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_maps_smooth.append(saliency_map_test)            
            
            num_attrs = len(attribution_maps)

            # check if any attribution maps are invalid
            for i in range(num_attrs):
                if np.sum(attribution_maps[i].reshape(1, 1, img_hw ** 2)) == 0 or np.sum(attribution_maps_smooth[i].reshape(1, 1, img_hw ** 2)) == 0:
                    print("Skipping Image due to 0 attribution in a method")
                    zero_attr_flag = 1
                    break

            # if we had any invalid attributions, try a new image
            if zero_attr_flag == 1:
                attribution_maps.clear()
                attribution_maps_smooth.clear()
                zero_attr_flag = 0
                continue

            # put the image in the form needed for attr eval
            ins_del_img = tensor_img.cpu()

            # capture embeddings and scores for the attributions then perform the main test operations
            for i in range(num_attrs):
                if "RISE" in test_type:
                    if test_type == "RISE":
                        embeddings_ins, _, curve_ins, _ = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size, return_embeddings=True)
                        embeddings_del, _, curve_del, _ = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size, return_embeddings=True)
                    elif test_type == "RISE_VIT":
                        embeddings_ins, _, curve_ins, _ = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size, return_embeddings=True)
                        embeddings_del, _, curve_del, _ = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size, return_embeddings=True)

                # mean curves along attributions
                curve_ins_total += curve_ins / num_attrs
                curve_del_total += curve_del / num_attrs

                # reset the model embeddings to the unperturbed image
                model_utils.getClass(tensor_img, model, "cuda:0")

                # for the output embeddings of every layer
                for layer in range(len(model.blocks)):
                    dist_ins_euclid = [0] * (num_patches ** 2 + 1)
                    dist_del_euclid = [0] * (num_patches ** 2 + 1)
                    dist_ins_cos = [0] * (num_patches ** 2 + 1)
                    dist_del_cos = [0] * (num_patches ** 2 + 1)

                    # measure how similar, on average across all tokens, the perturbed embedding of each token is to the original embedding 
                    for j in range(num_patches ** 2 + 1):
                        # gather embeddings
                        new_embedding = model.blocks[layer].get_block_out().detach()[0, j, :].cpu().numpy()

                        embed_ins_pert = embeddings_ins[layer, :, j, :]
                        embed_del_pert = embeddings_del[layer, :, j, :]

                        # measure this for every perturbation step using euclidean distance and cosine similarity
                        for k in range(embed_ins_pert.shape[0]):
                            dist_ins_euclid[k] += euc_dist(torch.from_numpy(new_embedding), torch.from_numpy(embed_ins_pert[k, :])) / (num_patches ** 2 + 1)
                            dist_del_euclid[k] += euc_dist(torch.from_numpy(new_embedding), torch.from_numpy(embed_del_pert[k, :])) / (num_patches ** 2 + 1)

                            dist_ins_cos[k] += cos_sim(torch.from_numpy(new_embedding), torch.from_numpy(embed_ins_pert[k, :]), dim = 0) / (num_patches ** 2 + 1)
                            dist_del_cos[k] += cos_sim(torch.from_numpy(new_embedding), torch.from_numpy(embed_del_pert[k, :]), dim = 0) / (num_patches ** 2 + 1)

                    # each array is a distance curve across all pert steps
                    # add it to the per layer tracking and divide by total number of attributions 
                    dist_ins_euclid_total[layer] += np.array(dist_ins_euclid) / num_attrs
                    dist_del_euclid_total[layer] += np.array(dist_del_euclid) / num_attrs
                    dist_ins_cos_total[layer] += np.array(dist_ins_cos) / num_attrs
                    dist_del_cos_total[layer] += np.array(dist_del_cos) / num_attrs

            # when all tests have passed, number of images used goes up by 1, reset the attr map lists
            images_used += 1
            pbar.update(1)
            attribution_maps.clear()
            attribution_maps_smooth.clear()

    # make sure to divide all of the curves by the total images
    curve_ins_total = curve_ins_total / images_used
    curve_del_total = curve_del_total / images_used
    dist_ins_euclid_total = np.array(dist_ins_euclid_total) / images_used
    dist_del_euclid_total = np.array(dist_del_euclid_total) / images_used
    dist_ins_cos_total = np.array(dist_ins_cos_total) / images_used
    dist_del_cos_total = np.array(dist_del_cos_total) / images_used

    # make the test folder if it doesn't exist
    folder = "test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the test results
    file_name = dataset_name + "_" + test_type + "_embed_dist_test_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["1 Insertion curve followed by 1 Deletion curve"])
        write.writerow(curve_ins_total)
        write.writerow(curve_del_total)
        write.writerow(["4 matrices, 12 rows each. Each row is layer x distance @ perturbation steps"])
        write.writerows(dist_ins_euclid_total)
        write.writerows(dist_ins_cos_total)
        write.writerows(dist_del_euclid_total)
        write.writerows(dist_del_cos_total)

    return

# main sets up model and transforms
def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform input images for model needs
    if FLAGS.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        num_patches = 14
        batch_size = 25
    elif FLAGS.model == "VIT32":
        model = vit_base_patch32_224(pretrained=True).to(device)
        num_patches = 7
        batch_size = 50
    elif FLAGS.model == "VIT8":
        model = vit_base_patch8_224(pretrained=True).to(device)
        num_patches = 28
        batch_size = 10

    # put model in eval mode and create the explainer class
    model = model.eval()
    explainer = Baselines(model)

    # standard transform 
    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, transform, FLAGS.image_count, batch_size, model, explainer, FLAGS.model, FLAGS.test_type, device, FLAGS.dataset_path, FLAGS.dataset_name, num_patches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Consistenty check the different metrics.')
    parser.add_argument('--image_count',
                        type = int, default = 1000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "VIT16",
                        help='Classifier to use: VIT16 or VIT32 or VIT8')
    parser.add_argument('--test_type',
                        type = str,
                        default = "RISE",
                        help='Test to use: RISE, RISE_VIT')
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