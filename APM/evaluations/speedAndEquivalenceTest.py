import torch
from torchvision import transforms
from PIL import Image
import os
import csv
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import time

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils

from util.test_methods import RISETestFunctions_VIT as RISE_VIT
from util.test_methods import RISETestFunctions_VIT_drop as RISE_VIT_drop
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm_alt import vit_base_patch32_224 as vit_base_patch32_224_alt
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm_alt import vit_base_patch16_224 as vit_base_patch16_224_alt
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch8_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm_alt import vit_base_patch8_224 as vit_base_patch8_224_alt

model = None

transform_normalize_VIT = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

num_patches = 0

def run_and_save_tests(img_hw, transform, image_count, batch_size, model, model_alt, explainer, model_name, device, dataset_path, dataset_name, num_patches):
    # two resizes, the first is for pixel-level attributions, the second for patch-level
    resize_square = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    test_attn = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'del')
    test_drop = RISE_VIT_drop.RISEMetric(model_alt, img_hw * img_hw, 'del')

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

    # set up test storage
    curve_del_attn_total = []
    curve_del_drop_total = []

    attn_speed = []
    drop_speed = []

    embed_difference = []

    with tqdm(total=image_count) as pbar:
        # look at validations starting from 1
        for image in sorted(os.listdir(dataset_path)):    
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
            trans_img = transform(PIL_img)
            tensor_img = normalize(trans_img)
            tensor_img = torch.unsqueeze(tensor_img, 0).to(device)

            # only rgb images can be classified
            if trans_img.shape != (3, img_hw, img_hw):
                continue

            # get the image's class
            target_class = model_utils.getClass(tensor_img, model, device)

            ########  Grad  ########
            attr = explainer.generate_grad(tensor_img.cuda(), target_class, "cuda:0")
            saliency_map = resize_square(attr.detach()).permute(1, 2, 0)
            saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
            attribution_map= saliency_map_test


            # check if any attribution maps are invalid
            if np.sum(attribution_map.reshape(1, 1, img_hw ** 2)) == 0:
                print("Skipping Image due to 0 attribution in a method")
                zero_attr_flag = 1
                break

            # if we had any invalid attributions, try a new image
            if zero_attr_flag == 1:
                zero_attr_flag = 0
                continue

            # put the image in the form needed for attr eval
            ins_del_img = tensor_img.cpu()

            start = time.time()
            test_attn.single_run(ins_del_img, saliency_map_test, patch_mask, device, max_batch_size = batch_size)
            end = time.time() - start
            attn_speed.append(end)

            start = time.time()
            test_drop.single_run(ins_del_img, saliency_map_test, patch_mask, device, max_batch_size = 1)
            end = time.time() - start
            drop_speed.append(end)

            embeddings_del_attn, _, curve_del_attn, _ = test_attn.single_run(ins_del_img, attribution_map, device = device, max_batch_size = batch_size, patch_mask = patch_mask, return_embeddings=True)
            embeddings_del_drop, _, curve_del_drop, _ = test_drop.single_run(ins_del_img, attribution_map, device = device, max_batch_size = 1, patch_mask = patch_mask, return_embeddings=True)
            
            # mean curves along attributions
            curve_del_attn_total.append(curve_del_attn)
            curve_del_drop_total.append(curve_del_drop)

            layer = 11
            embed_difference.append(np.abs(embeddings_del_attn[layer, :, 0, :] - embeddings_del_drop[layer, :, 0, :]))
                
            # when all tests have passed, number of images used goes up by 1, reset the attr map lists
            images_used += 1
            pbar.update(1)


    embed_difference = np.array(embed_difference)
    embed_diff_image_mean = np.mean(embed_difference, axis = 0)
    embed_diff_step_mean = np.mean(embed_diff_image_mean, axis = 0)
    embed_diff_step_std = np.std(embed_diff_image_mean, axis = 0)
    embed_diff_mean = np.mean(embed_diff_step_mean, axis = 0)
    embed_diff_std = np.std(embed_diff_step_mean, axis = 0)


    curve_del_attn_mean = np.mean(np.array(curve_del_attn_total), axis = 0)
    curve_del_attn_std = np.std(np.array(curve_del_attn_total), axis = 0)
    curve_del_drop_mean = np.mean(np.array(curve_del_drop_total), axis = 0)
    curve_del_drop_std = np.std(np.array(curve_del_drop_total), axis = 0)

    curve_diff_mean = np.mean(np.abs(np.array(curve_del_attn_total) - np.array(curve_del_drop_total)), axis = 0)
    curve_diff_std = np.std(np.abs(np.array(curve_del_attn_total) - np.array(curve_del_drop_total)), axis = 0)

    attn_speed_mean = np.mean(np.array(attn_speed), axis = 0)
    attn_speed_std = np.std(np.array(attn_speed), axis = 0)
    drop_speed_mean = np.mean(np.array(drop_speed), axis = 0)
    drop_speed_std = np.std(np.array(drop_speed), axis = 0)


    # make the test folder if it doesn't exist
    folder = "test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the test results
    file_name = dataset_name + "_equivalence_test_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["1 attn speed mean and std and 1 drop speed mean and std"])
        write.writerow([attn_speed_mean])
        write.writerow([attn_speed_std])
        write.writerow([drop_speed_mean])
        write.writerow([drop_speed_std])

        write.writerow(["1 Del Attn Pert Curve image mean and std followed by 1 Del Token Drop Pert Curve image mean and std"])
        write.writerow(curve_del_attn_mean)
        write.writerow(curve_del_attn_std)
        write.writerow(curve_del_drop_mean)
        write.writerow(curve_del_drop_std)

        write.writerow(["1 curve diff image mean and std"])
        write.writerow(curve_diff_mean)
        write.writerow(curve_diff_std)

        write.writerow(["1 curve diff mean and std"])
        write.writerow([np.mean(curve_diff_mean)])
        write.writerow([np.std(curve_diff_std)])

        write.writerow(["1 embed diff step mean and std"])
        write.writerow(embed_diff_step_mean)
        write.writerow(embed_diff_step_std)

        write.writerow(["1 embed diff mean and std"])
        write.writerow([embed_diff_mean])
        write.writerow([embed_diff_std])

    return

# main sets up model and transforms
def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform input images for model needs
    if FLAGS.model == "VIT16":
        model = vit_base_patch16_224(pretrained=True).to(device)
        model_alt = vit_base_patch16_224_alt(pretrained=True).to(device)
        num_patches = 14
        batch_size = 25
    elif FLAGS.model == "VIT32":
        model = vit_base_patch32_224(pretrained=True).to(device)
        model_alt = vit_base_patch32_224_alt(pretrained=True).to(device)
        num_patches = 7
        batch_size = 50
    elif FLAGS.model == "VIT8":
        model = vit_base_patch8_224(pretrained=True).to(device)
        model_alt = vit_base_patch8_224_alt(pretrained=True).to(device)
        num_patches = 28
        batch_size = 5

    # put model in eval mode and create the explainer class
    model = model.eval()
    model_alt = model_alt.eval()
    explainer = Baselines(model)

    # standard transform 
    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, transform, FLAGS.image_count, batch_size, model, model_alt, explainer, FLAGS.model, device, FLAGS.dataset_path, FLAGS.dataset_name, num_patches)

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