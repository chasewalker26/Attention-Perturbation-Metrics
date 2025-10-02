import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import csv
import time
import warnings
import argparse
import numpy as np

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils

from util.test_methods import RISETestFunctions as RISE
from util.test_methods import PosNegPertFunctions as PNP
from util.test_methods import AICTestFunctions as AIC
from util.test_methods import MonotonicityTest as Mono


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

resize_exact = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
resize_bicube = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.BICUBIC)
resize_bilin = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.BILINEAR)
num_patches = 0

def run_and_save_tests(img_hw, transform, image_count, batch_size, model, explainer, model_name, test_type, device, dataset_path, dataset_name, num_patches):
    # initialize RISE and RISE blur kernel
    klen = 11
    ksig = 11
    kern = RISE.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    normalize = transform_normalize_VIT

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    images_used = 0

    classes_used = [0] * 1000

    smoothing_functions = [resize_bicube, resize_bilin]
    kernels = [25, 55]
    for i in range(len(kernels)):
        klen_new = kernels[i]
        kern_new = RISE.gkern(klen_new, klen_new).to(device)
        blur_new = lambda y: nn.functional.conv2d(y, kern_new, padding = klen_new // 2).to(device)
        smoothing_functions.append(blur_new)

    # step size is the size of 1 patch for fairness
    step_size = int((img_hw ** 2) / (num_patches ** 2))
    if test_type == "RISE":
        test_a = RISE.RISEMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = blur)
        test_b = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
        test_a_name = test_type + "_INS"
        test_b_name = test_type + "_DEL"
    elif test_type == "POS_NEG_PERT":
        test_a = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'lerf', step_size, substrate_fn = torch.zeros_like) # negative
        test_b = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'morf', step_size, substrate_fn = torch.zeros_like) # positive
        test_a_name = test_type + "_LERF"
        test_b_name = test_type + "_MORF"
    elif test_type == "MONO":
        test_a = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = blur)
        test_b = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = blur)
        test_a_name = test_type + "_POS"
        test_b_name = test_type + "_POS"
    elif test_type == "AIC" or test_type == "DF":
        test_a = AIC.AICMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = blur)
        test_b = AIC.AICMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
        test_a_name = test_type + "_INS"
        test_b_name = test_type + "_DEL"

    test_a_scores = []
    test_b_scores = []

    attribution_maps = []
    zero_attr_flag = 0

    # look at test images in order from 1
    for image in sorted(os.listdir(dataset_path)):    
        start = time.time()

        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        image_path = dataset_path + "/" + image
        PIL_img = Image.open(image_path)

        # put the image in form needed for prediction
        trans_img = transform(PIL_img)
        tensor_img = normalize(trans_img)
        tensor_img = torch.unsqueeze(tensor_img, 0)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(tensor_img, model, device)
        original_pred = model_utils.getPrediction(tensor_img, model, device, target_class)[0] * 100
        if original_pred < 60:
            continue

        blur_pred = model_utils.getPrediction(blur(tensor_img.cpu()).to(device), model, device, target_class)[0] * 100
        black_pred = model_utils.getPrediction(torch.zeros_like(tensor_img.cpu()).to(device), model, device, target_class)[0] * 100
        black_classification = model_utils.getClass(torch.zeros_like(tensor_img.cpu()).to(device), model, device)
        blur_classification = model_utils.getClass(blur(tensor_img.cpu()).to(device), model, device)

        if blur_pred >= original_pred or black_pred >= original_pred or target_class == black_classification or target_class == blur_classification:
            continue    
        
        print(model_name + " " + test_type + " smoothing score test, image: " + image + " " + str(images_used + 1) + "/" + str(image_count))

        ########  Raw Attn  ########
        attr = explainer.generate_raw_attn(tensor_img.to(device), device = device)
        attr = attr * torch.ones((3, num_patches, num_patches)).to(device)
        attribution_maps.append(attr)

        ########  GC  ########
        attr = explainer.generate_cam_attn(tensor_img.to(device), target_class, device = device)
        attr = attr * torch.ones((3, num_patches, num_patches)).to(device)
        attribution_maps.append(attr)

        ########  IG  ########
        _, attr, _, _, _ = explainer.generate_transition_attention_maps(tensor_img.to(device), target_class, start_layer = 0, device = device)
        attr = attr * torch.ones((3, num_patches, num_patches)).to(device)
        attribution_maps.append(attr)

        ########  Rollout  ########
        attr, _, _ = explainer.generate_rollout(tensor_img.to(device))
        attr = attr * torch.ones((3, num_patches, num_patches)).to(device)
        attribution_maps.append(attr)

        ########  Bidirectional attn  ########
        attr, _ = explainer.bidirectional(tensor_img.to(device), target_class, device = device)
        attr = attr * torch.ones((3, num_patches, num_patches)).to(device)
        attribution_maps.append(attr)

        # check if any attribution maps are invalid
        for attribution_map in attribution_maps:
            if torch.sum(attribution_map.reshape(1, 1, 3 * num_patches ** 2)) == 0:
                print("Skipping Image due to 0 attribution in a method")
                zero_attr_flag = 1
                break

        if zero_attr_flag == 1:
            classes_used[target_class] -= 1
            attribution_maps.clear()
            zero_attr_flag = 0
            continue

        # Get attribution scores
        ins_del_img = tensor_img

        num_attrs = len(attribution_maps)

        score_list_a = np.zeros((len(smoothing_functions), num_attrs))
        score_list_b = np.zeros((len(smoothing_functions), num_attrs))

        # capture scores for the attributions for all smoothing functions
        for i in range(len(smoothing_functions)):
            for j in range(num_attrs):
                if i < 2:
                    attribution = smoothing_functions[i](attribution_maps[j].cpu().detach()).permute(1, 2, 0)  
                else:
                    attribution = smoothing_functions[i](resize_exact(attribution_maps[j])).permute(1, 2, 0).detach().cpu()
            
                attribution = np.abs(np.sum(attribution.numpy(), axis = 2))

                if test_type == "RISE":
                    _, _, RISE_ins = test_a.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    _, _, RISE_del = test_b.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(RISE_ins)
                    score_list_b[i, j] = RISE.auc(RISE_del)
                elif test_type == "RISE":
                    _, RISE_ins, _, _, _ = test_a.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    _, RISE_del, _, _, _ = test_b.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(RISE_ins)
                    score_list_b[i, j] = RISE.auc(RISE_del)
                elif test_type == "POS_NEG_PERT":
                    _, MORF = test_a.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    _, LERF = test_b.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(MORF)
                    score_list_b[i, j] = RISE.auc(LERF)
                elif test_type == "MONO":
                    _, pos = test_a.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    _, neg = test_b.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    score_list_a[i, j] = pos
                    score_list_b[i, j] = neg
                elif test_type == "AIC":
                    _, ins_curve = test_a.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    _, del_curve = test_b.single_run(ins_del_img, attribution, device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(ins_curve)
                    score_list_b[i, j] = RISE.auc(del_curve)
                elif test_type == "DF":
                    ins_score, _ = test_a.single_run(ins_del_img, attribution, device, decision_flip=True, max_batch_size = batch_size)
                    del_score, _ = test_b.single_run(ins_del_img, attribution, device, decision_flip=True, max_batch_size = batch_size)
                    score_list_a[i, j] = ins_score
                    score_list_b[i, j] = del_score

        test_a_scores.append(score_list_a)
        test_b_scores.append(score_list_b)

        print(time.time() - start)

        # when all tests have passed the number of images used can go up by 1 and clear map list
        images_used += 1
        attribution_maps.clear()

    test_a_scores = np.array(test_a_scores)
    test_b_scores = np.array(test_b_scores)

    test_a_mean = np.mean(test_a_scores, axis = 1)
    test_a_std = np.std(test_a_scores, axis = 1)
    test_a_percent_var = (test_a_std / test_a_mean) * 100

    test_b_mean = np.mean(test_b_scores, axis = 1)
    test_b_std = np.std(test_b_scores, axis = 1)
    test_b_percent_var = (test_b_std / test_b_mean) * 100

    test_a_std = np.mean(test_a_percent_var, axis = 0)
    test_b_std = np.mean(test_b_percent_var, axis = 0)

    # make the test folder if it doesn't exist
    folder = "test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the test results
    file_name = dataset_name + "_" + test_type + "_smoothing_score_test_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["Metric", "Attn", "GC", "IG", "Rollout", "Bi-Attn"])
        write.writerow([test_a_name + " STD", str(test_a_std), str(np.mean(test_a_std))])
        write.writerow([test_b_name + " STD", str(test_b_std), str(np.mean(test_b_std))])

    return

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


    explainer = Baselines(model)

    model = model.eval()

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
                        help='Test to use: RISE, RISE_VIT, POS_NEG_PERT, POS_NEG_PERT_VIT, AIC, AIC_VIT, DF, DF_VIT, MONO, MONO_VIT')
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