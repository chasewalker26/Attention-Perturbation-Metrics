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
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch8_224
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224

model = None

transform_normalize_VIT = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

resize = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
num_patches = 0

class full_like_standin():
    def __init__(self, fill_value):
        self.fill_value = fill_value
    
    def __call__(self, tensor):
        return self.forward(tensor)

    def forward(self, tensor):
        return torch.full_like(tensor, self.fill_value)

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
    step_size = int((img_hw ** 2) / (num_patches ** 2))

    # create all baselines
    baselines = [torch.rand_like, blur]
    for i in range(3):
        baselines.append(full_like_standin(i * 0.5))

    if test_type == "RISE" or test_type == "AIC" or test_type == "DF":
        test_a_name = test_type + "_INS"
        test_b_name = test_type + "_DEL"
    elif test_type == "POS_NEG_PERT":
        test_a_name = test_type + "_LERF"
        test_b_name = test_type + "_MORF"
    elif test_type == "MONO":
        test_a_name = test_type + "_POS"
        test_b_name = test_type + "_NEG"

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

        # check if confidence is high enough
        target_class = model_utils.getClass(tensor_img, model, device)
        original_pred = model_utils.getPrediction(tensor_img, model, device, target_class)[0] * 100
        if original_pred < 60:
            continue

        # check if all baselines are valid for the current image
        baseline_error = 0
        for i in range(len(baselines)):
            baseline_pred = model_utils.getPrediction(baselines[i](tensor_img.cpu()).to(device), model, device, target_class)[0] * 100
            baseline_class = model_utils.getClass(baselines[i](tensor_img.cpu()).to(device), model, device)
            if baseline_pred >= original_pred or baseline_class == target_class:
                baseline_error = 1
                break

        if baseline_error == 1:
            continue
        
        print(model_name + " " + test_type + " baseline test, image: " + image + " " + str(images_used + 1) + "/" + str(image_count))

        ########  Raw Attn  ########
        attr = explainer.generate_raw_attn(tensor_img.to(device), device = device)
        attr = resize(attr.cpu().detach())
        saliency_map = attr.permute(1, 2, 0)
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        ########  GC  ########
        attr = explainer.generate_cam_attn(tensor_img.to(device), target_class, device = device)
        attr = resize(attr.cpu().detach())
        saliency_map = attr.permute(1, 2, 0)
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        ########  IG  ########
        _, IG, _, _, _ = explainer.generate_transition_attention_maps(tensor_img.to(device), target_class, start_layer = 0, device = device)
        IG = resize(IG.cpu().detach())
        saliency_map = IG.permute(1, 2, 0)  
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        ########  Rollout  ########
        attr, _, _ = explainer.generate_rollout(tensor_img.to(device))
        attr = resize(attr.cpu().detach())
        saliency_map = attr.permute(1, 2, 0)  
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        ########  Bidirectional attn  ########
        attr, _ = explainer.bidirectional(tensor_img.to(device), target_class, device = device)
        attr = resize(attr.cpu().detach())
        saliency_map = (attr).permute(1, 2, 0)
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        # check if any attribution maps are invalid
        for attribution_map in attribution_maps:
            if np.sum(attribution_map.reshape(1, 1, img_hw ** 2)) == 0:
                print("Skipping Image due to 0 attribution in a method")
                zero_attr_flag = 1
                break

        if zero_attr_flag == 1:
            attribution_maps.clear()
            zero_attr_flag = 0
            continue

        # Get attribution scores
        ins_del_img = tensor_img

        num_attrs = len(attribution_maps)

        score_list_a = np.zeros((len(baselines), num_attrs))
        score_list_b = np.zeros((len(baselines), num_attrs))

        # capture scores for the attributions
        for i in range(len(baselines)):
            baseline = baselines[i]

            if test_type == "RISE":
                test_a = RISE.RISEMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = baseline)
                test_b = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = baseline)
            elif test_type == "POS_NEG_PERT":
                test_a = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'lerf', step_size, substrate_fn = baseline) # negative
                test_b = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'morf', step_size, substrate_fn = baseline) # positive
            elif test_type == "MONO":
                test_a = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = baseline)
                test_b = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = baseline)
            elif test_type == "AIC" or test_type == "DF":
                test_a = AIC.AICMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = baseline)
                test_b = AIC.AICMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = baseline)

            for j in range(num_attrs):
                if test_type == "RISE":
                    _, _, RISE_ins = test_a.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    _, _, RISE_del = test_b.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(RISE_ins)
                    score_list_b[i, j] = RISE.auc(RISE_del)
                elif test_type == "POS_NEG_PERT":
                    _, MORF = test_a.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    _, LERF = test_b.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(MORF)
                    score_list_b[i, j] = RISE.auc(LERF)
                elif test_type == "MONO":
                    _, pos = test_a.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    _, neg = test_b.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    score_list_a[i, j] = pos
                    score_list_b[i, j] = neg
                elif test_type == "AIC":
                    _, ins_curve = test_a.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    _, del_curve = test_b.single_run(ins_del_img, attribution_maps[j], device, max_batch_size = batch_size)
                    score_list_a[i, j] = RISE.auc(ins_curve)
                    score_list_b[i, j] = RISE.auc(del_curve)
                elif test_type == "DF":
                    ins_score, _ = test_a.single_run(ins_del_img, attribution_maps[j], device, decision_flip=True, max_batch_size = batch_size)
                    del_score, _ = test_b.single_run(ins_del_img, attribution_maps[j], device, decision_flip=True, max_batch_size = batch_size)
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
    file_name = dataset_name + "_" + test_type + "_baseline_test_" + str(image_count) + "_images"
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
                        help='Classifier to use: VIT16 or VIT32')
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