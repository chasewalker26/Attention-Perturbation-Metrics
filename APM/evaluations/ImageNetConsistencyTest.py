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
from util.test_methods import RISETestFunctions_VIT as RISE_VIT

from util.test_methods import AICTestFunctions as AIC
from util.test_methods import AICTestFunctions_VIT as AIC_VIT

from util.test_methods import PosNegPertFunctions as PNP
from util.test_methods import PosNegPertFunctions_VIT as PNP_VIT

from util.test_methods import MonotonicityTest as Mono
from util.test_methods import MonotonicityTest_VIT as Mono_VIT

from util.test_methods import sanityForMetrics as SFMetric

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
    resize = transforms.Resize((224, 224), antialias = True)
    resize_square = transforms.Resize((224, 224), antialias = True, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

    # initialize RISE and RISE blur kernel
    klen = 11
    ksig = 11
    kern = RISE.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    # step size is the size of 1 patch for fairness
    step_size = int((img_hw ** 2) / (num_patches ** 2))

    if "RISE" in test_type:
        if test_type == "RISE":
            test_a = RISE.RISEMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = blur)
            test_b = RISE.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
        elif test_type == "RISE_VIT" or test_type == "RISE_VIT":
            test_a = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'ins')
            test_b = RISE_VIT.RISEMetric(model, img_hw * img_hw, 'del')
        test_a_name = test_type + "_INS"
        test_b_name = test_type + "_DEL"
    elif "AIC" in test_type or "DF" in test_type:
        if test_type == "AIC" or test_type == "DF":
            test_a = AIC.RISEMetric(model, img_hw * img_hw, 'ins', step_size, substrate_fn = blur)
            test_b = AIC.RISEMetric(model, img_hw * img_hw, 'del', step_size, substrate_fn = torch.zeros_like)
        elif test_type == "AIC_VIT" or test_type == "DF_VIT":
            test_a = AIC_VIT.AICMetric(model, img_hw * img_hw, 'ins')
            test_b = AIC_VIT.AICMetric(model, img_hw * img_hw, 'del')
        test_a_name = test_type + "_INS"
        test_b_name = test_type + "_DEL"
    elif "POS_NEG_PERT" in test_type:
        if test_type == "POS_NEG_PERT":
            test_a = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'lerf', step_size, substrate_fn = torch.zeros_like) # negative
            test_b = PNP.PositiveNegativePerturbation(model, img_hw * img_hw, 'morf', step_size, substrate_fn = torch.zeros_like) # positive
        elif test_type == "POS_NEG_PERT_VIT":  
            test_a = PNP_VIT.PositiveNegativePerturbation(model, img_hw * img_hw, 'lerf') # negative
            test_b = PNP_VIT.PositiveNegativePerturbation(model, img_hw * img_hw, 'morf') # positive
        test_a_name = test_type + "_LERF"
        test_b_name = test_type + "_MORF"
    elif "MONO" in test_type:
        if test_type == "MONO":
            test_a = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = blur)
            test_b = Mono.MonotonicityMetric(model, img_hw * img_hw, 'negative', step_size, substrate_fn = torch.zeros_like)
        elif test_type == "MONO_patch":
            test_a = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = blur)
            test_b = Mono.MonotonicityMetric(model, img_hw * img_hw, 'positive', step_size, substrate_fn = blur)
        elif test_type == "MONO_VIT":  
            test_a = Mono_VIT.MonotonicityMetric(model, img_hw * img_hw, 'positive')
            test_b = Mono_VIT.MonotonicityMetric(model, img_hw * img_hw, 'positive')
        test_a_name = test_type + "_POS"
        test_b_name = test_type + "_POS"

    patch_ids = torch.arange(num_patches ** 2).reshape((num_patches, num_patches))
    patch_mask = patch_ids.repeat_interleave(int(img_hw / num_patches), dim=0).repeat_interleave(int(img_hw / num_patches), dim=1)

    normalize = transform_normalize_VIT

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    images_used = 0

    test_a_scores = []
    test_b_scores = []

    test_a_order = []
    test_b_order = []

    attribution_maps = []
    attribution_maps_smooth = []
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
        tensor_img = torch.unsqueeze(tensor_img, 0).to(device)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(tensor_img, model, device)
        original_pred = model_utils.getPrediction(tensor_img, model, device, target_class)[0] * 100
        if original_pred < 60:
            continue

        # checks that selected baselines result in expected behaviors for input perturbation
        blur_pred = model_utils.getPrediction(blur(tensor_img.cpu()).to(device), model, device, target_class)[0] * 100
        black_pred = model_utils.getPrediction(torch.zeros_like(tensor_img).to(device), model, device, target_class)[0] * 100
        black_classification = model_utils.getClass(torch.zeros_like(tensor_img).to(device), model, device)
        blur_classification = model_utils.getClass(blur(tensor_img.cpu()).to(device), model, device)
        if blur_pred >= original_pred or blur_pred > 10 or black_pred >= original_pred or target_class == black_classification or target_class == blur_classification:
            continue

        # checks that baselines result in expected behaviors for attention RISEking
        patch_mask_embed = torch.ones((1, 1, num_patches ** 2 + 1, num_patches ** 2 + 1))
        attn_RISEk = torch.full(patch_mask_embed.shape, torch.finfo(torch.float).min).to(device)
        attn_RISEk[:,:,:,0] = 0
        output = model(tensor_img, attn_RISEk)
        percentage = ((torch.nn.functional.softmax(output, dim = 1)[0])[target_class]).detach().cpu().numpy()
        _, index = torch.max(output, dim = 1)
        if target_class == index[0] or percentage >= original_pred:
            continue

        print(model_name + " " + test_type + " metric sanity test, image: " + image + " " + str(images_used + 1) + "/" + str(image_count))

        ######## Inverse bi Attn  ########
        attr, _ = explainer.bidirectional(tensor_img, target_class, device = device)            
        attr = torch.abs(attr - attr.max()).detach().cpu()
        saliency_map = resize_square(attr).permute(1, 2, 0)
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        saliency_map = resize(attr).permute(1, 2, 0)
        saliency_map_test = np.abs(np.sum(saliency_map.numpy(), axis = 2))
        attribution_maps_smooth.append(saliency_map_test)

        ########  IG  ########
        _, IG, _, _, _ = explainer.generate_transition_attention_maps(tensor_img, target_class, start_layer = 0, device = device)
        saliency_map = resize_square(IG.detach()).permute(1, 2, 0)  
        saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
        attribution_maps.append(saliency_map_test)

        saliency_map = resize(IG.detach()).permute(1, 2, 0)  
        saliency_map_test = np.abs(np.sum(saliency_map.cpu().numpy(), axis = 2))
        attribution_maps_smooth.append(saliency_map_test)

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

        if zero_attr_flag == 1:
            attribution_maps.clear()
            attribution_maps_smooth.clear()
            zero_attr_flag = 0
            continue

        # Get attribution scores
        ins_del_img = tensor_img.cpu()

        score_list_a = [0] * num_attrs
        score_list_b = [0] * num_attrs

        # capture scores for the attributions
        for i in range(num_attrs):
            if "RISE" in test_type:
                if test_type == "RISE":
                    _, _, RISE_ins = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                    _, _, RISE_del = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                elif test_type == "RISE_VIT":
                    _, _, RISE_ins = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                    _, _, RISE_del = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                score_list_a[i] = RISE.auc(RISE_ins)
                score_list_b[i] = RISE.auc(RISE_del)
            elif "AIC" in test_type:
                if test_type == "AIC":
                    _, AIC_ins = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                    _, AIC_del = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                elif test_type == "AIC_VIT":
                    _, AIC_ins = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                    _, AIC_del = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                score_list_a[i] = RISE.auc(AIC_ins)
                score_list_b[i] = RISE.auc(AIC_del)
            elif "DF" in test_type:
                if test_type == "DF":
                    DF_ins, _ = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, decision_flip=True, max_batch_size = batch_size)
                    DF_del, _ = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, decision_flip=True, max_batch_size = batch_size)
                elif test_type == "DF_VIT":
                    DF_ins, _ = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, decision_flip=True, max_batch_size = batch_size)
                    DF_del, _ = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, decision_flip=True, max_batch_size = batch_size)
                score_list_a[i] = DF_ins
                score_list_b[i] = DF_del
            elif "POS_NEG_PERT" in test_type:
                if test_type == "POS_NEG_PERT":
                    _, LERF = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                    _, MORF = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                elif test_type == "POS_NEG_PERT_VIT":
                    _, LERF = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                    _, MORF = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                score_list_a[i] = RISE.auc(LERF)
                score_list_b[i] = RISE.auc(MORF)
            elif "MONO" in test_type:
                if test_type == "MONO":
                    _, pos = test_a.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                    _, neg = test_b.single_run(ins_del_img, attribution_maps_smooth[i], device, max_batch_size = batch_size)
                elif test_type == "MONO_VIT":
                    _, pos = test_a.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                    _, neg = test_b.single_run(ins_del_img, attribution_maps[i], patch_mask, device, max_batch_size = batch_size)
                score_list_a[i] = pos
                score_list_b[i] = neg

        test_a_scores.append(score_list_a)
        test_b_scores.append(score_list_b)

        # find the oderings of the saliency maps
        test_a_order.append(np.argsort(test_a_scores[images_used])[::-1])
        test_b_order.append(np.argsort(test_b_scores[images_used]))

        print(time.time() - start)

        # when all tests have passed the number of images used can go up by 1 and clear map list
        images_used += 1
        attribution_maps.clear()
        attribution_maps_smooth.clear()

    test_a_IRR = SFMetric.inter_rater_reliability(np.asarray(test_a_order))
    test_b_IRR = SFMetric.inter_rater_reliability(np.asarray(test_b_order))
    test_a_b_ICR = SFMetric.internal_consistency_reliability(np.asarray(test_a_order), np.asarray(test_b_order))

    # make the test folder if it doesn't exist
    folder = "test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the test results
    file_name = dataset_name + "_" + test_type + "_sanity_test_" + str(image_count) + "_images"
    with open(folder + file_name + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow([test_a_name + " IRR", str(test_a_IRR)])
        write.writerow([test_b_name + " IRR", str(test_b_IRR)])
        write.writerow([test_a_name + " + " + test_b_name + " ICR", str(test_a_b_ICR)])

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
    elif FLAGS.model == "VIT8":
        model = vit_base_patch8_224(pretrained=True).to(device)
        num_patches = 28
        batch_size = 10

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