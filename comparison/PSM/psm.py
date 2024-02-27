import copy
import cv2
import numpy as np
import os
from torchvision.transforms import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam.utils.image import sgg
from skimage import morphology


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def psm_for_seg(x_fname, y_fname, model, args, tag, device):
    """
    modified from grad-cam
    """

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    target_layer_pos = model.layer1

    cam_pos = methods[args.method](model=model,
                                   target_layers=target_layer_pos,
                                   use_cuda=False)
    #
    # activation map fusion module to generate coarse segmentation
    if tag == 'test_set':
        path = args.data_test + '/images/' + x_fname
    elif tag == 'train_set':
        path = args.data_train + '/images/' + x_fname

    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)

    imsize = min(min(image.shape[0], image.shape[1]), 512)
    totensor_crop = transforms.Compose([transforms.ToTensor(),
                                        transforms.CenterCrop(imsize)])

    img = totensor_crop(image)

    # Shift the mean. This is important.
    # Without this fix, the train and test heatmap will be very different.
    img = img - img.mean() + 0.2

    cam_pos.batch_size = 1

    grayscale_cam_pos = cam_pos(input_tensor=img.unsqueeze(0).to(device),
                                target_category=0,
                                eigen_smooth=False,
                                aug_smooth=False)

    rgb_img = np.transpose(img.cpu().numpy(), (1, 2, 0))

    cam_image, cam_color = sgg(rgb_img, grayscale_cam_pos, x_fname, 1, use_rgb=True)

    cam_images = copy.deepcopy(cam_image)
    cam_images = morphology.remove_small_objects(cam_images, 200)
    cam_images = morphology.remove_small_holes(cam_images)
    cam_image_positive = cam_images * 255

    #####
    if tag == 'test_set':
        save_dir = '/'.join(args.data_test.split('/')[:-1]) + '/data_second_stage_test'
    elif tag == 'train_set':
        save_dir = '/'.join(args.data_train.split('/')[:-1]) + '/data_second_stage_train'
    os.makedirs(save_dir, exist_ok=True)

    basename = x_fname.split('.')[0]

    cv2.imwrite(os.path.join(save_dir, basename + '_pos.png'), cam_image_positive)
    cv2.imwrite(os.path.join(save_dir, basename + '_heat.png'), cam_color)
    cv2.imwrite(os.path.join(save_dir, basename + '_original.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if tag == 'test_set':
        gt_img = cv2.imread(os.path.join(args.data_test, 'masks', basename + '.png'), cv2.IMREAD_GRAYSCALE)
        gt_img = totensor_crop(gt_img)
        gt_img = np.transpose(gt_img.cpu().numpy(), (1, 2, 0)).squeeze(-1)
        gt_img = np.uint8(gt_img * 255)
        cv2.imwrite(os.path.join(save_dir, basename + '_gt.png'), gt_img)
    return
