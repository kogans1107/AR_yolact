# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:07:37 2020

@author: Bill
"""
from train import  NetWithLoss, CustomDataParallel, MultiBoxLoss, prepare_data

#
# I import Bolya's yolact/data subdirectory as D. This will pick
#   up information from config.py. 
#
import data as D  

from utils.augmentations import SSDAugmentation, FastBaseTransform #, BaseTransform
import torch
from yolact import Yolact
#from eval import prep_display # oops no, clone and modify here as local_prep_display
from layers.output_utils import postprocess, undo_image_transformation
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy
from utils import timer
import cv2
import random

#from torch.autograd import Variable
from collections import defaultdict
color_cache = defaultdict(lambda: {})


class NetWithLossPreds(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses, preds


def local_evalimage(net:Yolact, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    print('frame size is', frame.size())
    batch = FastBaseTransform()(frame.unsqueeze(0))
    print('Batch size is',batch.size())
    preds = net(batch)

    img_numpy = local_prep_display(preds, frame, None, None, undo_transform=False)
    
    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def local_prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    
    I don't have args available, so I need to create it and fill it with the defaults. 
    
    
    """
#    print('local_prep_display, type(dets_out) is',type(dets_out))
    
    top_k = 5
    score_threshold = 0.0
    display_masks = True
    display_text = True
    display_bboxes = True
    display_scores = True
    display_fps = False

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save =D.cfg.rescore_bbox
        D.cfg.rescore_bbox = True
#        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
#                                        crop_masks        = args.crop,
#                                        score_threshold   = args.score_threshold)
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = score_threshold)

        D.cfg.rescore_bbox = save

    with timer.env('Copy'):
#        idx = t[1].argsort(0, descending=True)[:args.top_k]
        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if D.cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(D.COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = D.COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if display_masks and D.cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    if display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if display_text or display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = D.cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
    
    return img_numpy

def gradinator(x):
    x.requires_grad = False
    return x

def local_prepare_data(datum, devices:list=None, allocation:list=None):
    batch_size = 4
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] #if args.cuda else ['cpu']
        if allocation is None:
#            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation = [batch_size // len(devices)] * (len(devices) - 1)
#            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
            allocation.append(batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
#        print(len(images))
        for device, alloc in zip(devices, allocation):
            for _ in range(len(images)):
#                print('cur_idx is ',cur_idx)
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

#        if D.cfg.preserve_aspect_ratio:
#            # Choose a random size from the batch
#            _, h, w = images[random.randint(0, len(images)-1)].size()
#
#            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
#                images[idx], targets[idx], masks[idx], num_crowds[idx] \
#                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds



def npscl(xin):
    x = copy.copy(xin)
    for i in range(x.shape[2]):
        xx = x[:,:,i]
        xmax = np.max(xx); xmin = np.min(xx)
        if xmax > xmin:
            x[:,:,i] = (xx-xmin)/(xmax-xmin)
        else:
            print('All the same!!',i)

    return x

def myshow(img):
    if img.size()[0] != 3:
        maskshow(img)
        
    ishow = img.cpu().numpy().transpose((1,2,0))
    ishow = (ishow-np.min(ishow))/(np.max(ishow) - np.min(ishow)).astype(np.int)
    plt.imshow(ishow)

def maskshow(img,pick=None):
    isize = img.size()
    if not pick:
        nmask = isize[0]
        pick = list(range(nmask))
    else:
        if type(pick) is not list:
            pick = list(pick)

    mask = np.zeros(isize[1:])
    img = img.cpu().detach().numpy()
    for i in pick:
        mask += img[i,:,:]
    
    plt.imshow(mask)
    rtn_size = list(mask.shape)
    rtn_size.append(1)
    return mask.reshape(rtn_size);

#import copy

#
# If I don't wrap this in if _name__ == '__main__', it will usually
#   fail with a broken pipe error. Something to do with multiprocessing 
#   on Windows. The phrase "Forking pickler" also appears in the 
#   traceback, and if that doesn't make you giggle, go take a break. 
#
if __name__ == '__main__':
    mode = 'eval'
    if mode == 'train':
        print('Testing net_cdp and custom loss...')
        
    if mode == 'eval':
        print('Testing net and preds...')
    
    
    
    backend_I_want = 'Qt5Agg'
    #
    # Define a dataset and a DataLoader, and get one datum. 
    #
    print('Before data set def, backend is',matplotlib.get_backend())
    dataset = D.COCODetection(image_path=D.cfg.dataset.train_images,
                            info_file='./data/coco/annotations/milliCOCO.json',
                            transform=SSDAugmentation(D.MEANS))
    
    img_ids = list(dataset.coco.imgToAnns.keys())
    
#                                info_file=D.cfg.dataset.train_info,

    print('After data set def, backend is',matplotlib.get_backend())

#
# Defining datset somehow sometiems changes matplotlib backend to Agg, I 
#   swear to God, so I reset it here if ncecessary.  
#
    if matplotlib.get_backend() != backend_I_want:
        print('WTF? Resetting matplotlib backend...')
        matplotlib.use(backend_I_want)    
    
    batch_size = 4
    num_workers = 0
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, 
                                  collate_fn=D.detection_collate,
                                  pin_memory=True)
    
    
    data_loader_iterator = iter(data_loader)
    
#    datum = next(data_loader_iterator)    
    
    #  datum itself is a list of 2 lists. The first has length batch_size and 
    #   contains images, the second has length 3 and contains targets, masks, 
    #   and num_crowds. 
    #
    #  images are 3x550x550 tensors, in a list of length batch_size.
    #
    #  targets are nx5 tensors, what is n?  in a list of length batch_size.
    #
    #  masks are tensors, size kx550x550, list of length batch_size, where k is
    #   the number of objects in the corresponding image. 
    #
    #  crowds are numpy.int32, in a list of length batch_size.
    #  
    
#    images, (targets, masks, num_crowds) = datum
    #                 img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
#                    {'num_crowds': num_crowds, 'labels': target[:, 4]})

    #
    # Define a Yolact net, with a fancy combined prediction and loss
    #   function, and use it to process datum. 
    #
    net = Yolact()
    net.init_weights(backbone_path='weights/' + D.cfg.backbone.path)
#    net.eval()
    
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
                         pos_threshold=D.cfg.positive_iou_threshold,
                         neg_threshold=D.cfg.negative_iou_threshold,
                         negpos_ratio=D.cfg.ohem_negpos_ratio)

    if mode == 'eval':
        for i_img, img_id in enumerate(img_ids):
            net.eval()
            file_name = dataset.coco.loadImgs(img_id)[0]['file_name']
            if file_name.startswith('COCO'):
                file_name = file_name.split('_')[-1]
        
            img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(i_img)
            batch = img.unsqueeze(0).cuda()
            preds = net(batch)
            img_numpy = local_prep_display(preds, img, h, w)
            plt.imshow(img_numpy)
            plt.pause(0.1)
    

#-----------------------
            try:
                datum = next(data_loader_iterator)
            except StopIteration:
                break
            
            images, targets, masks, num_crowds = local_prepare_data(datum)
            net.train()
            predsT = net(images[0])
            losses = criterion(net, predsT, targets[0], masks[0], num_crowds[0])
            loss = sum([losses[k] for k in losses])
            print(loss)
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
            loss.backward() # Do this to free up vram even if loss is not finite
#-----------------------



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
#    if mode == 'train':    
#       criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
#                                 pos_threshold=D.cfg.positive_iou_threshold,
#                                 neg_threshold=D.cfg.negative_iou_threshold,
#                                 negpos_ratio=D.cfg.ohem_negpos_ratio)
#     #
#    #    
#        net.train()
#        net_cdp = CustomDataParallel(NetWithLoss(net, criterion))
#        net_cdp.cuda()
#    #
##   Follwing is lifted from train.py...    
##        preds = self.net(images)
##        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
##  How does it get images? I think self.net is just Yolact().train(). But it takes 
##    images, rather than datum. 
##
#    #  images, targets, masks, num_crowds = prepare_data...??
#        for idx, datum in enumerate(data_loader):
##            images, targets, masks, num_crowds = local_prepare_data(datum)
#
##            preds = net(images)
##            losses = criterion(net, preds, targets, masks, num_crowds)
#            losses = net_cdp(datum)
##            for k,v in preds.items():
##                preds[k] = v.detach()
#                
##            losses = net_cdp(datum)
##            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
#            loss = sum([losses[k] for k in losses])
#            print('Loss',idx,'is',loss.item())
#            if idx > 0:
#                break
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
#
#  Below here I am trying to call the augmentation by hand, to quickly test
#    my work using a single datum.     

#    tform=SSDAugmentation(D.MEANS)
#    
#    img = np.array(images[0].cpu()).transpose((1,2,0))
#    img0 = copy.copy(img)
#    mask = np.array(masks[0].cpu())
#    target = targets[0]
#    target = np.array(target.cpu())
#    nc = num_crowds[0]
#    img, mask, boxes, labels=\
#    tform(img, mask, target[:, :4],
#                    {'num_crowds': nc, 'labels': target[:, 4]})
#
#    plt.imshow(npscl(img0))
#
#    plt.imshow(npscl(img))
#
#



