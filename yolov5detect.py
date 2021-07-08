from ctypes import *
import argparse
import os
from os.path import split
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

def sortSecond(val): 
    return val[1] 

def mkdir(path):
    from os import makedirs
    try:
        makedirs(path)
    except Exception as error:
        pass

def array_to_image(arr):
    arr = arr.transpose(2,0,1) 
    c = arr.shape[0] 
    h = arr.shape[1] 
    w = arr.shape[2] 
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0 
    data = arr.ctypes.data_as(POINTER(c_float)) 
    im = IMAGE(w,h,c,data) 
    return im, arr

def resize_keepratio( img, h,w):
    height, width = img.shape[:2]
    if(w is None):
        r = h/height
        newH = h
        newW = int(r*width)
    elif( h is None):
        r = w/width
        newW = w
        newH = int(r*height)
    # print(height,width, newH, newW)
    newimg = cv2.resize(img, (newW,newH))
    return newimg

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    count = 0
    video_name = split(source)[1]
    mkdir("./results/"+video_name)
    for path, img, im0s, vid_cap in dataset:
        # if count < 26000: continue
        ytext = 250
        ypic = 0
        xpic = 0

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            # print("save path", save_path )
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                plates = []
                plates_xy = []
                for *xyxy, conf, cls in reversed(det):
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        y1 = xyxy[1].type(torch.IntTensor)
                        y2 = xyxy[3].type(torch.IntTensor)+5
                        x1 = xyxy[0].type(torch.IntTensor)
                        x2 = xyxy[2].type(torch.IntTensor)+5
                        plate = im0[y1:y2,x1:x2]
                        plates.append(plate)
                        plates_xy.append(xyxy)

                for i, plate in enumerate(plates): 
                    # plate = img[yy1 : yy2, xx1 : xx2] #number plate
                    # try:
                    #     result = reader.readtext(plate)
                    #     y0 = 0
                    #     for res in result:
                    #         draw_text(img, res[1],(150, y0+ytext-50))
                    #     # cv2.putText(img,,, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2,cv2.LINE_AA)
                    #         y0+=100
                    # except Exception as err:
                    #     pass
                    # print("result", result)
                    # print("plate shape" ,plate.shape)
                    # cv2.rectangle(img,(xx1, yy1), (xx2, yy2), (0,255,0), 2)        
                    num_detect =""
                    num_im,_ = array_to_image(plate)
                    rgbgr_image(num_im)
                    predict_image(net_np,num_im)
                    dets_np = get_network_boxes(net_np, num_im.w, num_im.h, thresh_np, hier_thresh, None, 0, pnum_np)
                    # print("dets np", dets_np)
                    num_np = pnum_np[0]
                    if (dets_np):
                        do_nms_obj(dets_np, num_np, meta_np.classes, nms)
                    num_array = []

                    for j_np in range(num_np):
                        for i_np in range(meta_np.classes):
                            # print("dets np!!!!", dets_np[j_np].bbox)
                            if dets_np[j_np].prob[i_np] > 0:
                                bbn = dets_np[j_np].bbox
                                xn1 = int(bbn.x - bbn.w/2.0)
                                xn2 = int(bbn.x + bbn.w/2.0)
                                yn1 = int(bbn.y - bbn.h/2.0)
                                yn2 = int(bbn.y + bbn.h/2.0)
                                num_array.append((meta_np.names[i_np].decode('UTF-8'), xn1))
                    num_array.sort(key= sortSecond)
                    print("num_array", num_array)
                    # print(num_array[0][0])

                    for indx in range(len(num_array)):
                        num_detect+=(num_array[indx][0])
                    # cv2.putText(im0,num_detect,(20, ytext), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2,cv2.LINE_AA)
                    xyxy = plates_xy[i]
                    # new_xyxy = [xyxy[0],xyxy[1]-25,xyxy[2],xyxy[3]-25]
                    # print("hahahahahah", xyxy)
                    # plot_one_box(new_xyxy, im0, label=num_detect, color=colors[int(cls)], line_thickness=0)
                    # cv2.putText(im0,num_detect,(xyxy[0], xyxy[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2,cv2.LINE_AA)
                    h,w,d = plate.shape
                    print("plate shape", plate.shape)
                    if (h > 5 and w  > 5): 
                        # print(w,h)
                        try:
                            plate = resize_keepratio(plate,h = 200,w =None)
                            height,width = plate.shape[:2]
                            print("plate shape resize", height,width)
                            print("ypic", ypic)
                            im0[ypic:ypic+height,xpic:xpic+width] = plate
                            plot_one_box(xyxy, im0, label="plate", color=colors[int(cls)], line_thickness=3)
                            # if len(num_detect)==4:
                            cv2.putText(im0,num_detect,(xpic, ypic+height+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2,cv2.LINE_AA)
                            # count_plate +=1
                            # cv2.imwrite("./results/plates/"+str(count_plate)+".jpg", plate)
                            # # print(ypic + height, ytext)
                            if xpic < 1920:
                                xpic += width
                            else:
                                xpic = 0
                                ypic += 300
                                ytext = ypic+ 50
                        except Exception as err:
                            print("err", err)
                            print("ypic", ypic)
                            print("height", height)
                            # raise StopIteration
                            pass
            
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                try:
                    print("Save this frame ", count)
                    cv2.imwrite("./results/"+video_name+"/"+str(count)+".jpg", im0)
                    print("Save this frame succc ", count)
                    count += 1
                except Exception as err:
                    print("err", err)
                    raise StopIteration
                
                print("!!!!!!!!!!!!!!!", count)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        # fourcc = 'MJPG'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # print("save path", save_path)
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    thresh_np = 0.5
    hier_thresh =.5
    num_np = c_int(0)
    pnum_np = pointer(num_np)
    nms=.4

    lib = CDLL("/mnt/ELECOM/chi/chi/projects/train-yolov3/darknet/libdarknet.so",RTLD_GLOBAL)
    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]
    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA
    net_np = load_net("./number_recognition/yolov3-tiny.cfg".encode("utf-8"),"./number_recognition/checkpoints/yolov3-tiny_13000.weights".encode("utf-8"), 0)
    meta_np = load_meta("./number_recognition/yolov3-tiny.data".encode("utf-8"))

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()