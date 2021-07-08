from ctypes import *
import math
import random
import csv
import cv2
from typing import Tuple
import numpy as np
import argparse
import glob
import os
import easyocr
from os import makedirs
from PIL import Image, ImageDraw, ImageFont

def mkdir(path):
    try:
        makedirs(path)
    except Exception as error:
        pass

def init_parameters(fun, **init_dict):
    """
    help you to set the parameters in one's habits
    """
    def job(*args, **option):
        option.update(init_dict)
        return fun(*args, **option)
    return job

def cv2_img_add_text(img, text, left_corner: Tuple[int, int],
                     text_rgb_color=(255, 0, 0), text_size=24, font='/home/dl-box/chi/projects/train-yolov3/vnteam/nhat/darknet/HanaMinA.ttf', **option):
    """
    USAGE:
        cv2_img_add_text(img, '中文', (0, 0), text_rgb_color=(0, 255, 0), text_size=12, font='mingliu.ttc')
    """
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    font_text = ImageFont.truetype(font=font, size=text_size)
    draw.text(left_corner, text, fill=text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

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


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
draw_text = init_parameters(cv2_img_add_text, text_size=64, text_rgb_color=(0, 255, 0), font='/home/dl-box/chi/projects/train-yolov3/vnteam/nhat/darknet/HanaMinA.ttf', replace=True)
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so",RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def array_to_image(arr):
    arr = arr.transpose(2,0,1) 
    c = arr.shape[0] 
    h = arr.shape[1] 
    w = arr.shape[2] 
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0 
    data = arr.ctypes.data_as(POINTER(c_float)) 
    im = IMAGE(w,h,c,data) 
    return im, arr

def detect_np(net, meta, thresh = .5, hier_thresh = .5, nms = .45):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res

def detect(net, meta, image, thresh= .5, hier_thresh=.5, nms=.45):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res
    
def sortSecond(val): 
    return val[1] 

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

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def nms_over_class(boxs, nms_thresh):
    nms_vehicles =[] 
    overlaps = []
    for i in range(0, len(boxs)):
        is_overlap = False
        ixmin, iymin, ixmax, iymax,ilabel, iscore = boxs[i]
        ibox = (ixmin, iymin, ixmax, iymax)
        for j in range(0, len(boxs)):
            jxmin, jymin, jxmax, jymax, jlabel, jscore  = boxs[j]
            jbox = (jxmin, jymin, jxmax, jymax)
            if boxs[i] != boxs[j] and bb_intersection_over_union(ibox, jbox) > nms_thresh:
                index = i if iscore < jscore else j
                overlaps.append(index)
    nms_vehicles = [i for j, i in enumerate(boxs) if j not in overlaps]
    return nms_vehicles

def run_video(video = None, img_path=None, input_dir=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh",help="confidence threshold",default= 0.01,type = float)
    parser.add_argument("--weight","-w",default="./vehicle_det/checkpoints/ncxx.backup",type = str,help="weights file to load")
    parser.add_argument("--video","-v",default ="", type = str,help="path to video file")
    parser.add_argument("--img","-img",type = str,help="path to image file")
    parser.add_argument("--camera","-cam",action='store_false',help="turn on the camera function")
    parser.add_argument("--dir","-dir",default ="./data_images", type = str,help="path to directory of input image files")

    args = parser.parse_args()
    net = load_net("./vehicle_det/ncxx.cfg".encode("utf-8"), args.weight.encode("utf-8"), 0)
    meta = load_meta("./vehicle_det/ncxx.data".encode("utf-8"))
    
    net_lp = load_net("./plate_det/fastV2_ANPR.cfg".encode("utf-8"),"./plate_det/fastV2_ANPR_90000.weights".encode("utf-8"), 0)
    meta_lp = load_meta("./plate_det/fastV2_ANPR.data".encode("utf-8"))

    net_np = load_net("./number_recognition/yolov3-tiny.cfg".encode("utf-8"),"./number_recognition/checkpoints/yolov3-tiny_13000.weights".encode("utf-8"), 0)
    meta_np = load_meta("./number_recognition/yolov3-tiny.data".encode("utf-8"))
    # video = "demo_original_movie.mp4"
    #Chi
    #video = "/mnt/ELECOM/chi/videos/japan.mp4"
    video = args.video
    thresh = args.thresh
    img = args.img
    input_dir = args.dir
    cam = args.camera
    #if video:
    if img_path is None and video is None and input_dir is None:
            raise Exception("please input video or image or path to folder")
    if img_path is not None:
        fileName = os.path.basename(img_path)
        mkdir("./results")
        print("Yo dowg baazinga")

    #Chi
    # videoName = "clearPlate1080"
    #videoName = "japan"
    elif video:
        cap = cv2.VideoCapture(video)
        videoName = os.path.basename(video)
        video_prefix,extention = os.path.splitext(videoName)
        print("video prefix", video_prefix)
        # mkdir("./results/"+videoName)
        print("reading video: ",videoName)
        
        count_frame = 0
    #End Chi
    
        if(cap.isOpened() == False):
            print("Error opening video stream or file")
        if cap.isOpened:
            ret, img = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print("You dowg video debugging")
    
    elif input_dir and not cam:
        count_index = 0
        placeholder_path = os.path.join(input_dir, "*")
        folder_name = os.path.basename(os.path.normpath(input_dir))
        folder_path = os.path.join("./results", folder_name)
        mkdir(folder_path)
         
        # print(placeholder_path)

        files_list = glob.glob(placeholder_path)
        # print("OKOKOK")
        # print(files_path)
        # for file_path in files_list:

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    else:
        cap = cv2.VideoCapture('CAP_V4L2')
        # videoName = os.path.basename(video)
        # video_prefix,extention = os.path.splitext(videoName)
        # print("video prefix", video_prefix)
        # # mkdir("./results/"+videoName)
        # print("reading video: ",videoName)
        print("You dowg print debugging")
        count_frame = 0

    num = c_int(0)
    pnum = pointer(num)
    num_np = c_int(0)
    pnum_np = pointer(num_np)
    num_lp = c_int(0)
    pnum_lp = pointer(num_lp)
    #thresh=.4
    thresh_lp = 0.5
    thresh_np= 0.15
    hier_thresh=.5
    nms=.4
    nms_lp = .2

    #out = cv2.VideoWriter('./results/{}.avi'.format(videoName),cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (frame_width,frame_height))
    #reader = easyocr.Reader(['ja','en'])

    count_plate = 0
    ret = False
    while True:
        if img_path:
            #print("ok img")
            ret = True
            img = cv2.imread(img_path)
        elif video or cap.isOpened:
            ret, img = cap.read()
            print(img)
            print("tried to print image there")
            count_frame += 1
            if count_frame ==1:
                # height,width,layers=img.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_cap=cv2.VideoWriter("./results/"+video_prefix+".avi",fourcc,fps,(width,height))
        elif input_dir:
            ret = True
            if count_index > len(files_list)-1:
                break
            file_path = files_list[count_index]
            img = cv2.imread(file_path)
            file_name = os.path.basename(file_path)
            count_index+=1
            
        # elif video and ret == False:
        #     print("Done processing video")
        #     break
        
        #Chi
        
        # if count_frame < 100:
        #     continue
        #end
        # img = resize_keepratio(img,w= 1280,h=None)
        if ret:
            while True:
                ret, img = cap.read()
                im,_ = array_to_image(img)
                rgbgr_image(im)
                predict_image(net,im)
                # print("\nimg shape ",img.shape)
                dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
                num = pnum[0]
                if (nms): do_nms_obj(dets, num, meta.classes, nms)
                ytext = 250
                ypic = 0
                num_plates =[]
                for j in range(num):
                    for i in range(meta.classes):
                        if dets[j].prob[i] > 0:
                            b = dets[j].bbox
                            x1 = int(b.x - b.w/2.0)
                            x2 = int(b.x + b.w/2.0)
                            y1 = int(b.y - b.h/2.0)
                            y2 = int(b.y + b.h/2.0)
                            p1  = np.array((x1,y1))
                            p2  = np.array((x2,y2)) 
                            dst = np.sqrt(np.sum((p1-p2)**2))
                            if(dst >100):
                                score = dets[j].prob[i]
                                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)    ## draw rectangel to the vehicle 
                                detected = img[y1:y2,x1:x2] # vehicle
                                # print("detected shape ",detected.shape)
                                h,w = detected.shape[:2]
                                if (h > 10 and w > 10):
                                    vehi,_ = array_to_image(detected)
                                    rgbgr_image(vehi)
                                    predict_image(net_lp,vehi)
                                    dets_lp = get_network_boxes(net_lp, vehi.w, vehi.h, thresh_lp, hier_thresh, None, 0, pnum_lp)
                                    num_lp = pnum_lp[0]
                                    if (nms_lp): 
                                        do_nms_obj(dets_lp, num_lp, meta_lp.classes, nms_lp)
                                    for j_lp in range(num_lp):                  
                                        for i_lp in range(meta_lp.classes):                                
                                            if dets_lp[j_lp].prob[i_lp] > 0:
                                                bb = dets_lp[j_lp].bbox
                                                xx1 = int(bb.x - bb.w/2.0)
                                                xx2 = int(bb.x + bb.w/2.0)
                                                yy1 = int(bb.y - bb.h/2.0)
                                                yy2 = int(bb.y + bb.h/2.0)
                                                p1  = np.array((xx1,yy1))
                                                p2  = np.array((xx2,yy2)) 
                                                np_score = dets_lp[j_lp].prob[i_lp]
                                                dst_lp = np.sqrt(np.sum((p1-p2)**2))
                                                if (dst_lp  > 45):
                                                    num_plates.append([x1 + xx1, y1 + yy1, x1 + xx2, y1 +yy2, i_lp, np_score])
                nms_numPlates = nms_over_class(num_plates, 0.1)
                for numPlate in nms_numPlates: 
                    xx1, yy1, xx2, yy2,_,_ = numPlate
                    dst_lp = abs(xx2 - xx1)
                    xx1 -= int (0.1*dst_lp)
                    yy1 -= int (0.1*dst_lp)
                    xx2 += int (0.15*dst_lp)
                    yy2 += int (0.2*dst_lp)
                    plate = img[yy1 : yy2, xx1 : xx2] #number plate
                    try:
                        #result = reader.readtext(plate)
                        y0 = 0
                        for res in result:
                            draw_text(img, res[1],(150, y0+ytext-50))
                        # cv2.putText(img,,, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2,cv2.LINE_AA)
                            y0+=100
                    except Exception as err:
                        pass
                    
                    # print("result", result)
                    # print("plate shape" ,plate.shape)
                    cv2.rectangle(img,(xx1, yy1), (xx2, yy2), (0,255,0), 2)        
                    num_detect=""
                    num_im,_ = array_to_image(plate)
                    rgbgr_image(num_im)
                    predict_image(net_np,num_im)
                    dets_np = get_network_boxes(net_np, num_im.w, num_im.h, thresh_np, hier_thresh, None, 0, pnum_np)
                    num_np = pnum_np[0]
                    if (dets_np):
                        do_nms_obj(dets_np, num_np, meta_np.classes, nms)
                    num_array = []
                    for j_np in range(num_np):
                        for i_np in range(meta_np.classes):
                            if dets_np[j_np].prob[i_np] > 0:
                                bbn = dets_np[j_np].bbox
                                xn1 = int(bbn.x - bbn.w/2.0)
                                xn2 = int(bbn.x + bbn.w/2.0)
                                yn1 = int(bbn.y - bbn.h/2.0)
                                yn2 = int(bbn.y + bbn.h/2.0)
                                num_array.append((meta_np.names[i_np].decode('UTF-8'), xn1))
                    num_array.sort(key= sortSecond)
                    # print(num_array)
                    # print(num_array[0][0])

                    for indx in range(len(num_array)):
                        num_detect+=(num_array[indx][0])
                    cv2.putText(img,num_detect,(xx1, yy1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2,cv2.LINE_AA)
                    h,w,d = plate.shape
                    if(h > 5 and w  > 5): 
                        # print(w,h)
                        try:
                            plate = resize_keepratio(plate,h = 200,w =None)
                            height,width = plate.shape[:2]
                            #img[ypic:ypic+height,20:20+width] = plate
                            # count_plate +=1
                            # cv2.imwrite("./results/plates/"+str(count_plate)+".jpg", plate)
                            # # print(ypic + height, ytext)
                            ytext += 300
                            ypic += 300
                        except Exception as err:
                            print("ypic", ypic)
                            print("height", height)
                            pass
                #Chi
                if img_path:
                    cv2.imwrite(str(os.path.join(".","results",fileName)), img)
                    break
                elif video:
                    video_cap.write(img)
                    pc = int(count_frame/cap.get(cv2.CAP_PROP_FRAME_COUNT)*100)
                    #print()
                    print("="*pc+">", str(pc)+"%")
                    # cv2.imwrite("./results/"+ videoName+"/"+str(count_frame)+".jpg", img)
                    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                    # imS = cv2.resize(img, (1920, 1080))
                    # cv2.imshow("output", imS)
                    # cv2.waitKey(1)
                elif input_dir:
                    cv2.imwrite(str(os.path.join(folder_path, file_name)), img)
                else:
                    cv2.imshow('frame', img)
                    video_cap.write(img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                
                #End Chi
                # out.write(img)
                # img = resize_keepratio(img,w=1280,h=None)
                # cv2.imshow("img",img)
                # k = cv2.waitKey(1)
                # if k == 27:
                #     exit()
            cap.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
        else:
            if video:
                cap.release()
            #out.release()
                video_cap.release()
                # cv2.destroyAllWindows()
                break
            if input_dir:
                break

if __name__ == "__main__":
    # video = './short_video.mp4'
    run_video(video = None, img_path=None, input_dir=None)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--thresh",help="confidence threshold",default= 0.01,type = float)
#     parser.add_argument("--weight","-w",default="./vehicle_det/checkpoints/ncxx.backup",type = str,help="weights file to load")
#     parser.add_argument("--video","-v",default ="short_video.mp4", type = str,help="path to video file")
#     parser.add_argument("--img","-img",type = str,help="path to image file")
#     parser.add_argument("--dir","-dir",default ="./data_images", type = str,help="path to directory of input image files")

#     args = parser.parse_args()
#     net = load_net("./vehicle_det/ncxx.cfg".encode("utf-8"), args.weight.encode("utf-8"), 0)
#     meta = load_meta("./vehicle_det/ncxx.data".encode("utf-8"))
    
#     net_lp = load_net("./plate_det/fastV2_ANPR.cfg".encode("utf-8"),"./plate_det/fastV2_ANPR_90000.weights".encode("utf-8"), 0)
#     meta_lp = load_meta("./plate_det/fastV2_ANPR.data".encode("utf-8"))

#     net_np = load_net("./number_recognition/yolov3-tiny.cfg".encode("utf-8"),"./number_recognition/checkpoints/yolov3-tiny_13000.weights".encode("utf-8"), 0)
#     meta_np = load_meta("./number_recognition/yolov3-tiny.data".encode("utf-8"))
#     # video = "demo_original_movie.mp4"
#     #Chi
#     #video = "/mnt/ELECOM/chi/videos/japan.mp4"
#     video = args.video
#     thresh = args.thresh
#     img = args.img
#     input_dir = args.dir

# run_video(video, net, meta, net_lp, meta_lp, net_np, meta_np, thresh, img, input_dir)
    # video = "/mnt/ELECOM/chi/MVI_0098.MP4"
    #end Chi