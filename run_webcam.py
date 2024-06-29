import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov9-e.pt', help='model path or triton URL')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    margin_width = 50
    caption_height = 60
    is_file = Path(args.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = args.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = args.source.isnumeric() or args.source.endswith('.txt') or (is_url and not is_file)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_thickness=3
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    #########################################################################################
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    bs = 1  # batch_size
    # Load model
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    

    #########################################################################################
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, raw_image = vid.read() 
    
        # Display the resulting frame 
        #cv2.imshow('frame', frame) 
        
       
        #raw_image = cv2.imread(frame)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        ###############################################################################################
        with dt[0]:
            img = np.asanyarray(raw_image)
        
            im0 = img.copy()
            img = img[np.newaxis, :, :, :]
            img = np.stack(img, 0)
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch di

        ###############################################################################################
        with torch.no_grad():
            depth = depth_anything(image)
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False
                pred = model(img, augment=args.augment, visualize=visualize)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        #print(depth.shape)
        ####################################################################################
         # NMS
        with dt[2]:
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            final_index = 0
            final_mean = 0.0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            test =0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                
                for *xyxy, conf, cls in reversed(det):
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    sub_matrix = depth[y1:y2, x1:x2, :]
                    color_mean = np.mean(sub_matrix, dtype = np.float64)
                    if final_mean == 0.0:
                        final_mean = color_mean
                        final_index = i
                    else:
                        if final_mean < color_mean:
                            final_mean = color_mean
                            final_index = i

                # Write results
                *xyxy, conf, cls =  det[final_index]
                c = int(cls)  # integer class
                label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                annotator = Annotator(depth, line_width=line_thickness, example=str(names))
                annotator.box_label(xyxy, label, color=colors(c, True))
                
        
        #######################################################################################
        #filename = os.path.basename(filename)
        
        # if args.pred_only:
        #     cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        # else:
        #     split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        #     combined_results = cv2.hconcat([im0, split_region, depth])
            
        #     caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
        #     captions = ['Raw image', 'Depth Anything']
        #     segment_width = w + margin_width
            
        #     for i, caption in enumerate(captions):
        #         # Calculate text size
        #         text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        #         # Calculate x-coordinate to center the text
        #         text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        #         # Add text caption
        #         cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
        #     final_result = cv2.vconcat([caption_space, combined_results])
        #     cv2.imshow("Result", final_result) 
        cv2.imshow("Result", depth) 
            #cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
     # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
        