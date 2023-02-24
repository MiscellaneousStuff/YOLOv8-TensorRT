from models import TRTModule, TRTProfilerV1 # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import math
import numpy as np

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess_para, det_postprocess
from models.utils import blob, letterbox, path_to_list

# Use YOLOv8 Image Loader to get video frames
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages

def create_batch(images, W, H, batch_size=1, device=None):
    source_s, draw_s, im0_s, dwdh_s, ratio_s = [], [], [], [], []
    for _ in range(batch_size):
        try:
            source, _, im0, _, _ = next(images)
            bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            tensor = torch.asarray(tensor, device=device)
            tensor = torch.squeeze(tensor, dim=0)
            source_s.append(source)
            draw_s.append(im0)
            im0_s.append(tensor)
            dwdh_s.append(dwdh)
            ratio_s.append(ratio)
        except StopIteration:
            source_s.append(None)
            draw_s.append(None)
            im0_s.append(torch.tensor(np.zeros_like(im0_s[0].cpu()), device=device))
            dwdh_s.append(None)
            ratio_s.append(None)

    im0_s = torch.stack(im0_s)
    # print("create_batch->im0_s.shape", im0_s.shape)

    return source_s, draw_s, im0_s, dwdh_s, ratio_s

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    profiler = TRTProfilerV1()
    Engine = TRTModule(args.engine, device)
    Engine.set_profiler(profiler)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    save_path = Path(args.out_dir)
    
    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    
    images = LoadImages(args.source,
                        imgsz=640,
                        stride=32,
                        auto=True,
                        transforms=None,
                        vid_stride=1)
    
    print("images.frames:", images.frames)
    batch_size = args.batch_size
    batch_idxs = int(math.ceil(images.frames / batch_size))

    images.__iter__()

    frame_count = 0

    for batch_idx in range(batch_idxs):
        print("BATCH_IDX:", batch_idx)

        source_s, draw_s, im0_s, dwdh_s, ratio_s = \
            create_batch(images, W, H, batch_size, device)
        
        tensor = im0_s
        print("tensor.shape:", tensor.shape)
        data = Engine(tensor)
        print("len(data), len(data[0]), len(data[1]), len(data[2]), len(data[3]):", len(data), len(data[0]), len(data[1]), len(data[2]), len(data[3]))
        print("data.shapes->", [data[i].shape for i in range(4)])

        for i in range(tensor.shape[0]):
            frame_count += 1
            
            cur_data = (data[0][i], data[1][i], data[2][i], data[3][i])
            bboxes, scores, labels = det_postprocess_para(cur_data)
            
            bboxes -= dwdh_s[i]
            bboxes /= ratio_s[i]

            save_image = save_path / (source_s[i].split(".")[0] + str(frame_count+1) + ".jpg")

            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES[cls_id]
                color = COLORS[cls]
                cv2.rectangle(
                    img=draw_s[i],
                    pt1=tuple(bbox[:2]),
                    pt2=tuple(bbox[2:]),
                    color=color,
                    thickness=2)
                cv2.putText(draw_s[i],
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
            if args.show:
                cv2.imshow('result', draw_s[i])
                cv2.waitKey(0)
            else:
                cv2.imwrite(str(save_image), draw_s[i])

    Engine.context.profiler.report()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--source', type=str, help='YOLOv8 source')
    parser.add_argument('--batch-size', type=int, help='Number of frames per inference')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)