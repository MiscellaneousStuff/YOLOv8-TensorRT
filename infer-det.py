from models import TRTModule, TRTProfilerV1 # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

# Use YOLOv8 Image Loader to get video frames
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages

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
    
    frame_count = 0
    for image in images:
        frame_count += 1
        source, _, im0, _, _ = image
        
        save_image = save_path / (source.split(".")[0] + str(frame_count+1) + ".jpg")
        print("save image:", save_image)
        
        bgr = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)

        print("SHAPE TO INFER:", frame_count+1, tensor.shape)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            """
            cv2.rectangle(
                img=draw,
                pt1=tuple(bbox[:2]),
                pt2=tuple(bbox[2:]),
                color=color,
                thickness=2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
            """
        
        """
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)
        """

    Engine.context.profiler.report()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--source', type=str, help='YOLOv8 source')
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