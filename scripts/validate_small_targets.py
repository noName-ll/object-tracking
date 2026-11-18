"""Run from repository root with the Columbia COIL-100 ZIP and test_bus.jpg.
Usage: python scripts/validate_small_targets.py /path/to/coil-100.zip > results.csv
"""
import csv
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch

sys.path.insert(0, str(Path("legacy/python").resolve()))
from track import estimate

archive = zipfile.ZipFile(sys.argv[1])
torch.set_num_threads(2)
device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)
background = cv2.imread("models/yoloe/test_bus.jpg")
angles = [0, 5, 15, 30, 60, 90, 135, 180, None, 180, 0]


from small_target_scene import paste, iou

def features(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return extractor.extract(numpy_image_to_torch(rgb).to(device))


writer=csv.writer(sys.stdout, lineterminator="\n")
writer.writerow("method object size fill frame angle present emitted iou status".split())
with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
    folder=Path(tmp)
    for obj in (1, 5, 20):
        video=str(folder/"frames.avi")
        out=cv2.VideoWriter(video,cv2.VideoWriter_fourcc(*"MJPG"),24,(320,240))
        truth=[]
        for i,angle in enumerate(angles):
            frame=cv2.resize(background[500+i*4:900+i*4,100+i*7:640+i*7],(320,240))
            paste(archive, frame,2,0,58,245,130)
            truth.append(None if angle is None else paste(archive, frame,obj,angle,70,20+12*i,45+i*3))
            out.write(frame)
        out.release()
        cap=cv2.VideoCapture(video)
        frames=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            frames.append(frame)
        cap.release()
        assert len(frames)==len(truth)
        frame_features=[features(frame) for frame in frames]
        for size in (40,90):
            for fill in (0.95,0.60):
                ref=cv2.resize(background[:250,:250],(size,size))
                side=round(size*fill)
                paste(archive, ref,obj,0,side,(size-side)//2,(size-side)//2)
                path=str(folder/"target.png")
                cv2.imwrite(path,ref)
                proc=subprocess.run(["target/debug/mobile-object-tracker","--search-backend","features","--video",video,"--target",path],capture_output=True,text=True)
                if proc.returncode and "at least 12 SIFT" not in proc.stderr:
                    raise RuntimeError(proc.stderr)
                rows=[json.loads(line) for line in proc.stdout.splitlines()]
                assert proc.returncode or len(rows)==len(truth)
                ref_features=features(ref)
                for i,(gt,ff) in enumerate(zip(truth,frame_features)):
                    row=rows[i] if rows else {"bbox":None,"status":"reference_rejected"}
                    box=row["bbox"]
                    a=None if box is None else [box[k] for k in ("x","y","width","height")]
                    pairs=matcher({"image0":ref_features,"image1":ff})["matches"][0].cpu().numpy()
                    src=ref_features["keypoints"][0].cpu().numpy()[pairs[:,0]]
                    dst=ff["keypoints"][0].cpu().numpy()[pairs[:,1]]
                    result=estimate(src,dst,ref.shape,frames[i].shape,12,0.35)
                    b=None if result is None else cv2.boundingRect(result["quad"])
                    for method,pred,status in [("rust",a,row["status"]),("lightglue",b,"match" if b else "missing")]:
                        writer.writerow([method,obj,size,fill,i,angles[i],int(gt is not None),int(pred is not None),round(iou(pred,gt),4),status])
        sys.stdout.flush()
