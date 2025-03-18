
import os
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import torch
import argparse

# Paths
MODEL_DIR = "./models"
INPUT_DIR = "./inputs"
OUTPUT_DIR = "./outputs"
DEBUG_DIR = "./debug"
TEMP_FRAMES = "./temp_frames"
for d in [MODEL_DIR, INPUT_DIR, OUTPUT_DIR, DEBUG_DIR, TEMP_FRAMES]:
    os.makedirs(d, exist_ok=True)

# Load models
use_cuda = torch.cuda.is_available()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
face_analyzer = FaceAnalysis(name="buffalo_l", root=MODEL_DIR, providers=providers)
face_analyzer.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
swapper = ort.InferenceSession(f"{MODEL_DIR}/inswapper_128.onnx", providers=providers)

def swap_faces(source_img, target_img, prev_flow=None):
    source_faces = face_analyzer.get(source_img)
    target_faces = face_analyzer.get(target_img)
    if not source_faces or not target_faces:
        print("No faces detected!")
        return target_img, prev_flow

    source_face = source_faces[0]
    result = target_img.copy()
    for i, target_face in enumerate(target_faces):
        src_pts = np.float32([source_face.kps[0], source_face.kps[1], source_face.kps[2]])
        dst_pts = np.float32([target_face.kps[0], target_face.kps[1], target_face.kps[2]])
        M = cv2.getAffineTransform(dst_pts, src_pts)
        target_crop = target_img[int(target_face.bbox[1]):int(target_face.bbox[3]), int(target_face.bbox[0]):int(target_face.bbox[2])]
        if target_crop.size == 0:
            print(f"Target crop empty for face {i}, skipping!")
            continue
        target_aligned = cv2.warpAffine(target_crop, M, (128, 128), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_LANCZOS4)

        cv2.imwrite(f"{DEBUG_DIR}/aligned_target_{i}.jpg", target_aligned)

        target_input = target_aligned.astype(np.float32) / 255.0
        target_input = (target_input * 2) - 1
        target_input = target_input.transpose(2, 0, 1)[np.newaxis, ...]

        swapped = swapper.run(None, {
            "source": source_face.normed_embedding.reshape(1, -1).astype(np.float32),
            "target": target_input
        })[0][0].transpose(1, 2, 0)
        swapped = (np.clip(swapped, -1, 1) + 1) / 2 * 255
        swapped = swapped.astype(np.uint8)

        cv2.imwrite(f"{DEBUG_DIR}/swapped_raw_{i}.jpg", swapped)

        swapped_resized = cv2.resize(swapped, (target_crop.shape[1], target_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        kps = target_face.kps.astype(np.int32)
        hull = cv2.convexHull(kps)
        mask = np.zeros_like(target_crop, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        center = (target_crop.shape[1] // 2, target_crop.shape[0] // 2)
        target_crop = cv2.seamlessClone(swapped_resized, target_crop, mask, center, cv2.MIXED_CLONE)
        result[int(target_face.bbox[1]):int(target_face.bbox[3]), int(target_face.bbox[0]):int(target_face.bbox[2])] = target_crop

    if prev_flow is not None:
        gray_prev = cv2.cvtColor(prev_flow, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        flow = None

    return result, flow

def process_image(source_path, target_path, output_path):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    if source_img is None or target_img is None:
        print("Image load failed!")
        return
    result, _ = swap_faces(source_img, target_img, None)
    cv2.imwrite(output_path, result)
    print(f"Saved image: {output_path}")

def process_video(source_path, target_video, output_video):
    clip = VideoFileClip(target_video)
    frame_count = int(clip.fps * clip.duration)
    source_img = cv2.imread(source_path)
    if source_img is None:
        print("Source image load failed!")
        return
    prev_frame = None

    for i, frame in tqdm(enumerate(clip.iter_frames(fps=clip.fps, dtype="uint8")), total=frame_count):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result, flow = swap_faces(source_img, frame_bgr, prev_frame)
        cv2.imwrite(f"{TEMP_FRAMES}/frame_{i:05d}.jpg", result)
        prev_frame = result.copy()

    frame_files = sorted([f for f in os.listdir(TEMP_FRAMES) if f.endswith('.jpg')])
    def make_frame(t):
        frame_idx = int(t * clip.fps)
        frame_path = f"{TEMP_FRAMES}/frame_{frame_idx:05d}.jpg"
        return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)

    output_clip = clip.set_make_frame(make_frame)
    output_clip.write_videofile(output_video, fps=clip.fps, codec="libx264", audio=True, threads=4)
    print(f"Saved video: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Face Swap Tool")
    parser.add_argument('--source', type=str, required=True, help="Path to source image")
    parser.add_argument('--target', type=str, required=True, help="Path to target image or video")
    parser.add_argument('--output', type=str, default=f"{OUTPUT_DIR}/result.jpg", help="Path to output")
    args = parser.parse_args()

    if args.target.endswith(('.jpg', '.jpeg', '.png')):
        process_image(args.source, args.target, args.output)
    elif args.target.endswith(('.mp4', '.avi', '.mov')):
        process_video(args.source, args.target, args.output.replace('.jpg', '.mp4'))
    else:
        print("Target must be an image (.jpg, .png) or video (.mp4, .avi)!")

if __name__ == "__main__":
    main()
