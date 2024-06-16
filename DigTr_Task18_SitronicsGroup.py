import cv2
import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import os

def load_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3, 4])  # Reading all four channels
    return np.transpose(image, (1, 2, 0))  # Transposing to match height, width, channels format

def normalize_image(image):
    image_normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        image_normalized[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return image_normalized

def resize_image(image, target_resolution):
    return cv2.resize(image, (target_resolution[1], target_resolution[0]), interpolation=cv2.INTER_LINEAR)

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_transformation(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask

def warp_image(image, M, target_shape):
    warped_image = cv2.warpPerspective(image, M, (target_shape[1], target_shape[0]))
    return warped_image

def main(scene_folder, sentinel_folder):
    # Assume file names are consistent and correspond in both folders
    scene_files = sorted([os.path.join(scene_folder, f) for f in os.listdir(scene_folder) if f.endswith('.tif')])
    sentinel_files = sorted([os.path.join(sentinel_folder, f) for f in os.listdir(sentinel_folder) if f.endswith('.tif')])
    
    for scene_path, sentinel_path in zip(scene_files, sentinel_files):
        # Load images
        scene = load_image(scene_path)
        sentinel = load_image(sentinel_path)

        # Normalize images
        scene_norm = normalize_image(scene)
        sentinel_norm = normalize_image(sentinel)

        # Resize scene to Sentinel resolution for feature matching
        scene_resized = resize_image(scene_norm, sentinel_norm.shape)

        # Extract features
        kp_scene, desc_scene = extract_features(scene_resized)
        kp_sentinel, desc_sentinel = extract_features(sentinel_norm)

        # Match features
        matches = match_features(desc_scene, desc_sentinel)

        # Estimate transformation
        M, mask = estimate_transformation(kp_scene, kp_sentinel, matches)

        # Warp scene to align with Sentinel image
        warped_scene = warp_image(scene_norm, M, sentinel_norm.shape)

        # Show results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Scene")
        plt.imshow(scene_norm[:, :, :3].astype(np.uint8))  # Display RGB channels
        plt.subplot(1, 2, 2)
        plt.title("Warped Scene")
        plt.imshow(warped_scene[:, :, :3].astype(np.uint8))  # Display RGB channels
        plt.show()

if __name__ == '__main__':
    scene_folder = r'C:\Users\Isaac.Choma\OneDrive - Telestream Communications\Desktop\DOCS\New folder\Russian Hackathon\1_20'
    sentinel_folder = r'C:\Users\Isaac.Choma\OneDrive - Telestream Communications\Desktop\DOCS\New folder\Russian Hackathon\layouts'
    main(scene_folder, sentinel_folder)
