import os
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3, 4]) 
        meta = src.meta
    return np.transpose(image, (1, 2, 0)), meta  

def normalize_image(image):
    norm_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        norm_image[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return norm_image

def extract_features(image, method='ORB', nfeatures=1000):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=nfeatures)
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
  
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_transformation(kp1, kp2, matches):
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask

def warp_image(image, M, shape):
    warped = cv2.warpPerspective(image, M, (shape[1], shape[0]))
    return warped

def calculate_corners(M, transform):
    height, width = 10980, 10980  
    corners = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_corners = cv2.perspectiveTransform(corners, M)
    
    geo_corners = []
    for pt in transformed_corners:
        x_pixel, y_pixel = pt[0]
        x_geo, y_geo = transform * (x_pixel, y_pixel)
        geo_corners.append((x_geo, y_geo))
    
    return geo_corners

def detect_and_log_dead_pixels(original, corrected):
    dead_pixels_log = []
    threshold_under = 0.15
    threshold_over = 5.0

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            for k in range(original.shape[2]):
                original_value = original[i, j, k]
                corrected_value = corrected[i, j, k]
                if original_value == 0 or original_value < threshold_under * np.mean(original[:, :, k]) or original_value > threshold_over * np.mean(original[:, :, k]):
                    dead_pixels_log.append((i, j, k + 1, original_value, corrected_value))
    
    return dead_pixels_log

def correct_dead_pixels(image):
    corrected_image = np.copy(image)
    for k in range(image.shape[2]):
        mask = (image[:, :, k] == 0) | (image[:, :, k] < 0.15 * np.mean(image[:, :, k])) | (image[:, :, k] > 5 * np.mean(image[:, :, k]))
        corrected_image[mask, k] = np.median(image[~mask, k])
    return corrected_image

def generate_output(scene_path, corners, corrected_pixels, corrected_image, output_dir):
    base_name = os.path.basename(scene_path).replace('.tif', '')
    corners_path = os.path.join(output_dir, f"{base_name}_corners.txt")
    dead_pixels_path = os.path.join(output_dir, f"{base_name}_dead_pixels.txt")
    geo_tiff_output_path = os.path.join(output_dir, f"{base_name}_corrected.tif")

    with open(corners_path, 'w') as f:
        for corner in corners:
            f.write(f"{corner[0]:.3f}; {corner[1]:.3f}\n")
    
    with open(dead_pixels_path, 'w') as f:
        for dp in corrected_pixels:
            f.write(f"{dp[0]}; {dp[1]}; {dp[2]}; {dp[3]}; {dp[4]}\n")
    
 
    corrected_image = corrected_image.astype(np.float32)

 
    with rasterio.open(scene_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=4)
        with rasterio.open(geo_tiff_output_path, 'w', **profile) as dst:
            dst.write(corrected_image.transpose(2, 0, 1))

def process_scene(scene_path, sentinel_files, output_dir):
    try:
        logging.info(f"Processing scene file: {scene_path}")
        scene, scene_meta = load_image(scene_path)
        scene_norm = normalize_image(scene)
        
        for sentinel_path in sentinel_files:
            logging.info(f"Processing sentinel file: {sentinel_path}")
            sentinel, sentinel_meta = load_image(sentinel_path)
            sentinel_norm = normalize_image(sentinel)

            scene_resized = cv2.resize(scene_norm, (sentinel_norm.shape[1], sentinel_norm.shape[0]))

         
            kp_scene, desc_scene = extract_features(scene_resized, method='SIFT', nfeatures=1000)
            kp_sentinel, desc_sentinel = extract_features(sentinel_norm, method='SIFT', nfeatures=1000)

            matches = match_features(desc_scene, desc_sentinel)

            if len(matches) < 4:
                logging.warning(f"Not enough matches found with SIFT between {os.path.basename(scene_path)} and {os.path.basename(sentinel_path)}. Skipping this pair.")
                continue

            M, mask = estimate_transformation(kp_scene, kp_sentinel, matches)
            if M is None:
                logging.warning(f"Failed to compute homography for {os.path.basename(scene_path)} and {os.path.basename(sentinel_path)}. Skipping this pair.")
                continue

            warped_scene = warp_image(scene_norm, M, sentinel_norm.shape)

      
            transform = sentinel_meta['transform']
            corners = calculate_corners(M, transform)


            corrected_image = correct_dead_pixels(scene)
            corrected_pixels = detect_and_log_dead_pixels(scene, corrected_image)

            generate_output(scene_path, corners, corrected_pixels, corrected_image, output_dir)

            logging.info(f"Finished processing scene file: {scene_path} with sentinel file: {sentinel_path}")
    
    except Exception as e:
        logging.error(f"Error processing {scene_path}: {e}")

def main(scene_folder, sentinel_folder, output_dir):
    scene_files = sorted([os.path.join(scene_folder, f) for f in os.listdir(scene_folder) if f.endswith(('.tif', '.tiff'))])
    sentinel_files = sorted([os.path.join(sentinel_folder, f) for f in os.listdir(sentinel_folder) if f.endswith(('.tif', '.tiff'))])

    for scene_path in scene_files:
        process_scene(scene_path, sentinel_files, output_dir)

if __name__ == "__main__":
    scene_folder = r"C:\Users\Isaac.Choma\Russian Hackathon\Russian Hackathon\1_20"
    sentinel_folder = r"C:\Users\Isaac.Choma\Russian Hackathon\Russian Hackathon\layouts"
    output_dir = r"C:\Users\Isaac.Choma\Russian Hackathon\Russian Hackathon\output"

    main(scene_folder, sentinel_folder, output_dir)
