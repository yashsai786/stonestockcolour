import cv2
import numpy as np
import httpx
import asyncio
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService
from src.infrastructure.cv.cv_dominant_color_analyzer import CVDominantColorAnalyzer
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService
from src.infrastructure.repositories.json_color_profile_repository import JsonColorProfileRepository

def touches_border(contour, w, h, border_dist=3):
    """Returns True if the contour touches the outer boundaries of the image."""
    for pt in contour:
        x, y = pt[0][0], pt[0][1]
        if x <= border_dist or x >= w - 1 - border_dist or y <= border_dist or y >= h - 1 - border_dist:
            return True
    return False

async def main():
    # Load the local green marble slab image directly
    img_path = "/home/biz26/.gemini/antigravity/brain/d65caf73-5502-47ac-9eb0-f094cea4a595/media__1779778195409.jpg"
    print(f"Loading {img_path}...")
    
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image!")
        return
        
    h, w = img.shape[:2]
    
    # Resize
    max_dimension = 600
    scale = max_dimension / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    total_area = rw * rh
        
    # 1. Detect Slab
    detector = OpenCVSlabDetectionService()
    contour, slab_mask = detector.detect_slab(resized)
    
    # 2. Skin removal with boundary + area filtering
    skin_remover = HSVSkinRemovalService()
    raw_skin_mask = skin_remover.remove_skin(resized)
    
    conts, _ = cv2.findContours(raw_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_skin_mask = np.zeros_like(raw_skin_mask)
    
    masked_count = 0
    preserved_count = 0
    for c in conts:
        area = cv2.contourArea(c)
        touches = touches_border(c, rw, rh)
        if touches and (area < 0.10 * total_area):
            cv2.drawContours(filtered_skin_mask, [c], -1, 255, thickness=cv2.FILLED)
            masked_count += 1
        else:
            preserved_count += 1
            
    print(f"Total skin contours: {len(conts)}")
    print(f"Masked (hands at boundary): {masked_count}")
    print(f"Preserved (slab textures): {preserved_count}")
    
    valid_mask = cv2.bitwise_and(slab_mask, cv2.bitwise_not(filtered_skin_mask))
    print(f"valid_mask non-zero pixels: {np.sum(valid_mask > 0)}")
    
    # 3. Extract LAB pixels
    lab_img = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
    raw_lab_pixels = lab_img[valid_mask > 0].astype(np.float32)
    standard_lab_pixels = np.empty_like(raw_lab_pixels)
    standard_lab_pixels[:, 0] = raw_lab_pixels[:, 0] * (100.0 / 255.0)
    standard_lab_pixels[:, 1] = raw_lab_pixels[:, 1] - 128.0
    standard_lab_pixels[:, 2] = raw_lab_pixels[:, 2] - 128.0
    
    # Sample pixels
    max_sample_size = 10000
    if len(standard_lab_pixels) > max_sample_size:
        indices = np.random.choice(len(standard_lab_pixels), max_sample_size, replace=False)
        sampled_pixels = standard_lab_pixels[indices]
    else:
        sampled_pixels = standard_lab_pixels
        
    # 4. Dominant color analysis
    analyzer = CVDominantColorAnalyzer()
    dominant_colors = analyzer.analyze(sampled_pixels, max_colors=3)
    
    print("\n--- Dominant Colors Extracted (LAB) ---")
    for idx, c in enumerate(dominant_colors):
        print(f"Cluster {idx+1}: L={round(c.lab.l, 2)}, a={round(c.lab.a, 2)}, b={round(c.lab.b, 2)} | weight={round(c.percentage*100, 2)}%")
        
    # 5. Matching against refined color catalog
    repo = JsonColorProfileRepository()
    matcher = CVColorMatcherService(repo)
    matches = matcher.match_palette(dominant_colors)
    
    print("\n--- Catalog Matching Results ---")
    for idx, (profile, pct) in enumerate(matches):
        print(f"Match {idx+1}: {profile.name} (weight={round(pct*100, 2)}%) [Catalog LAB: L={profile.lab.l}, a={profile.lab.a}, b={profile.lab.b}]")

if __name__ == "__main__":
    asyncio.run(main())
