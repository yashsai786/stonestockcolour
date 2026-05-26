import cv2
import numpy as np
import httpx
import json
import asyncio
from src.infrastructure.cv.opencv_slab_detection_service import OpenCVSlabDetectionService
from src.infrastructure.cv.hsv_skin_removal_service import HSVSkinRemovalService
from src.infrastructure.cv.cv_dominant_color_analyzer import CVDominantColorAnalyzer
from src.infrastructure.cv.cv_color_matcher_service import CVColorMatcherService
from src.infrastructure.repositories.json_color_profile_repository import JsonColorProfileRepository

def touches_border(contour, w, h, border_dist=3):
    for pt in contour:
        x, y = pt[0][0], pt[0][1]
        if x <= border_dist or x >= w - 1 - border_dist or y <= border_dist or y >= h - 1 - border_dist:
            return True
    return False

async def main():
    url = "https://images.stonestocks.com/cdn-cgi/image/width=1200,quality=85/https://images.stonestocks.com/images/ooeccdeiv7/23af2ca3-2b4d-5a09-994f-165dacd111f6.jpg"
    print(f"Downloading {url}...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            print("Failed to download image!")
            return
        img_bytes = response.content
        
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode image!")
        return
        
    h, w = img.shape[:2]
    max_dimension = 600
    scale = max_dimension / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    total_area = rw * rh
    
    # 1. Slab detection
    detector = OpenCVSlabDetectionService()
    contour, slab_mask = detector.detect_slab(resized)
    
    # 2. Skin removal
    skin_remover = HSVSkinRemovalService()
    raw_skin_mask = skin_remover.remove_skin(resized)
    conts, _ = cv2.findContours(raw_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_skin_mask = np.zeros_like(raw_skin_mask)
    for c in conts:
        area = cv2.contourArea(c)
        if touches_border(c, rw, rh) and (area < 0.10 * total_area):
            cv2.drawContours(filtered_skin_mask, [c], -1, 255, thickness=cv2.FILLED)
            
    valid_mask = cv2.bitwise_and(slab_mask, cv2.bitwise_not(filtered_skin_mask))
    
    # 3. LAB extraction
    lab_img = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)
    raw_lab_pixels = lab_img[valid_mask > 0].astype(np.float32)
    standard_lab_pixels = np.empty_like(raw_lab_pixels)
    standard_lab_pixels[:, 0] = raw_lab_pixels[:, 0] * (100.0 / 255.0)
    standard_lab_pixels[:, 1] = raw_lab_pixels[:, 1] - 128.0
    standard_lab_pixels[:, 2] = raw_lab_pixels[:, 2] - 128.0
    
    # 4. KMeans
    analyzer = CVDominantColorAnalyzer()
    dominant_colors = analyzer.analyze(standard_lab_pixels, max_colors=3)
    
    # 5. Matching
    repo = JsonColorProfileRepository()
    matcher = CVColorMatcherService(repo)
    matches = matcher.match_palette(dominant_colors)
    
    # Map using the router functions
    def map_to_master_color(color_name):
        if not color_name or color_name.lower() == "none":
            return None
        c_lower = color_name.lower()
        if "white" in c_lower:
            return "White"
        elif any(x in c_lower for x in ["grey", "gray", "charcoal", "slate"]):
            return "Grey"
        elif any(x in c_lower for x in ["beige", "ivory", "cream", "sand", "taupe"]):
            return "Beige"
        elif any(x in c_lower for x in ["brown", "coffee", "chocolate", "rust", "terracotta"]):
            return "Brown"
        elif "black" in c_lower:
            return "Black"
        elif any(x in c_lower for x in ["red", "burgundy"]):
            return "Red"
        elif any(x in c_lower for x in ["green", "sage", "olive", "emerald", "forest"]):
            return "Green"
        elif any(x in c_lower for x in ["pink", "rose", "blush"]):
            return "Pink / Rose"
        else:
            return "Multi"
            
    m_primary = map_to_master_color(matches[0][0].name)
    m_secondary = map_to_master_color(matches[1][0].name) if len(matches) > 1 else None
    m_accent = map_to_master_color(matches[2][0].name) if len(matches) > 2 else None
    
    primary_pct = round(matches[0][1], 1)
    secondary_pct = round(matches[1][1], 1) if len(matches) > 1 else 0.0
    accent_pct = round(matches[2][1], 1) if len(matches) > 2 else 0.0
    
    # 2. Determine geological category
    p_name_l = m_primary.lower()
    s_name_l = (m_secondary or "").lower()
    a_name_l = (m_accent or "").lower()
    
    if primary_pct >= 85.0:
        stone_category = "Uniform"
    elif any(x in p_name_l or x in s_name_l or x in a_name_l for x in ["red", "brown", "green", "pink"]):
        stone_category = "Breccia"
    elif "white" in p_name_l and any(y in s_name_l or y in a_name_l for y in ["black", "grey"]):
        stone_category = "Bold Veined"
    else:
        stone_category = "Linear"
        
    # 3. Determine master pattern matching the 6 UI filters
    if primary_pct >= 85.0:
        pattern = "Uniform"
    elif stone_category == "Breccia":
        if accent_pct > 0 and accent_pct <= 12.0:
            pattern = "Webbed"
        else:
            pattern = "Breccia"
    elif stone_category == "Bold Veined":
        pattern = "Bold Veined"
    else: # Veined
        if accent_pct > 0 and accent_pct <= 8.0:
            pattern = "Linear"
        else:
            pattern = "Cloudy"
            
    result = {
        "primary_color": m_primary,
        "secondary_color": m_secondary,
        "accent_color": m_accent,
        "primary_percentage": primary_pct,
        "secondary_percentage": secondary_pct,
        "accent_percentage": accent_pct,
        "confidence": 0.95,
        "pattern": pattern,
        "stone_category": stone_category
    }
    
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
