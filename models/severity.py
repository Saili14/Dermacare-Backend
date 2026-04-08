import cv2
import numpy as np
import json
from datetime import datetime

# ----------------------------
# ENHANCED SCORING FUNCTIONS (3 params each)
# ----------------------------

def score_param(value, low, high):
    """Score parameter with smoother transitions"""
    if value < low:
        return 1  # mild
    elif value < high:
        return 2  # moderate
    else:
        return 3  # severe


def preprocess_image(img):
    """Preprocess image for better analysis"""
    # Resize to standard size
    img = cv2.resize(img, (512, 512))
    # Apply slight blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def detect_skin_mask(img):
    """Create mask for skin regions"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for skin detection
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


# Acne: lesion count, redness intensity, texture roughness
def analyze_acne(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Count dark spots/lesions using adaptive threshold (less sensitive)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    thresh_in_skin = cv2.bitwise_and(thresh, skin_mask)
    contours, _ = cv2.findContours(thresh_in_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 100])
    
    # 2. Redness intensity
    red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_in_skin = cv2.bitwise_and(red_mask, skin_mask)
    redness = cv2.countNonZero(red_in_skin)
    
    # 3. Texture roughness using edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges_in_skin = cv2.bitwise_and(edges, skin_mask)
    texture = cv2.countNonZero(edges_in_skin)
    
    # More realistic thresholds for acne classification
    score = (score_param(lesion_count, 15, 35) + 
             score_param(redness, 15000, 40000) + 
             score_param(texture, 8000, 20000))
    
    details = {
        'lesion_count': lesion_count,
        'redness_pixels': redness,
        'texture_roughness': texture
    }
    
    return classify_score(score), score, details


# Eczema: redness, dryness/scaling, affected area
def analyze_eczema(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Redness (more sensitive for eczema)
    red_mask = cv2.inRange(hsv, (0, 30, 30), (15, 255, 255))
    red_in_skin = cv2.bitwise_and(red_mask, skin_mask)
    redness = cv2.countNonZero(red_in_skin)
    
    # 2. Dryness/scaling (bright patches)
    bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    scaling_in_skin = cv2.bitwise_and(bright_mask, skin_mask)
    scaling = cv2.countNonZero(scaling_in_skin)
    
    # 3. Affected area irregularity
    irregularity = np.std(gray[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
    
    score = (score_param(redness, 8000, 30000) + 
             score_param(scaling, 2000, 10000) + 
             score_param(irregularity, 25, 60))
    
    details = {
        'redness_pixels': redness,
        'scaling_pixels': scaling,
        'irregularity_score': round(irregularity, 2)
    }
    
    return classify_score(score), score, details


# Lichen: raised patches, edge definition, color variation
def analyze_lichen(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge density (well-defined patches)
    edges = cv2.Canny(gray, 100, 200)
    edges_in_skin = cv2.bitwise_and(edges, skin_mask)
    edge_density = cv2.countNonZero(edges_in_skin)
    
    # 2. Patch contrast (raised appearance)
    contrast = np.std(gray[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
    
    # 3. Color uniformity within patches
    mean_intensity = np.mean(gray[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 128
    
    score = (score_param(edge_density, 4000, 15000) + 
             score_param(contrast, 30, 70) + 
             score_param(abs(mean_intensity - 128), 20, 50))
    
    details = {
        'edge_density': edge_density,
        'contrast_score': round(contrast, 2),
        'mean_intensity': round(mean_intensity, 2)
    }
    
    return classify_score(score), score, details


# Psoriasis: scaling, plaque thickness, affected area
def analyze_psoriasis(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Scaling (silvery-white patches)
    bright_mask = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    scaling_in_skin = cv2.bitwise_and(bright_mask, skin_mask)
    scaling = cv2.countNonZero(scaling_in_skin)
    
    # 2. Texture variation (plaque thickness)
    texture_var = np.var(gray[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
    
    # 3. Affected area size
    affected_area = cv2.countNonZero(skin_mask)
    
    score = (score_param(scaling, 3000, 12000) + 
             score_param(texture_var, 800, 2000) + 
             score_param(affected_area, 15000, 50000))
    
    details = {
        'scaling_pixels': scaling,
        'texture_variance': round(texture_var, 2),
        'affected_area': affected_area
    }
    
    return classify_score(score), score, details


# Rosacea: persistent redness, papules, affected area
def analyze_rosacea(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Persistent redness (central face focus)
    red_mask = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
    red_in_skin = cv2.bitwise_and(red_mask, skin_mask)
    redness = cv2.countNonZero(red_in_skin)
    
    # 2. Papules/bumps
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    papules_in_skin = cv2.bitwise_and(thresh, skin_mask)
    papules = cv2.countNonZero(papules_in_skin)
    
    # 3. Saturation intensity (flushed appearance)
    saturation_mean = np.mean(hsv[:, :, 1][skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
    
    score = (score_param(redness, 10000, 35000) + 
             score_param(papules, 5000, 20000) + 
             score_param(saturation_mean, 60, 140))
    
    details = {
        'redness_pixels': redness,
        'papules_pixels': papules,
        'saturation_intensity': round(saturation_mean, 2)
    }
    
    return classify_score(score), score, details


# Vitiligo: depigmented area, patch count, border definition
def analyze_vitiligo(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. White/light patches
    light_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    light_in_skin = cv2.bitwise_and(light_mask, skin_mask)
    white_area = cv2.countNonZero(light_in_skin)
    
    # 2. Patch count
    contours, _ = cv2.findContours(light_in_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patch_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 100])
    
    # 3. Contrast difference
    contrast = np.std(gray[skin_mask > 0]) if np.sum(skin_mask > 0) > 0 else 0
    
    score = (score_param(white_area, 2000, 15000) + 
             score_param(patch_count, 2, 8) + 
             score_param(contrast, 30, 80))
    
    details = {
        'white_area_pixels': white_area,
        'patch_count': patch_count,
        'contrast_score': round(contrast, 2)
    }
    
    return classify_score(score), score, details


# Warts: raised lesions, texture, cluster density
def analyze_warts(img):
    img = preprocess_image(img)
    skin_mask = detect_skin_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Raised spots detection
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    warts_in_skin = cv2.bitwise_and(thresh, skin_mask)
    contours, _ = cv2.findContours(warts_in_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wart_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 50])
    
    # 2. Surface roughness
    edges = cv2.Canny(gray, 80, 160)
    edges_in_skin = cv2.bitwise_and(edges, skin_mask)
    roughness = cv2.countNonZero(edges_in_skin)
    
    # 3. Lesion density
    total_wart_area = sum([cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 50])
    
    score = (score_param(wart_count, 2, 8) + 
             score_param(roughness, 2000, 8000) + 
             score_param(total_wart_area, 500, 3000))
    
    details = {
        'wart_count': wart_count,
        'surface_roughness': roughness,
        'total_wart_area': round(total_wart_area, 2)
    }
    
    return classify_score(score), score, details


# ----------------------------
# Enhanced classification with confidence
# ----------------------------
def classify_score(score):
    """Classify total score into severity with adjusted thresholds"""
    score = max(0, score)   # only prevent negative
    if score <= 6:  # More lenient for Low
        severity = "Low"
    elif score <= 15:  # More lenient for Moderate  
        severity = "Moderate"
    else:
        severity = "High"
    
    return severity


def get_confidence_score(score):
    """Get confidence score for the classification"""
    if score <= 4:
        return min(0.85 + (4 - score) * 0.03, 0.95)
    elif score <= 6:
        return 0.80 + abs(5 - score) * 0.05
    else:
        return min(0.85 + (score - 6) * 0.02, 0.95)


# ----------------------------
# Enhanced analyzer selector with detailed output
# ----------------------------
analyzers = {
    "acne": analyze_acne,
    "eczema": analyze_eczema,
    "lichen": analyze_lichen,
    "psoriasis": analyze_psoriasis,
    "rosacea": analyze_rosacea,
    "vitiligo": analyze_vitiligo,
    "warts": analyze_warts,
}


# ----------------------------
# Home remedies database (Home remedies for Low/Moderate, Dermat for High)
# ----------------------------
HOME_REMEDIES = {
    'acne': {
        'Low': [
            "Apply tea tree oil (diluted) as spot treatment",
            "Use honey face masks 2-3 times per week",
            "Cleanse with gentle salicylic acid cleanser",
            "Apply aloe vera gel to reduce inflammation"
        ],
        'Moderate': [
            "Use clay masks with bentonite clay twice weekly",
            "Apply diluted apple cider vinegar as toner",
            "Try turmeric paste with honey for spot treatment",
            "Use green tea compress on affected areas",
            "Apply ice to reduce inflammation",
            "Use oatmeal masks for gentle exfoliation"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'eczema': {
        'Low': [
            "Apply coconut oil or shea butter as moisturizer",
            "Use oatmeal baths for soothing relief",
            "Try aloe vera gel for mild inflammation",
            "Use gentle, fragrance-free soaps"
        ],
        'Moderate': [
            "Apply cool wet compresses for 15-20 minutes",
            "Use thick moisturizers like petroleum jelly",
            "Try chamomile tea compresses",
            "Add baking soda to lukewarm baths",
            "Apply diluted apple cider vinegar",
            "Use evening primrose oil"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'psoriasis': {
        'Low': [
            "Apply aloe vera gel daily",
            "Use coconut oil as moisturizer",
            "Try turmeric paste applications",
            "Take lukewarm oatmeal baths"
        ],
        'Moderate': [
            "Apply dead sea salt baths",
            "Use coal tar soap if available",
            "Try apple cider vinegar diluted applications",
            "Apply thick moisturizers after bathing",
            "Get moderate sun exposure (15-20 minutes)",
            "Use mahonia aquifolium cream"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'rosacea': {
        'Low': [
            "Apply green tea compresses",
            "Use gentle, fragrance-free moisturizers",
            "Apply aloe vera gel",
            "Use mineral sunscreen daily"
        ],
        'Moderate': [
            "Apply cool compresses during flare-ups",
            "Use oatmeal masks for soothing",
            "Try diluted apple cider vinegar",
            "Avoid hot beverages and spicy foods",
            "Use gentle cleansers only",
            "Apply licorice root extract"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'vitiligo': {
        'Low': [
            "Use broad-spectrum sunscreen on affected areas",
            "Apply aloe vera gel daily",
            "Try turmeric with mustard oil",
            "Use copper-rich foods in diet"
        ],
        'Moderate': [
            "Apply ginkgo biloba extract",
            "Try copper peptide creams",
            "Apply walnut oil to affected areas",
            "Use vitamin B12 and folic acid supplements",
            "Try psoralen from figs with sun exposure",
            "Apply black cumin oil"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'warts': {
        'Low': [
            "Apply duct tape occlusion therapy",
            "Use diluted apple cider vinegar",
            "Apply tea tree oil (diluted)",
            "Try banana peel applications"
        ],
        'Moderate': [
            "Apply salicylic acid treatments (over-the-counter)",
            "Use garlic paste applications",
            "Try castor oil with baking soda",
            "Apply vitamin E oil regularly",
            "Use crushed aspirin paste",
            "Try pineapple juice applications"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    },
    'healthy': {
    'Low': [
        "Your skin looks healthy!Great.",
        "Maintain hydration",
        "Use sunscreen daily",
        "Follow a gentle skincare routine",
        "Eat a balanced diet for glowing skin"
    ]
    },
    'lichen': {
        'Low': [
            "Apply aloe vera gel for soothing",
            "Use oatmeal baths",
            "Apply coconut oil as moisturizer",
            "Try cool compresses"
        ],
        'Moderate': [
            "Apply turmeric paste with honey",
            "Use apple cider vinegar (diluted)",
            "Try chamomile tea compresses",
            "Apply thick moisturizers regularly",
            "Use calendula cream",
            "Apply witch hazel extract"
        ],
        'High': [
            "Consult a dermatologist for proper treatment"
        ]
    }
}

def get_home_remedies(condition, severity):
    """Get home remedies for specific condition and severity"""
    condition = condition.lower()
    if condition in HOME_REMEDIES and severity in HOME_REMEDIES[condition]:
        return HOME_REMEDIES[condition][severity]
    return ["Apply aloe vera gel and use gentle skincare routine"]

def analyze_skin_image(image_data, condition):
    """
    Analyze uploaded skin image - structured output (FINAL VERSION)
    """

    try:
        # ----------------------------
        # LOAD IMAGE
        # ----------------------------
        if isinstance(image_data, str):
            img = cv2.imread(image_data)
        elif isinstance(image_data, np.ndarray):
            img = image_data
        else:
            return {"error": "Invalid image data format"}

        if img is None:
            return {"error": "Unable to process image"}

        condition = condition.lower()

        # ----------------------------
        # ✅ HANDLE HEALTHY
        # ----------------------------
        if condition == "healthy":
            severity = "Low"
            score = 0
            remedies = [
                "Your skin looks healthy! Great job maintaining it.",
                "Maintain hydration",
                "Use sunscreen daily",
                "Follow a gentle skincare routine",
                "Eat a balanced diet for glowing skin"
            ]

            return {
                "condition": "healthy",
                "severity": severity,
                "remedies": remedies
            }

        # ----------------------------
        # VALIDATE CONDITION
        # ----------------------------
        if condition not in analyzers:
            return {"error": f"Condition '{condition}' not supported"}

        # ----------------------------
        # RUN ANALYZER
        # ----------------------------
        severity, raw_score, details = analyzers[condition](img)

        # ----------------------------
        # GET REMEDIES
        # ----------------------------
        remedies = get_home_remedies(condition, severity)

        # ----------------------------
        # FINAL STRUCTURED OUTPUT
        # ----------------------------
        return {
            "condition": condition,
            "severity": severity,
            "remedies": remedies
        }

    except Exception as e:
        return {"error": f"Failed to analyze image - {str(e)}"}

# Legacy function for backward compatibility
def analyze_condition(image_path, condition):
    """Legacy function - use analyze_skin_image for new implementations"""
    return analyze_skin_image(image_path, condition)


def generate_recommendations(severity, condition, details):
    """Generate condition-specific recommendations"""
    recommendations = []
    
    # General recommendations based on severity
    if severity == "Low":
        recommendations.append("Continue current skincare routine and monitor changes")
        recommendations.append("Maintain good hygiene practices")
    elif severity == "Moderate":
        recommendations.append("Consider consulting a dermatologist for proper treatment")
        recommendations.append("Avoid known triggers and irritants")
    else:  # High
        recommendations.append("Seek immediate medical attention from a dermatologist")
        recommendations.append("Follow prescribed treatment regimen strictly")
    
    # Condition-specific recommendations
    condition_specific = {
        'acne': {
            'general': "Use gentle, non-comedogenic skincare products",
            'high': "Avoid picking or squeezing lesions"
        },
        'eczema': {
            'general': "Keep skin moisturized with fragrance-free products",
            'high': "Identify and avoid allergens or irritants"
        },
        'psoriasis': {
            'general': "Use gentle exfoliation and moisturizing",
            'high': "Consider phototherapy or systemic treatments"
        },
        'rosacea': {
            'general': "Avoid sun exposure and use gentle skincare",
            'high': "Identify and avoid personal triggers (food, stress, etc.)"
        },
        'vitiligo': {
            'general': "Use broad-spectrum sunscreen on affected areas",
            'high': "Consider repigmentation therapies"
        },
        'warts': {
            'general': "Avoid touching or picking at warts",
            'high': "Consider professional removal procedures"
        },
        'lichen': {
            'general': "Avoid scratching and use anti-inflammatory treatments",
            'high': "Monitor for changes in lesion appearance"
        }
    }
    
    if condition in condition_specific:
        recommendations.append(condition_specific[condition]['general'])
        if severity == "High":
            recommendations.append(condition_specific[condition]['high'])
    
    return recommendations


def get_supported_conditions():
    """Get list of supported conditions"""
    return list(analyzers.keys())


def batch_analyze(image_folder, condition):
    """Analyze multiple images in a folder"""
    import os
    
    results = {}
    if not os.path.exists(image_folder):
        return {'error': f'Folder not found: {image_folder}'}
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(image_folder) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        result = analyze_condition(image_path, condition)
        results[image_file] = result
    
    return results


# ----------------------------
# App-ready example usage
# ----------------------------
if __name__ == "__main__":
    # Example for app integration
    image_path = "download (1).jpeg"  # This would be the uploaded image path
    condition = "acne"  # This would come from user selection
    
    # Analyze uploaded image
    result = analyze_skin_image(image_path, condition)
    print(result)
    
    