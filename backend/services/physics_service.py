import numpy as np
import cv2
# Note: Ensure the 'ml' folder is accessible in your Python path 
# (uv workspaces usually handle sibling imports if configured, or run from root)
from ml.fusion.sar_physics import process_sar_physics, classify_modality
from ml.verification.physics_verifier import verify_water_claim

def process_image_physics(image_bytes: bytes, query: str):
    """
    Parses the image, determines modality, applies physics, and prepares the VLM input.
    """
    # 1. Decode image bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_array = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    
    # 2. Classify Modality
    modality_info = classify_modality(img_array)
    
    sar_data = None
    optical_data = None
    vlm_input = img_array
    
    if modality_info["modality"] == "SAR":
        # Apply True Physics Calibration
        sar_data = process_sar_physics(img_array)
        vlm_input = sar_data.pseudo_rgb_uint8  # 3-channel physics tensor for Gemma
    else:
        optical_data = img_array
        
    return {
        "modality_info": modality_info,
        "sar_data": sar_data,
        "optical_data": optical_data,
        "vlm_input": vlm_input
    }

def run_verification(query: str, vlm_confidence: float, vlm_bbox: tuple, physics_context: dict):
    """
    Runs deterministic spectral/SAR verification if the query targets specific land covers.
    """
    if "water" in query.lower() or "flood" in query.lower():
        opt_data = physics_context["optical_data"]
        sar_data = physics_context["sar_data"]
        
        # Assuming optical is RGBA/RGB/Multispectral. Adjust band indices as per your actual data setup.
        green_band = opt_data[:, :, 1] if opt_data is not None and opt_data.shape[-1] >= 3 else None
        nir_band = opt_data[:, :, 3] if opt_data is not None and opt_data.shape[-1] >= 4 else None
        
        verification = verify_water_claim(
            model_confidence=vlm_confidence,
            bbox=vlm_bbox,
            green=green_band,
            nir=nir_band,
            sar_vv_db=sar_data.sigma0_vv_db if sar_data else None,
            sar_homogeneity=sar_data.glcm_homogeneity if sar_data else None
        )
        return verification
    
    return None