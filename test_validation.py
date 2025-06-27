import os
import sys
from main import ProductDescriptionGenerator
from PIL import Image
import io

def test_validation():
    """Test the validation logic with a sample SKU"""
    
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=False)
    
    # Test SKU
    test_sku = "SHAN__KARAHI_FRY_GOSHAT_MASALA50_G"
    
    print(f"Testing validation for SKU: {test_sku}")
    
    # Check if we have a test image
    test_image_path = "product_images/shan__karahi_fry_goshat_masala50_g.jpg"
    
    if os.path.exists(test_image_path):
        print(f"Found test image: {test_image_path}")
        
        # Read the image
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get mime type
        img = Image.open(io.BytesIO(image_bytes))
        mime_type = Image.MIME[img.format]
        
        print(f"Image format: {mime_type}")
        
        # Test validation
        print("Running validation...")
        validation_result = generator.enhanced_image_validation(test_sku, image_bytes, mime_type)
        
        print(f"Validation result: {validation_result}")
        
        if validation_result.get('match'):
            print("✅ VALIDATION PASSED - Image matches SKU")
        else:
            print("❌ VALIDATION FAILED - Image does not match SKU")
            print(f"Reason: {validation_result.get('reason', 'No reason provided')}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Available images:")
        if os.path.exists("product_images"):
            for file in os.listdir("product_images"):
                print(f"  - {file}")

if __name__ == "__main__":
    test_validation() 