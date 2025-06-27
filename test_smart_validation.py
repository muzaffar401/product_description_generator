import os
import pandas as pd
from main import ProductDescriptionGenerator
from PIL import Image
import io

def test_smart_validation():
    """Test the smart validation system"""
    
    print("🧪 Testing Smart Validation System...")
    
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=False)
    
    # Test cases
    test_cases = [
        {
            'sku': 'SHAN__KARAHI_FRY_GOSHAT_MASALA50_G',
            'image_path': 'product_images/shan__karahi_fry_goshat_masala50_g.jpg',
            'expected': 'match',
            'description': 'Correct food product image'
        },
        {
            'sku': 'SHAN__KARAHI_FRY_GOSHAT_MASALA50_G',
            'image_path': 'product_images/national__garlic_powder_50g.jpg',
            'expected': 'mismatch',
            'description': 'Different food product (should be mismatch)'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"SKU: {test_case['sku']}")
        print(f"Image: {test_case['image_path']}")
        print(f"Expected: {test_case['expected']}")
        
        if os.path.exists(test_case['image_path']):
            print("✅ Image found")
            
            # Read image
            with open(test_case['image_path'], 'rb') as f:
                image_bytes = f.read()
            
            # Get mime type
            img = Image.open(io.BytesIO(image_bytes))
            mime_type = Image.MIME[img.format]
            
            # Run validation
            print("🔍 Running smart validation...")
            validation_result = generator.enhanced_image_validation(test_case['sku'], image_bytes, mime_type)
            
            print(f"Validation result: {validation_result}")
            
            is_match = validation_result.get('match', False)
            confidence = validation_result.get('confidence', 'unknown')
            reason = validation_result.get('reason', 'No reason')
            
            print(f"Match: {is_match}")
            print(f"Confidence: {confidence}")
            print(f"Reason: {reason}")
            
            # Check if result matches expectation
            if test_case['expected'] == 'match' and is_match:
                print("✅ CORRECT: Expected match, got match")
            elif test_case['expected'] == 'mismatch' and not is_match:
                print("✅ CORRECT: Expected mismatch, got mismatch")
            elif test_case['expected'] == 'match' and not is_match:
                print("❌ INCORRECT: Expected match, got mismatch")
            elif test_case['expected'] == 'mismatch' and is_match:
                print("❌ INCORRECT: Expected mismatch, got match")
            
        else:
            print(f"❌ Image not found: {test_case['image_path']}")
    
    print("\n🎉 Smart validation testing completed!")

if __name__ == "__main__":
    test_smart_validation() 