import os
import sys
from main import ProductDescriptionGenerator
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def test_shoe_validation_fix():
    """Test the web image comparison fix for shoe products"""
    
    print("🧪 Testing Web Image Comparison Fix for Shoes...")
    
    # Check for required API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("❌ Error: Either GEMINI_API_KEY or OPENAI_API_KEY must be set")
        return
    
    if not serpapi_key:
        print("⚠️  Warning: SERPAPI_API_KEY not set. Web image comparison will be disabled.")
        print("   Get your free API key from: https://serpapi.com/")
    
    # Initialize the generator
    use_openai = bool(openai_key)
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Test case for shoes
    test_sku = "Men_Black_Sports_Walking_Shoes"
    
    print(f"\n--- Testing Shoe Validation ---")
    print(f"SKU: {test_sku}")
    
    # Check if we have a test shoe image
    test_image_path = "pro-img/shoe_test.jpg"  # You'll need to add a shoe image here
    
    if os.path.exists(test_image_path):
        print("✅ Test shoe image found")
        
        # Read image
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get mime type
        img = Image.open(io.BytesIO(image_bytes))
        mime_type = Image.MIME[img.format]
        
        print(f"Image format: {mime_type}")
        
        # Test 1: Enhanced validation with web comparison (original method)
        print("\n🔍 Test 1: Enhanced validation with web comparison...")
        try:
            validation_result = generator.enhanced_image_validation_with_web_comparison(
                test_sku, image_bytes, mime_type
            )
            
            print(f"Web comparison result: {validation_result}")
            
            is_match = validation_result.get('match', False)
            confidence = validation_result.get('confidence', 'unknown')
            web_search_used = validation_result.get('web_search_used', False)
            web_comparison_used = validation_result.get('web_comparison_used', False)
            reason = validation_result.get('reason', 'No reason')
            
            print(f"Match: {is_match}")
            print(f"Confidence: {confidence}")
            print(f"Web search used: {web_search_used}")
            print(f"Web comparison used: {web_comparison_used}")
            print(f"Reason: {reason}")
            
            if is_match:
                print("✅ SUCCESS: Web comparison validation passed")
            else:
                print("❌ FAILED: Web comparison validation failed")
                
        except Exception as e:
            print(f"❌ Error in web comparison test: {str(e)}")
        
        # Test 2: Standard enhanced validation (fallback method)
        print("\n🔍 Test 2: Standard enhanced validation...")
        try:
            validation_result = generator.enhanced_image_validation(
                test_sku, image_bytes, mime_type
            )
            
            print(f"Standard validation result: {validation_result}")
            
            is_match = validation_result.get('match', False)
            confidence = validation_result.get('confidence', 'unknown')
            reason = validation_result.get('reason', 'No reason')
            
            print(f"Match: {is_match}")
            print(f"Confidence: {confidence}")
            print(f"Reason: {reason}")
            
            if is_match:
                print("✅ SUCCESS: Standard validation passed")
            else:
                print("❌ FAILED: Standard validation failed")
                
        except Exception as e:
            print(f"❌ Error in standard validation test: {str(e)}")
        
        # Test 3: Simple validation (most lenient)
        print("\n🔍 Test 3: Simple validation...")
        try:
            validation_result = generator.simple_image_validation(
                test_sku, image_bytes, mime_type
            )
            
            print(f"Simple validation result: {validation_result}")
            
            is_match = validation_result.get('match', False)
            reason = validation_result.get('reason', 'No reason')
            
            print(f"Match: {is_match}")
            print(f"Reason: {reason}")
            
            if is_match:
                print("✅ SUCCESS: Simple validation passed")
            else:
                print("❌ FAILED: Simple validation failed")
                
        except Exception as e:
            print(f"❌ Error in simple validation test: {str(e)}")
            
    else:
        print(f"❌ Test shoe image not found: {test_image_path}")
        print("Please add a shoe image to test with.")
        print("Available images in pro-img/ directory:")
        if os.path.exists("pro-img"):
            for file in os.listdir("pro-img"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    print(f"  - {file}")

if __name__ == "__main__":
    test_shoe_validation_fix() 