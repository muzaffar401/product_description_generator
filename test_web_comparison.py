import os
import sys
from main import ProductDescriptionGenerator
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def test_web_image_comparison():
    """Test the web image comparison functionality"""
    
    print("🧪 Testing Web Image Comparison System...")
    
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
    
    # Test cases
    test_cases = [
        {
            'sku': 'SHAN__KARAHI_FRY_GOSHAT_MASALA50_G',
            'image_path': 'product_images/shan__karahi_fry_goshat_masala50_g.jpg',
            'description': 'Correct food product image'
        },
        {
            'sku': 'BAISAN_FLOUR',
            'image_path': 'product_images/baisan_flour.jpg',
            'description': 'Another food product'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['description']} ---")
        print(f"SKU: {test_case['sku']}")
        print(f"Image: {test_case['image_path']}")
        
        if os.path.exists(test_case['image_path']):
            print("✅ Image found")
            
            # Read image
            with open(test_case['image_path'], 'rb') as f:
                image_bytes = f.read()
            
            # Get mime type
            img = Image.open(io.BytesIO(image_bytes))
            mime_type = Image.MIME[img.format]
            
            # Test web image scraping
            print("🔍 Testing web image scraping...")
            web_image_info = generator.get_image_from_web(test_case['sku'])
            
            if web_image_info:
                print(f"✅ Found web image: {web_image_info['url']}")
                print(f"   MIME type: {web_image_info['mime_type']}")
                print(f"   Size: {len(web_image_info['data'])} bytes")
                
                # Test image comparison
                print("🔍 Testing image comparison...")
                comparison_result = generator.compare_images_with_ai(
                    web_image_info['data'],
                    image_bytes,
                    web_image_info['mime_type'],
                    mime_type
                )
                
                if comparison_result is not None:
                    if comparison_result:
                        print("✅ Web image comparison: MATCH")
                    else:
                        print("❌ Web image comparison: MISMATCH")
                else:
                    print("⚠️  Web image comparison failed")
            else:
                print("❌ No web image found")
            
            # Test enhanced validation with web comparison
            print("🔍 Testing enhanced validation with web comparison...")
            validation_result = generator.enhanced_image_validation_with_web_comparison(
                test_case['sku'], image_bytes, mime_type
            )
            
            print(f"Validation result: {validation_result}")
            
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
            
        else:
            print(f"❌ Image not found: {test_case['image_path']}")
            print("Available images:")
            if os.path.exists("product_images"):
                for file in os.listdir("product_images"):
                    print(f"  - {file}")

if __name__ == "__main__":
    test_web_image_comparison() 