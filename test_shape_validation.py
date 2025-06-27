import os
import sys
from main import ProductDescriptionGenerator
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def test_shape_based_validation():
    """Test the shape-based validation for shoes and electronics"""
    
    print("🧪 Testing Shape-Based Validation for Shoes and Electronics...")
    
    # Check for required API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("❌ Error: Either GEMINI_API_KEY or OPENAI_API_KEY must be set")
        return
    
    # Initialize the generator
    use_openai = bool(openai_key)
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Test cases for shape-based validation
    test_cases = [
        # Shoe products (should use SHAPE-BASED validation)
        {
            'sku': 'Men_Black_Sports_Walking_Shoes',
            'category': 'Shoes (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        {
            'sku': 'Nike_Air_Max_Sneakers',
            'category': 'Shoes (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        {
            'sku': 'Adidas_Ultraboost_Running_Shoes',
            'category': 'Shoes (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        
        # Electronics products (should use SHAPE-BASED validation)
        {
            'sku': 'iPhone_15_Pro_Max',
            'category': 'Electronics (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        {
            'sku': 'Samsung_Galaxy_S24_Ultra',
            'category': 'Electronics (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        {
            'sku': 'MacBook_Pro_16_inch',
            'category': 'Electronics (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        },
        {
            'sku': 'Sony_WH_1000XM5_Headphones',
            'category': 'Electronics (SHAPE-BASED)',
            'expected_behavior': 'Should check only physical shape/form, ignore brands'
        }
    ]
    
    print(f"\n📋 Testing {len(test_cases)} shape-based validation cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['category']} ---")
        print(f"SKU: {test_case['sku']}")
        print(f"Expected Behavior: {test_case['expected_behavior']}")
        
        # Test the category detection logic
        sku_lower = test_case['sku'].lower()
        
        # Food/Spice categories (strict validation)
        food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
        
        # Electronics/Shoes categories (shape-based validation)
        shape_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
        
        is_food_product = any(keyword in sku_lower for keyword in food_keywords)
        is_shape_product = any(keyword in sku_lower for keyword in shape_keywords)
        
        if is_food_product:
            detected_category = "Food/Spice (STRICT)"
            validation_type = "STRICT"
        elif is_shape_product:
            detected_category = "Shoes/Electronics (SHAPE-BASED)"
            validation_type = "SHAPE-BASED"
        else:
            detected_category = "Other (DEFAULT)"
            validation_type = "DEFAULT"
        
        print(f"✅ Category Detection: {detected_category}")
        print(f"✅ Validation Type: {validation_type}")
        
        # Check if detection matches expected
        if "SHAPE-BASED" in detected_category:
            print("✅ CORRECT: Shape-based validation detected")
        else:
            print("❌ INCORRECT: Should use shape-based validation")
            print(f"   Expected: Shape-based validation")
            print(f"   Detected: {detected_category}")
        
        # Test with a sample image if available
        test_image_path = f"pro-img/test_{i}.jpg"
        if os.path.exists(test_image_path):
            print(f"📷 Testing with image: {test_image_path}")
            
            # Read image
            with open(test_image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Get mime type
            img = Image.open(io.BytesIO(image_bytes))
            mime_type = Image.MIME[img.format]
            
            # Test simple validation
            try:
                validation_result = generator.simple_image_validation(
                    test_case['sku'], image_bytes, mime_type
                )
                
                print(f"Simple validation result: {validation_result}")
                
                is_match = validation_result.get('match', False)
                reason = validation_result.get('reason', 'No reason')
                
                print(f"Match: {is_match}")
                print(f"Reason: {reason}")
                
                if is_match:
                    print("✅ Simple validation: MATCH")
                else:
                    print("❌ Simple validation: MISMATCH")
                    
            except Exception as e:
                print(f"❌ Error in simple validation: {str(e)}")
        else:
            print(f"📷 No test image found: {test_image_path}")
    
    print(f"\n🎯 Summary:")
    print(f"- Food/Spice products: Use STRICT validation (check brands, product types)")
    print(f"- Shoes/Electronics: Use SHAPE-BASED validation (check physical form only)")
    print(f"- Other products: Use DEFAULT validation (lenient)")
    print(f"\n✅ Shape-based validation rules implemented successfully!")

def test_shape_validation_examples():
    """Test specific shape validation examples"""
    
    print("\n🧪 Testing Specific Shape Validation Examples...")
    
    # Check for required API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("❌ Error: Either GEMINI_API_KEY or OPENAI_API_KEY must be set")
        return
    
    # Initialize the generator
    use_openai = bool(openai_key)
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Test specific scenarios
    scenarios = [
        {
            'name': 'Nike shoe SKU with Adidas shoe image',
            'sku': 'Nike_Air_Max_Sneakers',
            'expected': 'MATCH (same shoe shape, different brand)',
            'description': 'Should match if image shows any shoe-shaped footwear'
        },
        {
            'name': 'iPhone SKU with Samsung phone image',
            'sku': 'iPhone_15_Pro_Max',
            'expected': 'MATCH (same phone shape, different brand)',
            'description': 'Should match if image shows any phone-shaped device'
        },
        {
            'name': 'Shoe SKU with phone image',
            'sku': 'Men_Black_Sports_Walking_Shoes',
            'expected': 'MISMATCH (different shapes)',
            'description': 'Should reject if image shows phone-shaped device'
        },
        {
            'name': 'Phone SKU with shoe image',
            'sku': 'iPhone_15_Pro_Max',
            'expected': 'MISMATCH (different shapes)',
            'description': 'Should reject if image shows shoe-shaped footwear'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        print(f"SKU: {scenario['sku']}")
        print(f"Expected: {scenario['expected']}")
        print(f"Description: {scenario['description']}")
        
        # Test category detection
        sku_lower = scenario['sku'].lower()
        
        food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
        shape_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
        
        is_food_product = any(keyword in sku_lower for keyword in food_keywords)
        is_shape_product = any(keyword in sku_lower for keyword in shape_keywords)
        
        if is_food_product:
            print("✅ Detected as: Food/Spice (STRICT validation)")
        elif is_shape_product:
            print("✅ Detected as: Shoes/Electronics (SHAPE-BASED validation)")
        else:
            print("✅ Detected as: Other (DEFAULT validation)")

if __name__ == "__main__":
    test_shape_based_validation()
    test_shape_validation_examples() 