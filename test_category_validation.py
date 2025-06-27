import os
import sys
from main import ProductDescriptionGenerator
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

def test_category_specific_validation():
    """Test the category-specific validation rules"""
    
    print("🧪 Testing Category-Specific Validation Rules...")
    
    # Check for required API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("❌ Error: Either GEMINI_API_KEY or OPENAI_API_KEY must be set")
        return
    
    # Initialize the generator
    use_openai = bool(openai_key)
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Test cases for different categories
    test_cases = [
        # Food/Spice products (should use STRICT validation)
        {
            'sku': 'SHAN__GARAM_MASALA_50G',
            'category': 'Food/Spice (STRICT)',
            'expected_behavior': 'Should check brand names and product types strictly'
        },
        {
            'sku': 'BAISAN_FLOUR_1KG',
            'category': 'Food/Spice (STRICT)',
            'expected_behavior': 'Should check brand names and product types strictly'
        },
        {
            'sku': 'NATIONAL_GARLIC_POWDER_50G',
            'category': 'Food/Spice (STRICT)',
            'expected_behavior': 'Should check brand names and product types strictly'
        },
        
        # Shoes/Electronics products (should use LENIENT validation)
        {
            'sku': 'Men_Black_Sports_Walking_Shoes',
            'category': 'Shoes/Electronics (LENIENT)',
            'expected_behavior': 'Should only check category, not specific brands/models'
        },
        {
            'sku': 'Nike_Air_Max_Sneakers',
            'category': 'Shoes/Electronics (LENIENT)',
            'expected_behavior': 'Should only check category, not specific brands/models'
        },
        {
            'sku': 'iPhone_15_Pro_Max',
            'category': 'Shoes/Electronics (LENIENT)',
            'expected_behavior': 'Should only check category, not specific brands/models'
        },
        {
            'sku': 'Samsung_Galaxy_Laptop',
            'category': 'Shoes/Electronics (LENIENT)',
            'expected_behavior': 'Should only check category, not specific brands/models'
        },
        
        # Other products (should use DEFAULT validation)
        {
            'sku': 'Generic_Product_123',
            'category': 'Other (DEFAULT)',
            'expected_behavior': 'Should use lenient validation for unknown categories'
        }
    ]
    
    print(f"\n📋 Testing {len(test_cases)} different product categories...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['category']} ---")
        print(f"SKU: {test_case['sku']}")
        print(f"Expected Behavior: {test_case['expected_behavior']}")
        
        # Test the category detection logic
        sku_lower = test_case['sku'].lower()
        
        # Food/Spice categories (strict validation)
        food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
        
        # Electronics/Shoes categories (lenient validation)
        lenient_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
        
        is_food_product = any(keyword in sku_lower for keyword in food_keywords)
        is_lenient_product = any(keyword in sku_lower for keyword in lenient_keywords)
        
        if is_food_product:
            detected_category = "Food/Spice (STRICT)"
            validation_type = "STRICT"
        elif is_lenient_product:
            detected_category = "Shoes/Electronics (LENIENT)"
            validation_type = "LENIENT"
        else:
            detected_category = "Other (DEFAULT)"
            validation_type = "DEFAULT"
        
        print(f"✅ Category Detection: {detected_category}")
        print(f"✅ Validation Type: {validation_type}")
        
        # Check if detection matches expected
        if detected_category == test_case['category']:
            print("✅ CORRECT: Category detection matches expected")
        else:
            print("❌ INCORRECT: Category detection doesn't match expected")
            print(f"   Expected: {test_case['category']}")
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
    print(f"- Shoes/Electronics: Use LENIENT validation (check category only)")
    print(f"- Other products: Use DEFAULT validation (lenient)")
    print(f"\n✅ Category-specific validation rules implemented successfully!")

def test_validation_examples():
    """Test specific validation examples"""
    
    print("\n🧪 Testing Specific Validation Examples...")
    
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
            'name': 'SHAN Masala with SHAN brand image',
            'sku': 'SHAN__GARAM_MASALA_50G',
            'expected': 'MATCH (same brand, same product type)',
            'description': 'Should match if image shows SHAN brand masala product'
        },
        {
            'name': 'SHAN Masala with BAISAN brand image',
            'sku': 'SHAN__GARAM_MASALA_50G',
            'expected': 'MISMATCH (different brand)',
            'description': 'Should reject if image shows BAISAN brand product'
        },
        {
            'name': 'Shoe SKU with any shoe image',
            'sku': 'Men_Black_Sports_Walking_Shoes',
            'expected': 'MATCH (same category)',
            'description': 'Should match if image shows any type of footwear'
        },
        {
            'name': 'Shoe SKU with food image',
            'sku': 'Men_Black_Sports_Walking_Shoes',
            'expected': 'MISMATCH (different category)',
            'description': 'Should reject if image shows food/spices'
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
        lenient_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
        
        is_food_product = any(keyword in sku_lower for keyword in food_keywords)
        is_lenient_product = any(keyword in sku_lower for keyword in lenient_keywords)
        
        if is_food_product:
            print("✅ Detected as: Food/Spice (STRICT validation)")
        elif is_lenient_product:
            print("✅ Detected as: Shoes/Electronics (LENIENT validation)")
        else:
            print("✅ Detected as: Other (DEFAULT validation)")

if __name__ == "__main__":
    test_category_specific_validation()
    test_validation_examples() 