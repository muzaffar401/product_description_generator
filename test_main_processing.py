import os
import pandas as pd
from main import ProductDescriptionGenerator

def test_main_processing():
    """Test the main processing function with a simple SKU"""
    
    print("Testing main processing function...")
    
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=False)
    
    # Create a simple test dataframe
    test_data = {
        'sku': ['SHAN__KARAHI_FRY_GOSHAT_MASALA50_G'],
        'description': [''],
        'related_products': ['']
    }
    
    df = pd.DataFrame(test_data)
    
    print(f"Test dataframe created with {len(df)} products")
    
    # Test description generation
    print("\nTesting description generation...")
    try:
        description = generator.generate_product_description('SHAN__KARAHI_FRY_GOSHAT_MASALA50_G')
        print(f"✅ Description generated successfully: {description[:100]}...")
    except Exception as e:
        print(f"❌ Description generation failed: {str(e)}")
    
    # Test related products
    print("\nTesting related products generation...")
    try:
        related = generator.find_related_products('SHAN__KARAHI_FRY_GOSHAT_MASALA50_G', ['SHAN__KARAHI_FRY_GOSHAT_MASALA50_G', 'NATIONAL_GARLIC_POWDER_50G'])
        print(f"✅ Related products found: {related}")
    except Exception as e:
        print(f"❌ Related products generation failed: {str(e)}")
    
    print("\n🎉 Main processing test completed!")

if __name__ == "__main__":
    test_main_processing() 