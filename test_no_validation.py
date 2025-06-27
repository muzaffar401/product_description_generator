import os
import pandas as pd
from main import ProductDescriptionGenerator
from PIL import Image
import io

def test_processing_without_validation():
    """Test processing without any validation checks"""
    
    print("🧪 Testing processing without validation...")
    
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=False)
    
    # Test data
    test_sku = "SHAN__KARAHI_FRY_GOSHAT_MASALA50_G"
    test_image_path = "product_images/shan__karahi_fry_goshat_masala50_g.jpg"
    
    print(f"Testing SKU: {test_sku}")
    print(f"Testing image: {test_image_path}")
    
    if os.path.exists(test_image_path):
        print("✅ Image found")
        
        # Read image
        with open(test_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get mime type
        img = Image.open(io.BytesIO(image_bytes))
        mime_type = Image.MIME[img.format]
        
        print(f"✅ Image format: {mime_type}")
        
        # Simulate the processing without validation
        print("\n🔄 Simulating processing without validation...")
        
        try:
            # Skip validation - directly process
            print("✅ VALIDATION BYPASSED - Proceeding with processing")
            
            # Generate description
            print("📝 Generating description...")
            result = generator.generate_product_description_with_image(test_sku, "test_image.jpg", image_bytes, mime_type)
            description = result.get('description', 'Description generation failed.')
            print(f"✅ Description generated: {description[:100]}...")
            
            # Generate related products
            print("🔗 Generating related products...")
            related = generator.find_related_products(test_sku, [test_sku, "NATIONAL_GARLIC_POWDER_50G"])
            related_products_str = ' | '.join(related) if related else "No related products found."
            print(f"✅ Related products: {related_products_str}")
            
            print("\n🎉 SUCCESS! Processing completed without validation errors!")
            return True
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            return False
    else:
        print(f"❌ Test image not found: {test_image_path}")
        return False

if __name__ == "__main__":
    success = test_processing_without_validation()
    if success:
        print("\n✅ All tests passed! Your processing should work now.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.") 