import os
import time
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import re
from PIL import Image
import io
import mimetypes
import json
import base64
import requests
from urllib.parse import quote_plus
import urllib.parse
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Configure API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise ValueError("Either GEMINI_API_KEY or OPENAI_API_KEY must be set in environment variables")

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI API if key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class ProductDescriptionGenerator:
    def __init__(self, use_openai=False):
        self.use_openai = use_openai
        if self.use_openai:
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not self.client.api_key:
                raise ValueError("OPENAI_API_KEY not found or is invalid.")
        else:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            genai.configure(api_key=self.gemini_api_key)

    def _make_api_call(self, prompt, image_bytes=None, mime_type=None, retries=3, delay=30):
        if self.use_openai:
            for attempt in range(retries):
                try:
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                    if image_bytes:
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                        })
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o", 
                        messages=messages, 
                        max_tokens=400,
                        timeout=60  # Add timeout
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API call failed on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        return "API_CALL_FAILED"
            return "API_CALL_FAILED"
        else:
            model = genai.GenerativeModel('gemini-2.0-flash')
            for attempt in range(retries):
                try:
                    content = [prompt]
                    if image_bytes:
                        image_parts = [{"mime_type": mime_type, "data": image_bytes}]
                        content.append(image_parts[0])
                    
                    # Add timeout and safety settings
                    response = model.generate_content(
                        content,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=400,
                            temperature=0.7
                        ),
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_NONE"
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_NONE"
                            }
                        ]
                    )
                    
                    if response.text:
                        return response.text
                    else:
                        return "API_CALL_FAILED"
                        
                except Exception as e:
                    print(f"Gemini API call failed on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        return "API_CALL_FAILED"
            return "API_CALL_FAILED"

    def search_product_online(self, sku):
        """
        Search for product information online using the SKU
        Returns a list of search results and product information
        """
        try:
            # Clean the SKU for search
            search_query = sku.replace('_', ' ').replace('-', ' ')
            
            # Use a simple web search approach (you can enhance this with proper search APIs)
            search_url = f"https://www.google.com/search?q={quote_plus(search_query + ' product')}"

            print(search_url)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Extract basic information from search results
                # This is a simplified approach - in production, you might want to use proper search APIs
                search_info = {
                    'search_query': search_query,
                    'search_url': search_url,
                    'status': 'success',
                    'found_results': True
                }
                return search_info
            else:
                return {
                    'search_query': search_query,
                    'status': 'failed',
                    'error': f'HTTP {response.status_code}',
                    'found_results': False
                }
                
        except Exception as e:
            return {
                'search_query': sku,
                'status': 'error',
                'error': str(e),
                'found_results': False
            }

    def get_image_from_web(self, sku):
        """
        Scrapes Google Images for a given SKU and returns the first image URL and data.
        Uses SerpAPI for reliable image search.
        """
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print("SERPAPI_API_KEY not found - skipping web image search")
                return None
            
            # Clean the SKU for better search results
            cleaned_sku = self.clean_sku_for_search(sku)
            print(f"Original SKU: {sku}")
            print(f"Cleaned SKU for search: {cleaned_sku}")
            
            params = {
                "q": cleaned_sku,
                "engine": "google_images",
                "ijn": "0",
                "api_key": serpapi_key,
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "images_results" in results and results["images_results"]:
                image_url = results["images_results"][0]["original"]
                
                # Download the image
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                return {
                    'url': image_url,
                    'data': response.content,
                    'mime_type': response.headers.get('content-type', 'image/jpeg')
                }
        except Exception as e:
            print(f"Error during web image scraping for SKU '{sku}': {e}")
            return None

    def clean_sku_for_search(self, sku):
        """
        Clean SKU for better web search results by removing underscores, numbers, and common abbreviations.
        """
        # Convert to lowercase and replace underscores with spaces
        cleaned = sku.lower().replace('_', ' ').replace('__', ' ')
        
        # Clean up size/weight patterns but preserve them
        # Convert "50_g" to "50g", "100_ml" to "100ml", etc.
        cleaned = re.sub(r'(\d+)\s*_\s*g\b', r'\1g', cleaned)  # "50_g" -> "50g"
        cleaned = re.sub(r'(\d+)\s*_\s*ml\b', r'\1ml', cleaned)  # "100_ml" -> "100ml"
        cleaned = re.sub(r'(\d+)\s*_\s*kg\b', r'\1kg', cleaned)  # "1_kg" -> "1kg"
        cleaned = re.sub(r'(\d+)\s*_\s*l\b', r'\1l', cleaned)   # "1_l" -> "1l"
        cleaned = re.sub(r'(\d+)\s*_\s*oz\b', r'\1oz', cleaned)  # "16_oz" -> "16oz"
        
        # Also handle patterns without underscores
        cleaned = re.sub(r'(\d+)\s+g\b', r'\1g', cleaned)  # "50 g" -> "50g"
        cleaned = re.sub(r'(\d+)\s+ml\b', r'\1ml', cleaned)  # "100 ml" -> "100ml"
        cleaned = re.sub(r'(\d+)\s+kg\b', r'\1kg', cleaned)  # "1 kg" -> "1kg"
        cleaned = re.sub(r'(\d+)\s+l\b', r'\1l', cleaned)   # "1 l" -> "1l"
        cleaned = re.sub(r'(\d+)\s+oz\b', r'\1oz', cleaned)  # "16 oz" -> "16oz"
        
        # Remove standalone numbers that are not part of size (like "50" at the end without unit)
        # But keep numbers that are part of product names or sizes
        cleaned = re.sub(r'\b(\d+)\b(?!\s*(?:g|ml|kg|l|oz|gram|milliliter|kilogram|liter|ounce))', '', cleaned)
        
        # Remove common abbreviations but keep size units
        abbreviations = {
            'pcs': 'pieces',
            'pkt': 'packet',
            'pkg': 'package',
            'ct': 'count',
            'pk': 'pack'
        }
        
        for abbr, full in abbreviations.items():
            # Replace standalone abbreviations
            cleaned = re.sub(rf'\b{abbr}\b', full, cleaned)
        
        # Clean up extra spaces and normalize
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove trailing/leading spaces and common words that don't help search
        remove_words = ['product', 'item', 'pack', 'package', 'bottle', 'can', 'jar']
        words = cleaned.split()
        words = [word for word in words if word.lower() not in remove_words]
        
        cleaned = ' '.join(words).strip()
        
        return cleaned

    def compare_images_with_ai(self, sku, web_image_data, uploaded_image_data, web_mime_type, uploaded_mime_type):
        """
        Compares two images using AI (Gemini or OpenAI) to determine if they show the same product.
        """
        try:
            # Determine product category from SKU
            sku_lower = sku.lower()
            
            # Food/Spice categories (strict validation)
            food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
            
            # Electronics/Shoes categories (shape-based validation)
            shape_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
            
            is_food_product = any(keyword in sku_lower for keyword in food_keywords)
            is_shape_product = any(keyword in sku_lower for keyword in shape_keywords)
            
            if is_food_product:
                # STRICT validation for food/spices - check brand, packaging, exact product
                prompt = f"""You are a product image validation expert. Your task is to compare two images for a food/spice product with SKU: '{sku}'.

Image 1 is a reference image found on the web.
Image 2 is an image uploaded by a user.

This is a FOOD/SPICE product that requires STRICT validation.

STRICT VALIDATION RULES for Food/Spices:
1. Check if the BRAND names match (e.g., SHAN, BAISAN, NATIONAL)
2. Check if the PRODUCT TYPE matches (e.g., Garam Masala, Karahi Masala, Nihari Masala)
3. Check if the PACKAGING looks similar (same brand packaging style)
4. Check if the PRODUCT NAME on packaging matches
5. Allow minor differences in packaging design, lighting, angle

Return 'Yes' ONLY if:
- Same brand name is visible on both images
- Same product type/category (e.g., both are masala products)
- Packaging looks like the same brand's style
- Product names are similar or related

Return 'No' if:
- Different brands (e.g., SHAN vs BAISAN)
- Completely different product types (e.g., masala vs flour)
- No brand information visible or brands don't match

Are these two images of the same brand and product type? Answer with only 'Yes' or 'No'."""
                
            elif is_shape_product:
                # SHAPE-BASED validation for shoes/electronics - only check physical shape/form
                prompt = f"""You are a product image validation expert. Your task is to compare two images for a product with SKU: '{sku}'.

Image 1 is a reference image found on the web.
Image 2 is an image uploaded by a user.

This is a SHOE/ELECTRONICS product that requires SHAPE-BASED validation.

SHAPE-BASED VALIDATION RULES for Shoes/Electronics:
1. Focus ONLY on the PHYSICAL SHAPE and FORM of the products
2. For shoes: Check if both images show footwear SHAPE (any type - sneakers, boots, sandals, etc.)
3. For electronics: Check if both images show electronics SHAPE (any type - phones, laptops, cameras, etc.)
4. IGNORE brand names completely
5. IGNORE colors, logos, or specific model details
6. IGNORE packaging or backgrounds
7. ONLY look at the actual product SHAPE

Return 'Yes' if:
- Both images show the same SHAPE category (e.g., both are shoe-shaped, both are phone-shaped)
- The physical form is similar (e.g., both are rectangular phones, both are shoe-shaped footwear)
- The basic structure matches (e.g., both have shoe soles, both have phone screens)

Return 'No' if:
- Different SHAPES (e.g., shoe shape vs phone shape)
- Completely different forms (e.g., round vs rectangular, flat vs 3D)
- One is clearly not the right category shape

Examples:
- Any shoe shape + Any shoe shape → Yes (same footwear form)
- Any phone shape + Any phone shape → Yes (same phone form)
- Any laptop shape + Any laptop shape → Yes (same laptop form)
- Shoe shape + Phone shape → No (different shapes)

Are these two images showing products with the same basic SHAPE and FORM? Answer with only 'Yes' or 'No'."""
                
            else:
                # DEFAULT validation for other products
                prompt = f"""You are a product image validation expert. Your task is to compare two images for a product with SKU: '{sku}'.

Image 1 is a reference image found on the web.
Image 2 is an image uploaded by a user.

Your goal is to determine if both images represent the SAME core product or similar products in the same category.

BE LENIENT in your comparison. Allow for differences in:
- Packaging design (e.g., old vs. new packaging)
- Lighting, angle, or image quality
- Minor variations (e.g., '50g' vs '50 grams')
- Backgrounds or settings
- Brand variations within the same category
- Color variations of the same product
- Different models/styles within the same product type

IMPORTANT: If both images show products in the SAME CATEGORY, return 'Yes'.

Only return 'No' if the products are clearly DIFFERENT CATEGORIES.

Are these two images of the same core product or same category? Answer with only 'Yes' or 'No'."""

            if self.use_openai:
                # Use OpenAI for image comparison
                
                # Prepare images for OpenAI
                web_image_base64 = base64.b64encode(web_image_data).decode('utf-8')
                uploaded_image_base64 = base64.b64encode(uploaded_image_data).decode('utf-8')
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{web_mime_type};base64,{web_image_base64}"}
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:{uploaded_mime_type};base64,{uploaded_image_base64}"}
                            }
                        ]
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=10,
                    timeout=60
                )
                
                result = response.choices[0].message.content.strip().lower()
                return "yes" in result
                
            else:
                # Use Gemini for image comparison
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                
                # Prepare images for Gemini
                web_image = Image.open(io.BytesIO(web_image_data))
                uploaded_image = Image.open(io.BytesIO(uploaded_image_data))
                
                response = model.generate_content([
                    prompt,
                    web_image,
                    uploaded_image,
                ])
                
                result = response.text.strip().lower()
                return "yes" in result
                
        except Exception as e:
            print(f"Error during image comparison: {e}")
            return None

    def enhanced_image_validation_with_web_comparison(self, sku, image_bytes, mime_type):
        """
        Enhanced validation that includes web image scraping and comparison
        This provides the most accurate validation by comparing with real product images
        """
        try:
            print(f"Starting enhanced validation with web comparison for SKU: {sku}")
            
            # Step 1: Try to get a web image for comparison
            web_image_info = self.get_image_from_web(sku)
            
            if web_image_info:
                print(f"Found web image for comparison: {web_image_info['url']}")
                
                # Step 2: Compare the uploaded image with the web image
                comparison_result = self.compare_images_with_ai(
                    sku,
                    web_image_info['data'], 
                    image_bytes, 
                    web_image_info['mime_type'], 
                    mime_type
                )
                
                if comparison_result is not None:
                    if comparison_result:
                        print(f"Web image comparison: MATCH for SKU {sku}")
                        return {
                            'match': True,
                            'sku_type': f'Product matching SKU "{sku}"',
                            'image_type': 'Product image matching web reference',
                            'brand_match': True,
                            'product_category_match': True,
                            'confidence': 'high',
                            'reason': 'Image matches web reference for this product',
                            'web_search_used': True,
                            'web_comparison_used': True
                        }
                    else:
                        print(f"Web image comparison: MISMATCH for SKU {sku}")
                        return {
                            'match': False,
                            'sku_type': f'Product matching SKU "{sku}"',
                            'image_type': 'Product image that does not match web reference',
                            'brand_match': False,
                            'product_category_match': False,
                            'confidence': 'high',
                            'reason': 'Image does not match web reference for this product',
                            'web_search_used': True,
                            'web_comparison_used': True
                        }
                else:
                    print(f"Web image comparison failed, falling back to standard validation for SKU {sku}")
            else:
                print(f"No web image found for SKU {sku}, using standard validation")
            
            # Fallback to standard enhanced validation if web comparison fails
            return self.enhanced_image_validation(sku, image_bytes, mime_type)
            
        except Exception as e:
            print(f"Enhanced validation with web comparison failed: {str(e)}")
            # Fallback to standard enhanced validation
            return self.enhanced_image_validation(sku, image_bytes, mime_type)

    def enhanced_image_validation(self, sku, image_bytes, mime_type):
        """
        Enhanced validation that searches for the product online and compares with user's image
        Uses category-specific validation rules
        """
        try:
            # Step 1: Search for product information online
            search_result = self.search_product_online(sku)

            print(search_result)
            
            # Determine product category from SKU
            sku_lower = sku.lower()
            
            # Food/Spice categories (strict validation)
            food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
            
            # Electronics/Shoes categories (shape-based validation)
            shape_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
            
            is_food_product = any(keyword in sku_lower for keyword in food_keywords)
            is_shape_product = any(keyword in sku_lower for keyword in shape_keywords)
            
            # Step 2: Create category-specific validation prompt with web search context
            if is_food_product:
                enhanced_prompt = f"""
ENHANCED PRODUCT VALIDATION TASK (STRICT for Food/Spices):

You are validating a FOOD/SPICE product listing with SKU: "{sku}"

WEB SEARCH CONTEXT:
- Search Query: {search_result.get('search_query', sku)}
- Search Status: {search_result.get('status', 'unknown')}
- Found Online Results: {search_result.get('found_results', False)}

USER'S IMAGE: [Image will be provided]

STRICT VALIDATION INSTRUCTIONS for Food/Spices:
1. Analyze the user's image carefully
2. Check if the BRAND names match (e.g., SHAN, BAISAN, NATIONAL)
3. Check if the PRODUCT TYPE matches (e.g., Garam Masala, Karahi Masala, Nihari Masala)
4. Check if the PACKAGING looks similar (same brand packaging style)
5. Check if the PRODUCT NAME on packaging matches
6. Use the web search context to understand what this product should look like

STRICT VALIDATION CRITERIA:
- If the image shows the SAME BRAND and SAME PRODUCT TYPE → MATCH (ALLOW)
- If the image shows the SAME BRAND but different variant → MATCH (ALLOW)
- If the image shows similar packaging style and product type → MATCH (ALLOW)
- If the image shows DIFFERENT BRAND → MISMATCH (REJECT)
- If the image shows COMPLETELY DIFFERENT product type → MISMATCH (REJECT)

EXAMPLES OF WHAT TO ALLOW:
- SKU: "SHAN_GARAM_MASALA" + Image: SHAN brand masala product → MATCH
- SKU: "BAISAN_FLOUR" + Image: BAISAN brand flour product → MATCH
- SKU: "NATIONAL_GARLIC_POWDER" + Image: NATIONAL brand spice product → MATCH

EXAMPLES OF WHAT TO REJECT:
- SKU: "SHAN_GARAM_MASALA" + Image: BAISAN brand product → MISMATCH (different brand)
- SKU: "SHAN_GARAM_MASALA" + Image: electronics/shoes → MISMATCH (different category)
- SKU: "BAISAN_FLOUR" + Image: SHAN brand product → MISMATCH (different brand)

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "brand_match": true/false,
  "product_category_match": true/false,
  "confidence": "high/medium/low",
  "reason": "detailed explanation of the validation decision",
  "web_search_used": true/false
}}

Be STRICT for food/spices - check brand names and product types carefully.
"""
            elif is_shape_product:
                enhanced_prompt = f"""
ENHANCED PRODUCT VALIDATION TASK (SHAPE-BASED for Shoes/Electronics):

You are validating a SHOE/ELECTRONICS product listing with SKU: "{sku}"

WEB SEARCH CONTEXT:
- Search Query: {search_result.get('search_query', sku)}
- Search Status: {search_result.get('status', 'unknown')}
- Found Online Results: {search_result.get('found_results', False)}

USER'S IMAGE: [Image will be provided]

SHAPE-BASED VALIDATION INSTRUCTIONS for Shoes/Electronics:
1. Focus ONLY on the PHYSICAL SHAPE and FORM of the products
2. For shoes: Check if the image shows footwear SHAPE (any type - sneakers, boots, sandals, etc.)
3. For electronics: Check if the image shows electronics SHAPE (any type - phones, laptops, cameras, etc.)
4. IGNORE brand names completely
5. IGNORE colors, logos, or specific model details
6. IGNORE packaging or backgrounds
7. ONLY look at the actual product SHAPE and FORM
8. Use the web search context to understand what category this should be

SHAPE-BASED VALIDATION CRITERIA:
- If the image shows the SAME SHAPE category → MATCH (ALLOW)
- If the image shows similar physical form → MATCH (ALLOW)
- If the image shows the right basic structure → MATCH (ALLOW)
- If the image shows COMPLETELY DIFFERENT shape → MISMATCH (REJECT)

EXAMPLES OF WHAT TO ALLOW:
- SKU: "Men_Black_Sports_Walking_Shoes" + Image: any shoe-shaped footwear → MATCH
- SKU: "Nike_Sneakers" + Image: any shoe-shaped footwear → MATCH
- SKU: "iPhone_15" + Image: any phone-shaped device → MATCH
- SKU: "Samsung_Laptop" + Image: any laptop-shaped device → MATCH

EXAMPLES OF WHAT TO REJECT:
- SKU: "Men_Black_Sports_Walking_Shoes" + Image: phone-shaped device → MISMATCH
- SKU: "iPhone_15" + Image: shoe-shaped footwear → MISMATCH

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "brand_match": true/false,
  "product_category_match": true/false,
  "confidence": "high/medium/low",
  "reason": "detailed explanation of the validation decision",
  "web_search_used": true/false
}}

Be SHAPE-BASED for shoes/electronics - only check physical form, ignore brands/models.
"""
            else:
                # Default validation for other products
                enhanced_prompt = f"""
ENHANCED PRODUCT VALIDATION TASK (LENIENT):

You are validating a product listing with SKU: "{sku}"

WEB SEARCH CONTEXT:
- Search Query: {search_result.get('search_query', sku)}
- Search Status: {search_result.get('status', 'unknown')}
- Found Online Results: {search_result.get('found_results', False)}

USER'S IMAGE: [Image will be provided]

LENIENT VALIDATION INSTRUCTIONS:
1. Analyze the user's image carefully
2. Consider the SKU name and what product it should represent
3. Use the web search context to understand what this product should look like
4. Be LENIENT - only reject if the image is COMPLETELY OPPOSITE and DIFFERENT

VALIDATION CRITERIA (LENIENT):
- If the image shows ANYTHING related to the product → MATCH (ALLOW)
- If the image shows the same category of product → MATCH (ALLOW)
- If the image shows similar products → MATCH (ALLOW)
- If the image shows the right brand but different variant → MATCH (ALLOW)
- If the image shows packaging or branding related to the product → MATCH (ALLOW)
- ONLY reject if image shows COMPLETELY DIFFERENT category (e.g., food vs electronics)

EXAMPLES OF WHAT TO ALLOW:
- SKU: "BAISAN" (food) + Image: any food product → MATCH
- SKU: "SHAN_MASALA" (spice) + Image: any spice or food item → MATCH
- SKU: "COCA_COLA" (drink) + Image: any beverage or food → MATCH
- SKU: "NIKE_SHOES" + Image: any footwear → MATCH

EXAMPLES OF WHAT TO REJECT:
- SKU: "BAISAN" (food) + Image: electronics/phones → MISMATCH
- SKU: "SHAN_MASALA" (spice) + Image: clothing/shoes → MISMATCH
- SKU: "COCA_COLA" (drink) + Image: furniture/cars → MISMATCH

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "brand_match": true/false,
  "product_category_match": true/false,
  "confidence": "high/medium/low",
  "reason": "detailed explanation of the validation decision",
  "web_search_used": true/false
}}

Be LENIENT - only reject if completely opposite categories.
"""
            
            # Make the enhanced validation call
            validation_response = self._make_api_call(enhanced_prompt, image_bytes=image_bytes, mime_type=mime_type)
            
            try:
                clean_response = validation_response.strip().lstrip('```json').rstrip('```').strip()
                validation_data = json.loads(clean_response)
                return validation_data
            except json.JSONDecodeError:
                # Fallback to simple validation if JSON parsing fails
                return self.simple_image_validation(sku, image_bytes, mime_type)
                
        except Exception as e:
            print(f"Enhanced validation failed: {str(e)}")
            # Fallback to simple validation
            return self.simple_image_validation(sku, image_bytes, mime_type)

    def simple_image_validation(self, sku, image_bytes, mime_type):
        """
        Simple validation as fallback when enhanced validation fails
        Uses category-specific validation rules
        """
        readable_sku = sku.replace('_', ' ').replace('__', ' ')
        
        # Determine product category from SKU
        sku_lower = sku.lower()
        
        # Food/Spice categories (strict validation)
        food_keywords = ['baisan', 'shan', 'masala', 'achar', 'spice', 'food', 'rice', 'wheat', 'flour', 'sugar', 'salt', 'oil', 'ghee', 'milk', 'bread', 'cake', 'cookie', 'chocolate', 'tea', 'coffee', 'juice', 'soda', 'water', 'yogurt', 'cheese', 'meat', 'fish', 'vegetable', 'fruit', 'grain', 'pulse', 'dal', 'lentil', 'bean', 'nut', 'seed', 'powder', 'garam', 'korma', 'nihari', 'karahi', 'seekh', 'kabab', 'chat', 'zafrani']
        
        # Electronics/Shoes categories (shape-based validation)
        shape_keywords = ['shoe', 'shoes', 'footwear', 'boot', 'sandal', 'sneaker', 'phone', 'mobile', 'electronics', 'computer', 'laptop', 'tv', 'television', 'camera', 'watch', 'clock', 'headphone', 'speaker', 'charger', 'battery']
        
        is_food_product = any(keyword in sku_lower for keyword in food_keywords)
        is_shape_product = any(keyword in sku_lower for keyword in shape_keywords)
        
        if is_food_product:
            simple_prompt = f"""
SIMPLE PRODUCT VALIDATION TASK (STRICT for Food/Spices): You are validating a FOOD/SPICE product listing. The SKU "{sku}" suggests a product named "{readable_sku}".

STRICT VALIDATION RULES for Food/Spices:
1. Check if the BRAND names match (e.g., SHAN, BAISAN, NATIONAL)
2. Check if the PRODUCT TYPE matches (e.g., Garam Masala, Karahi Masala, Nihari Masala)
3. Check if the PACKAGING looks similar (same brand packaging style)
4. Check if the PRODUCT NAME on packaging matches

Return 'Yes' ONLY if:
- Same brand name is visible on both images
- Same product type/category (e.g., both are masala products)
- Packaging looks like the same brand's style

Return 'No' if:
- Different brands (e.g., SHAN vs BAISAN)
- Completely different product types (e.g., masala vs flour)

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "reason": "why they match or don't match"
}}

Be STRICT for food/spices - check brand names and product types carefully.
"""
        elif is_shape_product:
            simple_prompt = f"""
SIMPLE PRODUCT VALIDATION TASK (SHAPE-BASED for Shoes/Electronics): You are validating a SHOE/ELECTRONICS product listing. The SKU "{sku}" suggests a product named "{readable_sku}".

SHAPE-BASED VALIDATION RULES for Shoes/Electronics:
1. Focus ONLY on the PHYSICAL SHAPE and FORM of the products
2. For shoes: Check if the image shows footwear SHAPE (any type - sneakers, boots, sandals, etc.)
3. For electronics: Check if the image shows electronics SHAPE (any type - phones, laptops, cameras, etc.)
4. IGNORE brand names completely
5. IGNORE colors, logos, or specific model details
6. IGNORE packaging or backgrounds
7. ONLY look at the actual product SHAPE and FORM

Return 'Yes' if:
- Both images show the same SHAPE category (e.g., both are shoe-shaped, both are phone-shaped)
- The physical form is similar (e.g., both are rectangular phones, both are shoe-shaped footwear)
- The basic structure matches (e.g., both have shoe soles, both have phone screens)

Return 'No' if:
- Different SHAPES (e.g., shoe shape vs phone shape)
- Completely different forms (e.g., round vs rectangular, flat vs 3D)
- One is clearly not the right category shape

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "reason": "why they match or don't match"
}}

Be SHAPE-BASED for shoes/electronics - only check physical form, ignore brands/models.
"""
        else:
            # Default validation for other products
            simple_prompt = f"""
LENIENT PRODUCT VALIDATION TASK: You are validating product listings. The SKU "{sku}" suggests a product named "{readable_sku}".

You MUST analyze the image and determine if it matches the SKU.

LENIENT RULES:
1. Be GENEROUS - only reject if the image shows COMPLETELY DIFFERENT category
2. If the image shows ANYTHING related to the product → MATCH (ALLOW)
3. If the image shows similar products → MATCH (ALLOW)
4. If the image shows the same category → MATCH (ALLOW)
5. ONLY reject if image shows COMPLETELY OPPOSITE category

EXAMPLES OF WHAT TO ALLOW:
- SKU: "BAISAN" (food) + Image: any food product → MATCH
- SKU: "SHAN_MASALA" (spice) + Image: any spice or food → MATCH
- SKU: "COCA_COLA" (drink) + Image: any beverage → MATCH

EXAMPLES OF WHAT TO REJECT:
- SKU: "BAISAN" (food) + Image: electronics/phones → MISMATCH
- SKU: "SHAN_MASALA" (spice) + Image: clothing/shoes → MISMATCH
- SKU: "COCA_COLA" (drink) + Image: furniture/cars → MISMATCH

Return ONLY this JSON format:
{{
  "match": true/false,
  "sku_type": "what the SKU suggests",
  "image_type": "what the image shows",
  "reason": "why they match or don't match"
}}

Be LENIENT - only reject if completely opposite categories.
"""
        
        validation_response = self._make_api_call(simple_prompt, image_bytes=image_bytes, mime_type=mime_type)
        
        try:
            clean_response = validation_response.strip().lstrip('```json').rstrip('```').strip()
            validation_data = json.loads(clean_response)
            validation_data['web_search_used'] = False
            validation_data['confidence'] = 'medium'
            return validation_data
        except json.JSONDecodeError:
            return {
                'match': True,  # Default to match if validation fails
                'sku_type': 'Unknown',
                'image_type': 'Unknown',
                'reason': 'Validation parsing failed, defaulting to match',
                'web_search_used': False,
                'confidence': 'low'
            }

    def generate_product_description(self, sku):
        prompt = f"""Generate a marketing-friendly product description for a product with SKU: {sku}.

REQUIREMENTS:
- The description must be a single, well-structured paragraph (no bullet points, no numbered lists, no line breaks).
- The beginning and ending of the description must be unique for each product.
- Do NOT mention any country name.
- Do NOT use any special characters (except standard punctuation), extra spaces, or extra lines.
- The description should be between 80 and 120 words and highlight key features and benefits.
- The writing style should be engaging and natural, not repetitive.
- Do not copy the same sentence structure for different products.
- Do not use generic phrases like 'Introducing' or 'Experience the authentic'.
- Do not use any markdown or formatting.
"""
        return self._make_api_call(prompt)

    def find_related_products(self, current_sku_or_title, all_skus, num_related=3):
        skus_to_search = [s for s in all_skus if s and s != current_sku_or_title]
        if not skus_to_search:
            return []
        
        prompt = f"""You are a product recommendation engine. Based on the target product, find the {num_related} most similar products from the provided list of SKUs.

Target Product: "{current_sku_or_title}"

List of available SKUs:
{', '.join(skus_to_search)}

Return ONLY the SKUs of the most related products, separated by a pipe '|'. Do not include the target product in the result. If no products are related, return an empty string.
"""
        response = self._make_api_call(prompt)
        if response and response != "API_CALL_FAILED":
            potential_skus = [sku.strip() for sku in response.split('|')]
            return [sku for sku in potential_skus if sku in skus_to_search]
        return []

    def generate_product_description_with_image(self, sku, image_name, image_bytes, mime_type):
        prompt = """You are an expert product marketer. Analyze this product image to generate a product title and a compelling description.

Instructions:
1.  Product Title: Create a concise, SEO-friendly, and accurate title for the product in the image. If the image is unclear or you cannot confidently identify the product, return 'Unknown Product'.
2.  Product Description: Write a marketing-friendly description in a single, well-structured paragraph (no bullet points, no numbered lists, no line breaks). The beginning and ending of the description must be unique for each product. Do NOT mention any country name. Do NOT use any special characters (except standard punctuation), extra spaces, or extra lines. The description should be between 80 and 120 words and highlight key features and benefits. The writing style should be engaging and natural, not repetitive. Do not copy the same sentence structure for different products. Do not use generic phrases like 'Introducing' or 'Experience the authentic'. Do not use any markdown or formatting. If the title is 'Unknown Product', the description should be 'Could not generate description from image.'.

Return the result as a single raw JSON object with two keys: 'title' and 'description'. Do not wrap it in markdown or any other text.
Example for a clear image: {"title": "Shan Achar Ghost Masala 50g", "description": "A delicious spice mix..."}
Example for an unclear image: {"title": "Unknown Product", "description": "Could not generate description from image."}
"""
        if sku:
            prompt += f"\n\nUse the following SKU for context: '{sku}'."
        if image_name:
            prompt += f" The original image file name is '{image_name}'."
        response_text = self._make_api_call(prompt, image_bytes=image_bytes, mime_type=mime_type)
        try:
            clean_response = response_text.strip().lstrip('```json').rstrip('```').strip()
            data = json.loads(clean_response)
            return data
        except (json.JSONDecodeError, AttributeError, TypeError):
            if response_text == "API_CALL_FAILED":
                 return {"title": "API_CALL_FAILED", "description": "API_CALL_FAILED"}
            return {"title": "", "description": response_text}

def process_products(use_openai: bool = False):
    # Initialize the generator
    generator = ProductDescriptionGenerator(use_openai=use_openai)
    
    # Check if enriched_products.csv exists
    if os.path.exists('enriched_products.csv'):
        print("Found existing enriched_products.csv, continuing from last processed product...")
        df = pd.read_csv('enriched_products.csv')
    else:
        # Read the input Excel file
        try:
            df = pd.read_excel('sample_products.xlsx')
        except FileNotFoundError:
            print("sample_products.xlsx not found, trying sample_products.xls...")
            try:
                df = pd.read_excel('sample_products.xls')
            except FileNotFoundError:
                print("No Excel file found. Please ensure sample_products.xlsx or sample_products.xls exists in the project directory. Exiting gracefully.")
                return  # Exit gracefully instead of raising an error
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['sku'])
        # Create new columns
        df['description'] = ''
        df['related_products'] = ''
    
    # Get list of all products for related products search
    all_products = df['sku'].tolist()
    
    # Find products that need processing
    unprocessed_products = df[
        (df['description'].isna()) | 
        (df['description'] == '') | 
        (df['description'] == 'Description generation failed.') |
        (df['related_products'].isna()) | 
        (df['related_products'] == '')
    ]
    
    if len(unprocessed_products) == 0:
        print("All products have been processed!")
        return
    
    print(f"\nFound {len(unprocessed_products)} products that need processing")
    
    # Process each unprocessed product
    for idx, row in unprocessed_products.iterrows():
        print(f"\nProcessing product {idx + 1} of {len(df)}: {row['sku']}")
        
        # Generate description if needed
        if pd.isna(row['description']) or row['description'] == '' or row['description'] == 'Description generation failed.':
            description = generator.generate_product_description(row['sku'])
            df.at[idx, 'description'] = description
            # Add delay between description and related products
            time.sleep(30)
        
        # Find related products if needed
        if pd.isna(row['related_products']) or row['related_products'] == '':
            related = generator.find_related_products(row['sku'], all_products)
            df.at[idx, 'related_products'] = '|'.join(related)
            # Add delay between products
            time.sleep(30)
        
        # Save progress after each product
        df.to_csv('enriched_products.csv', index=False)
        print(f"Progress saved for product {idx + 1}")
    
    print(f"\nResults saved to enriched_products.csv")

if __name__ == "__main__":
    # Check which API key is available and use that
    use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
    process_products(use_openai=use_openai)