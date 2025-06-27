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

    def find_related_products_with_image(self, current_sku, current_image_bytes, current_mime_type, all_products_data, num_related=3):
        """
        Find related products using PURE image analysis only.
        SKU is completely ignored - only visual content matters.
        """
        # Filter out the current product from the list
        other_products = [(sku, img_bytes, mime) for sku, img_bytes, mime in all_products_data if sku != current_sku]
        
        if not other_products:
            return []
        
        # STEP 1: Analyze what the current image shows (pure visual analysis)
        current_image_prompt = """Look at this image and tell me ONLY what type of product/object you see.

CRITICAL: 
- IGNORE ALL TEXT, LABELS, OR PRODUCT NAMES
- Look ONLY at the visual content: shapes, colors, materials, objects
- Do NOT read any text on the product or packaging

What do you see? Choose the most specific category:
- running shoes / sneakers / athletic footwear
- dress shoes / formal footwear
- spice packets / seasoning containers
- electronics / gadgets / devices
- clothing / apparel / garments
- food items / beverages
- cosmetics / beauty products
- kitchen items / utensils
- other (be specific)

Respond with ONLY the category name, nothing else."""

        # Get what the current image shows
        current_category_response = self._make_api_call(current_image_prompt, image_bytes=current_image_bytes, mime_type=current_mime_type)
        
        if not current_category_response or current_category_response == "API_CALL_FAILED":
            return []
        
        current_category = current_category_response.strip().lower()
        print(f"Current image category: {current_category}")
        
        # STEP 2: Analyze each other product image individually to find matches
        matching_products = []
        
        for other_sku, other_img_bytes, other_mime_type in other_products:
            # Analyze each other product image
            other_image_prompt = f"""Look at this image and tell me ONLY what type of product/object you see.

CRITICAL: 
- IGNORE ALL TEXT, LABELS, OR PRODUCT NAMES
- Look ONLY at the visual content: shapes, colors, materials, objects
- Do NOT read any text on the product or packaging

What do you see? Choose the most specific category:
- running shoes / sneakers / athletic footwear
- dress shoes / formal footwear
- spice packets / seasoning containers
- electronics / gadgets / devices
- clothing / apparel / garments
- food items / beverages
- cosmetics / beauty products
- kitchen items / utensils
- other (be specific)

Respond with ONLY the category name, nothing else."""

            other_category_response = self._make_api_call(other_image_prompt, image_bytes=other_img_bytes, mime_type=other_mime_type)
            
            if other_category_response and other_category_response != "API_CALL_FAILED":
                other_category = other_category_response.strip().lower()
                print(f"Product {other_sku} image category: {other_category}")
                
                # Check if categories match (allowing for variations)
                if self._categories_match(current_category, other_category):
                    matching_products.append(other_sku)
                    print(f"✓ Match found: {other_sku} (both are {current_category})")
                else:
                    print(f"✗ No match: {other_sku} ({other_category} vs {current_category})")
        
        # Return top matches
        return matching_products[:num_related]

    def _categories_match(self, category1, category2):
        """
        Check if two categories match, allowing for variations in naming.
        """
        # Normalize categories
        cat1 = category1.lower().strip()
        cat2 = category2.lower().strip()
        
        # Direct match
        if cat1 == cat2:
            return True
        
        # Shoe variations
        shoe_keywords = ['shoe', 'sneaker', 'footwear', 'athletic', 'running', 'dress']
        if any(keyword in cat1 for keyword in shoe_keywords) and any(keyword in cat2 for keyword in shoe_keywords):
            return True
        
        # Spice variations
        spice_keywords = ['spice', 'seasoning', 'masala', 'condiment']
        if any(keyword in cat1 for keyword in spice_keywords) and any(keyword in cat2 for keyword in spice_keywords):
            return True
        
        # Electronics variations
        electronic_keywords = ['electronic', 'gadget', 'device', 'phone', 'laptop']
        if any(keyword in cat1 for keyword in electronic_keywords) and any(keyword in cat2 for keyword in electronic_keywords):
            return True
        
        # Clothing variations
        clothing_keywords = ['clothing', 'apparel', 'garment', 'shirt', 'pants']
        if any(keyword in cat1 for keyword in clothing_keywords) and any(keyword in cat2 for keyword in clothing_keywords):
            return True
        
        return False

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