import streamlit as st
import pandas as pd
import time
from main import ProductDescriptionGenerator
import os
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as genai
import datetime
import json
import threading
import queue
import zipfile

# Configure page - must be the first Streamlit command
st.set_page_config(
    page_title="Product Description Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Status file path
STATUS_FILE = 'processing_status.json'
PROGRESS_FILE = 'processing_progress.csv'

def save_status(status_data):
    """Save processing status to file with atomic write"""
    temp_file = f"{STATUS_FILE}.tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(status_data, f)
        os.replace(temp_file, STATUS_FILE)
    except Exception as e:
        print(f"Error saving status: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_status():
    """Load processing status from file with error handling"""
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading status: {str(e)}")
    return None

def save_progress(df, output_file):
    """Save progress to CSV with atomic write"""
    temp_file = f"{output_file}.tmp"
    try:
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, output_file)
    except Exception as e:
        print(f"Error saving progress: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_progress(output_file):
    """Load progress from CSV with error handling"""
    try:
        if os.path.exists(output_file):
            return pd.read_csv(output_file)
    except Exception as e:
        print(f"Error loading progress: {str(e)}")
    return None

def process_products_in_background(generator, df, image_name_mapping, output_file, status_queue):
    """
    A single, robust background processing function for all scenarios.
    Handles SKU-only, image-only, and SKU+image cases.
    """
    try:
        total_products = len(df)
        all_skus = []
        if 'sku' in df.columns:
            all_skus = df['sku'].dropna().tolist()

        # Ensure description and related_products columns exist
        if 'description' not in df.columns:
            df['description'] = ''
        if 'related_products' not in df.columns:
            df['related_products'] = ''

        for processed_count, (i, row) in enumerate(df.iterrows(), 1):
            sku = None
            image_name = None
            
            try:
                # Get SKU and image name if they exist
                if 'sku' in df.columns and pd.notna(row.get('sku')) and str(row.get('sku')).strip():
                    sku = str(row['sku'])
                if 'image_name' in df.columns and pd.notna(row.get('image_name')) and str(row.get('image_name')).strip():
                    image_name = str(row['image_name'])
                
                image_file = image_name_mapping.get(image_name) if image_name and image_name_mapping else None
                current_item_identifier = sku or image_name or f"row {i+1}"

                # Update status
                status = {
                    'current': processed_count, 'total': total_products,
                    'current_sku': current_item_identifier, 'status': 'processing', 'error': None
                }
                status_queue.put(status)
                save_status(status)

                description = ""
                related_products_str = ""
                
                if sku and image_file:
                    image_bytes = image_file.read()
                    image_file.seek(0)
                    img = Image.open(io.BytesIO(image_bytes)); mime_type = Image.MIME[img.format]
                    
                    readable_sku = sku.replace('_', ' ').replace('__', ' ')
                    
                    validation_prompt = f"""
You are a highly analytical AI system for verifying product listings. Your task is to determine if a product image matches its SKU by following a strict, logical process and returning a JSON object.

**Input:**
1.  **SKU:** `{sku}` (which is for a product named "{readable_sku}")
2.  **IMAGE:** [An image will be provided]

**Instructions:**
1.  **Analyze ONLY the SKU:** What is the general category of the product based *only* on the SKU text?
2.  **Analyze ONLY the Image:** What is the general category of the product shown *only* in the image?
3.  **Compare and Decide:** Based on the two categories you just identified, do they represent the same type of product? A mismatch occurs if the categories are fundamentally different (e.g., 'Food' vs. 'Footwear').

**Output Format:**
You MUST return a single, raw JSON object with the following three keys:
- `sku_category`: Your conclusion from Instruction 1.
- `image_category`: Your conclusion from Instruction 2.
- `decision`: Your final verdict, which must be either the single word `MATCH` or `MISMATCH`.

**Example 1 (Mismatch):**
Input:
- SKU: `BAISAN_HALF_1_2KG`
- Image: [Image of shoes]
Expected JSON Output:
{{
  "sku_category": "Food/Groceries",
  "image_category": "Footwear/Shoes",
  "decision": "MISMATCH"
}}

**Example 2 (Match):**
Input:
- SKU: `SHAN_MASALA_50G`
- Image: [Image of Shan Masala spice mix]
Expected JSON Output:
{{
  "sku_category": "Food/Groceries",
  "image_category": "Food/Groceries",
  "decision": "MATCH"
}}

Now, perform the analysis for the provided SKU and image.
"""
                    validation_response_text = generator._make_api_call(validation_prompt, image_bytes=image_bytes, mime_type=mime_type)
                    
                    try:
                        clean_response = validation_response_text.strip().lstrip('```json').rstrip('```').strip()
                        validation_data = json.loads(clean_response)
                        decision = validation_data.get("decision", "MISMATCH").upper()

                        if decision != "MATCH":
                            # Provide a detailed error message for debugging
                            sku_cat = validation_data.get('sku_category', 'Unknown')
                            img_cat = validation_data.get('image_category', 'Unknown')
                            error_message = f"Image-SKU Mismatch for '{sku}'. AI decided categories do not match. SKU Category: '{sku_cat}', Image Category: '{img_cat}'."
                            raise ValueError(error_message)

                    except (json.JSONDecodeError, ValueError) as e:
                        # Reraise the ValueError with the detailed message, or create a new one for JSON errors
                        if isinstance(e, ValueError):
                            raise e
                        else:
                            error_message = f"Failed to validate image for '{sku}'. AI returned an invalid response: '{validation_response_text[:200]}...'"
                            raise ValueError(error_message)

                    result = generator.generate_product_description_with_image(sku, image_name, image_bytes, mime_type)
                    description = result.get('description', 'Description generation failed.')
                    related = generator.find_related_products(sku, all_skus)
                    related_products_str = ' | '.join(related) if related else "No related products found."

                elif sku and not image_file:
                    description = generator.generate_product_description(sku)
                    related = generator.find_related_products(sku, all_skus)
                    related_products_str = ' | '.join(related) if related else "No related products found."

                elif image_file and not sku:
                    image_bytes = image_file.read()
                    image_file.seek(0)
                    img = Image.open(io.BytesIO(image_bytes)); mime_type = Image.MIME[img.format]
                    
                    result = generator.generate_product_description_with_image("", image_name, image_bytes, mime_type)
                    description = result.get('description', 'Description generation failed.')
                    
                    identified_title = result.get('title')
                    if identified_title and identified_title.lower() not in ["unknown product", "api_call_failed"]:
                        related = generator.find_related_products(identified_title, all_skus)
                        related_products_str = ' | '.join(related) if related else "No related products found."
                    else:
                        related_products_str = 'Could not identify product from image to find related.'
                
                else:
                    description = 'No SKU or image provided for this row.'
                    related_products_str = 'Not applicable.'

                df.at[i, 'description'] = description
                df.at[i, 'related_products'] = related_products_str
                
                save_progress(df, output_file)
                time.sleep(30)

            except Exception as e:
                error_message = str(e)
                status = {'current': processed_count, 'total': total_products, 'current_sku': current_item_identifier, 'status': 'error', 'error': error_message}
                status_queue.put(status)
                save_status(status)
                if "Image-SKU Mismatch" in error_message:
                    return
                continue

        final_status = {'current': total_products, 'total': total_products, 'status': 'complete', 'error': None}
        status_queue.put(final_status)
        save_status(final_status)

    except Exception as e:
        error_status = {'status': 'error', 'error': str(e)}
        status_queue.put(error_status)
        save_status(error_status)

def reset_all_data():
    """Reset all data files and clear history"""
    files_to_remove = [
        'enriched_products.csv',
        'enriched_products_with_images.csv',
        'processing_status.json',
        'processing_progress.csv',
        'enriched_products.csv.tmp',
        'enriched_products_with_images.csv.tmp',
        'processing_status.json.tmp'
    ]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing {file_path}: {str(e)}")
    
    # Clear session state
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'scenario' in st.session_state:
        del st.session_state['scenario']
    if 'uploaded_images' in st.session_state:
        del st.session_state['uploaded_images']

# Simple, modern, theme-adaptive CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&display=swap');
    html, body, .stApp {
        font-family: 'Montserrat', sans-serif;
    }
    .simple-card {
        background: var(--background-color);
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(60,60,60,0.07);
        padding: 28px 24px 24px 24px;
        margin: 18px 0;
        border: 1px solid var(--secondary-background-color);
    }
    .simple-title {
        font-size: 2.2em;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.2em;
        margin-top: 0.2em;
        text-align: center;
        letter-spacing: 1px;
    }
    .simple-subtitle {
        color: var(--text-color);
        font-size: 1.1em;
        text-align: center;
        margin-bottom: 2em;
    }
    .simple-btn button {
        background: var(--primary-color);
        color: var(--background-color);
        border: none;
        border-radius: 8px;
        font-size: 1.05em;
        font-weight: 600;
        padding: 10px 32px;
        margin-top: 16px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.10);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .simple-btn button:hover {
        box-shadow: 0 4px 16px rgba(60,60,60,0.13);
        transform: translateY(-2px);
    }
    .simple-progress .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        border-radius: 8px;
        height: 14px !important;
    }
    .simple-upload .stFileUploader {
        background: var(--secondary-background-color);
        border: 1.5px solid var(--primary-color);
        border-radius: 10px;
        color: var(--text-color);
        margin-bottom: 16px;
    }
    .simple-info {
        background: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 14px 20px;
        margin: 14px 0;
        color: var(--text-color);
        font-size: 1em;
    }
    .simple-error {
        background: var(--secondary-background-color);
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 14px 20px;
        margin: 14px 0;
        color: #e74c3c;
        font-size: 1em;
    }
    .stSelectbox, .stTextInput, .stDownloadButton button {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
        border: 1.5px solid var(--primary-color) !important;
    }
    .stSelectbox > div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    .stSelectbox input {
        pointer-events: none !important;
        user-select: none !important;
        background: transparent !important;
        color: var(--text-color) !important;
        caret-color: transparent !important;
    }
    /* Target the dropdown arrow button for pointer cursor */
    .stSelectbox [data-baseweb="select"] button,
    .stSelectbox [aria-label="Open"] {
        cursor: pointer !important;
    }
    /* Make the entire selectbox area a pointer on hover */
    .stSelectbox:hover, .stSelectbox:active, .stSelectbox:focus {
        cursor: pointer !important;
    }
    /* Target the dropdown indicator button and SVG */
    .stSelectbox [data-baseweb="select"] button,
    .stSelectbox [aria-label="Open"],
    .stSelectbox svg,
    .stSelectbox [data-testid="stSelectbox"] svg {
        cursor: pointer !important;
    }
    /* Target the BaseWeb dropdown indicator container */
    .stSelectbox [data-baseweb="select"] > div[role="button"] {
        cursor: pointer !important;
    }
    .stats-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 24px;
        margin: 24px 0 32px 0;
        justify-content: flex-start;
    }
    .stat-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 18px 28px;
        min-width: 220px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.07);
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .stat-icon {
        font-size: 1.6em;
        opacity: 0.85;
    }
    .stat-label {
        color: var(--text-color);
        font-size: 1em;
        margin-bottom: 2px;
    }
    .stat-value {
        font-size: 1.3em;
        font-weight: 700;
        color: var(--primary-color);
    }
    .styled-download {
        margin: 32px 0 0 0;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .styled-download .stDownloadButton button {
        background: var(--primary-color);
        color: var(--background-color);
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: 600;
        padding: 12px 36px;
        margin-top: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(60,60,60,0.10);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .styled-download .stDownloadButton button:hover {
        box-shadow: 0 4px 16px rgba(60,60,60,0.13);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

def process_dataframe(df):
    """Process dataframe to remove duplicates and prepare for analysis"""
    original_count = len(df)
    df = df.drop_duplicates(subset=['sku'], keep='first')
    cleaned_count = len(df)
    return df, original_count, cleaned_count

def poll_status_and_update_ui(processing_file, download_label):
    status_data = load_status()
    if status_data:
        current = status_data.get('current', 0)
        total = status_data.get('total', 0)
        status = status_data.get('status', '')
        error = status_data.get('error', None)
        current_sku = status_data.get('current_sku', '')
        if total > 0:
            progress = max(0, min(100, int((current / total) * 100)))
            st.progress(progress)
        if status == 'complete':
            st.markdown("<div class='simple-info'>Processing completed!</div>", unsafe_allow_html=True)
            if os.path.exists(processing_file):
                with open(processing_file, 'rb') as f:
                    st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                    st.download_button(
                        label=download_label,
                        data=f,
                        file_name=os.path.basename(processing_file),
                        mime="text/csv",
                        key="download_final"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        elif status == 'error':
            st.markdown(f"<div class='simple-error'>Error processing product {current_sku}: {error}</div>", unsafe_allow_html=True)
        elif status in ['starting', 'processing']:
            st.markdown(f"<span style='color:var(--primary-color);'>Processing product <b>{current}</b> of <b>{total}</b>: <b>{current_sku}</b></span>", unsafe_allow_html=True)
            time.sleep(2)
            st.experimental_rerun()

def main():
    st.markdown("""
        <div class='simple-title'>📝 Product Description Generator</div>
        <div class='simple-subtitle'>Transform your product data into compelling descriptions using AI</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🔄 Reset All Data", type="secondary", help="Clear all processed files and start fresh"):
            reset_all_data()
            st.success("✅ All data has been reset! Please refresh the page.")
            st.rerun()

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("""
            <div class='simple-card'>
                <h2 style='color: var(--primary-color); font-weight: 700;'>📋 Input Options</h2>
            </div>
        """, unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Choose AI Model",
            ("Gemini", "OpenAI"),
            index=0,
            key="model_select",
            help="Select the AI model. For OpenAI, ensure a valid API key is in your .env file."
        )
        scenario_options = [
            "Select your scenario",
            "Only Product SKUs",
            "Product SKUs with Image Names"
        ]
        scenario = st.selectbox(
            "Choose Input Type",
            scenario_options,
            index=0,
            key="scenario_select",
            help="Select how you want to provide your product information"
        )

    with col2:
        if scenario == "Select your scenario":
            st.markdown("""
                <div class='simple-info'>
                    <b>👉 Please select a scenario to begin</b><br>
                    Choose how you want to provide your product information to get started.
                </div>
            """, unsafe_allow_html=True)
            return
        if scenario == "Only Product SKUs":
            st.markdown("<div class='simple-card'><h3 style='color:var(--primary-color);'>📤 Upload Product Data</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload your product SKUs file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload a file containing your product SKUs",
                key="sku_file",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state['df'] = df
                    st.session_state['scenario'] = 'sku_only'
                    st.success("✅ File uploaded successfully!")
                except Exception as e:
                    st.markdown(f"<div class='simple-error'>❌ Error reading file: {str(e)}</div>", unsafe_allow_html=True)
                    return
        else:
            st.markdown("<div class='simple-card'><h3 style='color:var(--primary-color);'>📤 Upload Product Data & Images</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload file with SKUs and Image Names",
                type=['xlsx', 'xls', 'csv'],
                help="Upload a file containing SKUs and corresponding image names",
                key="sku_img_file",
                label_visibility="collapsed"
            )
            uploaded_images = st.file_uploader(
                "Upload Product Images",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                accept_multiple_files=True,
                help="Upload all product images (no ZIP support)",
                key="img_files",
                label_visibility="collapsed"
            )
            st.session_state['uploaded_images'] = uploaded_images
            st.markdown("</div>", unsafe_allow_html=True)
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state['df'] = df
                    st.session_state['scenario'] = 'sku_image'
                    st.session_state['uploaded_images'] = uploaded_images
                    st.success("✅ File uploaded successfully!")
                except Exception as e:
                    st.markdown(f"<div class='simple-error'>❌ Error reading file: {str(e)}</div>", unsafe_allow_html=True)
                    return

    # Data Processing Section
    if 'df' in st.session_state and 'scenario' in st.session_state:
        df = st.session_state['df']
        scenario = st.session_state['scenario']
        st.markdown("""
            <div class='simple-card' style='margin-top: 32px;'>
                <h3 style='color:var(--primary-color);'>📊 Data Processing</h3>
            </div>
        """, unsafe_allow_html=True)
        if scenario == 'sku_only':
            if 'sku' not in df.columns:
                st.markdown("<div class='simple-error'>❌ The file must contain a 'sku' column!</div>", unsafe_allow_html=True)
                return
            cleaned_df, original_count, cleaned_count = process_dataframe(df)
            st.markdown(f"""
                <div class='stats-bar'>
                    <div class='stat-card'>
                        <span class='stat-icon'>📦</span>
                        <div>
                            <div class='stat-label'>Original number of products</div>
                            <div class='stat-value'>{original_count}</div>
                        </div>
                    </div>
                    <div class='stat-card'>
                        <span class='stat-icon'>✅</span>
                        <div>
                            <div class='stat-label'>After removing duplicates</div>
                            <div class='stat-value'>{cleaned_count}</div>
                        </div>
                    </div>
                    <div class='stat-card'>
                        <span class='stat-icon'>🗑️</span>
                        <div>
                            <div class='stat-label'>Duplicates removed</div>
                            <div class='stat-value'>{original_count - cleaned_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            processing_file = 'enriched_products.csv'
            poll_status_and_update_ui(processing_file, "⬇️ Download Results")
        elif scenario == 'sku_image':
            if 'sku' not in df.columns or 'image_name' not in df.columns:
                st.markdown("<div class='simple-error'>❌ The file must contain both 'sku' and 'image_name' columns!</div>", unsafe_allow_html=True)
                return
            uploaded_images = st.session_state.get('uploaded_images', None)
            if not uploaded_images or len(uploaded_images) == 0:
                st.markdown("<div class='simple-info'>Please upload all product images before starting processing.</div>", unsafe_allow_html=True)
                return
            image_name_mapping = {}
            for img in uploaded_images:
                base_name = os.path.splitext(img.name)[0]
                image_name_mapping[base_name] = img
            if 'image_name' in df.columns:
                image_name_set = set(str(x) for x in df['image_name'] if pd.notna(x) and str(x).strip() != '')
                uploaded_image_bases = set(image_name_mapping.keys())
                missing_images = image_name_set - uploaded_image_bases
            else:
                missing_images = set()
            st.markdown(f"<div class='simple-info'>Total products: <b>{len(df)}</b><br>Total images uploaded: <b>{len(uploaded_images)}</b>" + (f"<br>Image matching: <b>{len(image_name_set)}</b> required, <b>{len(uploaded_image_bases)}</b> found" if 'image_name' in df.columns else "") + "</div>", unsafe_allow_html=True)
            if missing_images:
                st.markdown(f"<div class='simple-error'>The following images are missing in the uploaded files: {', '.join(missing_images)}</div>", unsafe_allow_html=True)
                return
            cleaned_df, original_count, cleaned_count = process_dataframe(df)
            st.markdown(f"""
                <div class='stats-bar'>
                    <div class='stat-card'>
                        <span class='stat-icon'>📦</span>
                        <div>
                            <div class='stat-label'>Original number of products</div>
                            <div class='stat-value'>{original_count}</div>
                        </div>
                    </div>
                    <div class='stat-card'>
                        <span class='stat-icon'>✅</span>
                        <div>
                            <div class='stat-label'>After removing duplicates</div>
                            <div class='stat-value'>{cleaned_count}</div>
                        </div>
                    </div>
                    <div class='stat-card'>
                        <span class='stat-icon'>🗑️</span>
                        <div>
                            <div class='stat-label'>Duplicates removed</div>
                            <div class='stat-value'>{original_count - cleaned_count}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            processing_file = 'enriched_products_with_images.csv'
            poll_status_and_update_ui(processing_file, "⬇️ Download Results (With Images)")

    # --- Always show download button if output file exists (for user reliability) ---
    if os.path.exists('enriched_products.csv'):
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime('enriched_products.csv')).strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"<div class='styled-download'><b>✅ A completed results file was found (last updated: {last_modified}).</b><br>You can download it below:</div>", unsafe_allow_html=True)
        with open('enriched_products.csv', 'rb') as f:
            st.download_button(
                label="⬇️ Download Results (SKU Only)",
                data=f,
                file_name="enriched_products.csv",
                mime="text/csv",
                key="download_sku_always"
            )
    if os.path.exists('enriched_products_with_images.csv'):
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime('enriched_products_with_images.csv')).strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"<div class='styled-download'><b>✅ A completed results file with images was found (last updated: {last_modified}).</b><br>You can download it below:</div>", unsafe_allow_html=True)
        with open('enriched_products_with_images.csv', 'rb') as f:
            st.download_button(
                label="⬇️ Download Results (With Images)",
                data=f,
                file_name="enriched_products_with_images.csv",
                mime="text/csv",
                key="download_image_always"
            )

if __name__ == "__main__":
    main() 