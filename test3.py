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

def process_products_background(generator, df, output_file, status_queue):
    """Background processing function with enhanced error handling and state management"""
    try:
        total_products = len(df)
        processed_count = 0
        
        # Load existing progress if available
        existing_df = load_progress(output_file)
        if existing_df is not None:
            # Merge with existing progress
            df = pd.merge(df, existing_df, on='sku', how='left', suffixes=('', '_existing'))
            df['description'] = df['description_existing'].combine_first(df['description'])
            df['related_products'] = df['related_products_existing'].combine_first(df['related_products'])
            df = df[['sku', 'description', 'related_products']]
            processed_count = len(df[df['description'].notna() & (df['description'] != '')])
        
        for i, row in df.iterrows():
            try:
                # Skip if already processed
                if pd.notna(row['description']) and row['description'] != '' and row['description'] != 'Description generation failed.':
                    processed_count += 1
                    continue
                
                # Update status
                status = {
                    'current': processed_count + 1,
                    'total': total_products,
                    'current_sku': str(row['sku']),
                    'status': 'processing',
                    'error': None,
                    'last_updated': datetime.datetime.now().isoformat()
                }
                status_queue.put(status)
                save_status(status)

                # Process product with retries
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        description = generator.generate_product_description(row['sku'])
                        if description and description != "Description generation failed.":
                            df.at[i, 'description'] = description
                            related = generator.find_related_products(row['sku'], df['sku'].tolist())
                            df.at[i, 'related_products'] = '|'.join(related)
                            break
                        elif retry < max_retries - 1:
                            time.sleep(30 * (retry + 1))  # Exponential backoff
                            continue
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(30 * (retry + 1))
                            continue
                        raise e
                
                # Save progress after each successful product
                save_progress(df, output_file)
                processed_count += 1
                time.sleep(30)  # Rate limiting
                
            except Exception as e:
                status = {
                    'current': processed_count + 1,
                    'total': total_products,
                    'current_sku': str(row['sku']),
                    'status': 'error',
                    'error': str(e),
                    'last_updated': datetime.datetime.now().isoformat()
                }
                status_queue.put(status)
                save_status(status)
                # Save progress even on error
                save_progress(df, output_file)
                continue

        # Mark as complete
        final_status = {
            'current': total_products,
            'total': total_products,
            'current_sku': None,
            'status': 'complete',
            'error': None,
            'last_updated': datetime.datetime.now().isoformat()
        }
        status_queue.put(final_status)
        save_status(final_status)
        
    except Exception as e:
        error_status = {
            'status': 'error',
            'error': str(e),
            'last_updated': datetime.datetime.now().isoformat()
        }
        status_queue.put(error_status)
        save_status(error_status)

def process_products_with_images_background(generator, df, uploaded_images, output_file, status_queue):
    """Background processing function for products with images"""
    try:
        total_products = len(df)
        processed_count = 0
        image_file_map = {img.name: img for img in uploaded_images}
        
        for i, row in df.iterrows():
            try:
                # Update status
                status = {
                    'current': i + 1,
                    'total': total_products,
                    'current_sku': str(row['sku']),
                    'status': 'processing',
                    'error': None,
                    'last_updated': datetime.datetime.now().isoformat()
                }
                status_queue.put(status)
                save_status(status)

                # Process product
                sku = str(row['sku']) if pd.notna(row['sku']) and row['sku'] != '' else None
                image_name = str(row['image_name']) if pd.notna(row['image_name']) and row['image_name'] != '' else None
                image_file = image_file_map.get(image_name) if image_name else None
                
                if image_file:
                    image_bytes = image_file.read()
                    image_file.seek(0)
                    img = Image.open(io.BytesIO(image_bytes))
                    mime_type = Image.MIME[img.format]
                    
                    if sku:
                        description = generator.generate_product_description_with_image(sku, image_name, image_bytes, mime_type=mime_type)
                        df.at[i, 'description'] = description
                        related = generator.find_related_products(sku, df['sku'].tolist())
                        df.at[i, 'related_products'] = '|'.join(related)
                    else:
                        description = generator.generate_product_description_with_image("", image_name, image_bytes, mime_type=mime_type)
                        df.at[i, 'description'] = description
                        df.at[i, 'related_products'] = ''
                elif sku:
                    description = generator.generate_product_description(sku)
                    df.at[i, 'description'] = description
                    related = generator.find_related_products(sku, df['sku'].tolist())
                    df.at[i, 'related_products'] = '|'.join(related)
                else:
                    df.at[i, 'description'] = 'No SKU or image.'
                    df.at[i, 'related_products'] = ''
                
                # Save progress
                save_progress(df, output_file)
                time.sleep(30)  # Rate limiting
                
            except Exception as e:
                status = {
                    'current': i + 1,
                    'total': total_products,
                    'current_sku': str(row['sku']),
                    'status': 'error',
                    'error': str(e),
                    'last_updated': datetime.datetime.now().isoformat()
                }
                status_queue.put(status)
                save_status(status)
                continue

        # Mark as complete
        final_status = {
            'current': total_products,
            'total': total_products,
            'current_sku': None,
            'status': 'complete',
            'error': None,
            'last_updated': datetime.datetime.now().isoformat()
        }
        status_queue.put(final_status)
        save_status(final_status)
        
    except Exception as e:
        error_status = {
            'status': 'error',
            'error': str(e),
            'last_updated': datetime.datetime.now().isoformat()
        }
        status_queue.put(error_status)
        save_status(error_status)

def check_processing_state():
    """Check if there's an ongoing processing task and restore its state"""
    status = load_status()
    if status and status['status'] == 'processing':
        # Calculate time since last update
        last_updated = datetime.datetime.fromisoformat(status['last_updated'])
        time_since_update = datetime.datetime.now() - last_updated
        
        # If no update in last 5 minutes, consider it stale
        if time_since_update.total_seconds() > 300:
            return None
            
        return status
    return None

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

def main():
    st.markdown("""
        <div class='simple-title'>📝 Product Description Generator</div>
        <div class='simple-subtitle'>Transform your product data into compelling descriptions using AI</div>
    """, unsafe_allow_html=True)

    # Add reset button in the top right
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🔄 Reset All Data", type="secondary", help="Clear all processed files and start fresh"):
            reset_all_data()
            st.success("✅ All data has been reset! Please refresh the page.")
            st.rerun()

    # Check for ongoing processing
    processing_state = check_processing_state()
    if processing_state:
        st.markdown("""
            <div class='simple-info'>
                <b>🔄 Processing in progress...</b><br>
                A previous processing task was detected and is still running.
            </div>
        """, unsafe_allow_html=True)
        
        # Show progress
        progress = int((processing_state['current'] / processing_state['total']) * 100)
        st.progress(progress)
        st.markdown(
            f"<span style='color:var(--primary-color);'>Processing product <b>{processing_state['current']}</b> of <b>{processing_state['total']}</b>: <b>{processing_state['current_sku']}</b></span>",
            unsafe_allow_html=True
        )
        
        # Show download button if results exist
        if os.path.exists('enriched_products.csv'):
            with open('enriched_products.csv', 'rb') as f:
                st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Download Current Results",
                    data=f,
                    file_name="enriched_products.csv",
                    mime="text/csv",
                    key="download_ongoing"
                )
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Auto-refresh the page
        time.sleep(5)
        st.rerun()

    # Centered layout with two simple cards
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("""
            <div class='simple-card'>
                <h2 style='color: var(--primary-color); font-weight: 700;'>📋 Input Options</h2>
            </div>
        """, unsafe_allow_html=True)
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
                help="Upload all product images",
                key="img_files",
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
            total_products = len(cleaned_df)
            progress_bar = st.progress(0)
            status_text = st.empty()
            if st.button("Start Processing", key="start_btn", type="primary"):
                with st.spinner("Processing products..."):
                    try:
                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                        use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
                        generator = ProductDescriptionGenerator(use_openai=use_openai)
                        
                        # Load or initialize progress
                        if os.path.exists('enriched_products.csv'):
                            enriched_df = pd.read_csv('enriched_products.csv')
                            for col in ['sku', 'description', 'related_products']:
                                if col not in enriched_df.columns:
                                    enriched_df[col] = ''
                            merged_df = pd.merge(cleaned_df, enriched_df, on='sku', how='left', suffixes=('', '_enriched'))
                            if 'description_enriched' not in merged_df.columns:
                                merged_df['description_enriched'] = ''
                            if 'related_products_enriched' not in merged_df.columns:
                                merged_df['related_products_enriched'] = ''
                            merged_df['description'] = merged_df['description_enriched'].combine_first(merged_df['description'])
                            merged_df['related_products'] = merged_df['related_products_enriched'].combine_first(merged_df['related_products'])
                            merged_df = merged_df[['sku', 'description', 'related_products']]
                        else:
                            merged_df = cleaned_df.copy()
                            merged_df['description'] = ''
                            merged_df['related_products'] = ''
                        
                        # Find products that need processing
                        to_process = merged_df[
                            (merged_df['description'].isna()) | (merged_df['description'] == '') |
                            (merged_df['description'] == 'Description generation failed.') |
                            (merged_df['related_products'].isna()) | (merged_df['related_products'] == '')
                        ]
                        total_to_process = len(to_process)
                        
                        if total_to_process == 0:
                            st.markdown("<div class='simple-info'>All products have already been processed!</div>", unsafe_allow_html=True)
                            with open('enriched_products.csv', 'rb') as f:
                                st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                                st.download_button(
                                    label="⬇️ Download Results",
                                    data=f,
                                    file_name="enriched_products.csv",
                                    mime="text/csv",
                                    key="download_sku"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            return
                        
                        # Initialize status
                        status_queue = queue.Queue()
                        initial_status = {
                            'current': 0,
                            'total': total_to_process,
                            'current_sku': None,
                            'status': 'starting',
                            'error': None,
                            'last_updated': datetime.datetime.now().isoformat()
                        }
                        save_status(initial_status)
                        
                        # Start background processing
                        thread = threading.Thread(
                            target=process_products_background,
                            args=(generator, to_process, 'enriched_products.csv', status_queue)
                        )
                        thread.daemon = True
                        thread.start()
                        
                        # Monitor progress with auto-refresh
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        error_placeholder = st.empty()
                        
                        while True:
                            try:
                                status = status_queue.get_nowait()
                                if status['status'] == 'complete':
                                    st.markdown("<div class='simple-info'>Processing completed!</div>", unsafe_allow_html=True)
                                    with open('enriched_products.csv', 'rb') as f:
                                        st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                                        st.download_button(
                                            label="⬇️ Download Results",
                                            data=f,
                                            file_name="enriched_products.csv",
                                            mime="text/csv",
                                            key="download_sku"
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    return
                                elif status['status'] == 'error':
                                    error_placeholder.markdown(
                                        f"<div class='simple-error'>Error processing product {status['current_sku']}: {status['error']}</div>",
                                        unsafe_allow_html=True
                                    )
                                    continue
                                
                                progress = int((status['current'] / status['total']) * 100)
                                progress_placeholder.progress(progress)
                                status_placeholder.markdown(
                                    f"<span style='color:var(--primary-color);'>Processing product <b>{status['current']}</b> of <b>{status['total']}</b>: <b>{status['current_sku']}</b></span>",
                                    unsafe_allow_html=True
                                )
                                
                                # Auto-refresh the page every 5 seconds
                                time.sleep(5)
                                st.rerun()
                                
                            except queue.Empty:
                                # Check if processing is still running
                                current_status = load_status()
                                if current_status and current_status['status'] == 'complete':
                                    st.rerun()
                                time.sleep(1)
                                continue
                                
                    except Exception as e:
                        st.markdown(f"<div class='simple-error'>An error occurred during processing: {str(e)}</div>", unsafe_allow_html=True)
        elif scenario == 'sku_image':
            if 'sku' not in df.columns or 'image_name' not in df.columns:
                st.markdown("<div class='simple-error'>❌ The file must contain both 'sku' and 'image_name' columns!</div>", unsafe_allow_html=True)
                return
            uploaded_images = st.session_state.get('uploaded_images', None)
            if not uploaded_images or len(uploaded_images) == 0:
                st.markdown("<div class='simple-info'>Please upload all product images before starting processing.</div>", unsafe_allow_html=True)
                return
            image_name_set = set(df['image_name'].astype(str))
            uploaded_image_names = set([img.name for img in uploaded_images])
            missing_images = image_name_set - uploaded_image_names
            st.markdown(f"<div class='simple-info'>Total products: <b>{len(df)}</b><br>Total images uploaded: <b>{len(uploaded_images)}</b></div>", unsafe_allow_html=True)
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
            total_products = len(cleaned_df)
            progress_bar = st.progress(0)
            status_text = st.empty()
            download_ready = False
            if st.button("Start Processing", key="start_btn_img", type="primary"):
                with st.spinner("Processing products with images..."):
                    try:
                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                        use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
                        generator = ProductDescriptionGenerator(use_openai=use_openai)
                        
                        # Load or initialize progress
                        if os.path.exists('enriched_products_with_images.csv'):
                            enriched_df = pd.read_csv('enriched_products_with_images.csv')
                            for col in ['sku', 'image_name', 'description', 'related_products']:
                                if col not in enriched_df.columns:
                                    enriched_df[col] = ''
                            merged_df = pd.merge(cleaned_df, enriched_df, on='sku', how='left', suffixes=('', '_enriched'))
                            if 'description_enriched' not in merged_df.columns:
                                merged_df['description_enriched'] = ''
                            if 'related_products_enriched' not in merged_df.columns:
                                merged_df['related_products_enriched'] = ''
                            merged_df['description'] = merged_df['description_enriched'].combine_first(merged_df['description'])
                            merged_df['related_products'] = merged_df['related_products_enriched'].combine_first(merged_df['related_products'])
                            merged_df = merged_df[['sku', 'image_name', 'description', 'related_products']]
                        else:
                            merged_df = cleaned_df.copy()
                            merged_df['description'] = ''
                            merged_df['related_products'] = ''
                        
                        # Find products that need processing
                        to_process = merged_df[
                            (merged_df['description'].isna()) | (merged_df['description'] == '') |
                            (merged_df['description'] == 'Description generation failed.') |
                            (merged_df['related_products'].isna()) | (merged_df['related_products'] == '')
                        ]
                        total_to_process = len(to_process)
                        
                        if total_to_process == 0:
                            st.markdown("<div class='simple-info'>All products have already been processed!</div>", unsafe_allow_html=True)
                            with open('enriched_products_with_images.csv', 'rb') as f:
                                st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                                st.download_button(
                                    label="⬇️ Download Results",
                                    data=f,
                                    file_name="enriched_products_with_images.csv",
                                    mime="text/csv",
                                    key="download_image"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            return
                        
                        # Initialize status
                        status_queue = queue.Queue()
                        initial_status = {
                            'current': 0,
                            'total': total_to_process,
                            'current_sku': None,
                            'status': 'starting',
                            'error': None,
                            'last_updated': datetime.datetime.now().isoformat()
                        }
                        save_status(initial_status)
                        
                        # Start background processing
                        thread = threading.Thread(
                            target=process_products_with_images_background,
                            args=(generator, to_process, uploaded_images, 'enriched_products_with_images.csv', status_queue)
                        )
                        thread.daemon = True
                        thread.start()
                        
                        # Monitor progress with auto-refresh
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        error_placeholder = st.empty()
                        
                        while True:
                            try:
                                status = status_queue.get_nowait()
                                if status['status'] == 'complete':
                                    st.markdown("<div class='simple-info'>Processing completed!</div>", unsafe_allow_html=True)
                                    with open('enriched_products_with_images.csv', 'rb') as f:
                                        st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                                        st.download_button(
                                            label="⬇️ Download Results",
                                            data=f,
                                            file_name="enriched_products_with_images.csv",
                                            mime="text/csv",
                                            key="download_image"
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    break
                                elif status['status'] == 'error':
                                    error_placeholder.markdown(
                                        f"<div class='simple-error'>Error processing product {status['current_sku']}: {status['error']}</div>",
                                        unsafe_allow_html=True
                                    )
                                    continue
                                
                                progress = int((status['current'] / status['total']) * 100)
                                progress_placeholder.progress(progress)
                                status_placeholder.markdown(
                                    f"<span style='color:var(--primary-color);'>Processing product <b>{status['current']}</b> of <b>{status['total']}</b>: <b>{status['current_sku']}</b></span>",
                                    unsafe_allow_html=True
                                )
                                
                                # Auto-refresh the page every 5 seconds
                                time.sleep(5)
                                st.rerun()
                                
                            except queue.Empty:
                                # Check if processing is still running
                                current_status = load_status()
                                if current_status and current_status['status'] == 'complete':
                                    st.rerun()
                                time.sleep(1)
                                continue
                                
                    except Exception as e:
                        st.markdown(f"<div class='simple-error'>An error occurred during processing: {str(e)}</div>", unsafe_allow_html=True)
                        download_ready = False
            if os.path.exists('enriched_products_with_images.csv'):
                with open('enriched_products_with_images.csv', 'rb') as f:
                    st.markdown("<div class='styled-download'>", unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download Results",
                        data=f,
                        file_name="enriched_products_with_images.csv",
                        mime="text/csv",
                        key="download_image"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

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