import streamlit as st
import pandas as pd
import time
from main import ProductDescriptionGenerator
import os
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as genai

# Load environment variables
load_dotenv()

def process_dataframe(df):
    """Process dataframe to remove duplicates and prepare for analysis"""
    original_count = len(df)
    df = df.drop_duplicates(subset=['sku'], keep='first')
    cleaned_count = len(df)
    return df, original_count, cleaned_count

def main():
    # Configure page
    st.set_page_config(
        page_title="Product Description Generator",
        page_icon="📝",
        layout="wide"
    )

    # Title and description
    st.title("Product Description Generator")
    st.markdown("""
    This tool helps you generate product descriptions and find related products using AI.\n
    **Choose your scenario and upload the required files to get started.**
    """)

    # Scenario selection
    scenario_options = [
        "Select your scenario",
        "Only Product SKUs",
        "Product SKUs with Image Names"
    ]
    scenario = st.selectbox(
        "Select Input Scenario",
        scenario_options,
        index=0,
        key="scenario_select",
        help="Choose the type of input you want to provide."
    )

    if scenario == "Select your scenario":
        st.info("Please select a scenario to continue.")
        return

    if scenario == "Only Product SKUs":
        uploaded_file = st.file_uploader("Upload file with product SKUs", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['df'] = df
                st.session_state['scenario'] = 'sku_only'
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    else:
        uploaded_file = st.file_uploader("Upload file with SKUs and Image Names (columns: sku, image_name)", type=['xlsx', 'xls', 'csv'])
        uploaded_images = st.file_uploader(
            "Upload all product images (select multiple files)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['df'] = df
                st.session_state['scenario'] = 'sku_image'
                st.session_state['uploaded_images'] = uploaded_images
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return

    # Process data if we have it (either from upload or sample)
    if 'df' in st.session_state and 'scenario' in st.session_state:
        df = st.session_state['df']
        scenario = st.session_state['scenario']
        if scenario == 'sku_only':
            # Check if 'sku' column exists
            if 'sku' not in df.columns:
                st.error("The file must contain a 'sku' column!")
                return
            # Process and clean data
            cleaned_df, original_count, cleaned_count = process_dataframe(df)
            st.info(f"Original number of products: {original_count}")
            st.info(f"Number of products after removing duplicates: {cleaned_count}")
            st.info(f"Number of duplicates removed: {original_count - cleaned_count}")
            total_products = len(cleaned_df)
            progress_bar = st.progress(0)
            status_text = st.empty()
            if st.button("Start Processing"):
                with st.spinner("Processing products..."):
                    try:
                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                        use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
                        generator = ProductDescriptionGenerator(use_openai=use_openai)
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
                        to_process = merged_df[
                            (merged_df['description'].isna()) | (merged_df['description'] == '') |
                            (merged_df['description'] == 'Description generation failed.') |
                            (merged_df['related_products'].isna()) | (merged_df['related_products'] == '')
                        ]
                        total_to_process = len(to_process)
                        if total_to_process == 0:
                            st.success("All products have already been processed!")
                            with open('enriched_products.csv', 'rb') as f:
                                st.download_button(
                                    label="Download Results",
                                    data=f,
                                    file_name="enriched_products.csv",
                                    mime="text/csv"
                                )
                            return
                        for i, (row_idx, row) in enumerate(to_process.iterrows()):
                            progress = int(((i + 1) / total_to_process) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing product {i + 1} of {total_to_process}: {row['sku']}")
                            try:
                                description = generator.generate_product_description(row['sku'])
                                merged_df.at[row_idx, 'description'] = description
                                related = generator.find_related_products(row['sku'], merged_df['sku'].tolist())
                                merged_df.at[row_idx, 'related_products'] = '|'.join(related)
                                merged_df.to_csv('enriched_products.csv', index=False)
                            except Exception as e:
                                st.error(f"Error processing product {row['sku']}: {str(e)}")
                                continue
                            time.sleep(30)
                        st.success("Processing completed!")
                        with open('enriched_products.csv', 'rb') as f:
                            st.download_button(
                                label="Download Results",
                                data=f,
                                file_name="enriched_products.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
        elif scenario == 'sku_image':
            # Check for required columns
            if 'sku' not in df.columns or 'image_name' not in df.columns:
                st.error("The file must contain both 'sku' and 'image_name' columns!")
                return
            # Check if images are uploaded
            uploaded_images = st.session_state.get('uploaded_images', None)
            if not uploaded_images or len(uploaded_images) == 0:
                st.warning("Please upload all product images before starting processing.")
                return
            # Map image_name to uploaded file
            image_name_set = set(df['image_name'].astype(str))
            uploaded_image_names = set([img.name for img in uploaded_images])
            missing_images = image_name_set - uploaded_image_names
            # Show total images and products
            st.info(f"Total products: {len(df)}")
            st.info(f"Total images uploaded: {len(uploaded_images)}")
            if missing_images:
                st.error(f"The following images are missing in the uploaded files: {', '.join(missing_images)}")
                return
            # Process and clean data
            cleaned_df, original_count, cleaned_count = process_dataframe(df)
            st.info(f"Original number of products: {original_count}")
            st.info(f"Number of products after removing duplicates: {cleaned_count}")
            st.info(f"Number of duplicates removed: {original_count - cleaned_count}")
            total_products = len(cleaned_df)
            progress_bar = st.progress(0)
            status_text = st.empty()
            download_ready = False
            if st.button("Start Processing"):
                with st.spinner("Processing products with images..."):
                    try:
                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                        use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
                        generator = ProductDescriptionGenerator(use_openai=use_openai)
                        # Prepare output DataFrame
                        output_df = cleaned_df.copy()
                        output_df['description'] = ''
                        output_df['related_products'] = ''
                        # Map image_name to file object
                        image_file_map = {img.name: img for img in uploaded_images}
                        for i, row in output_df.iterrows():
                            progress = int(((i + 1) / total_products) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing product {i + 1} of {total_products}: {row['sku']}")
                            try:
                                sku = str(row['sku']) if pd.notna(row['sku']) and row['sku'] != '' else None
                                image_name = str(row['image_name']) if pd.notna(row['image_name']) and row['image_name'] != '' else None
                                image_file = image_file_map.get(image_name) if image_name else None
                                image_bytes = image_file.read() if image_file else None
                                if image_file:
                                    image_file.seek(0)
                                    try:
                                        img = Image.open(io.BytesIO(image_bytes))
                                        img.verify()  # Will raise if not a valid image
                                    except Exception as img_e:
                                        st.error(f"Image {image_name} is not a valid image: {img_e}")
                                        output_df.at[i, 'description'] = 'Image invalid.'
                                        output_df.at[i, 'related_products'] = ''
                                        continue
                                # Mismatch validation: check if SKU and image are unrelated
                                if sku and image_file:
                                    img = Image.open(io.BytesIO(image_bytes))
                                    mime_type = Image.MIME[img.format]
                                    mismatch_prompt = (
                                        f"Is the product in this image related to the SKU '{sku}'? "
                                        "If the image is completely unrelated (e.g., SKU is a food item but the image is a shoe), respond ONLY with 'MISMATCH'. "
                                        "If the image matches or is related, respond ONLY with 'OK'. Do not add any other text."
                                    )
                                    if not generator.use_openai:
                                        mismatch_result = generator._make_api_call(mismatch_prompt, image_bytes=image_bytes, mime_type=mime_type)
                                        print('Gemini mismatch result:', mismatch_result)  # For debugging
                                    else:
                                        mismatch_result = "OK"
                                    if mismatch_result.strip().upper() == 'MISMATCH':
                                        st.error(f"The image for product '{sku}' is mismatched. Processing stopped.")
                                        return
                                # Description generation logic
                                if sku and image_file:
                                    description = generator.generate_product_description_with_image(sku, image_name, image_bytes, mime_type=mime_type)
                                    output_df.at[i, 'description'] = description
                                    related = generator.find_related_products(sku, output_df['sku'].tolist())
                                    output_df.at[i, 'related_products'] = '|'.join(related)
                                elif sku and not image_file:
                                    description = generator.generate_product_description(sku)
                                    output_df.at[i, 'description'] = description
                                    related = generator.find_related_products(sku, output_df['sku'].tolist())
                                    output_df.at[i, 'related_products'] = '|'.join(related)
                                elif image_file and not sku:
                                    # Generate description from image only
                                    description = generator.generate_product_description_with_image("", image_name, image_bytes, mime_type=mime_type)
                                    output_df.at[i, 'description'] = description
                                    output_df.at[i, 'related_products'] = ''
                                else:
                                    output_df.at[i, 'description'] = 'No SKU or image.'
                                    output_df.at[i, 'related_products'] = ''
                                output_df.to_csv('enriched_products_with_images.csv', index=False)
                            except Exception as e:
                                st.error(f"Error processing product {row['sku']}: {str(e)}")
                                continue
                            time.sleep(30)
                        # Ensure output columns order
                        output_df = output_df[['sku', 'image_name', 'description', 'related_products']]
                        output_df.to_csv('enriched_products_with_images.csv', index=False)
                        st.success("Processing completed!")
                        download_ready = True
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
                        download_ready = False
            # Always show download button if file exists
            if os.path.exists('enriched_products_with_images.csv'):
                with open('enriched_products_with_images.csv', 'rb') as f:
                    st.download_button(
                        label="Download Results",
                        data=f,
                        file_name="enriched_products_with_images.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main() 