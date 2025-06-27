import streamlit as st
import pandas as pd
import time
from main import ProductDescriptionGenerator
import os
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

def process_dataframe(df):
    """Process dataframe to remove duplicates and prepare for analysis"""
    original_count = len(df)
    df = df.drop_duplicates(subset=['sku'], keep='first')
    cleaned_count = len(df)
    return df, original_count, cleaned_count

def smart_delay():
    """Add a randomized delay between API calls to prevent rate limiting"""
    base_delay = 45  # Base delay of 45 seconds
    jitter = random.uniform(0, 10)  # Add random jitter between 0-10 seconds
    time.sleep(base_delay + jitter)

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
    This tool helps you generate product descriptions and find related products using AI.
    Upload your Excel file containing product SKUs to get started.
    """)

    # File uploader only (no sample data)
    uploaded_file = st.file_uploader("Upload Excel file with product SKUs", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            st.session_state['df'] = df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

    # Process data if we have it (either from upload or sample)
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Check if 'sku' column exists
        if 'sku' not in df.columns:
            st.error("The file must contain a 'sku' column!")
            return

        # Process and clean data
        cleaned_df, original_count, cleaned_count = process_dataframe(df)
        
        # Display cleaning results
        st.info(f"Original number of products: {original_count}")
        st.info(f"Number of products after removing duplicates: {cleaned_count}")
        st.info(f"Number of duplicates removed: {original_count - cleaned_count}")
        
        # Save cleaned data to CSV for verification
        cleaned_df.to_csv('cleaned_products.csv', index=False)
        st.write("Cleaned data saved to 'cleaned_products.csv'")
            
        # Process the products
        total_products = len(cleaned_df)
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        retry_text = st.empty()
        
        # Start processing
        if st.button("Start Processing"):
            with st.spinner("Processing products..."):
                try:
                    # Check which API key is available
                    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                    use_openai = bool(OPENAI_API_KEY) and not bool(GEMINI_API_KEY)
                    
                    # Initialize the generator
                    generator = ProductDescriptionGenerator(use_openai=use_openai)
                    
                    # Resume logic: load enriched_products.csv if exists
                    if os.path.exists('enriched_products.csv'):
                        enriched_df = pd.read_csv('enriched_products.csv')
                        # Ensure all columns exist
                        for col in ['sku', 'description', 'related_products']:
                            if col not in enriched_df.columns:
                                enriched_df[col] = ''
                        # Merge with cleaned_df to ensure all SKUs are present
                        merged_df = pd.merge(cleaned_df, enriched_df, on='sku', how='left', suffixes=('', '_enriched'))
                        # Ensure the merged columns exist
                        if 'description_enriched' not in merged_df.columns:
                            merged_df['description_enriched'] = ''
                        if 'related_products_enriched' not in merged_df.columns:
                            merged_df['related_products_enriched'] = ''
                        # Use the most recent data for description and related_products
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
                        progress = min(int(((i + 1) / total_to_process) * 100), 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing product {i + 1} of {total_to_process}: {row['sku']}")
                        
                        max_retries = 3
                        retry_count = 0
                        success = False
                        
                        while not success and retry_count < max_retries:
                            try:
                                if retry_count > 0:
                                    retry_text.warning(f"Retry attempt {retry_count} of {max_retries}...")
                                
                                description = generator.generate_product_description(row['sku'])
                                merged_df.at[row_idx, 'description'] = description
                                related = generator.find_related_products(row['sku'], merged_df['sku'].tolist())
                                merged_df.at[row_idx, 'related_products'] = '|'.join(related)
                                merged_df.to_csv('enriched_products.csv', index=False)
                                success = True
                                retry_text.empty()
                                
                            except Exception as e:
                                retry_count += 1
                                if retry_count < max_retries:
                                    retry_delay = 120 * retry_count  # Increase delay with each retry
                                    retry_text.warning(f"Error occurred: {str(e)}\nWaiting {retry_delay} seconds before retry...")
                                    time.sleep(retry_delay)
                                else:
                                    st.error(f"Failed to process {row['sku']} after {max_retries} attempts: {str(e)}")
                                    continue
                        
                        # Add delay between products (with randomization to avoid pattern detection)
                        smart_delay()
                        
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

if __name__ == "__main__":
    main() 