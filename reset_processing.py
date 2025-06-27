import os
import json

def reset_processing_state():
    """Reset all processing state files to clear any cached errors"""
    
    files_to_remove = [
        'processing_status.json',
        'processing.lock',
        'enriched_products.csv',
        'enriched_products_with_images.csv',
        'processing_progress.csv'
    ]
    
    print("Resetting processing state...")
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
            else:
                print(f"ℹ️ Not found: {file_path}")
        except Exception as e:
            print(f"❌ Error removing {file_path}: {str(e)}")
    
    print("\n🎉 Processing state reset complete!")
    print("You can now restart processing without the mismatch error.")

if __name__ == "__main__":
    reset_processing_state() 