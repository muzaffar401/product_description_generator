# Product Description Generator

A powerful AI-powered tool for generating compelling product descriptions and validating product images using advanced machine learning models.

## 🚀 Features

### Core Functionality
- **AI-Powered Description Generation**: Generate marketing-friendly product descriptions using Google Gemini or OpenAI
- **Image Analysis**: Analyze product images to extract product information and generate descriptions
- **Related Products**: Find and suggest related products from your product catalog
- **Background Processing**: Process products in the background while you continue using other applications

### Advanced Image Validation
- **Web Image Comparison**: Compare uploaded images with web-scraped product images for accurate validation
- **AI-Powered Validation**: Use advanced AI models to detect mismatches between SKUs and images
- **Multi-Level Validation**: 
  - Pre-validation using keyword matching
  - Enhanced validation with web search context
  - Web image comparison (when SerpAPI key is provided)
  - Fallback to simple validation

### Input Options
- **SKU Only**: Generate descriptions using only product SKUs
- **SKU + Images**: Generate descriptions using both SKUs and product images
- **Image Only**: Generate descriptions using only product images

## 🔧 Setup

### Prerequisites
- Python 3.8 or higher
- Required API keys (see API Keys section)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd product-description-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Required (Choose One)
   GEMINI_API_KEY="your_gemini_api_key"
   # OR
   OPENAI_API_KEY="your_openai_api_key"
   
   # Optional (For Enhanced Web Image Comparison)
   SERPAPI_API_KEY="your_serpapi_api_key"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔑 API Keys

### Required API Keys (Choose One)

#### Google Gemini
- Visit [Google AI for Developers](https://ai.google.dev/)
- Create a new project and enable the Gemini API
- Generate an API key
- Add to `.env` file as `GEMINI_API_KEY`

#### OpenAI
- Visit [OpenAI Platform](https://platform.openai.com/)
- Create an account and add billing information
- Generate an API key
- Add to `.env` file as `OPENAI_API_KEY`

### Optional API Keys

#### SerpAPI (For Enhanced Web Image Comparison)
- Visit [SerpAPI](https://serpapi.com/)
- Sign up for a free account
- Get your API key from the dashboard
- Add to `.env` file as `SERPAPI_API_KEY`

**Note**: SerpAPI provides 100 free searches per month. The web image comparison feature will gracefully fall back to standard validation if the API key is not provided.

## 📖 Usage

### 1. Choose Input Type
- **Only Product SKUs**: Upload an Excel/CSV file with a 'sku' column
- **Product SKUs with Image Names**: Upload an Excel/CSV file with 'sku' and 'image_name' columns, plus the corresponding images

### 2. Upload Files
- **Data File**: Excel (.xlsx, .xls) or CSV files are supported
- **Images**: JPG, JPEG, PNG, WebP, BMP formats are supported
- **Image Naming**: Images should match the names in your Excel file (extensions are optional)

### 3. Configure Settings
- **AI Model**: Choose between Gemini or OpenAI
- **Debug Mode**: Enable for detailed logging
- **Test Validation**: Test specific SKU-image pairs before batch processing

### 4. Start Processing
- Click "Start Processing" to begin
- Processing runs in the background
- You can switch tabs or close the browser - processing continues
- Return to check progress and download results

### 5. Download Results
- Results are saved as CSV files
- Download links appear when processing is complete
- Files are automatically saved with timestamps

## 🔍 Image Validation Process

The application uses a multi-level validation system to ensure image-SKU matches:

### 1. Pre-Validation
- Checks for obvious mismatches using keyword analysis
- Compares food vs non-food categories
- Validates file extensions

### 2. Enhanced Validation
- Uses AI to analyze image content
- Considers web search context for product information
- Provides detailed confidence scores

### 3. Web Image Comparison (Optional)
- Searches for product images online using SerpAPI
- Downloads reference images from the web
- Compares uploaded images with web references using AI
- Provides highest accuracy validation

### 4. Fallback Validation
- Simple AI-based validation if enhanced methods fail
- Ensures processing continues even with API issues

## 🧪 Testing

### Test Web Image Comparison
```bash
python test_web_comparison.py
```

### Test Validation Logic
```bash
python test_validation.py
```

### Test Smart Validation
```bash
python test_smart_validation.py
```

## 📁 File Structure

```
product-description-generator/
├── app.py                          # Main Streamlit application
├── main.py                         # Core processing logic
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                           # Environment variables (create this)
├── test_web_comparison.py         # Web image comparison tests
├── test_validation.py             # Validation tests
├── test_smart_validation.py       # Smart validation tests
└── product_images/                # Test images directory
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key
- `OPENAI_API_KEY`: OpenAI API key
- `SERPAPI_API_KEY`: SerpAPI key for web image comparison

### Processing Settings
- **Delay between products**: 30 seconds (configurable in code)
- **API retries**: 3 attempts with exponential backoff
- **Timeout**: 60 seconds per API call
- **Max tokens**: 400 for AI responses

## 🚨 Error Handling

The application includes comprehensive error handling:

- **API Failures**: Automatic retries with exponential backoff
- **Image Processing Errors**: Graceful fallback to SKU-only processing
- **Validation Failures**: Detailed error messages with specific reasons
- **File Format Issues**: Clear error messages for unsupported formats
- **Missing Images**: Validation before processing starts

## 📊 Output Format

### CSV Output Columns
- **sku**: Original product SKU
- **image_name**: Image filename (if provided)
- **description**: Generated product description (1000 characters)
- **related_products**: Related product SKUs (pipe-separated)

### Processing Status
- Real-time progress updates
- Current product being processed
- Error messages with detailed explanations
- Processing completion status

## 🔒 Security

- API keys are stored in environment variables
- No sensitive data is logged or stored
- Temporary files are cleaned up automatically
- Processing lock files prevent duplicate runs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the error messages in the application
2. Review the console logs for detailed information
3. Ensure all API keys are correctly configured
4. Verify file formats and naming conventions

## 🔄 Updates

### Recent Updates
- **Web Image Comparison**: Added SerpAPI integration for enhanced validation
- **Multi-Model Support**: Added OpenAI GPT-4o support alongside Gemini
- **Background Processing**: Improved background processing with status tracking
- **Enhanced Error Handling**: Better error messages and recovery mechanisms
