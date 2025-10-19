# EHTool - Error Handling Tool

A comprehensive web application for error detection and proofreading in image segmentation workflows. EHTool integrates error detection capabilities with advanced proofreading tools to help users identify and correct segmentation errors efficiently.

## 🚀 Features

### **Dual Workflow System**
- **Error Detection Workflow**: Identify and categorize segmentation errors
- **Standalone Proofreading Workflow**: Direct image editing and mask correction

### **Error Detection Capabilities**
- Load and analyze 2D/3D image datasets
- Interactive layer-by-layer error detection
- Categorize layers as: Correct, Incorrect, or Unsure
- Batch operations for efficient labeling
- Pagination support for large datasets
- Progress tracking and statistics

### **Advanced Proofreading Tools**
- **Integrated Proofreading**: Seamlessly edit incorrect layers from error detection
- **Standalone Proofreading**: Direct image editing with full PFTool functionality
- Real-time mask editing with paint and erase tools
- Keyboard shortcuts for efficient editing
- Layer navigation for 3D datasets
- Automatic mask generation and saving

### **User Interface**
- Modern, responsive web interface
- Intuitive navigation between workflows
- Real-time progress tracking
- Comprehensive keyboard shortcuts
- Mac-specific UI adaptations

## 🛠️ Installation

### Prerequisites
- Python 3.9
- Flask
- Required Python packages (see requirements.txt)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd EHTool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5004
```

## 📁 Project Structure

```
EHTool/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── routes/                         # Flask route handlers
│   ├── landing.py                 # Home page and main navigation
│   ├── detection_workflow.py      # Error detection workflow
│   ├── proofreading_workflow.py   # Standalone proofreading
│   ├── proofreading.py           # Integrated proofreading
│   ├── detection.py               # Detection page logic
│   ├── review.py                  # Review and statistics
│   └── export.py                  # Export functionality
├── backend/                        # Core backend modules
│   ├── session_manager.py         # Session state management
│   ├── data_manager.py           # Data loading and processing
│   └── volume_manager.py         # Volume and mask operations
├── templates/                      # HTML templates
│   ├── base.html                 # Base template
│   ├── landing.html              # Home page
│   ├── detection_load.html        # Data loading interface
│   ├── detection.html            # Error detection interface
│   ├── proofreading_load.html    # Proofreading data loading
│   ├── proofreading_standalone.html # Standalone proofreading editor
│   └── proofreading.html         # Integrated proofreading editor
├── static/                        # Static assets
│   ├── css/                      # Stylesheets
│   └── js/                       # JavaScript files
└── _uploads/                      # Uploaded files directory
```

## 🎯 Usage Guide

### **Getting Started**

1. **Launch EHTool**: Open your browser to `http://localhost:5004`
2. **Choose Workflow**: Select either "Error Detection" or "Proofreading"
3. **Load Dataset**: Upload files or provide file paths
4. **Start Working**: Begin your error detection or proofreading workflow

### **Error Detection Workflow**

1. **Load Dataset**:
   - Upload image and mask files, or
   - Provide local file paths
   - Supports TIFF, PNG, JPG, JPEG formats

2. **Detect Errors**:
   - Navigate through layers using pagination
   - Mark layers as Correct, Incorrect, or Unsure
   - Use batch operations for efficiency
   - Track progress with statistics

3. **Proofread Incorrect Layers**:
   - Select incorrect layers for proofreading
   - Edit masks with paint and erase tools
   - Navigate between layers
   - Save corrections automatically

### **Standalone Proofreading Workflow**

1. **Load Dataset**:
   - Upload image and mask files
   - Or provide local file paths
   - Automatic mask generation if none provided

2. **Edit Masks**:
   - Use paint and erase tools
   - Navigate between slices (3D datasets)
   - Keyboard shortcuts for efficiency
   - Real-time editing feedback

3. **Save Changes**:
   - Automatic saving to original format
   - Preserves file integrity
   - Supports all major image formats

## ⌨️ Keyboard Shortcuts

### **General Navigation**
- `A` or `←` - Previous layer/slice
- `D` or `→` - Next layer/slice
- `⌘/Ctrl + Z` - Undo
- `⌘/Ctrl + Shift + Z` - Redo

### **Editing Tools**
- `P` - Paint mode
- `E` - Erase mode
- `Space` - Toggle between paint/erase

## 🔧 Technical Details

### **Supported Formats**
- **Images**: TIFF, PNG, JPG, JPEG
- **Masks**: TIFF, PNG, JPG, JPEG
- **3D Data**: Multi-slice TIFF files

### **Data Management**
- Session-based state management
- Automatic data clearing between workflows
- Persistent storage for large datasets
- Memory-efficient processing

### **Architecture**
- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, PIL, NumPy, Tifffile
- **State Management**: Custom session manager

## 🚨 Troubleshooting

### **Common Issues**

1. **White Box in Proofreading**:
   - Ensure data is properly loaded
   - Check file format compatibility
   - Verify mask file exists

2. **Data Not Loading**:
   - Check file paths are correct
   - Ensure files are accessible
   - Verify file format support

3. **Session Issues**:
   - Use "Reset" button to clear all data
   - Refresh browser if needed
   - Check browser console for errors

### **Performance Tips**
- Use TIFF format for best performance
- Close other applications for large datasets
- Use batch operations for efficiency

## 🔄 Workflow Separation

EHTool maintains separate data storage for different workflows:

- **Error Detection**: `DETECTION_*` config variables
- **Standalone Proofreading**: `PROOFREADING_*` config variables  
- **Integrated Proofreading**: `INTEGRATED_*` config variables
- **Landing Page**: `LANDING_*` config variables

This ensures no interference between workflows and maintains data integrity.

## 📊 Export Options

- **Session Data**: Complete workflow state
- **Annotations**: Layer classifications
- **Statistics**: Progress and accuracy metrics
- **Proofreading Queue**: Incorrect layers for review

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on Flask web framework
- Image processing with OpenCV and PIL
- UI/UX inspired by modern web applications
- Keyboard shortcuts optimized for Mac users

---

**EHTool** - Making error detection and proofreading efficient and intuitive! 🎯
