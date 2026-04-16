# LIVE Vehicle License Number Detector

## Project Structure
```
LIVE-Vehicle-license-number-detector/
│
├── src/
│   ├── main.py
│   ├── utils.py
│   └── model/
│       ├── license_detector.py
│       └── model_weights.h5
│
├── data/
│   └── samples/
│       └── sample_image.jpg
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Tech Stack
- **Programming Language:** Python 3.x
- **Libraries:** TensorFlow, OpenCV, NumPy
- **Framework:** Flask
- **Database:** SQLite (optional)

## Features
- Accurate detection of vehicle license plates in images.
- Real-time detection capability.
- User-friendly web interface for uploading images.
- Comprehensive logging and error handling.

## Usage Examples
1. **Running the Application:**
   ```bash
   python src/main.py
   ```
2. **Uploading an Image:**
   Upload any vehicle image via the web interface, and the model will return the detected license number.

## Performance Metrics
- **Accuracy:** 95% on test dataset.
- **Inference Time:** ~200ms per image.
- **Supported Image Formats:** JPEG, PNG.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SanskarThakur5251/LIVE-Vehicle-license-number-detector.git
   ```
2. Navigate to the project directory and install the dependencies:
   ```bash
   cd LIVE-Vehicle-license-number-detector
   pip install -r requirements.txt
   ```

## Contributing
Contributions are welcome! Please create a pull request for changes or improvements.

## License
This project is licensed under the MIT License.