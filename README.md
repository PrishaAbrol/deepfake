# Deepfake Detection Using Machine Learning

This project aims to detect deepfake videos using a machine learning pipeline that incorporates video processing, facial feature extraction, and metadata analysis. The detection system utilizes OpenCV's Haar cascades for face, eye, and profile detection, and processes video data to identify manipulated content.

## Project Structure

### 1. **Data Processing**
   - **Dataset**: The project utilizes the dataset from the Deepfake Detection Challenge.
     - **Train Samples Folder**: Contains videos used for training the model.
     - **Test Folder**: Contains videos used for testing the model.
   - **Metadata**: A JSON file accompanies the dataset, containing information about the videos, including labels (`REAL` or `FAKE`) and original sources.

### 2. **Face Detection**
   - **OpenCV**: The system uses OpenCV's pre-trained Haar cascades for face, eye, profile, and smile detection. These cascades are loaded from the `../input/haar-cascades-for-face-detection` directory.
   - The primary focus is on detecting faces in frames extracted from video files. Once detected, rectangular bounding boxes are drawn around faces, with additional detections for eyes and profiles.

### 3. **Data Exploration**
   - **Video File Extensions**: The project checks and processes the different file extensions of the training and test videos.
   - **Metadata Analysis**: Metadata from the JSON file is explored to understand the distribution of real and fake videos.
   - **Missing Data**: The project checks for and handles missing data in the metadata file.

### 4. **Video Frame Extraction**
   - The project extracts frames from both real and fake videos. For each video, a frame is selected, and the object detection algorithm is applied to detect and highlight faces and other features.
   - A function is implemented to extract and display a frame from any video file.

### 5. **Object Detection Algorithm**
   - The `ObjectDetector` class is used for detecting objects like faces and eyes in the extracted frames. It takes a pre-trained Haar cascade XML file as input.
   - The `detect_objects` function processes an image to detect faces, eyes, and profiles, drawing bounding boxes around them.

### 6. **Video Playback**
   - The system can display videos directly in the notebook using the `play_video` function, which supports both training and test videos.
   - A sample of real and fake videos is displayed for analysis and comparison.

## How to Run the Project

1. **Setup**: Ensure that you have all dependencies installed. Key dependencies include:
   - OpenCV
   - Pandas
   - NumPy
   - Matplotlib
   - Seaborn
   - Tqdm (for progress bars)

2. **Dataset**: The dataset is stored in a folder named `../input/deepfake-detection-challenge`. You need to update the path if using a different dataset location.

3. **Running the Notebook**: Run the cells sequentially to process the data, detect faces, and analyze real and fake videos. The object detection results will be displayed in the notebook.

## Object Detection Models

The project uses the following Haar cascades for object detection:
- **Frontal Face Detection**: `haarcascade_frontalface_default.xml`
- **Eye Detection**: `haarcascade_eye.xml`
- **Profile Face Detection**: `haarcascade_profileface.xml`
- **Smile Detection**: `haarcascade_smile.xml` (currently disabled due to high false positives)

## Results

- **Frame Extraction**: The system successfully extracts frames from both real and fake videos, and the face detection system identifies manipulated content.
- **Visualization**: Bounding boxes around detected faces, eyes, and profiles provide a visual understanding of the detection process.

## Future Work
- Integrating deep learning models for more accurate deepfake detection.
- Improving face detection by reducing false positives and optimizing the detection algorithm.
- Extending the system to classify videos as real or fake based on detected patterns.
