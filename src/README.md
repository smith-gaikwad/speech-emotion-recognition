# Speech Emotion Recognition

This project uses a 1D Convolutional Neural Network (CNN) to recognize emotions from speech audio files. The model is trained on the RAVDESS dataset.

## Features
- Extracts MFCC features from audio files.
- Trains a CNN to classify 8 different emotions (happy, sad, angry, etc.).
- Achieves high accuracy on the validation set.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/smith-gaikwad/speech-emotion-recognition.git](https://github.com/smith-gaikwad/speech-emotion-recognition.git)
    cd speech-emotion-recognition
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the Data:**
    Download the RAVDESS dataset from [here](https://zenodo.org/record/1188976) and extract the "Actor_XX" folders into a `data/` directory in the project root.

4.  **Train the model:**
    ```bash
    python src/train.py
    ```