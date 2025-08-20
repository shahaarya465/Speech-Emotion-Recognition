# 🎤 Speech Emotion Recognition

A full-stack application that recognizes human emotions from speech using a deep learning model. The system analyzes audio signals in real-time to classify emotions such as happiness, sadness, and anger. It features a modern React frontend that allows users to provide audio via file upload, microphone recording, or drag-and-drop.

---

## ✨ Key Features

-   **Modern React Frontend**: A responsive and intuitive user interface built with React.
-   **Multiple Input Methods**: Supports file uploads, direct microphone recording, and drag-and-drop.
-   **Advanced Emotion Model**: Utilizes an LSTM neural network and LightGBM classifiers for prediction.
-   **Multi-Lingual & Multi-Accent Support**: Trained on English, Hindi, and Indian-accented English datasets for robust performance.
-   **Automated Local Setup**: Includes simple startup scripts (`start.bat`/`start.sh`) for a one-click launch of the entire full-stack environment.

---

## 🛠️ Technologies Used

-   **Backend**: Python, Flask, TensorFlow, Keras, Librosa, Scikit-learn
-   **Frontend**: ReactJS, Vite, CSS
-   **Core Libraries**: Pandas, NumPy, Joblib, SoundFile
-   **Development**: Jupyter Notebook, Git

---

## 🗃️ Datasets

This project was trained on a combination of three distinct datasets to ensure a comprehensive and diverse model:

-   **RAVDESS Dataset**: The core English dataset featuring North American actors.
-   **Hindi SER Dataset**: A collection of emotional speech recordings in the Hindi language.
-   **Indian Accent English TTS Dataset**: A public dataset of emotional speech in an Indian English accent.

*Note: The original structure of these datasets was modified for streamlined data processing.*

---

## 🚀 How to Run Locally

This project includes automated startup scripts for a simple setup.

### Prerequisites

-   **Python 3.12.3** must be installed and available in your system's PATH.
-   **Node.js** (LTS version is recommended) must be installed.

### Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/KavyaJP/SGP-I.git
    cd SGP-I
    ```

2.  **Run the Startup Script:**
    -   **On Windows:** Double-click the `start.bat` file.
    -   **On macOS / Linux:** Make the script executable first (`chmod +x start.sh`), then run it (`./start.sh`).

The script will automatically perform a one-time setup to create a virtual environment and install all Python and Node.js dependencies. It will then start both the backend and frontend servers and open the application in your browser.

---

## 🤝 Contributors

-   Kavya Prajapati - [E-Mail](mailto:kavya31052005@gmail.com)
-   Aarya Shah - [E-Mail](mailto:shahaarya465@gmail.com)
-   Vansh Mehta - [E-Mail](mailto:vansh161976@gmail.com)
