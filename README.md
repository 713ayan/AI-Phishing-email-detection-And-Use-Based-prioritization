# AI Phishing Email Detection and Prioritization using Machine Learning

This repository focuses on building a machine learning model for phishing email detection and prioritization. The project leverages Natural Language Processing (NLP) techniques and machine learning algorithms to accurately detect phishing emails and rank them by importance, ensuring safe and effective email management. 

With the increasing sophistication of phishing attempts, this solution helps users detect malicious emails, protect their sensitive information, and prioritize important emails efficiently. The model is trained on datasets such as SpamAssassin and Enron, with potential deployment on cloud platforms like AWS, Azure, or Google Cloud.

## Installation Packages

### Google Colab
To run this project on Google Colab, you need to install the following packages:
```bash
!pip install pandas
!pip install scikit-learn
!pip install dvc
```

- **Pandas**: For handling tabular datasets like email metadata and content.
- **Scikit-learn**: Includes machine learning algorithms for building and evaluating phishing detection models.
- **DVC**: For data versioning and efficient management of datasets and models in your MLOps pipeline.

### Anaconda Prompt (Jupyter Notebook)
If you're running this project locally in a Jupyter Notebook (Anaconda), you can install the required packages via `pip` or `conda`:
```bash
pip install -r requirements.txt
```
Alternatively, you can manually install the packages listed in the `requirements.txt` file.

### Requirements.txt Example
If you have a `requirements.txt` file, it should look something like this:
```
pandas
scikit-learn
dvc
numpy
nltk
```

## Project Structure

The project is structured as follows:

```
AI-Phishing-Email-Detection/
│
├── data/                 # Folder for datasets (tracked via DVC)
├── src/                  # Folder for source code
│   ├── preprocessing.py  # Script for preprocessing email data
│   ├── train.py          # Script for training ML models
│   ├── evaluate.py       # Script for evaluating models
│   └── deploy.py         # Script for deployment
├── models/               # Folder for storing trained models
├── dvc.yaml              # DVC pipeline configuration
├── README.md             # Project description
└── requirements.txt      # Python dependencies
```

## Key Features
- **Phishing Detection**: Uses machine learning models to detect phishing emails from legitimate emails.
- **Email Prioritization**: Rank emails based on their urgency and importance.
- **Natural Language Processing (NLP)**: Feature extraction from email content using NLP techniques.
- **MLOps Integration**: Seamlessly integrates with MLOps practices using DVC for dataset management and cloud deployment.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-Phishing-Email-Detection.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script:
   ```bash
   python src/preprocessing.py
   ```

4. Train the model:
   ```bash
   python src/train.py
   ```

5. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```

6. Deploy the model:
   ```bash
   python src/deploy.py
   ```

## Data Versioning with DVC

The `data/` folder is tracked using DVC, and datasets are stored remotely in cloud storage (AWS S3, Azure Blob, or Google Cloud Storage). To pull the dataset, run:
```bash
dvc pull
```

## Contact Information
If you have any questions or feedback about this project, feel free to contact:
- **Collaborators**: Kartik Aggarwal,Arjun Khadikar, Mohammad Ayan, Aryan Patel
