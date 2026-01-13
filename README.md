# NLP-Supported Framework for Requirements Extraction (Hausa-to-English)

**Thesis Title:** An NLP-Supported Framework for Requirements Extraction and Specification in Low-Resource Languages  
**Language Focus:** Hausa (Low-Resource) $\rightarrow$ English (Structured Specification)

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20HuggingFace-orange)

##  Project Overview
This repository contains the source code, datasets, and experimental scripts for an MSc/PhD thesis focused on automating **Software Requirements Engineering (RE)** for low-resource languages. 

The framework successfully implements an end-to-end pipeline that:
1.  **Extracts** software requirements from unstructured Hausa text.
2.  **Classifies** them into Functional (FR) and Non-Functional (NFR) requirements using a **Multi-View Fusion Model** (mBERT + XLM-R).
3.  **Translates** them into technical English using a fine-tuned **NLLB-200** model.
4.  **Generates** a compliant **IEEE 830-1998** Software Requirements Specification (SRS) document.

##  Key Features
* **Hybrid Fusion Architecture:** Combines mBERT (global context) and XLM-R (local syntax) to solve the "Majority Class" problem in requirement classification.
* **Domain-Adaptive Translation:** Fine-tuned NLLB-200 model that correctly handles engineering terminology (e.g., *Tsaro* $\rightarrow$ *Security*, *Shiga* $\rightarrow$ *Log in*).
* **Automated SRS Generation:** Python-based module that auto-formats predictions into industry-standard documentation.
* **LaBSE Semantic Filtering:** Uses Language-Agnostic BERT Sentence Embeddings to clean noisy parallel datasets.

##  Performance Highlights

### 1. Classification (Hausa Requirements)
The proposed **Fusion Model** significantly outperformed single-encoder baselines, particularly in identifying safety-critical constraints.

| Model Architecture | Global Accuracy | NFR Recall (Safety Check) |
| :--- | :--- | :--- |
| **XLM-R (Baseline)** | 65.54% | 0.00 (Failed) |
| **mBERT (Baseline)** | 68.47% | 0.10 (Poor) |
| **Proposed Fusion Model** | **73.22%** | **0.71 (Strong)** |

### 2. Translation (Hausa $\rightarrow$ English)
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **BLEU** | 14.14 | Adequate fluency for low-resource data. |
| **LaBSE** | **0.74** | High semantic preservation of engineering intent. |

##  Repository Structure

## 📂 Project Workflow & File Description

This repository is organized sequentially to replicate the entire Requirements Engineering pipeline, from raw data collection to final documentation.

### **Phase 1: Data Preparation**
* **`01_Data_Collection.ipynb`**
    * **Purpose:** Handles the scraping of raw Hausa text from technical forums and the synthesis of parallel data.
    * **Key Output:** Generates the raw, uncleaned dataset containing mixed-quality Hausa-English pairs.
    
* **`02_Preprocessing.ipynb`**
    * **Purpose:** Performs rigorous cleaning, tokenization, and quality assurance.
    * **Methodology:** Implements **LaBSE (Language-Agnostic BERT Sentence Embeddings)** to compute semantic similarity scores between sentence pairs. Pairs with a similarity score below 0.6 are automatically discarded to remove "noisy" translations.
    * **Key Output:** `clean_dataset.csv` (High-quality, aligned data ready for training).

### **Phase 2: Model Training**
* **`03_Model_Training_Fusion.ipynb`**
    * **Purpose:** Trains the classification model to distinguish between Functional (FR) and Non-Functional (NFR) requirements.
    * **Architecture:** Implements the **Multi-View Fusion** strategy, loading both `mBERT` and `XLM-R` simultaneously. It freezes their base layers and trains a shared "Fusion Layer" to combine their embeddings.
    * **Key Output:** `fusion_model_state.bin` (The trained classifier weights).

* **`04_Model_Training_NLLB.ipynb`**
    * **Purpose:** Fine-tunes the translation model to convert Hausa requirements into technical English.
    * **Methodology:** Uses **LoRA (Low-Rank Adaptation)** to fine-tune the massive **NLLB-200** model on consumer hardware (e.g., Colab T4 GPU). It forces the model to learn engineering terminology (e.g., *Shiga* -> *Log in*).
    * **Key Output:** `nllb_lora_weights/` (Adapter weights for translation).

### **Phase 3: Evaluation & Deployment**
* **`05_Evaluation.ipynb`**
    * **Purpose:** Generates all statistical proofs and graphs found in the thesis.
    * **Metrics:** Calculates Accuracy, F1-Score, and generates the Confusion Matrix for classification. For translation, it computes BLEU, chrF, and LaBSE scores.
    * **Visuals:** Produces the t-SNE scatter plots showing the separation of FR/NFR clusters.

* **`06_SRS_Generator.py`**
    * **Purpose:** The final application script.
    * **Function:** Takes a raw text file of Hausa requirements, runs it through the Fusion and NLLB models, and automatically formats the output into a structured **IEEE 830-1998** PDF or DOCX file.
    * **Usage:** Run this script to demonstrate the system end-to-end.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* Google Colab (Recommended for GPU support) or a local machine with NVIDIA GPU (16GB+ VRAM).

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Hausa-NLP-Thesis-Framework.git](https://github.com/YOUR_USERNAME/Hausa-NLP-Thesis-Framework.git)
    cd Hausa-NLP-Thesis-Framework
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline
To run the full extraction and generation process on a new Hausa text file:
```python
python 06_SRS_Generator.py --input "hausa_requirements.txt" --output "Final_SRS.pdf"
Abdullah Ibrahim Haruna, "An NLP-Supported Framework for Requirements Extraction and Specification in Low-Resource Languages," Nile University of Nigeria, 2025.
