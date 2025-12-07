# 🎬 CS410 MovieMatch  
A TF-IDF Powered Movie Search Engine

MovieMatch is a lightweight full-stack search engine that lets users query Netflix movie & show descriptions and receive the most relevant results using a custom TF-IDF ranking pipeline.  
Built with **Python + Flask** and a simple **HTML/JS frontend**.

---

## 🚀 Features

- Search through Netflix movies using natural-language keywords  
- Custom TF-IDF + BM25-style relevance scoring  
- Flask backend with a `/search` GET endpoint  
- Clean and simple frontend UI (`frontend.html`)  
- Fully local — no external APIs required  


---

## 🛠 Installation & Setup

### 1. Install Dependencies

Run:

```bash
pip install flask pandas numpy nltk

### 2. Prepare the dataset

Run:

```bash
python dataset_preprocessing_script.py

This will produce: netflix_cleaned.csv


---

## Run the Backend
```bash
python app.py
