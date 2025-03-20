# Sentiment-Analysis-Using-LSTM-

## 📌 Project Overview  
This project focuses on **Sentiment Analysis of news articles** using **LSTM and GRU-based neural networks**. The dataset consists of **209,527 news articles** collected from HuffPost, covering topics like **U.S. News, Politics, and Comedy**.  

The project includes **data exploration, preprocessing, data visualization, and model development** using **LSTMs, Bidirectional LSTMs, and GRUs**, followed by **hyperparameter tuning** to optimize performance.

🔹 **Due to academic policies, the source code is not publicly available.**  
🔹 **However, this repository provides insights into the approach, methodology, and results.**  

---

## 📂 Dataset Details  
- The dataset consists of **209,527 news articles** with attributes:  
  - **Headline** (Title of the article)  
  - **Category** (e.g., Politics, Comedy, Sports, etc.)  
  - **Short Summary** (Brief content description)  
  - **Author Name**  
  - **Publication Date**  
- The dataset spans **multiple years**, making it ideal for **NLP tasks** such as **sentiment analysis, classification, and content summarization**.  

---

## 🛠️ Tech Stack  
- **Deep Learning Framework:** PyTorch / TensorFlow  
- **NLP Preprocessing:** NLTK, BERT Tokenization  
- **Neural Networks Used:** LSTM, Bidirectional LSTM, GRU  
- **Optimization Techniques:** Random Search Hyperparameter Tuning  
- **Libraries Used:**  
  - `NLTK` – Tokenization, stopword removal  
  - `Torchtext` – Text preprocessing & embeddings  
  - `Matplotlib & Seaborn` – Data visualization  
  - `Pandas & NumPy` – Data handling  

---

## 🔍 Model Training & Optimization  

### **1️⃣ Baseline LSTM Model**  
- Implemented an **LSTM-based model** for sentiment classification.  
- **Architecture Summary:**  
  ✅ **Embedding Layer** (Word embeddings for feature representation)  
  ✅ **LSTM Layer** (Captures sequential patterns in text)  
  ✅ **Batch Normalization & Dropout** (Prevents overfitting)  
  ✅ **Fully Connected Layer (Linear Layer) for Final Classification**  

### **2️⃣ Hyperparameter Tuning (Random Search)**  
- Used **random search** to optimize key parameters such as:  
  ✅ Learning Rate  
  ✅ Hidden Size  
  ✅ Dropout Rate  
  ✅ Batch Size  
  ✅ Number of Epochs  

### **3️⃣ Improved Model: Bidirectional LSTM & GRU**  
- Enhanced the **base LSTM model** by introducing:  
  ✅ **Bidirectional LSTMs** – Captures context in both forward and backward directions.  
  ✅ **GRU (Gated Recurrent Unit)** – A lighter and efficient alternative to LSTM.  
  ✅ **ReLU Activation Function** – Improves learning efficiency.  
  ✅ **Batch Normalization & Dropout** – Helps with better generalization.  

### **4️⃣ Further Optimization Using Hyperparameter Tuning**  
- The **Bidirectional GRU model** was further fine-tuned using **Random Search Hyperparameter Tuning**.  

---

## 📊 Results & Key Takeaways  

### **Best Hyperparameters (After Random Search Tuning):**  
| Hyperparameter | Best Value |
|---------------|------------|
| **Learning Rate** | 0.000327 |
| **Hidden Size** | 64 |
| **Dropout Rate** | 0.6 |
| **Batch Size** | 32 |
| **Epochs** | 5 |

### **Training & Validation Performance:**  
- The **Improved Model (Bidirectional GRU)** achieved **higher accuracy** and **lower validation loss** compared to the baseline LSTM.  
- **Early stopping was used** to prevent overfitting and stabilize the model.  

---

## 🚀 How to Use This Project  
🚫 **Due to academic regulations, the source code is not publicly available.**  
🔹 However, if you are interested in discussing the methodology or results, feel free to reach out!  

---

## 📌 Future Improvements  
- 🔹 Experiment with **Transformer-based models** (e.g., BERT, GPT) for better text representations.  
- 🔹 Fine-tune **pre-trained embeddings** (GloVe, FastText) to improve classification.  
- 🔹 Deploy the model as a **real-time sentiment analysis API**.  

---

## 📜 License  
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  

---

## 🔗 Connect with Me  
📧 Email: sindhuravel@gmail.com  
🔗 LinkedIn: [Sindhura Velmurugan](https://www.linkedin.com/in/sindhura-velmurugan/) 

---

### ⭐ If you find this project interesting, feel free to **star** this repository!
