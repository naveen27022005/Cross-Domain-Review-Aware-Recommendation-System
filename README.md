# Cross-Domain Review-Aware Recommendation System

## 📌 Project Overview
This project implements a **Cross-Domain Review-Aware Recommendation System** that leverages user reviews from multiple domains (e.g., books, movies) to improve recommendation quality. By learning shared representations across domains, the system addresses data sparsity and enhances personalization.

The project focuses on data preprocessing, review aggregation, feature extraction, and model experimentation for building an effective recommender system.

---

## 🎯 Objectives
- To analyze and preprocess large-scale review datasets
- To build a cross-domain recommendation pipeline
- To explore how user reviews improve recommendation accuracy
- To ensure scalability and reproducibility of experiments

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Environment:** Jupyter Notebook / Python scripts  
- **Version Control:** Git & GitHub  

## 📊 Dataset Information
⚠️ **Note:**  
Large datasets and trained model files are **not included in this repository** due to GitHub size limitations.

- Review datasets (e.g., `.jsonl`, `.csv`)
- Trained models (`.pt`)
- Processed data files (`.parquet`)

📦 These files can be regenerated using the provided notebooks or made available upon request.

---

## 🧠 Model Description
The recommendation system is designed to learn user preferences by leveraging **review-based features across multiple domains**. The model captures semantic information from textual reviews and combines it with user–item interaction patterns to generate personalized recommendations.

Key aspects of the model include:
- **Cross-Domain Learning:** Knowledge is transferred between different domains (e.g., books and movies) to mitigate data sparsity.
- **Review-Aware Representation:** User and item representations are enriched using aggregated review information.
- **Scalable Architecture:** The pipeline supports large-scale datasets and modular experimentation.

Multiple baseline and machine learning–based approaches were experimented with to evaluate performance and generalization across domains.

---

## 📊 Evaluation Metrics
The performance of the recommendation model is evaluated using standard metrics to assess both accuracy and ranking quality.

### Metrics Used
- **Precision@K:** Measures the proportion of relevant items among the top-K recommended items.
- **Recall@K:** Evaluates the ability of the model to retrieve relevant items within the top-K recommendations.
- **F1-Score@K:** Harmonic mean of Precision@K and Recall@K, providing a balanced evaluation.
- **Mean Average Precision (MAP):** Assesses ranking quality by considering the order of relevant items.
- **Normalized Discounted Cumulative Gain (NDCG):** Measures ranking effectiveness by giving higher importance to top-ranked items.

### Evaluation Strategy
- The dataset is split into training and testing sets.
- Models are evaluated on unseen user–item interactions.
- Metrics are computed at multiple cutoff values (e.g., K = 5, 10).

---

## 📈 Results Summary
The experimental results indicate that incorporating **cross-domain review information** improves recommendation performance compared to single-domain baselines, particularly in scenarios with limited interaction data.

Detailed metric values and comparison tables are available in the `detailed_results.json` directory.


### 👤 Author
Naveen E

