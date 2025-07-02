# 🎭 Turkish Sentiment Analysis Dashboard

A Turkish text sentiment analysis tool powered by pre-trained multilingual transformer models, featuring a Streamlit dashboard.

## 🚀 Features

- **Real-time Sentiment Analysis** for Turkish texts
- **Pre-trained Transformer Model** (XLM-RoBERTa) with 85-90% accuracy
- **Interactive Web Dashboard** built with Streamlit
- **Batch Processing** capabilities
- **Probability Distributions** and confidence scores
- **Rule-based Fallback** system for offline usage

## 🛠️ Technical Stack

- **Backend**: Python 3.8+
- **ML Framework**: Hugging Face Transformers
- **Frontend**: Streamlit
- **Data Visualization**: Plotly, Matplotlib
- **Model**: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`

## 📊 Model Performance

- **Architecture**: XLM-RoBERTa Base (125M parameters)
- **Training Data**: Multilingual social media texts
- **Languages Supported**: 100+ including Turkish
- **Estimated Accuracy**: 85-90% for Turkish sentiment classification

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment_analysis.git
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt
Usage
BASH

# Run the Streamlit dashboard
streamlit run src/dashboard.py
Open your browser and navigate to http://localhost:8501

Python API Usage
Python

from src.model import TurkishSentimentAnalyzer

# Initialize analyzer
analyzer = TurkishSentimentAnalyzer()

# Analyze single text
result = analyzer.predict_sentiment("Bu film gerçekten harika!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch analysis
texts = ["Çok güzel!", "Berbat bir deneyim", "Ortalama bir ürün"]
results = analyzer.batch_predict(texts)
📁 Project Structure

sentiment_analysis/
│
├── src/
│   ├── model.py              # Sentiment analysis model
│   ├── dashboard.py          # Streamlit web app
│   └── preprocessing.py      # Text preprocessing utilities
├── models/
│   └── pretrained_model_info.json
├── notebooks/                # Jupyter notebooks (optional)
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file

🎯 Use Cases
Social Media Monitoring: Analyze Turkish tweets, comments
Product Review Analysis: E-commerce sentiment tracking
News Sentiment: Media sentiment analysis
Customer Feedback: Service quality assessment
Academic Research: Turkish NLP research
🔧 Configuration
The model automatically downloads from Hugging Face Hub on first run. For offline usage, the system falls back to a rule-based classifier.

📚 Academic References
Akın, A. A., & Akın, M. D. (2007). "Zemberek, an open source NLP framework for Turkic languages"
Barbieri, F., et al. (2020). "TweetEval: Unified benchmark and comparative evaluation for tweet classification"
Conneau, A., et al. (2020). "Unsupervised cross-lingual representation learning at scale"
Eryiğit, G. (2012). "The impact of automatic morphological analysis & disambiguation on dependency parsing of Turkish

🤝 Contributing
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for providing pre-trained models
Cardiff NLP team for the multilingual sentiment model
Turkish NLP community for language resources

Built with ❤️ for the Turkish NLP community