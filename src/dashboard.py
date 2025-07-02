import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Local imports
from model import TurkishSentimentAnalyzer

class SentimentDashboard:
    def __init__(self):
        """
        Türkçe Sentiment Analizi Dashboard (Hazır Eğitilmiş Model)
        """
        self.load_analyzer()
        self.create_demo_data()
    
    def load_analyzer(self):
        """Sentiment analyzer'ı yükle"""
        with st.spinner("🤖 Model yükleniyor..."):
            self.analyzer = TurkishSentimentAnalyzer()
        
        if self.analyzer.model_loaded:
            st.success("✅ Hazır eğitilmiş model başarıyla yüklendi!")
        else:
            st.warning("⚠️ Kural tabanlı model kullanılıyor (internet bağlantısı gerekebilir)")
    
    def create_demo_data(self):
        """Dashboard için demo veri seti"""
        self.demo_texts = [
            "Bu film gerçekten harika, çok beğendim!",
            "Muhteşem bir yapım, herkese tavsiye ederim!",
            "Seni çok seviyorum aşkım",
            "Berbat bir film, zamanımı boşa harcadım",
            "Seni öldürürüm", 
            "Senden nefret ediyorum",
            "Film ortalama, ne iyi ne kötü",
            "Bu ürün fena değil ama çok da iyi sayılmaz",
            "Muhteşem bir performans! Tebrikler!",
            "Bu adaletsizlik beni çok üzdü"
        ]
    
    def predict_single_text(self, text):
        """Tek metin için prediction"""
        return self.analyzer.predict_sentiment(text)
    
    def create_sentiment_chart(self, probabilities):
        """Sentiment probability bar chart"""
        sentiments = ['Negative', 'Neutral', 'Positive']
        probs = [probabilities['negative'], probabilities['neutral'], probabilities['positive']]
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        fig = go.Figure(data=[
            go.Bar(x=sentiments, y=probs, marker_color=colors, text=[f'{p:.3f}' for p in probs], textposition='auto')
        ])
        
        fig.update_layout(
            title="Sentiment Probability Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def analyze_demo_batch(self):
        """Demo metinlerin toplu analizi"""
        results = []
        for text in self.demo_texts:
            result = self.analyzer.predict_sentiment(text)
            result['text'] = text
            results.append(result)
        return results
    
    def create_batch_analysis_chart(self, results):
        """Toplu analiz sonuçları grafiği"""
        df = pd.DataFrame(results)
        
        # Sentiment counts
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Demo Texts - Sentiment Distribution",
            color_discrete_map={
                'positive': '#6bcf7f',
                'negative': '#ff6b6b',
                'neutral': '#ffd93d'
            }
        )
        
        return fig
    
    def main(self):
        """Ana dashboard fonksiyonu"""
        # Sayfa konfigürasyonu
        st.set_page_config(
            page_title="Turkish Sentiment Analysis",
            page_icon="🎭",
            layout="wide"
        )
        
        # Başlık
        st.title("🎭 Turkish Sentiment Analysis Dashboard")
        st.markdown("**Powered by Pre-trained Multilingual Model**")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("📊 Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🔍 Single Text Analysis", "📊 Batch Analysis", "ℹ️ Model Information"]
        )
        
        if page == "🔍 Single Text Analysis":
            self.single_text_page()
        elif page == "📊 Batch Analysis":
            self.batch_analysis_page()
        elif page == "ℹ️ Model Information":
            self.model_info_page()
    
    def single_text_page(self):
        """Tek metin analizi sayfası"""
        st.header("🔍 Single Text Sentiment Analysis")
        
        # Örnek metinler
        st.subheader("💡 Quick Examples")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("😊 Positive Example"):
                st.session_state.example_text = "Bu film gerçekten harika, çok beğendim!"
        
        with col2:
            if st.button("😞 Negative Example"):
                st.session_state.example_text = "Senden nefret ediyorum!"
        
        with col3:
            if st.button("😐 Neutral Example"):
                st.session_state.example_text = "Bu konu hakkında emin değilim."
        
        # Text input
        user_text = st.text_area(
            "Enter Turkish text to analyze:",
            value=st.session_state.get('example_text', ''),
            placeholder="Örnek: Seni çok seviyorum aşkım!",
            height=100
        )
        
        if st.button("🚀 Analyze Sentiment", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing..."):
                    result = self.predict_single_text(user_text)
                
                # Sonuçları göster
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Results")
                    
                    # Sentiment badge
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    method = result.get('method', 'unknown')
                    
                    if sentiment == 'positive':
                        st.success(f"😊 **{sentiment.upper()}** (Confidence: {confidence:.3f})")
                    elif sentiment == 'negative':
                        st.error(f"😞 **{sentiment.upper()}** (Confidence: {confidence:.3f})")
                    else:
                        st.warning(f"😐 **{sentiment.upper()}** (Confidence: {confidence:.3f})")
                    
                    st.info(f"🔧 Method: {method}")
                    
                    # Detaylı probabilities
                    st.write("**Detailed Probabilities:**")
                    for sent, prob in result['probabilities'].items():
                        st.write(f"• {sent.title()}: {prob:.4f}")
                
                with col2:
                    # Probability chart
                    fig = self.create_sentiment_chart(result['probabilities'])
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    def batch_analysis_page(self):
        """Toplu analiz sayfası"""
        st.header("📊 Batch Analysis - Demo Texts")
        
        if st.button("🔄 Analyze All Demo Texts", type="primary"):
            with st.spinner("Analyzing demo texts..."):
                results = self.analyze_demo_batch()
            
            # Özet istatistikler
            st.subheader("📈 Summary Statistics")
            
            df = pd.DataFrame(results)
            sentiment_counts = df['sentiment'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Texts", len(results))
            
            with col2:
                st.metric("Positive", sentiment_counts.get('positive', 0))
            
            with col3:
                st.metric("Negative", sentiment_counts.get('negative', 0))
            
            # Pie chart
            fig = self.create_batch_analysis_chart(results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detaylı sonuçlar
            st.subheader("📝 Detailed Results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['text'][:50]}..."):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Full Text:** {result['text']}")
                        st.write(f"**Sentiment:** {result['sentiment'].upper()}")
                        st.write(f"**Confidence:** {result['confidence']:.4f}")
                        st.write(f"**Method:** {result.get('method', 'unknown')}")
                    
                    with col2:
                        # Mini chart
                        mini_fig = self.create_sentiment_chart(result['probabilities'])
                        mini_fig.update_layout(height=200)
                        st.plotly_chart(mini_fig, use_container_width=True)
    
    def model_info_page(self):
        """Model bilgileri sayfası"""
        st.header("ℹ️ Model Information")
        
        model_info = self.analyzer.get_model_info()
        
        st.subheader("🤖 Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Name:**", model_info['model_name'])
            st.write("**Status:**", "✅ Loaded" if model_info['model_loaded'] else "⚠️ Fallback")
            st.write("**Method:**", model_info['method'])
        
        with col2:
            st.write("**Description:**", model_info['description'])
            st.write("**Estimated Accuracy:**", model_info['accuracy_estimate'])
            st.write("**Last Updated:**", model_info['timestamp'])
        
        # Teknik detaylar
        st.subheader("🔧 Technical Details")
        
        if model_info['model_loaded']:
            st.success("""
            **Pre-trained Transformer Model:**
            - Architecture: XLM-RoBERTa Base
            - Training Data: Multilingual social media texts
            - Languages: 100+ languages including Turkish
            - Parameters: ~125M
            - Fine-tuned for sentiment analysis
            """)
        else:
            st.warning("""
            **Rule-based Fallback Model:**
            - Method: Keyword matching with Turkish sentiment lexicon
            - Coverage: Common positive/negative expressions
            - Accuracy: Lower than transformer model
            - Usage: When transformer model unavailable
            """)
        
        # Test örnekleri
        st.subheader("🧪 Model Performance Examples")
        
        test_cases = [
            ("Positive Strong", "Bu film muhteşem, kesinlikle izlenmeli!"),
            ("Negative Strong", "Bu ürün berbat, para kaybı!"),
            ("Neutral", "Bu konu hakkında net bir görüşüm yok."),
            ("Mixed", "Film güzeldi ama biraz uzundu.")
        ]
        
        for case_type, text in test_cases:
            result = self.analyzer.predict_sentiment(text)
            
            with st.expander(f"{case_type}: {text}"):
                sentiment_emoji = "😊" if result['sentiment'] == 'positive' else ("😞" if result['sentiment'] == 'negative' else "😐")
                st.write(f"{sentiment_emoji} **Result:** {result['sentiment'].upper()} (Confidence: {result['confidence']:.3f})")

# Ana uygulama
if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.main()