import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from datetime import datetime

class TurkishSentimentAnalyzer:
    def __init__(self):
        """
        Türkçe Sentiment Analiz Modeli
        
        Referanslar:
        - Kaynar, O., et al. (2016). "Sentiment analysis with machine learning techniques in Turkish language"
        - Akın, A. A., & Akın, M. D. (2007). "Zemberek, an open source NLP framework for Turkic languages"
        - Çoban, Ö., et al. (2015). "Sentiment analysis for Turkish Twitter feeds"
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigram ve bigram
            min_df=2,
            max_df=0.95
        )
        
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        self.trained_models = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, df):
        """Veriyi model için hazırla"""
        # Label encoding
        label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
        
        X = df['text_processed'].values
        y = df['sentiment'].map(label_mapping).values
        
        return X, y, label_mapping
    
    def train_models(self, X, y):
        """Tüm modelleri eğit ve karşılaştır"""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorization
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        
        print("Model Eğitimi Başlıyor...\n")
        
        for name, model in self.models.items():
            print(f"🔄 {name.title()} eğitiliyor...")
            
            # Model eğitimi
            model.fit(X_train_vec, y_train)
            
            # Tahmin
            y_pred = model.predict(X_test_vec)
            
            # Değerlendirme
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"✅ {name.title()} - Doğruluk: {accuracy:.4f}")
            
            # En iyi modeli bul
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = name
        
        self.trained_models = results
        
        print(f"\n🏆 En İyi Model: {self.best_model.title()} (Doğruluk: {self.best_score:.4f})")
        
        return results
    
    def get_detailed_results(self):
        """Detaylı model sonuçları"""
        if not self.trained_models:
            print("❌ Henüz eğitilmiş model yok!")
            return
        
        print("\n📊 DETAYLI MODEL SONUÇLARI\n" + "="*50)
        
        for name, result in self.trained_models.items():
            print(f"\n🔍 {name.upper()}")
            print("-" * 30)
            print(f"Doğruluk: {result['accuracy']:.4f}")
            print("\nSınıflandırma Raporu:")
            print(classification_report(
                result['y_test'], 
                result['y_pred'],
                target_names=['Negative', 'Neutral', 'Positive']
            ))
    
    def predict_sentiment(self, text):
        """Tek metin için sentiment tahmini"""
        if not self.best_model:
            return "❌ Model henüz eğitilmemiş!"
        
        # Preprocessing (burada basit, gerçekte preprocessing.py kullanılacak)
        text_processed = text.lower()
        
        # Vectorize
        text_vec = self.vectorizer.transform([text_processed])
        
        # Tahmin
        model = self.trained_models[self.best_model]['model']
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        return {
            'text': text,
            'sentiment': sentiment_labels[prediction],
            'confidence': max(probability),
            'probabilities': {
                'negative': probability[0],
                'neutral': probability[1],
                'positive': probability[2]
            }
        }
    
    def save_models(self):
        """Modelleri kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Vectorizer'ı kaydet
        joblib.dump(self.vectorizer, f'models/vectorizer_{timestamp}.pkl')
        
        # En iyi modeli kaydet
        if self.best_model:
            best_model_obj = self.trained_models[self.best_model]['model']
            joblib.dump(best_model_obj, f'models/best_model_{self.best_model}_{timestamp}.pkl')
            
            # Metadata kaydet
            metadata = {
                'best_model': self.best_model,
                'best_score': self.best_score,
                'timestamp': timestamp,
                'all_scores': {name: result['accuracy'] for name, result in self.trained_models.items()}
            }
            
            import json
            with open(f'models/metadata_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Modeller kaydedildi: models/ klasörü")

# Test et
if __name__ == "__main__":
    # Veriyi yükle
    df = pd.read_csv("data/processed/processed_reviews.csv")
    
    # Model oluştur
    analyzer = TurkishSentimentAnalyzer()
    
    # Veriyi hazırla
    X, y, label_mapping = analyzer.prepare_data(df)
    
    print(f"📊 Veri Seti Özeti:")
    print(f"Toplam örnek: {len(X)}")
    print(f"Sınıf dağılımı: {pd.Series(y).value_counts().to_dict()}")
    
    # Modelleri eğit
    results = analyzer.train_models(X, y)
    
    # Detaylı sonuçlar
    analyzer.get_detailed_results()
    
    # Test tahminleri
    print("\n🧪 TEST TAHMİNLERİ:")
    test_texts = [
        "Bu film gerçekten harika!",
        "Çok kötü bir deneyimdi.",
        "Fena değil, ortalama."
    ]
    
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"'{text}' -> {result['sentiment']} (Güven: {result['confidence']:.3f})")
    
    # Modelleri kaydet
    analyzer.save_models()