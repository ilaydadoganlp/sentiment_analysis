import pandas as pd
import re
import string
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

# NLTK data download (ilk çalıştırmada)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TurkishPreprocessor:
    def __init__(self):
        """
        Türkçe metin ön işleme sınıfı
        Referans: Eryiğit, G. (2012). "The impact of automatic morphological analysis & disambiguation on dependency parsing of Turkish"
        """
        # Türkçe stopwords
        self.turkish_stopwords = set([
            'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'bir', 'birşey', 'biz', 'bu', 'çok',
            'da', 'daha', 'de', 'diye', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise',
            'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'niye', 'niçin',
            'o', 'sanki', 'şey', 'siz', 've', 'veya', 'ya', 'yani'
        ])
        
        # Emoticon patterns
        self.emoticons = {
            'positive': [':-)', ':)', '=)', ':D', ':-D', '=D', ':P', ':-P'],
            'negative': [':-(', ':(', '=(', ':/', ':-/', ':|', ':-|']
        }
    
    def clean_text(self, text):
        """Temel metin temizleme"""
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # HTML tagları kaldır
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL'leri kaldır
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Email adreslerini kaldır
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Mention'ları kaldır (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Hashtag'leri temizle (#hashtag -> hashtag)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        return text.strip()
    
    def handle_emoticons(self, text):
        """Emoticon'ları kelimeye çevir"""
        for emotion, emoticons in self.emoticons.items():
            for emoticon in emoticons:
                text = text.replace(emoticon, f' {emotion}_emoticon ')
        return text
    
    def remove_punctuation(self, text):
        """Noktalama işaretlerini kaldır (Türkçe karakterleri koru)"""
        # Türkçe karakterleri koruyarak noktalama kaldır
        text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ]', ' ', text)
        return text
    
    def remove_stopwords(self, text):
        """Stop word'leri kaldır"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.turkish_stopwords]
        return ' '.join(filtered_words)
    
    def normalize_whitespace(self, text):
        """Fazla boşlukları temizle"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess_text(self, text):
        """Tam preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.handle_emoticons(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.normalize_whitespace(text)
        return text
    
    def preprocess_dataframe(self, df):
        """DataFrame'i işle"""
        df_processed = df.copy()
        df_processed['text_processed'] = df_processed['text'].apply(self.preprocess_text)
        return df_processed

# Test et
if __name__ == "__main__":
    # Sample test
    preprocessor = TurkishPreprocessor()
    
    # Ham veriyi yükle
    df = pd.read_csv("data/raw/sample_reviews.csv")
    
    # İşle
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Kaydet
    df_processed.to_csv("data/processed/processed_reviews.csv", index=False, encoding='utf-8')
    
    print("Önce ve sonra karşılaştırması:")
    for i in range(3):
        print(f"\nÖncesi: {df.iloc[i]['text']}")
        print(f"Sonrası: {df_processed.iloc[i]['text_processed']}")