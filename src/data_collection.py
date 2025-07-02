import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

class TurkishDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def collect_sample_data(self):
        """
        Başlangıç için manuel olarak hazırlanmış örnek veri seti
        Gerçek projede web scraping veya API kullanılacak
        """
        sample_data = [
            {"text": "Bu film gerçekten harika, çok beğendim!", "sentiment": "positive"},
            {"text": "Berbat bir film, zamanımı boşa harcadım.", "sentiment": "negative"},
            {"text": "Film ortalama, ne iyi ne kötü.", "sentiment": "neutral"},
            {"text": "Oyuncular çok iyiydi ama senaryo zayıftı.", "sentiment": "neutral"},
            {"text": "Muhteşem bir yapım, herkese tavsiye ederim!", "sentiment": "positive"},
            {"text": "Çok sıkıcıydı, yarıda bıraktım.", "sentiment": "negative"},
            # Daha fazla örnek ekleyeceğiz...
        ]
        
        df = pd.DataFrame(sample_data)
        return df
    
    def save_raw_data(self, df, filename):
        """Ham veriyi kaydet"""
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Veri kaydedildi: {filepath}")

# Test et
if __name__ == "__main__":
    collector = TurkishDataCollector()
    df = collector.collect_sample_data()
    collector.save_raw_data(df, "sample_reviews.csv")