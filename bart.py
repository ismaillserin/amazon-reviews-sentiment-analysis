import time
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from googletrans import Translator
from langdetect import detect
from transformers import pipeline


# Step 1: Set Up WebDriver
def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    )
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# Step 2: Get Product Links from Category Page
def get_product_links(category_url, driver):
    driver.get(category_url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_links = []

    for link in soup.find_all("a", class_="a-link-normal"):
        href = link.get("href")
        if href and "/dp/" in href:
            product_links.append("https://www.amazon.com.tr" + href.split("?")[0])

    return list(set(product_links))


# Step 3: Get Reviews from Product Page
def get_reviews(product_url, driver, max_pages=3):
    reviews = []
    driver.get(product_url)
    time.sleep(5)
    page = 1

    while page <= max_pages:
        print(f"Scraping reviews from {product_url} - Page {page}...")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        review_elements = soup.find_all("span", {"data-hook": "review-body"})

        for element in review_elements:
            review_text = element.text.strip()
            if review_text and review_text not in reviews:
                reviews.append(review_text)

        print(f"Scraped reviews on page {page}: {reviews}")

        # Try clicking "Next" button
        try:
            next_button = driver.find_element(By.CLASS_NAME, "a-last")
            if "a-disabled" in next_button.get_attribute("class"):
                break
            next_button.click()
            time.sleep(3)
        except Exception:
            break

        page += 1

    return reviews


# Step 4: Translate Reviews into English
def translate_reviews(reviews):
    translator = Translator()
    translated_reviews = []

    for review in reviews:
        if not review.strip():
            print("Skipping empty review")
            translated_reviews.append("")
            continue

        try:
            # **Önce Dil Tespiti Yap**
            lang = detect(review)
            if lang == "en":
                print(f"Skipping translation, already in English: {review}")
                translated_reviews.append(review)
                continue

            # **Google Translate API'yi Kullanarak Çeviri Yap**
            translated = translator.translate(review, src="tr", dest="en")
            if translated and translated.text:
                translated_reviews.append(translated.text)
                print(f"Original: {review} → Translated: {translated.text}")
            else:
                translated_reviews.append(review)  # Çeviri başarısızsa orijinal veriyi kaydet

            time.sleep(1)  # Google Translate engellemesini önlemek için

        except Exception as e:
            print(f"Translation error for: {review} → {e}")
            translated_reviews.append(review)  # Çeviri başarısızsa orijinal yorumu kullan

    return translated_reviews


# Step 5: Perform Sentiment Analysis with BART
def analyze_sentiment_bart(reviews):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = []

    for review in reviews:
        if not review.strip():
            print("Skipping empty review")
            continue

        try:
            print(f"Analyzing: {review}")
            output = classifier(review, candidate_labels=["positive", "negative", "neutral"])
            sentiment = output["labels"][0]
            score = output["scores"][0]

        except Exception as e:
            print(f"BART error for: {review} → {e}")
            sentiment, score = "neutral", 0.0

        results.append({"review": review, "sentiment": sentiment, "score": score})

    return results


# Step 6: Main Execution
def main():
    start_time = time.time()
    category_url = "https://www.amazon.com.tr/gp/bestsellers/sporting-goods/ref=zg_bs_sporting-goods_sm"
    driver = setup_driver()

    try:
        product_links = get_product_links(category_url, driver)
        print(f"Found {len(product_links)} products.")

        all_reviews = []
        for idx, product_link in enumerate(product_links):
            print(f"Scraping reviews for Product {idx + 1}: {product_link}...")
            reviews = get_reviews(product_link, driver)
            all_reviews.extend(reviews)
            print(f"Collected {len(reviews)} reviews.")

        print("Translating reviews into English...")
        translated_reviews = translate_reviews(all_reviews)

        print("Performing sentiment analysis with BART...")
        sentiment_results = analyze_sentiment_bart(translated_reviews)

        # **CSV'ye Kaydetme - Önce Terminalde Veriyi Kontrol Et**
        if sentiment_results:
            sentiment_df = pd.DataFrame(sentiment_results)
            print("\n✅ Final Data to be saved in CSV:")
            print(sentiment_df)

            sentiment_df.to_csv("amazon_reviews_bart_sentiment.csv", index=False)
            print("✅ Sentiment analysis complete. Results saved to 'amazon_reviews_bart_sentiment.csv'.")
        else:
            print("❌ No reviews to save! Check scraping and translation.")

    finally:
        driver.quit()
        end_time = time.time()
        print(f"Process completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
