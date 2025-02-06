import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize VADER Sentiment Analyzer
vader = SentimentIntensityAnalyzer()

# Step 1: Load the dataset
data = [
    {"review": "As shown sharply, but the handle is not very solid because it is plastic", "vader_sentiment": "neutral", "vader_score": -0.2479, "true_sentiment": "negative"},
    {"review": "LA garbage you know.I wish I had to claim with that money.It would come to the same account again", "vader_sentiment": "positive", "vader_score": 0.4019, "true_sentiment": "negative"},
    {"review": "I was very disappointed according to this product image.", "vader_sentiment": "negative", "vader_score": -0.5256, "true_sentiment": "negative"},
    {"review": "very well as promised", "vader_sentiment": "positive", "vader_score": 0.6318, "true_sentiment": "positive"},
    {"review": "The product is nice but amazonun packaging disgrace", "vader_sentiment": "negative", "vader_score": -0.5267, "true_sentiment": "negative"},
    {"review": "It sounded big in size, but it keeps it very warm and very good at its duration.", "vader_sentiment": "positive", "vader_score": 0.7952, "true_sentiment": "positive"},
    {"review": "Quality 10 number 5 stars I recommend to everyone ...", "vader_sentiment": "positive", "vader_score": 0.4215, "true_sentiment": "positive"},
    {"review": "When I got the product, I filled it for drinking water.However, when I turned it side by coincidence, the water began to leak leaks, especially when the rear holding hook was closed.There was also a small black stain in the product.I started the return process.I will change it with a new one.Let's see if I will encounter the same problem. The product was changed quickly and delivered to me.That's why I use the Amazon site.They didn't even cause any problems once.",
     "vader_sentiment": "negative", "vader_score": -0.7269, "true_sentiment": "negative"},
    {"review": "Beautiful and useful.", "vader_sentiment": "positive", "vader_score": 0.7783, "true_sentiment": "positive"},
    {"review": "Thanks Amazon.I liked the ice cube into the successful water and provided cooling for a long time.", "vader_sentiment": "positive", "vader_score": 0.8591, "true_sentiment": "positive"},
    {"review": "The seller immediately reached my cargo side without any problems.The product is absolutely original.The seller immediately solved the small billing problem.I make sure that the next shopping is from the same seller.", "vader_sentiment": "positive", "vader_score": 0.5859, "true_sentiment": "positive"},
    {"review": "There is known everywhere in the market and they sell poor quality products by saying real quality is absolutely original and giving the right to a product that gives the right to register for life warranty and I recommend using it.", "vader_sentiment": "positive", "vader_score": 0.5657, "true_sentiment": "positive"},
    {"review": "I ordered Saturday evening on Monday noon.The product came completely well packaged.", "vader_sentiment": "positive", "vader_score": 0.3384, "true_sentiment": "positive"},
    {"review": "It smells so bad", "vader_sentiment": "negative", "vader_score": -0.6696, "true_sentiment": "negative"},
    {"review": "Very good product I recommend it to everyone.", "vader_sentiment": "positive", "vader_score": 0.7146, "true_sentiment": "positive"},
    {"review": "Great perfect", "vader_sentiment": "positive", "vader_score": 0.8316, "true_sentiment": "positive"},
    {"review": "Fake trend", "vader_sentiment": "negative", "vader_score": -0.4767, "true_sentiment": "negative"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Evaluate VADER Predictions
true_labels = df["true_sentiment"]
predicted_labels = df["vader_sentiment"]

# Step 3: Calculate Accuracy, Precision, Recall, and F1-Score
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")

# Step 4: Print Results
print("\n📌 **Evaluation Results**:")
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1-Score: {f1:.2f}")

# Step 5: Display Classification Report
print("\n🔍 **Detailed Report:**\n")
print(classification_report(true_labels, predicted_labels))

# Step 6: Show Misclassified Cases
df_misclassified = df[df["true_sentiment"] != df["vader_sentiment"]]
if not df_misclassified.empty:
    print("\n🚨 **Misclassified Reviews:**")
    for _, row in df_misclassified.iterrows():
        print(f"\n❌ **Review:** {row['review']}")
        print(f"   → **True Sentiment:** {row['true_sentiment']}")
        print(f"   → **Predicted Sentiment:** {row['vader_sentiment']}")
        print(f"   → **VADER Score:** {row['vader_score']}")
else:
    print("\n✅ No misclassified reviews!")

