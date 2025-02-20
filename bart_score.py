import pandas as pd
from sklearn.metrics import f1_score, classification_report

# ✅ Burada test etmek istediğin verileri MANUEL gireceksin!
data = [
    ("Bu satıcı çok özensiz ve dikkatsiz , vidaları olmayan bir ürün gönderdi, paketlemeden önce kontrol etmemiş.", "negative", 0.5076263546943665),
    ("soğuk içecek termosu diye geçiyor fakat kahvede içiyorum ilk üç beş denememde ağzım ciddi yandı fakat artık alıştım eğer benim gibi sıcak içecek için kullanacaksanız deneme sürecinde ağzınız biraz yanacak ama işini hakkıyla yapıyor söylenen değerler doğru 6 7 saat kaynar tutuyor 12 saat soğuk", "negative", 0.5701226592063904),
    ("Daha iyi termos kullanmadım stanley number 1", "neutral", 0.5823155045509338),
    ("stanley kalitesi. başka yorum yok", "negative", 0.5720131397247314),
    ("günlük taşımak için ideal boy", "positive", 0.8173773884773254),
    ("Kesinlikle almaya değer", "negative", 0.602785587310791),
    ("Stanley kalitesi ortada. Çokta yoruma gerek yok. Harika bir ürün. Teşekkürler.", "negative", 0.5608403086662292),
    ("ORJİNAL", "negative", 0.5063351988792419),
    ("Suyum buz gibi", "negative", 0.6973511576652527),
    ("Es sorprendente el tiempo que dura exactamente igual de frío o caliente que cuando lo llenas. En general está marca es así de buena pero hay que revisar las indicaciones del tiempo porque cambian de producto a producto, y éste en especial tiene el tamaño adecuado para el portavasos del coche y es fácil de usar. Solamente hay que ser cuidadoso al momento de enroscar la tapa para no dañar el enroscado, ya que la tapa es de plástico, y porque no es una pieza de motor.", "positive", 0.687094509601593),
    ("very good water bottle, hot or cold. Wish it would keep hot for more than 5 hours though. No leak, slim and fits in car cup holder. Highly recommend.", "positive", 0.7661781311035156),
    ("We have 3 of these bottles now - the orange, white and citron. My sister bought one too. They are one of the best drinking bottles I’ve found for the gym.I hate struggling to drink at the gym and this has a lovely big glug style opening perfect for having a nice big drink when you are at the gym or on a hot day. My main Stanley’s I use at my desk to drink while I’m working. But they just did not cut it in the gym.This 0.71 litre bottle holds enough for a short session, easily filled up, and holds the cold temp ALL DAY! The wide opening means it’s easy to add my gym supplements to the water when I want to or add hydration salts.It has never leaked in my bag yet. I use it all the time - gym obv. walks, cinema, out and about in the car - it’s the perfect bottle. Just wish my cup holder in the Skoda Kodiaq was biggerThis bottle is a winner! Just buy it!!My kids love it so they also have the 0.71l and are drinking more too. This bottle is responsible for improving the hydration of 3 people!", "positive", 0.924)
]

# DataFrame'e çevir
df = pd.DataFrame(data, columns=["review", "sentiment", "score"])

# 🔹 Gerçek etiketleri sayısal hale getir (0 = positive, 1 = negative)
df["true_labels"] = df["sentiment"].apply(lambda x: 1 if x == "negative" else 0)

# 🔹 Modelin tahmin ettiği sınıfları belirle
threshold = 0.5  # Eğer pozitifleri kaçırıyorsa 0.6 veya 0.7 yap
df["pred_labels"] = df["score"].apply(lambda x: 1 if x >= threshold else 0)

# 🔹 F1 Skoru Hesaplama
f1 = f1_score(df["true_labels"], df["pred_labels"])

# 🔹 Detaylı Rapor
report = classification_report(df["true_labels"], df["pred_labels"], target_names=["Positive", "Negative"])

# Sonuçları Yazdır
print(f"📊 F1 Score: {f1:.4f}")
print("\n🔍 Classification Report:\n", report)
