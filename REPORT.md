# REPORT

## System Architecture

**Optimization**: Adaptif öğrenme oranı (Adaptive Learning Rate) mekanizması nedeniyle Adam Optimizer tercih edilmiştir. Bu, farklı parametreler için gradyanların karesel ortalamasına göre öğrenme hızını ayarlayarak daha hızlı yakınsama (convergence) sağlar.

**Loss Function**: Çok sınıflı sınıflandırma yapıldığı için Cross-Entropy Loss kullanılmıştır. Bu seçim, istatistiksel olarak Maximum Likelihood Estimation (MLE) prensibiyle uyumludur; modelin doğru sınıf üzerindeki olasılık değerini maksimize etmeyi hedefler.

**Activation**: Derin ağlarda gradyan akışını korumak ve Vanishing Gradient problemini engellemek için gizli katmanlarda ReLU kullanılmıştır.

**Dataset Quality (CIFAR-10)**: 10 farklı sınıfa ait 60.000 renkli (RGB) görüntüden oluşan CIFAR-10 veri seti kullanılmıştır. Veri setinin kalitesi; dengeli sınıf dağılımı (her sınıfta eşit sayıda görsel) sunması sayesinde modelde "sınıf dengesizliği (class imbalance)" kaynaklı yanlılıkları (bias) engeller. Ayrıca gerçek dünya nesnelerinin standartlaştırılmış formatı (32x32) ideal bir "benchmark" (kıyaslama) ortamı sağlamaktadır.

## Model Capacity Ablation (Student 1)

Model kapasitesi iddiasını kanıtlamak için sistematik bir ablation protokolü uygulanmıştır. Bu deneyde yalnızca kapasite parametreleri değiştirilmiş, diğer tüm koşullar sabit tutulmuştur:

- Aynı veri seti: CIFAR-10
- Aynı optimizer/scheduler: Adam + CosineAnnealingLR
- Aynı eğitim süreci: epoch sayısı, batch size ve eğitim döngüsü
- Değişen tek faktör: model kapasitesi (`base_channels`, `fc_hidden_dim`)

Karşılaştırılan konfigürasyonlar:

- **Small Capacity**: `base_channels=8`, `fc_hidden_dim=128`
- **Base Capacity**: `base_channels=16`, `fc_hidden_dim=256`
- **Large Capacity**: `base_channels=32`, `fc_hidden_dim=512`

Bu deneyin amacı, bias-variance tradeoff'u sayısal olarak görünür kılmaktır. Kapasite düşükken modelin temsili gücü sınırlanır ve underfitting riski artar; kapasite yükseldikçe eğitim doğruluğu artarken genelleme boşluğu (`train_acc - test_acc`) büyüyerek overfitting riski yükselir.

Çıktı artefaktları:

- `ogrenci1/capacity_ablation_summary.json`
- `results/student1_capacity_ablation_curves.png`

Deneyi yeniden üretmek için:

`python generate_student1_capacity_report.py`
