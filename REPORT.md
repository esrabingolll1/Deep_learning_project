# REPORT

## System Architecture

**Optimization**: Adaptif öğrenme oranı (Adaptive Learning Rate) mekanizması nedeniyle Adam Optimizer tercih edilmiştir. Bu, farklı parametreler için gradyanların karesel ortalamasına göre öğrenme hızını ayarlayarak daha hızlı yakınsama (convergence) sağlar.

**Loss Function**: Çok sınıflı sınıflandırma yapıldığı için Cross-Entropy Loss kullanılmıştır. Bu seçim, istatistiksel olarak Maximum Likelihood Estimation (MLE) prensibiyle uyumludur; modelin doğru sınıf üzerindeki olasılık değerini maksimize etmeyi hedefler.

**Activation**: Derin ağlarda gradyan akışını korumak ve Vanishing Gradient problemini engellemek için gizli katmanlarda ReLU kullanılmıştır.

**Dataset Quality (CIFAR-10)**: 10 farklı sınıfa ait 60.000 renkli (RGB) görüntüden oluşan CIFAR-10 veri seti kullanılmıştır. Veri setinin kalitesi; dengeli sınıf dağılımı (her sınıfta eşit sayıda görsel) sunması sayesinde modelde "sınıf dengesizliği (class imbalance)" kaynaklı yanlılıkları (bias) engeller. Ayrıca gerçek dünya nesnelerinin standartlaştırılmış formatı (32x32) ideal bir "benchmark" (kıyaslama) ortamı sağlamaktadır.
