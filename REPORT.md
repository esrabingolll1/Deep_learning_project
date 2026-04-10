# CIFAR-10 CNN — Proje Raporu (Özet + Deney Sonuçları)

Bu repo, CIFAR-10 üzerinde bir CNN eğitip farklı düzenlileştirme/optimizasyon stratejilerini ve FGSM adversarial saldırı dayanıklılığını kıyaslamak için hazırlanmıştır. Rapor; `responsibilities/` altındaki öğrenci raporları ve `ogrenci*/` / `results/` altındaki üretilmiş çıktı dosyalarına dayanır.

## Takım ve Roller (Dosya Kanıtı)

| Öğrenci No | Rol | Kanıt dosyası |
|---|---|---|
| 210911026 | Sistem Mimarı & Entegratör + Kapasite Ablation | `responsibilities/210911026.md` |
| 190722054 | Dropout + BatchNorm + Initialization analizi | `responsibilities/190722054.md` |
| 210911028 | Augmentation + Label Smoothing + Scheduler (Week 5) | `responsibilities/210911028.md` |
| 210911030 | FGSM Güvenlik Analizi + Optimizer kıyası | `responsibilities/210911030.md` |

## Veri Seti ve Ön İşleme

- **Veri seti**: CIFAR-10 (50k train, 10k test, 10 sınıf, 32×32 RGB)
- **Train transform**: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + `Normalize`
- **Test transform**: `Normalize`
- **Kaynak**: `data.py`

## Model (CNN)

`models.py` içinde parametrik bir CNN kullanılır:

- **Kapasite parametreleri**:
  - `base_channels` (varsayılan 16)
  - `fc_hidden_dim` (varsayılan 256)
- **Düzenlileştirme**:
  - `dropout_rate` (0 → kapalı)
  - `use_batchnorm` (False → kapalı)
  - `init_scheme`: `default`, `he`, `xavier`

## Eğitim Altyapısı

`train.py` tek eğitim çekirdeğidir.

- **Loss**: `CrossEntropyLoss(label_smoothing=...)`
- **L1**: loss’a manuel eklenir (`l1_lambda * Σ|w|`)
- **L2**: optimizer `weight_decay`
- **Optimizer**: Adam veya SGD(+momentum)
- **Scheduler**: CosineAnnealingLR veya ReduceLROnPlateau
- **Early stopping**: patience=5, en iyi ağırlıklara dönüş
- **Gradient clipping**: `grad_clip_norm > 0` ise `clip_grad_norm_`

## Deneyler ve Ölçülen Sonuçlar

### Öğrenci 1 — Model Capacity Ablation

Kaynak:
- `ogrenci1/capacity_ablation_summary.json`
- `ogrenci1/student1_capacity_ablation_curves.png`
- `ogrenci1/generalization_gap_bar.png`

Özet (best test acc):

| Konfig | base_channels | fc_hidden_dim | Parametre | Best Test Acc (%) |
|---|---:|---:|---:|---:|
| Small | 8 | 128 | 278,842 | 47.30 |
| Base | 16 | 256 | 1,111,914 | 56.48 |
| Large | 32 | 512 | 4,440,778 | **60.05** |

### Öğrenci 3 — Dropout + BatchNorm + Initialization (Default/He/Xavier)

Kaynak:
- `ogrenci3/student3_init_summary.json`
- `ogrenci3/student3_init_comparison_curves.png`

Özet (6 epoch, best test acc):

| Init | Best Test Acc (%) |
|---|---:|
| Default | **63.76** |
| Xavier | 49.04 |
| He | 10.28 |

### Öğrenci 4 — Augmentation + Label Smoothing + Scheduler (Week 5)

Kaynak:
- `ogrenci4/week5_scheduler_analysis.json`
- `ogrenci4/student4_scheduler_aug_smooth_comparison.png`

Özet:

| Deney | Best Test Acc (%) | %70’e ulaşma epoch |
|---|---:|---:|
| Cosine \| Base (No Aug, No Smooth) | 74.42 | 3 |
| Cosine \| Aug+Smooth | **79.43** | 3 |
| Plateau \| Base (No Aug, No Smooth) | 75.72 | 3 |
| Plateau \| Aug+Smooth | 78.92 | 4 |

### Öğrenci 5 — FGSM Robustness (Düzenleme kıyası + Optimizer kıyası)

Kaynak:
- `ogrenci5/fgsm_results.json` (ε başına doğruluklar)
- `ogrenci5/robustness_curve_regularization.png`
- `ogrenci5/robustness_curve_optimizer.png`
- `ogrenci5/accuracy_drop_bar_regularization.png`
- `ogrenci5/accuracy_drop_bar_optimizer.png`
- `ogrenci5/adversarial_samples.png`
- Ek özet: `ogrenci5/student5_report_data.json`, `ogrenci5/fgsm_optimizer_study.json`

FGSM tanımı:
\[
x_{\text{adv}} = \mathrm{clip}\bigl(x + \varepsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L})\bigr)
\]

**Not (kritik)**: Girişler normalize uzayda olduğundan `fgsm_eval.py` içinde clip işlemi kanal bazında normalize aralığına göre yapılır.

Örnek tablo (Optimizer kıyası — `fgsm_optimizer_study.json`, Clean / ε=0.1):

| Model | Clean (%) | ε=0.1 (%) |
|---|---:|---:|
| SGD (No Clip) | 44.30 | 14.72 |
| SGD (Clip=1.0) | 36.11 | 22.24 |
| Adam (No Clip) | 77.18 | 14.89 |
| Adam (Clip=1.0) | **77.75** | 14.51 |

Yorum:
- Clean doğrulukta Adam daha yüksek; fakat FGSM altında tüm modeller ciddi düşüş yaşıyor.
- SGD tarafında clipping bazı ε değerlerinde daha yumuşak düşüşe katkı verebiliyor (grafiklerden görülebilir).

## Projeyi Çalıştırma

Kurulum:

```bash
pip install -r requirements.txt
```

Tek model eğitimi (CLI):

```bash
python main.py --epochs 10 --optimizer adam --scheduler cosine --l2 0.001
```

Deney scriptleri:

```bash
python generate_student1_capacity_report.py
python generate_student3_init_report.py --quick
python generate_student4_report.py
python fgsm_eval.py
```

## Üretilen Çıktı Klasörleri

- `ogrenci1/`: kapasite ablation (grafikler + JSON)
- `ogrenci3/`: initialization karşılaştırması (grafikler + JSON)
- `ogrenci4/`: scheduler + augmentation/smoothing analizi (grafikler + JSON)
- `ogrenci5/`: FGSM robustness grafikleri + sonuç JSON’ları

