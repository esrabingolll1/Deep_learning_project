"""
train.py - Eğitim ve Doğrulama Döngüsü

Modeli eğiten ve her epoch sonunda doğrulama başarısını ölçen
fonksiyonları içerir. L1 ve L2 düzenlileştirme desteği mevcuttur.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy


def train_model(model, trainloader, testloader, epochs, device,
                lr=0.001, l1_lambda=0.0, l2_weight_decay=0.0,
                label_smoothing=0.0, verbose=True,
                attack_fn=None, input_grad_callback=None, track_input_grads=False,
                adv_train=False, adv_epsilon=0.03,
                optimizer_name="adam",
                scheduler_name="cosine",
                scheduler_patience=2,
                scheduler_factor=0.5,
                grad_clip_norm=0.0):
    """
    Modeli eğitir ve epoch bazında metrikleri döndürür.

    Args:
        model: Eğitilecek nn.Module modeli.
        trainloader: Eğitim veri yükleyicisi.
        testloader: Doğrulama (test) veri yükleyicisi.
        epochs: Toplam eğitim epoch sayısı.
        device: Hesaplama cihazı (cpu / mps / cuda).
        lr: Öğrenme oranı (learning rate).
        l1_lambda: L1 düzenlileştirme katsayısı (0 = kapalı).
        l2_weight_decay: L2 düzenlileştirme katsayısı (0 = kapalı).
        label_smoothing: Label Smoothing katsayısı (0 = kapalı).
        verbose: True ise her epoch'ta sonuçları ekrana yazdırır.
        attack_fn: Opsiyonel adversarial attack fonksiyonu. Eğer verilirse,
            eğitim sırasında "clean" ileri geçişinden sonra giriş gradyanları
            hesaplanır ve attack_fn bu gradyanları kullanarak adversarial
            girişleri üretir. Beklenen arayüz:

            ``attack_fn(model, inputs, labels, input_grads) -> adv_inputs``

            - inputs: (temiz) giriş tensorü
            - input_grads: inputs'a göre loss gradyanı (detach edilmiş)
            - adv_inputs: model için yeniden ileri geçişte kullanılacak tensör
        input_grad_callback: input gradyanlarını yakalamak/incelemek için
            çağrılan opsiyonel callback. İmza:
            ``callback(batch_idx: int, input_grads: torch.Tensor)``
        track_input_grads: True ise, clean pass sonrası inputs.grad snapshot'ı
            (detach+clone) alınır. attack_fn olmasa bile gradyan takibi yapılır.

    Returns:
        dict: {
            'train_loss': [...],
            'train_acc':  [...],
            'test_acc':   [...]
        }

    Notlar (metodolojik gerekçeler):
        - Neden Cross-Entropy Loss?
          Sınıflandırmada modelin logitlerinden elde edilen olasılık dağılımını,
          ground-truth etiket ile karşılaştırmak için MLE (Maximum Likelihood
          Estimation) perspektifi kullanılır. Cross-Entropy, gerçek etiketin
          olasılığını maksimize etmeye karşılık gelen negatif log-likelihood'ı
          minimize eder.

        - Neden Adam Optimizer?
          Adam, her parametre için ayrı bir adaptif öğrenme oranı üretir.
          Bu sayede özellikle farklı ölçeklerdeki gradyanlarda eğitim daha hızlı
          ve genellikle daha stabil ilerler; yerel minimum/plateau etkileri
          daha yumuşak deneyimlenebilir.

        - ReLU ve vanishing gradient ilişkisi:
          ReLU, aktifken (x > 0) gradyanı 1 seviyesinde tutarak bilgi/gradient
          akışını destekler. Bu, sigmoid benzeri fonksiyonların derin ağlarda
          gradyanları küçültmesiyle ortaya çıkan vanishing gradient riskini azaltır.
    """
    model.to(device)

    # Cross-Entropy kaybı (opsiyonel Label Smoothing desteği)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer seçimi: Adam veya SGD
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=l2_weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=l2_weight_decay,
        )

    # Learning-rate scheduler seçimi
    scheduler_name = scheduler_name.lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    history = {
        'train_loss': [],
        'train_acc':  [],
        'test_acc':   [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    def compute_loss(outputs, labels):
        """Veri kaybı + opsiyonel L1 regularization (parametre seviyesinde)."""
        data_loss = criterion(outputs, labels)
        if l1_lambda > 0:
            # L1, model parametrelerinin mutlak değerlerinin toplamı üzerinden
            # manuel olarak eklenir. inputs'a göre gradyanı etkilemez.
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            return data_loss + l1_lambda * l1_penalty
        return data_loss

    for epoch in range(epochs):
        # ─────────── Eğitim Aşaması ───────────
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Adversarial attack varsa giriş gradyanına ihtiyaç vardır.
            need_input_grads = (attack_fn is not None) or track_input_grads or adv_train
            if need_input_grads:
                inputs = inputs.detach()
                inputs.requires_grad_(True)

            # 1) Clean forward+backward -> inputs.grad elde et
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = compute_loss(outputs, labels)
            loss.backward()

            input_grads = None
            if need_input_grads:
                # FGSM gibi saldırılar için bir snapshot saklıyoruz.
                input_grads = (
                    inputs.grad.detach().clone()
                    if inputs.grad is not None
                    else None
                )
                if input_grad_callback is not None and input_grads is not None:
                    input_grad_callback(batch_idx=batch_idx, input_grads=input_grads)

            # 2) Eğer attack_fn verildiyse adversarial girişle ikinci adım
            if attack_fn is not None:
                with torch.no_grad():
                    adv_inputs = attack_fn(
                        model=model,
                        inputs=inputs.detach(),
                        labels=labels,
                        input_grads=input_grads,
                    )
                adv_inputs = adv_inputs.to(device).detach()

                optimizer.zero_grad(set_to_none=True)
                outputs_adv = model(adv_inputs)
                loss_adv = compute_loss(outputs_adv, labels)
                loss_adv.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

                running_loss += loss_adv.item()
                _, predicted = outputs_adv.max(1)
            elif adv_train and input_grads is not None:
                # 3) Adversarial Training Bonus: Clean + FGSM Loss
                adv_inputs = inputs.detach() + adv_epsilon * input_grads.sign()
                adv_inputs = torch.clamp(adv_inputs, 0.0, 1.0)
                outputs_adv = model(adv_inputs)
                loss_adv = compute_loss(outputs_adv, labels)
                loss_adv.backward() # Combine gradients
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                running_loss += (loss.item() + loss_adv.item()) / 2.0
                _, predicted = outputs.max(1)
            else:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)

            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # ─────────── Doğrulama Aşaması ───────────
        model.eval()
        correct_test = 0
        total_test = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_batch = criterion(outputs, labels)
                val_running_loss += val_loss_batch.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        # Metrikleri kaydet
        epoch_loss = running_loss / len(trainloader)
        val_loss_epoch = val_running_loss / len(testloader)
        train_acc  = 100.0 * correct_train / total_train
        test_acc   = 100.0 * correct_test / total_test

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if verbose:
            print(f"  Epoch [{epoch+1:>2}/{epochs}]  "
                  f"Loss: {epoch_loss:.4f}  |  "
                  f"Train Acc: {train_acc:.2f}%  |  "
                  f"Test Acc: {test_acc:.2f}%  |  "
                  f"Val Loss: {val_loss_epoch:.4f}")

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss_epoch)
            else:
                scheduler.step()

        if patience_counter >= 5:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch+1}. Restoring best weights.")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return history
