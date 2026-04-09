"""
data.py - CIFAR-10 Veri Seti Yükleme ve Ön İşleme

CIFAR-10 veri setini indirip PyTorch DataLoader nesnelerine dönüştürür.
Eğitim seti için veri artırma (data augmentation), test seti için
yalnızca normalizasyon uygulanır.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# macOS Python kurulumlarında SSL sertifika doğrulama hatası çözümü
ssl._create_default_https_context = ssl._create_unverified_context

# CIFAR-10 istatistikleri (kanal bazında ortalama ve standart sapma)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_dataloaders(batch_size=128, num_workers=2, use_augmentation=True):
    """
    CIFAR-10 eğitim ve test veri yükleyicilerini oluşturur.

    Args:
        batch_size: Mini-batch boyutu.
        num_workers: Paralel veri yükleme için iş parçacığı sayısı.
        use_augmentation: Eğitim setinde veri artırma kullanılsın mı?

    Returns:
        (trainloader, testloader) tuple'ı.
    """
    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def denormalize(tensor):
    """
    Normalize edilmiş bir tensor'ı orijinal piksel aralığına [0, 1] geri döndürür.
    Görselleştirme için gereklidir.

    Args:
        tensor: Normalize edilmiş görüntü tensörü (C, H, W).

    Returns:
        [0, 1] aralığına clamp'lenmiş tensor.
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
