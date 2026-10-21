import os
import sys
import urllib.request
import zipfile
import tarfile

# Define a pasta 'data' na raiz do repositório
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def ensure_data_dir():
    """Garante a criação da pasta data se ela não existir."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[+] Pasta de dados criada em: {DATA_DIR}")
    else:
        print(f"[i] Pasta de dados detectada: {DATA_DIR}")


def download_cifar100():
    """Verifica, baixa e extrai o CIFAR-100."""
    cifar_folder = os.path.join(DATA_DIR, "cifar-100-python")
    cifar_tar = os.path.join(DATA_DIR, "cifar-100-python.tar.gz")

    print("\n--- [1/3] Verificando CIFAR-100 ---")

    # 1. Checa se a pasta descompactada já existe
    if os.path.exists(cifar_folder):
        print(f"[✓] CIFAR-100 já está extraído e pronto em: {cifar_folder}")
        return

    # 2. Baixa se o arquivo tar.gz não existir
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    if not os.path.exists(cifar_tar):
        print("[↓] Baixando CIFAR-100 (~160 MB)...")
        urllib.request.urlretrieve(url, cifar_tar)
        print("[✓] Download concluído.")

    # 3. Descompacta
    print("[⚙] Descompactando CIFAR-100...")
    with tarfile.open(cifar_tar, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)

    # 4. Remove o arquivo compactado para economizar disco
    if os.path.exists(cifar_tar):
        os.remove(cifar_tar)

    print("[✓] CIFAR-100 pronto para uso!")


def download_tiny_imagenet():
    """Verifica, baixa e extrai o Tiny-ImageNet-200."""
    tiny_folder = os.path.join(DATA_DIR, "tiny-imagenet-200")
    tiny_zip = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")

    print("\n--- [2/3] Verificando Tiny-ImageNet-200 ---")

    # 1. Checa se a pasta descompactada já existe
    if os.path.exists(tiny_folder):
        print(f"[✓] Tiny-ImageNet já está extraído e pronto em: {tiny_folder}")
        return

    # 2. Baixa se o zip não existir
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    if not os.path.exists(tiny_zip):
        print("[↓] Baixando Tiny-ImageNet-200 (~230 MB)...")
        urllib.request.urlretrieve(url, tiny_zip)
        print("[✓] Download concluído.")

    # 3. Descompacta
    print("[⚙] Descompactando Tiny-ImageNet...")
    with zipfile.ZipFile(tiny_zip, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # 4. Remove o arquivo zip
    if os.path.exists(tiny_zip):
        os.remove(tiny_zip)

    print("[✓] Tiny-ImageNet pronto para uso!")


def download_dermamnist():
    """Verifica e baixa o DermaMNIST via MedMNIST."""
    derma_file = os.path.join(DATA_DIR, "dermamnist.npz")

    print("\n--- [3/3] Verificando DermaMNIST ---")

    # 1. Checa se o arquivo .npz já existe
    if os.path.exists(derma_file):
        print(f"[✓] DermaMNIST já está salvo e pronto em: {derma_file}")
        return

    # 2. Usa a biblioteca medmnist para baixar
    try:
        from medmnist import DermaMNIST
        print("[↓] Baixando DermaMNIST (~20 MB)...")
        DermaMNIST(split="train", download=True, root=DATA_DIR)
        DermaMNIST(split="val", download=True, root=DATA_DIR)
        DermaMNIST(split="test", download=True, root=DATA_DIR)
        print("[✓] DermaMNIST pronto para uso!")
    except ImportError:
        print("[!] Erro: A biblioteca 'medmnist' não está instalada.")
        print("    Execute `pip install medmnist` e rode o script novamente.")


if __name__ == "__main__":
    print("==================================================")
    print("      PREPARAÇÃO AUTOMÁTICA DOS DATASETS          ")
    print("==================================================")
    ensure_data_dir()
    download_cifar100()
    download_tiny_imagenet()
    download_dermamnist()
    print("\n==================================================")
    print(" [✓] TODOS OS DATASETS ESTÃO PRONTOS EM data/     ")
    print("==================================================")