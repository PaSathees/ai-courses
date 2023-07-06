"""
PyTorch simple function associated with environment setup and device configuration
"""
import torch
import matplotlib
import pandas as pd
import numpy as np

def print_versions():
    """Prints the available packages versions, e.g., PyTorch, Torchinfo...
    """
    # PyTorch
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
    except ImportError:
        print("[INFO] No PyTorch found")

    # Matplotlib
    try:
        import matplotlib
        print(f"Matplotlib Version: {matplotlib.__version__}")
    except ImportError:
        print("[INFO] No Matplotlib found")

    # Pandas
    try:
        import pandas
        print(f"Pandas Version: {pandas.__version__}")
    except ImportError:
        print("[INFO] No Pandas found")

    # Numpy
    try:
        import numpy
        print(f"Numpy Version: {numpy.__version__}")
    except ImportError:
        print("[INFO] No Numpy found")

    # Torchvision
    try:
        import torchvision
        print(f"[INFO] Torchvision Version: {torchvision.__version__}")
    except ImportError:
        print("[INFO] No Torchvision found")

    # Torchaudio
    try:
        import torchaudio
        print(f"Torchaudio Version: {torchaudio.__version__}")
    except ImportError:
        print("[INFO] No Torchaudio found")

    # Scikit-learn
    try:
        import sklearn
        print(f"Scikit-learn Version: {sklearn.__version__}")
    except ImportError:
        print("[INFO] No Scikit-learn found")

    # Torchmetrics
    try:
        import torchmetrics
        print(f"Torchmetrics Version: {torchmetrics.__version__}")
    except ImportError:
        print("[INFO] No Torchmetrics found")

    # TQDM
    try:
        import tqdm
        print(f"TQDM Version: {tqdm.__version__}")
    except ImportError:
        print("[INFO] No TQDM found")

    # MLXTEND
    try:
        import mlxtend
        print(f"MLEXTEND Version: {mlxtend.__version__}")
    except ImportError:
        print("[INFO] No MLEXTEND found")

    # PIL
    try:
        import PIL
        print(f"PIL Version: {PIL.__version__}")
    except ImportError:
        print("[INFO] No PIL found")

    # Torchinfo
    try:
        import torchinfo
        print(f"Torchinfo Version: {torchinfo.__version__}")
    except ImportError:
        print("[INFO] No Torchinfo found")

    # Gradio
    try:
        import gradio
        print(f"Gradio Version: {gradio.__version__}")
    except ImportError:
        print("[INFO] No Gradio found")


def install_essential_packages():
    """Installs essential packages that are missing
    """
    # PyTorch
    try:
        import torch
    except ImportError:
        print("[INFO] No PyTorch found, Installing it...")
        !pip -q install torch

    # Matplotlib
    try:
        import matplotlib
    except ImportError:
        print("[INFO] No Matplotlib found, Installing it...")
        !pip -q nstall matplotlib

    # Pandas
    try:
        import pandas
    except ImportError:
        print("[INFO] No Pandas found, Installing it...")
        !pip -q install pandas

    # Numpy
    try:
        import numpy
    except ImportError:
        print("[INFO] No Numpy found, Installing it...")
        !pip -q install numpy

    # Torchvision
    try:
        import torchvision
    except ImportError:
        print("[INFO] No Torchvision found, Installing it...")
        !pip -q install torchvision

    # Torchaudio
    try:
        import torchaudio
    except ImportError:
        print("[INFO] No Torchaudio found, Installing it...")
        !pip -q install torchaudio

    # Torchmetrics
    try:
        import torchmetrics
    except ImportError:
        print("[INFO] No Torchmetrics found, Installing it...")
        !pip -q install torchmetrics

    # TQDM
    try:
        import tqdm
    except ImportError:
        print("[INFO] No TQDM found, Installing it...")
        !pip -q install tqdm

    # MLEXTEND
    try:
        import mlxtend
    except ImportError:
        print("[INFO] No MLEXDEND found, Installing it...")
        !pip -q install mlxtend

    # Torchinfo
    try:
        import torchinfo
    except ImportError:
        print("[INFO] No Torchinfo found, Installing it...")
        !pip -q install torchinfo

    # Gradio
    try:
        import gradio
    except ImportError:
        print("[INFO] No Gradio found, Installing it...")
        !pip -q install gradio



def print_gpu_status():
    """Prints whether a CUDA GPU is available & number of GPUs
    """
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} Supported CUDA GPU are available")
    else:
        print("No Supported CUDA GPU found")


def get_agnostic_device():
    """Returns device name as "cuda" if supported GPU is available or will return "cpu"

    Returns:
        string: name of the device ("cuda" or "cpu")
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

print_versions()