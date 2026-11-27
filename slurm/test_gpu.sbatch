#!/bin/bash
# Teste rápido de GPU/CUDA no Apuana. Uso (da raiz do repositório): sbatch slurm/test_gpu.sbatch
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --time=00:10:00
source slurm/common.sh
python - <<'EOF'
import torch, torchvision
print('torch', torch.__version__, '| build CUDA', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())
print('torchvision', torchvision.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device='cuda'); print('matmul ok:', float((x @ x).sum()) == float((x @ x).sum()))
    conv = torch.nn.Conv2d(3, 16, 3).cuda(); print('conv ok:', tuple(conv(torch.randn(2, 3, 32, 32, device='cuda')).shape))
    from torchvision.ops import nms
    b = torch.tensor([[0, 0, 1, 1.], [0, 0, 1, 1.]], device='cuda')
    print('torchvision nms cuda ok:', nms(b, torch.tensor([1., .5], device='cuda'), .5).tolist())
EOF
