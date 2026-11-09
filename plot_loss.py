import sys
import re
import matplotlib.pyplot as plt

log_path = sys.argv[1] 

epochs_train = []
train_losses = []
val_losses = []

current_epoch = None
batch_losses = []
train_epoch_avg = {}

with open(log_path, 'r') as f:
    for line in f:
        # Extrai Loss de treino: Batch [...] | [Epoch X/100] ... Loss: 3.4034e+00
        m_train = re.search(r'\[Epoch\s+(\d+)/\d+\].*?Loss:\s*([0-9.eE+-]+)', line)
        if m_train:
            ep = int(m_train.group(1))
            loss = float(m_train.group(2))
            if ep not in train_epoch_avg:
                train_epoch_avg[ep] = []
            train_epoch_avg[ep].append(loss)

        # Extrai Loss de validação: V Loss 1.6341...
        m_val = re.search(r'V\s+Loss\s+([0-9.eE+-]+)', line)
        if m_val:
            val_losses.append(float(m_val.group(1)))

# Calcula média da loss de treino por época
x_train = sorted(train_epoch_avg.keys())
y_train = [sum(train_epoch_avg[ep]) / len(train_epoch_avg[ep]) for ep in x_train]

x_val = list(range(1, len(val_losses) + 1))

# Gerar o gráfico
plt.figure(figsize=(9, 5))
if y_train:
    plt.plot(x_train, y_train, label='Treino (Train Loss)', color='#1f77b4', linewidth=2)
if val_losses:
    plt.plot(x_val, val_losses, label='Validação (Val Loss)', color='#ff7f0e', linewidth=2)

plt.xlabel('Época', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Curva de Aprendizado - {log_path.split("/")[-1]}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_file = 'loss_curve.png'
plt.savefig(output_file, dpi=300)
print(f"✅ Gráfico salvo com sucesso: {output_file}")
