# import os
import sys
# import time
# import shutil

# try:
#     term_width = shutil.get_terminal_size((80, 20)).columns
# except Exception:
#     term_width = 80

# TOTAL_BAR_LENGTH = 65.

# def progress_bar(current, total, msg=None):
  
#     L = []
#     if msg:
#         L.append(msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

# def format_time(seconds):
#     days = int(seconds / 3600/24)
#     seconds = seconds - days*3600*24
#     hours = int(seconds / 3600)
#     seconds = seconds - hours*3600
#     minutes = int(seconds / 60)
#     seconds = seconds - minutes*60
#     secondsf = int(seconds)
#     seconds = seconds - secondsf
#     millis = int(seconds*1000)

#     f = ''
#     i = 1
#     if days > 0:
#         f += str(days) + 'D'
#         i += 1
#     if hours > 0 and i <= 2:
#         f += str(hours) + 'h'
#         i += 1
#     if minutes > 0 and i <= 2:
#         f += str(minutes) + 'm'
#         i += 1
#     if secondsf > 0 and i <= 2:
#         f += str(secondsf) + 's'
#         i += 1
#     if millis > 0 and i <= 2:
#         f += str(millis) + 'ms'
#         i += 1
#     if f == '':
#         f = '0ms'
#     return f

def progress_bar(batch_idx, total_batches, msg=None):
    # Imprime apenas no último batch da época ou a cada 50 batches para não poluir o log
    if batch_idx == total_batches - 1 or batch_idx % 50 == 0:
        log_msg = f"Batch [{batch_idx + 1}/{total_batches}]"
        if msg:
            log_msg += f" | {msg}"
        print(log_msg)
        sys.stdout.flush() # Força a escrita imediata no arquivo de log do Slurm