import torch
import time

torch.manual_seed(74)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

matmulMatrix1CPU = torch.rand(512, 512)
matmulMatrix2CPU = torch.rand(512, 512)
matmulMatrix1GPU = matmulMatrix1CPU.to(device)
matmulMatrix2GPU = matmulMatrix2CPU.to(device)

matrixA = torch.rand(64, 1024, 1024)
matrixB = torch.rand(128, 512, 512)
matrixC = torch.rand(256, 256, 256)

matrixCPU = torch.rand(1024, 1024, 1024)
matrixGPU = matrixCPU.to(device)

def timeComparison(matrixEventCPU, matrixEventGPU):    #Измеряет время выполнения операций на CPU и GPU.
    startTimeCPU = time.time()
    matrixEventCPU()
    endTimeCPU = time.time()

    # Переводим в миллисекунды (умножаем на 1000)
    elapsedTimeCPU = (endTimeCPU - startTimeCPU) * 1000

    startEvent = torch.cuda.Event(enable_timing=True)
    endEvent = torch.cuda.Event(enable_timing=True)
    startEvent.record()
    matrixEventGPU()
    endEvent.record()
    endEvent.synchronize()

    elapsedTimeGPU = startEvent.elapsed_time(endEvent)

    return elapsedTimeCPU, elapsedTimeGPU

print(f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (ms)':<10} | {'Ускорение':<10}")
print("-" * 64)

for op in [
    ("Матричное умножение", 
     lambda: torch.matmul(matmulMatrix1CPU, matmulMatrix2CPU), 
     lambda: torch.matmul(matmulMatrix1GPU, matmulMatrix2GPU)
    ),
    ("Поэлементное сложение",
     lambda: matrixCPU + matrixCPU,
     lambda: matrixGPU + matrixGPU
    ),
    ("Поэлементное умножение",
     lambda: matrixCPU * matrixCPU,
     lambda: matrixGPU * matrixGPU
    ),
    ("Транспонирование",
     lambda: matrixCPU.transpose(1, 2),
     lambda: matrixGPU.transpose(1, 2)
    ),
    ("Вычисление sum всех эл.",
     lambda: matrixCPU.sum(),
     lambda: matrixGPU.sum()
    )]:

    timeCPU, timeGPU = timeComparison(op[1], op[2])
    
    if device == 'cuda' and timeGPU > 0:
        speedup = f"{timeCPU / timeGPU:.2f}x"
        gpu_str = f"{timeGPU:.2f}"
    else:
        speedup = "1.00x (CPU)"
        gpu_str = f"{timeGPU:.2f}" if device == 'cuda' else "N/A"
        
    print(f"{op[0]:<25} | {timeCPU:<10.2f} | {gpu_str:<10} | {speedup:<10}")