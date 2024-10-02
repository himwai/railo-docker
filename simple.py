import torch
import time

# 檢查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 定義模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 創建模型
model = SimpleModel().to(device)

# 生成隨機數據
x = torch.randn(1000, 1024).to(device)
y = torch.randint(0, 10, (1000,)).to(device)

# 定義損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 預熱
for _ in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# 計時
start_time = time.time()
for _ in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
end_time = time.time()

# 計算 FLOPS
total_params = sum(p.numel() for p in model.parameters())
total_flops = total_params * 2 * 1000 * 100  # 參數數 * 2 (乘加) * 樣本數 * iterations
tops = total_flops / (end_time - start_time) / 1e12

if __name__ == "__main__":
    sizes = [1000, 2000, 3000, 4000, 5000]
    cpu_times, gpu_times = run_comparison(sizes)
    plot_results(sizes, cpu_times, gpu_times)
    print("結果已保存到 'performance_comparison.png'")


# 檢查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 定義模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 創建模型
model = SimpleModel().to(device)

# 生成隨機數據
x = torch.randn(1000, 1024).to(device)
y = torch.randint(0, 10, (1000,)).to(device)

# 定義損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 預熱
for _ in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# 計時
start_time = time.time()
for _ in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
end_time = time.time()

# 計算 FLOPS
total_params = sum(p.numel() for p in model.parameters())
total_flops = total_params * 2 * 1000 * 100  # 參數數 * 2 (乘加) * 樣本數 * iterations
tops = total_flops / (end_time - start_time) / 1e12

print(f"估計的 TOPS: {tops:.2f}")    
