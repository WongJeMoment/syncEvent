import os

# === 参数设置 ===
folder_path = '/home/wangzhe/ICRA2025/MY/train/frame'  # 替换为你的文件夹路径
ext = '.jpg'                   # 文件扩展名

# === 获取并排序所有jpg文件 ===
files = [f for f in os.listdir(folder_path) if f.endswith(ext)]
files.sort()  # 默认按名字排序

# === 依次重命名为 1.jpg, 2.jpg, ... ===
for i, filename in enumerate(files, 1):
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, f"{i}{ext}")
    os.rename(old_path, new_path)
    print(f"🔁 {filename} → {i}{ext}")

print("✅ 重命名完成！")
