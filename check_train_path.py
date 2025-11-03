import os

print('=== 获取正确的绝对路径 ===')

# 使用相对路径，兼容不同操作系统
project_root = os.path.abspath('.')  # 当前目录
train_path = os.path.join(project_root, 'train', 'image_samples')
cache_path = os.path.join(project_root, 'train', 'image_samples', 'cache')

print(f'项目根目录: {project_root}')
print(f'训练数据路径: {train_path}')
print(f'缓存路径: {cache_path}')

print(f'\n路径存在检查:')
print(f'  训练数据路径存在: {os.path.exists(train_path)}')
print(f'  缓存路径存在: {os.path.exists(cache_path)}')

if os.path.exists(train_path):
    files = os.listdir(train_path)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    txt_files = [f for f in files if f.lower().endswith('.txt')]
    
    print(f'\n文件统计:')
    print(f'  总文件数: {len(files)}')
    print(f'  图片文件: {len(image_files)}')
    print(f'  视频文件: {len(video_files)}')
    print(f'  文本文件: {len(txt_files)}')
    
    print(f'\nTOML配置应该使用的路径:')
    toml_path = train_path.replace('\\', '/')
    toml_cache = cache_path.replace('\\', '/')
    print(f'  video_directory = "{toml_path}"')
    print(f'  cache_directory = "{toml_cache}"')

