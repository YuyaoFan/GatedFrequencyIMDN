import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 配置区域 =================
# 1. 结果根目录
RESULTS_ROOT = 'results'
GT_FOLDER = 'datasets/benchmark/Urban100/HR'

# 2. 想要对比的模型列表 (对应文件夹名称中的 test_前缀后部分)
# 注意：你需要确保运行了 test.py 生成了这些文件夹
SELECTED_MODELS = [
    'IMDN',
    'FreqIMDN',
    'GateFreqIMDN'
]

# 3. 图片标识
IMG_NAME_KEY = 'img051'  # 图片核心关键词
SUFFIX = 'x2'

# ================= 工具函数 =================
def read_image(path):
    img = cv2.imread(path)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_y_channel(img):
    """简单转换为Y通道 (类似 MATLAB 逻辑)"""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        # RGB to YCbCr conversion
        img = 65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2] + 16.0
    return img

def calculate_psnr(img1, img2, crop_border=2):
    """计算 PSNR (Y通道, 去除边缘)"""
    # 1. Crop border
    h, w = img1.shape[:2]
    img1 = img1[crop_border:h-crop_border, crop_border:w-crop_border]
    img2 = img2[crop_border:h-crop_border, crop_border:w-crop_border]
    
    # 2. To Y channel
    y1 = to_y_channel(img1)
    y2 = to_y_channel(img2)
    
    # 3. MSE & PSNR
    mse = np.mean((y1 - y2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def get_error_map(img_sr, img_gt):
    """计算与GT的误差热力图"""
    # 确保尺寸一致
    h, w = img_gt.shape[:2]
    if img_sr.shape[:2] != (h, w):
        img_sr = cv2.resize(img_sr, (w, h))
        
    diff = cv2.absdiff(img_sr, img_gt)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_enhanced = cv2.normalize(diff_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # 放大误差显示
    diff_enhanced = np.uint8(diff_enhanced * 5.0) 
    heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def crop_detail(img):
    """统一裁剪位置"""
    h, w = img.shape[:2]
    crop_size = 80
    cx, cy = w // 2 + 30, h // 2 + 50
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    return img[y1:y1+crop_size, x1:x1+crop_size]

# ================= 主程序 =================
def main():
    # 1. 准备 GT
    gt_path = glob.glob(os.path.join(GT_FOLDER, f"*{IMG_NAME_KEY}*.png"))[0]
    img_gt = read_image(gt_path)
    
    # 2. 准备数据列表
    data_list = []
    
    # 添加 GT
    data_list.append({'name': 'GT', 'img': img_gt, 'psnr': float('inf'), 'is_gt': True})
    
    # 添加 Bicubic
    h, w = img_gt.shape[:2]
    img_lr = cv2.resize(img_gt, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
    img_bic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    psnr_bic = calculate_psnr(img_bic, img_gt)
    data_list.append({'name': 'Bicubic', 'img': img_bic, 'psnr': psnr_bic, 'is_gt': False})
    
    # 添加模型结果
    for model_name in SELECTED_MODELS:
        # 路径构造: results/test_DeepResNet/visualization/Set5/butterflyx2_test_DeepResNet.png
        # 注意: 这里的文件夹名假设是 test_{model_name}，这是 BasicSR test.py 默认行为
        # 同时也处理可能的命名变体
        folder_name = f"test_{model_name}"
        file_pattern = f"*{IMG_NAME_KEY}*{SUFFIX}*{model_name}*.png"
        
        search_path = os.path.join(RESULTS_ROOT, folder_name, 'visualization', 'Urban100', file_pattern)
        found = glob.glob(search_path)
        
        if not found:
            # 尝试另一种路径 (如果没有 test_ 前缀)
            search_path_alt = os.path.join(RESULTS_ROOT, model_name, 'visualization', 'Urban100', file_pattern)
            found = glob.glob(search_path_alt)
            
        if found:
            img_sr = read_image(found[0])
            psnr = calculate_psnr(img_sr, img_gt)
            data_list.append({'name': model_name, 'img': img_sr, 'psnr': psnr, 'is_gt': False})
            print(f"Loaded: {model_name} | PSNR: {psnr:.2f} dB")
        else:
            print(f"Warning: 未找到模型 {model_name} 的结果图")

    # ================= 绘图逻辑 =================
    num_models = len(data_list)
    fig, axes = plt.subplots(3, num_models, figsize=(3 * num_models, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    for i, item in enumerate(data_list):
        img = item['img']
        
        # --- Row 1: 整体 + PSNR ---
        ax_main = axes[0, i]
        ax_main.imshow(img)
        title = item['name']
        if not item['is_gt']:
            title += f"\n({item['psnr']:.2f} dB)"
        ax_main.set_title(title, fontsize=12, fontweight='bold')
        ax_main.axis('off')
        
        # --- Row 2: 局部细节 ---
        ax_zoom = axes[1, i]
        img_crop = crop_detail(img)
        ax_zoom.imshow(img_crop)
        if i == 0: ax_zoom.set_ylabel("Zoom", fontsize=12)
        ax_zoom.axis('off')
        
        # --- Row 3: 误差热力图 ---
        ax_err = axes[2, i]
        if item['is_gt']:
            ax_err.text(0.5, 0.5, "Reference", ha='center')
            ax_err.axis('off')
        else:
            err_map = get_error_map(img, img_gt)
            ax_err.imshow(err_map)
        if i == 0: ax_err.set_ylabel("Error Map", fontsize=12)
        ax_err.axis('off')

    plt.suptitle(f"Final Model Comparison: {IMG_NAME_KEY} x2", fontsize=16, y=0.98)
    save_path = "comparison/final_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"最终对比图已保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()