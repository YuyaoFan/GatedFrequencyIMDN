import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================= 配置区域 =================
EXP_ROOT = 'experiments'
GT_ROOT = 'datasets/benchmark/Set5/HR' 
MODEL_NAME = 'GateFreqIMDN'
IMG_NAME = 'butterflyx2'
GT_KEY = 'butterfly'  # 用于匹配 GT 文件名的关键字

# 步数列表
STEPS = [2000, 4000, 10000, 20000]

# ================= 工具函数 =================
def read_image(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def crop_detail(img, crop_size=100):
    """
    固定位置裁剪：选取蝴蝶翅膀边缘纹理丰富的区域
    """
    if img is None: return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    h, w, _ = img.shape
    # 针对 butterflyx2 优化裁剪坐标
    cx, cy = w // 2 + 30, h // 2 + 50
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    return img[y1:y1+crop_size, x1:x1+crop_size]

def get_error_heatmap(img_sr, img_gt):
    """
    计算并返回伪彩色误差图
    蓝色: 误差小 | 红色: 误差大
    """
    if img_sr is None or img_gt is None: 
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 确保尺寸一致
    if img_sr.shape != img_gt.shape:
        img_sr = cv2.resize(img_sr, (img_gt.shape[1], img_gt.shape[0]))
        
    # 计算绝对差值
    diff = cv2.absdiff(img_sr.astype(np.float32), img_gt.astype(np.float32))
    # 转为灰度误差能量
    diff_mag = np.mean(diff, axis=2).astype(np.uint8)
    
    # 放大误差增益以便观察 (x10倍)
    diff_mag = cv2.normalize(diff_mag, None, 0, 255, cv2.NORM_MINMAX)
    # 使用 JET 色彩映射 (低-蓝，高-红)
    heatmap = cv2.applyColorMap(diff_mag, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ================= 主程序 =================
def main():
    # 1. 搜寻并读取 GT
    gt_files = glob.glob(os.path.join(GT_ROOT, f"*{GT_KEY}*.png"))
    if not gt_files:
        print("Error: 找不到 GT 文件")
        return
    img_gt = read_image(gt_files[0])
    
    # 2. 准备对比列表
    # 格式: (标题, 图像对象)
    compare_list = [("Ground Truth", img_gt)]
    
    base_dir = os.path.join(EXP_ROOT, MODEL_NAME, 'visualization', IMG_NAME)
    for step in STEPS:
        p = os.path.join(base_dir, f"{IMG_NAME}_{step}.png")
        images_step = read_image(p)
        compare_list.append((f"Iter {step}", images_step))

    # 3. 绘图：3行 (整体, 局部, 热力图) x N列
    num_cols = len(compare_list)
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * num_cols, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for i, (title, img) in enumerate(compare_list):
        # --- 第一行：整体 ---
        ax_top = axes[0, i]
        if img is not None:
            ax_top.imshow(img)
        ax_top.set_title(title, fontsize=14, fontweight='bold')
        ax_top.axis('off')
        if i == 0: ax_top.text(-50, 100, "Global", rotation=90, va='center', fontsize=12)

        # --- 第二行：局部放大 ---
        ax_mid = axes[1, i]
        crop = crop_detail(img)
        ax_mid.imshow(crop)
        ax_mid.axis('off')
        if i == 0: ax_mid.set_ylabel("Zoom Detail", fontsize=12, labelpad=20)

        # --- 第三行：热力图 ---
        ax_bot = axes[2, i]
        if i == 0: # GT 自身没有误差，显示黑色或文字
            ax_bot.fill_between([0, 1], 0, 1, color='black')
            ax_bot.text(0.5, 0.5, "REFERENCE", ha='center', va='center', color='white')
        else:
            h_map = get_error_heatmap(img, img_gt)
            ax_bot.imshow(h_map)
        ax_bot.axis('off')
        if i == 0: ax_bot.set_ylabel("Error Map", fontsize=12, labelpad=20)

    plt.suptitle(f"Quantitative & Qualitative Evolution: {MODEL_NAME}", fontsize=22, y=0.95)
    
    save_path = f"evolution/evolution_{MODEL_NAME}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已生成并保存至: {save_path}")
    plt.show()

if __name__ == '__main__':
    main()