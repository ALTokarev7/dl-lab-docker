import os
import torch
import random
import numpy as np
import subprocess
import cv2
import shutil

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Фиксируем CUDA seed только если CUDA доступна
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_test():
    print("Начинаем тестирование проекта Neural Style Transfer...")
    set_seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    cmd = [
        "python", "neural_style_transfer.py",
        "--content_img_name", "lion.jpg", 
        "--style_img_name", "vg_starry_night.jpg",
        "--height", "128",            
        "--optimizer", "adam",
        "--saving_freq", "-1"   
    ]
    
    print("---Генерация картинки---")
    subprocess.run(cmd, check=True)
    
    generated_img_path = "data/output-images/combined_lion_vg_starry_night/lion_vg_starry_night_o_adam_i_content_h_128_m_vgg19_cw_100000.0_sw_30000.0_tv_1.0.jpg" 
    expected_dir = "test_data"
    expected_img_path = f"{expected_dir}/expected_output.jpg"

    if not os.path.exists(generated_img_path):
        print("ОШИБКА: Модель не создала выходной файл!")
        return

    print("Сравниваем результат с эталоном...")
    gen_img = cv2.imread(generated_img_path)
    exp_img = cv2.imread(expected_img_path)

    if gen_img is None or exp_img is None:
        print("ОШИБКА: Не удалось прочитать картинки для сравнения.")
        return

    diff = np.mean(np.abs(gen_img.astype(np.float32) - exp_img.astype(np.float32)))
    print(f"Средняя ошибка (MAE): {diff:.4f}")

    if diff < 2.0: 
        print("ТЕСТ УСПЕШНО ПРОЙДЕН! Модель работает корректно и воспроизводимо.")
    else:
        print("ТЕСТ ПРОВАЛЕН! Картинка сильно отличается от эталона.")

if __name__ == "__main__":
    run_test()
    