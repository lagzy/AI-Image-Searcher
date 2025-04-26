import tkinter as tk
from tkinter import filedialog, Label, Text, OptionMenu, StringVar
from PIL import Image, ImageTk
import cv2
import numpy as np
import mss
import time
from threading import Thread
import imagehash
import pyautogui
import keyboard

# Настройка pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image search SIFT")
        
        # Переменные
        self.template_path = None
        self.is_running = False
        self.screenshot = None
        self.check_count = 0
        self.found_region = None
        self.frequency = 32  # Частота по умолчанию (Гц)
        
        # Интерфейс
        self.label = tk.Label(root, text="Stock image")
        self.label.pack(pady=10)
        
        self.btn_load = tk.Button(root, text="Change", command=self.load_template)
        self.btn_load.pack(pady=5)
        
        # Настройка частоты
        self.freq_label = tk.Label(root, text="Scaning speed (Ghz):")
        self.freq_label.pack(pady=5)
        self.freq_var = StringVar(value=str(self.frequency))
        self.freq_menu = OptionMenu(root, self.freq_var, *["12", "16", "24", "32", "48", "60", "90", "120"], command=self.update_frequency)
        self.freq_menu.pack(pady=5)
        
        self.btn_start = tk.Button(root, text="START CTRL+Z", command=self.start_search)
        self.btn_start.pack(pady=5)
        
        self.btn_stop = tk.Button(root, text="STOP CTRL+Z", command=self.stop_search, state="disabled")
        self.btn_stop.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=400, height=300, bg="white")
        self.canvas.pack(pady=10)
        
        self.log_text = Text(root, height=5, width=50)
        self.log_text.pack(pady=10)
        
        self.image_label = None
        
        # Настройка шортката Ctrl+Z
        self.keyboard_thread = Thread(target=self.monitor_shortcut)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def update_frequency(self, value):
        self.frequency = int(value)
        self.log_message(f"Ghz UPDATE! {self.frequency} Гц")
    
    def load_template(self):
        self.template_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if self.template_path:
            self.btn_load.config(text=f"Image: {self.template_path.split('/')[-1]}")
            self.log_message(f"Uploaded Image: {self.template_path.split('/')[-1]}")
    
    def monitor_shortcut(self):
        while True:
            keyboard.wait("ctrl+z")
            if self.template_path:
                if self.is_running:
                    self.stop_search()
                else:
                    self.start_search()
            else:
                self.log_message("ERROR please change one image")
            time.sleep(0.1)  # Защита от множественных срабатываний
    
    def start_search(self):
        if not self.template_path:
            self.log_message("ERROR please change one image")
            return
        
        self.is_running = True
        self.check_count = 0
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log_message("SEARCHING STARTED...")
        
        Thread(target=self.search_image).start()
    
    def stop_search(self):
        self.is_running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.log_message("SEARCHING STOPED")
    
    def move_mouse_to(self, target_x, target_y, screen_width, screen_height):
        # Текущая позиция мыши
        current_x, current_y = pyautogui.position()
        
        # Вычисляем относительное смещение
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Плавное перемещение мыши (несколько шагов)
        steps = 10
        for i in range(steps):
            step_x = dx / steps
            step_y = dy / steps
            pyautogui.moveRel(step_x, step_y, duration=0.01)
            time.sleep(0.01)
    
    def search_image(self):
        # Загрузка эталонного изображения
        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        template_color = cv2.imread(self.template_path, cv2.IMREAD_COLOR)
        template_pil = Image.open(self.template_path)
        
        # Масштабирование шаблона для маленьких изображений
        scale_factor = 2.0 if template.shape[0] < 100 or template.shape[1] < 100 else 1.0
        if scale_factor > 1.0:
            template = cv2.resize(template, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            template_color = cv2.resize(template_color, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            self.log_message(f"Шаблон масштабирован в {scale_factor}x для улучшения детекции")
        
        # Инициализация SIFT
        sift = cv2.SIFT_create()
        kp_template, des_template = sift.detectAndCompute(template, None)
        
        # Проверка на ключевые точки
        if des_template is None or len(kp_template) < 5:
            self.log_message("SIFT error, program use yet hash matching")
            use_sift = False
        else:
            use_sift = True
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        with mss.mss() as sct:
            while self.is_running:
                start_time = time.time()
                self.check_count += 1
                
                # Захват экрана
                monitor = sct.monitors[1]
                screen = sct.grab(monitor)
                screen_np = np.array(screen)
                screen_bgr = cv2.cvtColor(screen_np, cv2.COLOR_RGBA2BGR)
                screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
                screen_width, screen_height = screen_gray.shape[1], screen_gray.shape[0]
                
                found = False
                hash_diff = None
                best_match_val = 0
                
                if use_sift:
                    # Поиск с помощью SIFT
                    kp_screen, des_screen = sift.detectAndCompute(screen_gray, None)
                    
                    if des_screen is None:
                        self.root.after(0, lambda: self.log_message(f"Searching.. {self.check_count}: nope"))
                        time.sleep(1/self.frequency)
                        continue
                    
                    matches = flann.knnMatch(des_template, des_screen, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                    
                    if len(good_matches) > 5:
                        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_screen[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            h, w = template.shape
                            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            
                            self.found_region = dst
                            
                            x, y = np.int32(dst[0][0])
                            x2, y2 = np.int32(dst[2][0])
                            if x2 > x and y2 > y and x >= 0 and y >= 0 and x2 <= screen_gray.shape[1] and y2 <= screen_gray.shape[0]:
                                found_region = screen_bgr[y:y2, x:x2]
                                found_pil = Image.fromarray(cv2.cvtColor(found_region, cv2.COLOR_BGR2RGB))
                                found_hash = imagehash.average_hash(found_pil)
                                
                                hash_diff = imagehash.average_hash(template_pil) - found_hash
                                if hash_diff < 20:
                                    found = True
                
                if not found:
                    # Шаблонное соответствие
                    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
                    best_match_loc = None
                    best_scale = 1.0
                    
                    for scale in scales:
                        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
                            continue
                        
                        result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > best_match_val and max_val > 0.7:
                            best_match_val = max_val
                            best_match_loc = max_loc
                            best_scale = scale
                    
                    if best_match_val > 0.7:
                        h, w = template.shape
                        top_left = best_match_loc
                        bottom_right = (top_left[0] + int(w * best_scale), top_left[1] + int(h * best_scale))
                        
                        pts = np.float32([
                            [top_left[0], top_left[1]],
                            [top_left[0], bottom_right[1]],
                            [bottom_right[0], bottom_right[1]],
                            [bottom_right[0], top_left[1]]
                        ]).reshape(-1, 1, 2)
                        self.found_region = pts
                        
                        found_region = screen_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                        found_pil = Image.fromarray(cv2.cvtColor(found_region, cv2.COLOR_BGR2RGB))
                        found_hash = imagehash.average_hash(found_pil)
                        
                        hash_diff = imagehash.average_hash(template_pil) - found_hash
                        if hash_diff < 20:
                            found = True
                
                if found:
                    self.screenshot = screen_np
                    
                    # Вычисление центра найденной области
                    pts = np.int32(self.found_region).reshape(-1, 2)
                    x_center = int(np.mean(pts[:, 0]))
                    y_center = int(np.mean(pts[:, 1]))
                    
                    # Плавное перемещение мыши к центру
                    self.move_mouse_to(x_center, y_center, screen_width, screen_height)
                    self.log_message(f"\Mouse moved to position: ({x_center}, {y_center})")
                    
                    # Сохранение скриншота
                    self.root.after(0, lambda: self.display_screenshot(hash_diff))
                
                # Логирование
                elapsed = time.time() - start_time
                self.root.after(0, lambda: self.log_message(f"Searching {self.check_count}: SIFT maching {len(good_matches) if use_sift else 0}, Template {best_match_val:.3f}, время {elapsed:.3f}с"))
                
                # Частота проверки
                time.sleep(max(0, 1/self.frequency - elapsed))
    
    def display_screenshot(self, hash_diff):
        if self.screenshot is not None and self.found_region is not None:
            screen_bgr = cv2.cvtColor(self.screenshot, cv2.COLOR_RGBA2BGR)
            overlay = screen_bgr.copy()
            
            pts = self.found_region
            pts = np.int32(pts).reshape(-1, 2)
            
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, screen_bgr, 1 - alpha, 0, screen_bgr)
            
            cv2.imwrite("screenshot.png", screen_bgr)
            
            img = Image.fromarray(cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB))
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            self.screenshot_img = ImageTk.PhotoImage(img)
            if self.image_label:
                self.canvas.delete(self.image_label)
            self.image_label = self.canvas.create_image(200, 150, image=self.screenshot_img)
            self.log_message(f"Object found!! Image has saved as 'screenshot.png'")

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.geometry("500x650")
    root.mainloop()
