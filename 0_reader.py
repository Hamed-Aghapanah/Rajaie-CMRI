import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
from tqdm import tqdm  # برای نمایش پیشرفت

# ساخت پوشه‌ها
# mask_RV = 'RV_mask'
image_RV = 'RV_image'
SAMPLE = 'sample'
CONTOURS = 'Contours'

# os.makedirs(mask_RV, exist_ok=True)
os.makedirs(image_RV, exist_ok=True)
os.makedirs(SAMPLE, exist_ok=True)
os.makedirs(CONTOURS, exist_ok=True)

# مسیر پوشه ماسک ها
path_masks = 'masks'
path_images = 'images'

# بررسی وجود پوشه‌ها
if not os.path.exists(path_masks) or not os.path.exists(path_images):
    print("پوشه ماسک‌ها یا تصاویر وجود ندارد. لطفا بررسی کنید.")
    exit()

# لیست تمام ماسک ها
mask_files = [f for f in os.listdir(path_masks) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"تعداد کل ماسک‌ها: {len(mask_files)}")

# پیدا کردن مقادیر unique
unique_values_set = set()
for mask_file in mask_files:
    mask_path = os.path.join(path_masks, mask_file)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    unique_values_set.update(np.unique(mask_array))

uniqe = sorted(list(unique_values_set))
print(f"مقادیر unique در ماسک‌ها: {uniqe}")

# تابع برای پیدا کردن کانتور از ماسک
def find_contours(mask_array):
    contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# تعریف کلیدهای مورد نظر (RV, LV, Myo)
RV_value, LV_value, Myo_value = None, None, None

# تعیین مقادیر RV, LV, Myo اگر موجود باشند
if len(uniqe) > 3:
    RV_value = uniqe[-3]  # فرض بر این است که مقدار RV دومین مقدار unique است
if len(uniqe) > 2:
    LV_value = uniqe[-1]  # فرض بر این است که مقدار LV سومین مقدار unique است
if len(uniqe) > 1:
    Myo_value = uniqe[-2]  # فرض بر این است که مقدار Myo چهارمین مقدار unique است

# تغییر ماسک‌ها و رسم کانتور در صورت وجود مقادیر RV, LV, Myo
# استفاده از tqdm برای نمایش پیشرفت
for mask_file in tqdm(mask_files, desc="در حال پردازش فایل‌ها"):
    # باز کردن ماسک
    mask_path = os.path.join(path_masks, mask_file)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    
    # باز کردن تصویر مربوطه
    image_path = os.path.join(path_images, mask_file)
    if not os.path.exists(image_path):
        print(f"تصویر مربوط به {mask_file} یافت نشد.")
        continue
    
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # تبدیل تصویر به RGB اگر سیاه و سفید است
    if len(image_array.shape) == 2 or image_array.shape[2] == 1:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    
    # رسم کانتورها در صورت وجود RV, LV, Myo
    contour_image = image_array.copy()
    
    if RV_value is not None:
        new_mask_array_RV = np.where(mask_array == RV_value, 255, 0)
        contours_RV = find_contours(new_mask_array_RV)
        cv2.drawContours(contour_image, contours_RV, -1, (255, 0, 0), 2)  # Red for RV
    
    if LV_value is not None:
        new_mask_array_LV = np.where(mask_array == LV_value, 255, 0)
        contours_LV = find_contours(new_mask_array_LV)
        cv2.drawContours(contour_image, contours_LV, -1, (0, 0, 255), 2)  # Blue for LV
    
    if Myo_value is not None:
        new_mask_array_Myo = np.where(mask_array == Myo_value, 255, 0)
        contours_Myo = find_contours(new_mask_array_Myo)
        cv2.drawContours(contour_image, contours_Myo, -1, (0, 255, 0), 2)  # Green for Myo
    
    # تبدیل آرایه تصویر به فرمت PIL برای ذخیره
    contour_image_pil = Image.fromarray(contour_image.astype(np.uint8))
    
    # مسیر ذخیره کانتور در پوشه Contours
    contour_save_path = os.path.join(CONTOURS, mask_file)
    
    # ذخیره تصویر با کانتورها به صورت رنگی
    contour_image_pil.save(contour_save_path)
    
    # نمونه‌سازی رنگی برای RV, LV, Myo
    sample = np.zeros([np.shape(mask_array)[0], np.shape(mask_array)[1], 3])
    
    if RV_value is not None:
        sample[:, :, 0] = np.where(mask_array == RV_value, 255, 0)  # Red for RV
    if LV_value is not None:
        sample[:, :, 2] = np.where(mask_array == LV_value, 255, 0)  # Green for LV
    if Myo_value is not None:
        sample[:, :, 1] = np.where(mask_array == Myo_value, 255, 0)  # Blue for Myo
    
    # تبدیل نمونه به تصویر و ذخیره آن
    sample_image_pil = Image.fromarray(sample.astype(np.uint8))
    sample_save_path = os.path.join(SAMPLE, mask_file)
    sample_image_pil.save(sample_save_path)
