import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

def prepare_and_split_data():
    # Dosya ismindeki tek/çift sayıya göre Normal (0) ve Anomali (1) ayırımı
    data_folder = "../data"
    labels = []
    
    if os.path.exists(data_folder):
        for file_name in os.listdir(data_folder):
            if file_name.endswith(".tim"):
                numbers = re.findall(r'\d+', file_name)
                if len(numbers) >= 2:
                    loop_index = int(numbers[-2])
                    if loop_index % 2 == 1:
                        labels.append(1) # Anomali
                    else:
                        labels.append(0) # Normal

    print("Etiketleme tamamlandı. Toplam veri:", len(labels))