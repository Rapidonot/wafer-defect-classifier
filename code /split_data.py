import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    # Dummy fallback if tqdm isn't installed
    tqdm = lambda x, **kwargs: x

WM_PATH = 'LSWMD.pkl'
MIXED_PATH = 'MixedWm38.npz'
IMAGE_SIZE = (224, 224)
NUM_NORMAL_WAFERS = 15000 

MODEL_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", 
    "Loc", "Near-full", "Random", "Scratch"
]

#loading wm811k with python 2
def load_wm811k():
    try:
        import pandas.core.indexes.base
        sys.modules['pandas.indexes'] = pandas.core.indexes
        sys.modules['pandas.indexes.base'] = pandas.core.indexes.base
    except ImportError:
        pass 

    with open(WM_PATH, 'rb') as f:
        df = pickle.load(f, encoding='latin1')

    def clean_label(x):
        try:
            if isinstance(x, (list, np.ndarray, tuple)):
                if len(x) > 0:
                    if isinstance(x[0], (list, np.ndarray, tuple)):
                         if len(x[0]) > 0: return x[0][0]
                    else: return x[0]
            return 'none'
        except: 
            return 'none'

    df['failureType'] = df['failureType'].apply(clean_label)
    
    valid_classes = set(MODEL_CLASSES)
    defective_df = df[df['failureType'].isin(valid_classes)].copy()
    
    normal_df = df[df['failureType'] == 'none'].sample(n=NUM_NORMAL_WAFERS, random_state=42)
    final_df = pd.concat([defective_df, normal_df]).reset_index(drop=True)
    
    total_wafers = len(final_df)
    X = np.zeros((total_wafers, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
    Y = np.zeros((total_wafers, len(MODEL_CLASSES)), dtype=np.float32)

    for idx, row in tqdm(final_df.iterrows(), total=total_wafers):
        w_map = row['waferMap']
        if np.max(w_map) <= 2:
            w_map = (w_map * 127).astype(np.uint8)
            
        img = Image.fromarray(w_map).resize(IMAGE_SIZE, Image.NEAREST)
        X[idx] = np.array(img)
        
        label_str = row['failureType']
        if label_str != 'none':
            class_idx = MODEL_CLASSES.index(label_str)
            Y[idx, class_idx] = 1.0 

    return X, Y

def load_mixedwm():
    data = np.load(MIXED_PATH)
    
    x_raw = data['arr_0'] if 'arr_0' in data else data['x']
    y_raw = data['arr_1'] if 'arr_1' in data else data['y']
    
    total_wafers = len(x_raw)
    X = np.zeros((total_wafers, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.uint8)
    
    for i in tqdm(range(total_wafers)):
        w_map = x_raw[i]
        if len(w_map.shape) == 3: 
            w_map = w_map[:, :, 0]
            
        if np.max(w_map) <= 2:
            w_map = (w_map * 127).astype(np.uint8)
        else:
            w_map = w_map.astype(np.uint8)
            
        img = Image.fromarray(w_map).resize(IMAGE_SIZE, Image.NEAREST)
        X[i] = np.array(img)

    return X, y_raw.astype(np.float32)

def build_and_split_dataset():
    X_wm, Y_wm = load_wm811k()
    X_mix, Y_mix = load_mixedwm()

    X_all = np.concatenate((X_wm, X_mix), axis=0)
    Y_all = np.concatenate((Y_wm, Y_mix), axis=0)
    
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_all, Y_all, test_size=0.30, random_state=42, shuffle=True
    )
    
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.50, random_state=42, shuffle=True
    )

    np.savez_compressed('dataset_70_TRAIN.npz', x=X_train, y=Y_train)
    np.savez_compressed('dataset_15_VAL.npz', x=X_val, y=Y_val)
    np.savez_compressed('dataset_15_TEST.npz', x=X_test, y=Y_test)

if __name__ == '__main__':
    build_and_split_dataset()