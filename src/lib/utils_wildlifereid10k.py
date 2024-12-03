# %% [code]
import json
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def load_dataframe_heads(root_dir):
    # Load annotations JSON file
    with open(os.path.join(root_dir, 'annotations.json')) as file:
        data = json.load(file)

    # Extract data from the JSON file
    create_dict = lambda i: {'id': i['id'], 'image_id': i['image_id'], 'identity': i['identity'], 'position': i['position']}
    df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
    create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
    df_images = pd.DataFrame([create_dict(i) for i in data['images']])

    # Merge the information from the JSON file
    df = pd.merge(df_annotation, df_images, on='image_id')
    df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
    df = df.drop(['image_id', 'file_name'], axis=1)
    df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:])

    return df

def load_dataframe_full(root_dir):
    # Load annotations JSON file
    with open(os.path.join(root_dir, 'annotations.json')) as file:
        data = json.load(file)

    # Extract data from the JSON file
    create_dict = lambda i: {'id': i['id'], 'bbox': i['bbox'], 'image_id': i['image_id'], 'identity': i['identity'], 'segmentation': i['segmentation'], 'position': i['position']}
    df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])
    create_dict = lambda i: {'file_name': i['path'].split('/')[-1], 'image_id': i['id'], 'date': i['date']}
    df_images = pd.DataFrame([create_dict(i) for i in data['images']])

    # Merge the information from the JSON file
    df = pd.merge(df_annotation, df_images, on='image_id')
    df['path'] = 'images' + os.path.sep + df['identity'] + os.path.sep + df['file_name']        
    df = df.drop(['image_id', 'file_name'], axis=1)
    df['date'] = df['date'].apply(lambda x: x[:4] + '-' + x[5:7] + '-' + x[8:])

    return df

def load_dataframe(root_dir, img_type='heads'):
    if img_type == 'heads':
        return load_dataframe_heads(root_dir)
    elif img_type == 'full':
        return load_dataframe_full(root_dir)
    else:
        raise(Exception('Choose img_type from (heads, full).'))

def get_image(path: str) -> Image:
    '''
    Loads and image and converts it into PIL.Image.
    We load it with OpenCV because PIL does not apply metadata.
    '''    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def plot_grid(
        df: pd.DataFrame,
        root: str,
        n_rows: int = 5,
        n_cols: int = 8,
        offset: float = 10,
        img_min: float = 100,
        rotate: bool = True
        ) -> Image:
    """Plots a grid of size (n_rows, n_cols) with images from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with column `path` (relative path).
        root (str): Root folder where the images are stored. 
        n_rows (int, optional): The number of rows in the grid.
        n_cols (int, optional): The number of columns in the grid.
        offset (float, optional): The offset between images.
        img_min (float, optional): The minimal size of the plotted images.
        rotate (bool, optional): Rotates the images to have the same orientation.

    Returns:
        The plotted grid.
    """

    # Select indices of images to be plotted
    n = min(len(df), n_rows*n_cols)
    idx = np.random.permutation(len(df))[:n]

    # Load images and compute their ratio
    ratios = []
    for k in idx:
        file_path = os.path.join(root, df.iloc[k]['path'])
        im = get_image(file_path)
        ratios.append(im.size[0] / im.size[1])

    # Get the size of the images after being resized
    ratio = np.median(ratios)
    if ratio > 1:    
        img_w, img_h = int(img_min*ratio), int(img_min)
    else:
        img_w, img_h = int(img_min), int(img_min/ratio)

    # Create an empty image grid
    im_grid = Image.new('RGB', (n_cols*img_w + (n_cols-1)*offset, n_rows*img_h + (n_rows-1)*offset))

    # Fill the grid image by image
    for k in range(n):
        i = k // n_cols
        j = k % n_cols

        # Load the image
        file_path = os.path.join(root, df.iloc[idx[k]]['path'])
        im = get_image(file_path)

        # Possibly rotate the image
        if rotate and ((ratio > 1 and im.size[0] < im.size[1]) or (ratio < 1 and im.size[0] > im.size[1])):
            im = im.transpose(Image.ROTATE_90)

        # Rescale the image
        im.thumbnail((img_w,img_h))

        # Place the image on the grid
        pos_x = j*img_w + j*offset
        pos_y = i*img_h + i*offset        
        im_grid.paste(im, (pos_x,pos_y))
    return im_grid