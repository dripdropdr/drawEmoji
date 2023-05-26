import pandas as pd
from matplotlib import pyplot as plt
import ast
from PIL import Image , ImageDraw
from sklearn.preprocessing import OneHotEncoder
import time
import numpy as np
import os


def inspection(file_name):
    df = pd.read_csv(file_name)
    df['word'] = df['word'].replace(' ','_',regex = True)
    print(type(df['recognized'][0])) 

    idx= df.iloc[:5].index 
    print(df.loc[idx,'recognized'].values) 

    for i in range(len(df.loc[idx,'drawing'].values)) : 
        if df.loc[idx,'recognized'].values[i] == True : 
            print(i, end=' ') 
            
    idx= df.iloc[:2000].index 
    T_cnt = 0 
    F_cnt = 0 

    for i in range(len(df.loc[idx,'drawing'].values)) : 
        if df.loc[idx,'recognized'].values[i] == True : 
            T_cnt += 1 
        else : 
            F_cnt += 1 
            
    print('\nTrue Count :',T_cnt)
    print('False Count :',F_cnt)
    print(df.head())


def visualize(file_name):
    def check_draw(img_arr) : 
        k=3 
        for i in range(len(img_arr[k])): 
            img = plt.plot(img_arr[k][i][0],img_arr[k][i][1]) 
            plt.scatter(img_arr[k][i][0],img_arr[k][i][1]) 
            
        plt.xlim(0,256) 
        plt.ylim(0,256) 
        plt.gca().invert_yaxis()
        # plt.show()
    
    df = pd.read_csv(file_name)
    ten_ids = df.iloc[:10].index
    img_arr = [ast.literal_eval(lst) for lst in df.loc[ten_ids,'drawing'].values] 
    #ast.literal_eval is squence data made string to array 
    # print(img_arr[0])
    # check_draw(img_arr)
    img = make_img(img_arr[3]) 
    # img = img.resize((64,64))
    plt.imshow(img)
    plt.show()


def make_img(img_arr) : 
        image = Image.new("P", (256,256), color=255) 
        image_draw = ImageDraw.Draw(image) 
        for stroke in img_arr: 
            for i in range(len(stroke[0])-1): 
                image_draw.line([stroke[0][i], stroke[1][i], stroke[0][i+1], stroke[1][i+1]], fill=0, width=5) 
        
        return image


def preprocessing(data_dir) : 
    img_batch = 3000
    class_label = []
    val= []
    
    for fname in [f for f in os.listdir(data_dir) if f.endswith('csv')] :
        df = pd.read_csv(os.path.join(data_dir, fname), encoding='CP949') 
        df['word'] = df['word'].replace(' ','_',regex = True)
        class_label.append(df['word'][0])
        df = df[df['recognized'] == True]
        keys = df.iloc[img_batch:img_batch+1000].index
        
        for i in range(len(df.loc[keys, 'drawing'].values)) :
            drawing = ast.literal_eval(df.loc[keys,'drawing'].values[i])
            key_id = df.loc[keys, 'key_id'].values[i]
            img = make_img(drawing)
            img = img.resize((64,64))
            img.save(f'./val/images/{key_id}.jpg', "BMP")
            val.append([key_id, df.loc[keys, 'word'].values[i]])
        
    labels = pd.DataFrame(val, columns=['key_id', 'label'])
    labels.to_csv('./val/test.csv')


if __name__ == '__main__':
    dirname = './train_simplified'
    file_name = dirname+'/'+'angel.csv'
    # inspection(file_name)
    # visualize(file_name)
    preprocessing(dirname)