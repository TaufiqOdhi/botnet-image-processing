import pandas as pd
import os
from config import BASE_DATASET_DIR, BASE_RESULT_DIR, BASE_GENERATED_IMAGE_DIR
from convert import get_list_pics, convert_pict


def _generate_image(df, image_dir):
    n_feature = df.shape[1]
    list_pics_1, list_pics_2 = get_list_pics(df)
    os.system(f'mkdir -p {image_dir}')
    convert_pict(image_dir=image_dir, list_pics=list_pics_1, n_features=n_feature, pict_prefix='pic1')
    convert_pict(image_dir=image_dir, list_pics=list_pics_2, n_features=n_feature, pict_prefix='pic2')

def convert(benign_filename, mirai_filename, threshold, filter_column=None, corr_method='pearson'):
    # Combine benign dataset and mirai dataset, also add the label (y)
    df_benign = pd.read_csv(f'{BASE_DATASET_DIR}/{benign_filename}')
    df_mirai = pd.read_csv(f'{BASE_DATASET_DIR}/{mirai_filename}')
    if filter_column:
        result_dir = f'{BASE_GENERATED_IMAGE_DIR}/v2/{filter_column}/{threshold}'
        df_benign = df_benign[df_benign.columns[df_benign.columns.str.contains(filter_column)].to_list()]
        df_mirai = df_mirai[df_mirai.columns[df_mirai.columns.str.contains(filter_column)].to_list()]
    else:
        result_dir = f'{BASE_GENERATED_IMAGE_DIR}/v2/all/{threshold}'
    df_benign["y"]=0
    df_mirai['y']=1
    df_all = pd.concat([df_benign, df_mirai])

    # get list of selected features
    corr_all = df_all.corr(method=corr_method)
    abs_corr_all_y = corr_all['y'].apply(lambda x: abs(x))
    abs_corr_all_y = abs_corr_all_y[abs_corr_all_y > threshold]
    feature_selected = abs_corr_all_y.index.to_list()

    # save the correlation desctibtion
    os.system(f'mkdir -p {result_dir}')
    with open(f'{result_dir}/abs_corr_{corr_method}_describe.txt', 'w') as file:
        file.write(abs_corr_all_y.describe().__str__())
        file.close()

    # Generate the image
    _generate_image(df=df_benign[feature_selected[:-1]], image_dir=f'{result_dir}/{benign_filename[:-4]}')
    _generate_image(df=df_mirai[feature_selected[:-1]], image_dir=f'{result_dir}/{mirai_filename[:-4]}')
    