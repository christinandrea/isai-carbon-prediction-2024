import cv2
# from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import boto3
import pickle
import json
import os
from sklearn.preprocessing import MaxAbsScaler

# app = Flask(__name__)

# @app.route('/preprocess-test', methods=['POST'])
client = boto3.client('lambda')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    # TODO implement
    bucket = "kalkulator-isai"
    key = event['file']
    # location = boto3.client('s3').get_bucket_location(Bucket=bucket)['LocationConstraint']
    url = "https://%s.s3.us-east-1.amazonaws.com/%s" % (bucket, key)
    uuid = key.split(".")[0]
    #   if 'image' not in
    df = pd.DataFrame(columns=['Feature Extraction', 'Descriptors'])

    os.chdir('/tmp')
    with open('filename.png', 'wb') as data:
        s3.download_fileobj(bucket, key, data)

    img  = cv2.imread(data.name,cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    # resized = cv2.resize(img, (30, 30))
    height, width = img.shape

# Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the starting and ending points of the crop
    start_x = max(center_x - 15, 0)
    start_y = max(center_y - 15, 0)
    end_x = start_x + 30
    end_y = start_y + 30

    # Make sure the crop dimensions are within the image bounds
    end_x = min(end_x, width)
    end_y = min(end_y, height)

    # Crop the image
    cropped_image = img[start_y:end_y, start_x:end_x]
    print(cropped_image.shape)
    if cropped_image.shape != (30, 30):
            cropped_image = cv2.resize(cropped_image, (30, 30))

    cropped_image = img[start_y:end_y, start_x:end_x]
    print(cropped_image.shape)
    if cropped_image.shape != (30, 30):
            cropped_image = cv2.resize(cropped_image, (30, 30))
        # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    cl_img = clahe.apply(cropped_image)

    # Detect FAST keypoints
    fast = cv2.FastFeatureDetector_create()
    img_kp = fast.detect(cl_img)
    pts_new = [img_kp[x].pt for x in range(len(img_kp))]
    np_newlist = np.array(pts_new, dtype=np.float32).tolist()
    sift = cv2.SIFT_create()
    kp, des = sift.compute(cl_img, img_kp, descriptors=np.array([]))

    # Convert descriptors to a list
    sift_descriptors = np.array(des,dtype=np.float32).tolist()

    df = df._append(pd.Series([np_newlist,sift_descriptors], index = ['Feature Extraction', 'Descriptors']), ignore_index = True )

    df['Feature Extraction'] = df['Feature Extraction'].apply(lambda x: [item for sublist in x for item in sublist])
    df['Descriptors'] = df['Descriptors'].apply(lambda x: [item for sublist in x for item in sublist])

    df_new = pd.DataFrame()

    for index, row in df.iterrows():
        # Extract 'Feature Extraction' and 'Descriptors'
        feature_extraction = row['Feature Extraction']
        descriptors = row['Descriptors']

        max_length = max(len(feature_extraction), len(descriptors))

        padded_feature_extraction = np.pad(feature_extraction, (0, max_length - len(feature_extraction) ), mode='constant')

        padded_descriptors = np.pad(descriptors, (0, max_length - len(descriptors)), mode='constant')

        feature_extraction_cols = [f'Feature Extraction {i}' for i in range(max_length)]
        descriptors_cols = [f'Descriptors {i}' for i in range(max_length)]

        new_row_data = {
            **{col: val for col, val in zip(feature_extraction_cols, padded_feature_extraction)},
            **{col: val for col, val in zip(descriptors_cols, padded_descriptors)}
        }

        df_new = pd.concat([df_new, pd.DataFrame(new_row_data, index=[0])], ignore_index=True)
        
    with open('/opt/model/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    value_df = df_new

    value_df_flattened = np.array([x.flatten() for x in value_df.values])

    value_df_normalized = scaler.fit_transform(value_df_flattened)

    value_df_reshaped = value_df_normalized.reshape((value_df_normalized.shape[0], value_df_normalized.shape[1], 1))

    keras_model = load_model('/opt/model/model_fast_v11.h5')

    prediction = keras_model.predict(value_df_reshaped)

    result = float(prediction[0][0])
    s3.delete_object(Bucket=bucket, Key=key)
    result = round(result,2)   

    return {
        'statusCode': 200,
        'body': {
	     	'carbon' : result
	}
   }
