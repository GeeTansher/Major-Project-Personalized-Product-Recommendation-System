from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import torch
from utils import processing
import json
from tensorflow.keras.models import load_model


app = Flask(__name__)

PAD = 0
MASK = 1

def predictId(list_products:list, model, id2mapid, map2id):
    
    ids = [PAD] * (120 - len(list_products) - 1) + [id2mapid[a] for a in list_products] + [MASK]
    
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(src)
    
    masked_pred = prediction[0, -1].numpy()
    
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    
    return [map2id[a] for a in sorted_predicted_ids[:5] if a in map2id]

@app.route('/')
def home():
    return "Hello world"

# API ROUTING 
@app.route('/recommend' ,methods=['POST'])
def getRecommend():
    data = request.form.get('interaction')
    category = request.form.get('category')
    data = json.loads(data)
    
    # print(category)
    # print(data)
    
    data_csv_path = ""
    featureModel = ""
    model_path=""
    if category == 'Apparel':
        data_csv_path = "./Dataset/Apparel.csv"
        featureModel = "./Models/recommender-apparal.ckpt"
        model_path = './Models/ApparelNew2.h5'
    elif category == 'Jewellery':
        data_csv_path = "./Dataset/Jewelry.csv"
        featureModel = "./Models/recommender-jewellery-v1.ckpt"
        model_path = './Models/AugJewelry.h5'
    elif category == 'Luggage':
        data_csv_path = "./Dataset/Luggage.csv"
        featureModel = "./Models/recommender-luggage.ckpt"
        model_path = './Models/Luggage.h5'
    elif category == 'Watches':
        data_csv_path = "./Dataset/Watches.csv"
        featureModel = "./Models/recommender-watches.ckpt"
        model_path = './Models/Watches.h5'
    elif category == 'Shoes':
        data_csv_path = "./Dataset/Shoes.csv"
        featureModel = "./Models/recommender-shoes.ckpt"
        model_path = './Models/Shoes.h5'
    elif category == 'Beauty':
        data_csv_path = "./Dataset/Beauty.csv"
        featureModel = "./Models/recommender-beauty.ckpt"
        model_path = './Models/Beauty.h5'
    elif category == 'GiftCard':
        data_csv_path = "./Dataset/GiftCard.csv"
        featureModel = "./Models/recommender-gift-card.ckpt"
        model_path = './Models/AugGiftCard.h5'
    else:
        data_csv_path = "./Dataset/Global.csv"
        featureModel = "./Models/recommender-main.ckpt"
        model_path = './Models/WholeDatasetModel.h5'

    
    model = load_model(model_path)
    response_by_category = processing(data_csv_path, featureModel, data,model)
    
    
    data_csv_path = "./Dataset/Global.csv"
    featureModel = "./Models/recommender-main.ckpt"
    model_path = './Models/WholeDatasetModel.h5'
    model = load_model(model_path)
    response_by_global = processing(data_csv_path, featureModel, data,model)

    return jsonify({"categoryProductIds":response_by_category,"globalProductIds":response_by_global})
  


@app.route('/productslist')
def sendData():
    category = request.args.get('category')
    data = pd.read_csv("./Dataset/"+category+".csv")
    products_set = []

    unique_product_ids = set() 

    for i in range(len(data)):
        rating = 4
        if len(str(data.iloc[i]['rating'])) == 1:
            rating = int(data.iloc[i]['rating'])

        product_id = data.iloc[i]['product_id']
        
        # Check if the product ID is unique and the set does not exceed 20 items
        if product_id not in unique_product_ids:
            product_dict = {
                "product_id": product_id,
                "product_title": data.iloc[i]['product_title'],
                "product_category": data.iloc[i]['product_category'],
                "rating": rating
            }
            products_set.append(product_dict)
            unique_product_ids.add(product_id)
        
        if len(products_set) >= 20:
            break
    
    print(len(products_set))
    print(category)

    return jsonify({"products_list": products_set})
    

# if __name__ == '__main__':
#     app.run(debug=False)

