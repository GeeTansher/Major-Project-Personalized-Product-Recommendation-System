import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl


PAD = 0
MASK = 1

def processing(data_csv_path, featureModel, list,modelRating):
    data = pd.read_csv(data_csv_path)
    data, mapping, _ = map_column(data, col_name="product_id")
    model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )
    
    model.eval()
    model.load_state_dict(torch.load(featureModel,map_location ='cpu')["state_dict"])
    
    id2mapid = {a: mapping[b] for a, b in zip(data.product_id.tolist(), data.product_id.tolist()) if b in mapping}
    map2id = {v: k for k, v in id2mapid.items()}


    top_products = predictId(list, model, id2mapid, map2id)

    item_encoder = LabelEncoder()
    data['items'] = item_encoder.fit_transform(data['product_id'])
    items = item_encoder.fit_transform(top_products)
    # Example user and item data
    target_user_id = 32158956

    user_encoder = LabelEncoder()
    user_encoder.fit([target_user_id])

    item_mapping = {item: index for index, item in enumerate(items)}

    encoded_user_id = user_encoder.transform([target_user_id])[0]

    encoded_item_ids = np.array([item_mapping[item] for item in items])

    num_users = 1
    num_items = len(items)
    model_predictions = np.random.rand(num_users, num_items)
    user_ids = np.full(num_items, encoded_user_id)
    predictions = modelRating.predict([user_ids, encoded_item_ids]).flatten()
    predictions = model_predictions[encoded_user_id]

    N = 3 # Number of recommendations you want to provide
    recommended_item_indices = predictions.argsort()[::-1][:N]
    decoded_items = item_encoder.inverse_transform(recommended_item_indices)
    top_products_title = idToTitle(decoded_items.tolist()[:3], data)
    
    # return decoded_items.tolist()[:3],top_products_title
    return top_products_title

def map_column(df: pd.DataFrame, col_name: str):
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def predictId(list_products, model, id2mapid, map2id):
    
    ids = [PAD] * (120 - len(list_products) - 1) + [id2mapid[a] for a in list_products] + [MASK]
    
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(src)
    
    masked_pred = prediction[0, -1].numpy()
    
    sorted_predicted_ids = np.argsort(masked_pred).tolist()[::-1]
    
    sorted_predicted_ids = [a for a in sorted_predicted_ids if a not in ids]
    
    return [map2id[a] for a in sorted_predicted_ids[:30] if a in map2id]

def idToTitle(list, data):
    products_set = []
    
    for id in list:
        for i in range(len(data.product_id)):
            if id == data.product_id[i]:
                rating = 4
                if len(str(data.iloc[i]['rating'])) == 1:
                    rating = int(data.iloc[i]['rating'])
                
                product_dict = {
                    "product_id": data.iloc[i]['product_id'],
                    "product_title": data.iloc[i]['product_title'],
                    "product_category": data.iloc[i]['product_category'],
                    "rating": rating
                }
                products_set.append(product_dict)
                break
    
    return products_set

class Recommender(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=128,
        cap=0,
        mask=1,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=channels
        )

        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.linear_out = Linear(channels, self.vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        src_items = self.item_embeddings(src_items)

        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src_items += pos_encoder

        src = src_items.permute(1, 0, 2)

        src = self.encoder(src)

        return src.permute(1, 0, 2)

    def forward(self, src_items):

        src = self.encode_src(src_items)

        out = self.linear_out(src)

        return out
