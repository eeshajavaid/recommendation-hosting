import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/get_place_recommendation', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        prediction = get_place_recommendation(data)
        def get_place_recommendation(place_name):
                n_places_to_reccomend = 2
                place_list = place[place['title'].str.contains(place_name)]  
                if len(place_list):        
                    place_idx= place_list.iloc[0]['placeId']
                    place_idx = final_dataset[final_dataset['placeId'] == place_idx].index[0]
                    distances , indices = knn.kneighbors(csr_data[place_idx],n_neighbors=n_places_to_reccomend+1)    
                    rec_place_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
                    recommend_frame = []
                    for val in rec_place_indices:
                        place_idx = final_dataset.iloc[val[0]]['placeId']
                        idx = place[place['placeId'] == place_idx].index
                        recommend_frame.append({'Title':place.iloc[idx]['title'].values[0],'Distance':val[1]})
                        df = pd.DataFrame(recommend_frame,index=range(1,n_places_to_reccomend+1))
                        return df
                else:
                        return "No Places found. Please check your input"

    return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)