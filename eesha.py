# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import os
import pickle
import requests
import json
from scipy.sparse import csr_matrix
os.system("python recommendation_model.py")

# Your API definition
app = Flask(__name__)
knn = pickle.load(open('trained_model.pkl', 'rb'))
place = pd.read_csv("places.csv")
ratings = pd.read_csv("ratings.csv")
final_dataset = ratings.pivot(index='placeId',columns='user_id',values='ratings')
final_dataset.fillna(0,inplace=True)
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)


def get_place_recommendation(place_name):
	n_places_to_reccomend = 2
	place_list = place[place['title'].str.contains(place_name)]
	if len(place_list):
		place_idx = place_list.iloc[0]['placeId']
		place_idx = final_dataset[final_dataset['placeId'] == place_idx].index[0]
		distances, indices = knn.kneighbors(csr_data[place_idx], n_neighbors=n_places_to_reccomend + 1)
		rec_place_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
								   key=lambda x: x[1])[:0:-1]
		recommend_frame = []
		dict = {}
		for val in rec_place_indices:
			place_idx = final_dataset.iloc[val[0]]['placeId']
			idx = place[place['placeId'] == place_idx].index
			recommend_frame.append({'Title': place.iloc[idx]['title'].values[0], 'Distance': val[1]})
			dict[place.iloc[idx]['title'].values[0]] = val[1]
			#df = pd.DataFrame(recommend_frame, index=range(1, n_places_to_reccomend + 1))
		return dict
	else:
		return "No Places found. Please check your input"


@app.route('/results', methods=['POST', 'GET'])
def results():
	dict = request.args
	for item in dict:
		a = dict.get(item)
		break
	prediction = get_place_recommendation(a)
	return (prediction)
	#result = data['placeName']
	# sending get request and saving the response as response object
	# extracting data in json format
	#data = data['placeName']



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)

