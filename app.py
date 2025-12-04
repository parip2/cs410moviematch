from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from cs410_project_netflix_shows_and_movies import TFIDFSearch, df_revised

app = Flask(__name__)

dataset = list(zip(df_revised['title'], df_revised['tokenized_description']))
ratings = df_revised["rating"].tolist()
descriptions = df_revised["description"].tolist()

tr = TFIDFSearch(dataset)

@app.route("/search")
def search():
    q = request.args.get("q", "").lower().split()
    scores = tr.execute_search_TF_IDF(q)

    top_idx = np.argsort(scores)[-10:][::-1]

    results = []
    for i in top_idx:
        results.append({
            "title": tr.dataset.iloc[i, 0],
            "description": descriptions[i],
            "rating": ratings[i],
            "score": float(scores[i])
        })

    return jsonify(results)

@app.route("/")
def home():
    return send_from_directory(".", "frontend.html")

if __name__ == "__main__":
    app.run(debug=True)

