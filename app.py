# from flask import Flask, request, jsonify, send_from_directory
# import numpy as np
# from cs410_project_netflix_shows_and_movies import TFIDFSearch, df_revised

# app = Flask(__name__)

# dataset = list(zip(df_revised['title'], df_revised['tokenized_description']))
# ratings = df_revised["rating"].tolist()
# descriptions = df_revised["description"].tolist()

# tr = TFIDFSearch(dataset)
# tr.compute_IDF()

# '''
#     Search endpoint to execute tf-idf ranking pipeline and return top 10 ranked movie results
# '''
# @app.route("/search")
# def search():
#     q = request.args.get("q", "").lower().split()
#     scores = tr.execute_search_TF_IDF(q)

#     top_idx = np.argsort(scores)[-10:][::-1]

#     results = []
#     for i in top_idx:
#         results.append({
#             "title": tr.dataset.iloc[i, 0],
#             "description": descriptions[i],
#             "rating": ratings[i],
#             "score": float(scores[i])
#         })

#     return jsonify(results)

# @app.route("/")
# def home():
#     return send_from_directory(".", "frontend.html")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from cs410_project_netflix_shows_and_movies import TFIDFSearch, df_revised

app = Flask(__name__)

dataset = list(zip(df_revised['title'], df_revised['tokenized_description']))
ratings = df_revised["rating"].tolist()
descriptions = df_revised["description"].tolist()

tr = TFIDFSearch(dataset)
tr.compute_IDF()

'''
    Search endpoint to execute tf-idf ranking pipeline and return top 10 ranked movie results
'''
@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    
    # Check if query is empty
    if not q:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    query_tokens = q.lower().split()
    scores = tr.execute_search_TF_IDF(query_tokens)

    top_idx = np.argsort(scores)[-10:][::-1]

    results = []
    # Check if any relevant documents were found (score > 0)
    has_relevant_docs = any(scores[i] > 0 for i in top_idx)
    
    if not has_relevant_docs:
        return jsonify({"message": "No movies were found with your search. Try different keywords."})
    
    for i in top_idx:
        if scores[i] > 0:  # Only include documents with relevance
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
