from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation
import collaborative_recommendation
import hybrid_recommendation

app = Flask(__name__)
CORS(app)


@app.route('/movie/cbf', methods=['GET'])
def recommend():
    print("cbf: ", request.args.getlist('movies[]'))
    print('\n')
    res = recommendation.recommend(request.args.getlist('movies[]'))
    return jsonify({
        "code": 200,
        "status": "Success",
        "message": "Data fetched successfully",
        "result": res
    })


@app.route('/movie/cf', methods=['GET'])
def recommend_collaborative():
    print("cf: ", request.args.getlist('movies[]'))
    print('\n')
    res = collaborative_recommendation.recommend_collaborative(
        request.args.getlist('movies[]'))

    return jsonify({
        "code": 200,
        "status": "Success",
        "message": "Data fetched successfully",
        "result": res
    })


@app.route('/movie/hybrid', methods=['GET'])
def recommend_hybrid():
    print("Hybrid: ", request.args.getlist('movies[]'))
    print('\n')
    res = hybrid_recommendation.recommend_hybrid(
        request.args.getlist('movies[]'))

    return jsonify({
        "code": 200,
        "status": "Success",
        "message": "Data fetched successfully",
        "result": res
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
