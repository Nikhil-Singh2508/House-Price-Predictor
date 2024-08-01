from flask import jsonify, request, Flask
import util2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

util2.load_saved_artifacts()


@app.route('/get_location_names')
def get_location_names():
    response = jsonify({'locations': util2.get_location_names()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_prices():
    total_sqft = float(request.form['total_sqft'])
    location = (request.form['location'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util2.get_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Flask Application")
    app.run()
