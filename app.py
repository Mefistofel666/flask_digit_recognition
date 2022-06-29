from flask import Flask, redirect, render_template, url_for, request, jsonify
from PIL import Image
from io import BytesIO
import base64

from preprocess_image import preprocess
from prediction import predict

app = Flask(__name__)

pred = 0    

data = {
    'predict': pred
}



@app.route('/home')
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data_url = request.form['save_image'].replace('data:image/png;base64,' ,'')
        # decode base64 string to bytes object
        img_bytes = base64.b64decode(data_url)
        img = Image.open(BytesIO(img_bytes))
        img = preprocess(img)
        # get predict from NN
        pred = predict(img)
        print(pred)
        data['predict'] = pred
        return jsonify(data)
    return render_template('index.html')
   
    
# @app.route('/', methods=['GET'])
# def test():
#     return jsonify(data)


@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)