from flask import Flask, request, render_template
from inference import get_pred
from fastai.vision import *
# from inference import get_pred

#init class
app = Flask(__name__)

# route
@app.route('/', methods = ['GET', 'POST'])

def covid_func():
    if request.method == 'GET':
        return render_template('index.html', value = 'We Got You!')

    if request.method == 'POST':
         print(request.files)
         if 'file' not in request.files:
             print('Image not uploaded')
             return
         file = request.files['file']
         img = open_image(file)
         model = load_learner('models')
         pred_class, pred_idx, outputs = model.predict(img)
         return render_template('result.html',
         label = pred_class, category = outputs,idx = pred_idx)

if __name__ == "__main__":
    app.run(debug=True)