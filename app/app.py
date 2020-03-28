from flask import Flask, request, render_template
from fastai.vision import *
# from inference import get_pred

#init class
app = Flask(__name__)

# route
@app.route('/', methods = ['GET', 'POST'])

def covid_func():
    model = load_learner('models')
    if request.method == 'GET':
        return render_template('index.html', value = 'We Got You!')

    if request.method == 'POST':
         print(request.files)
         if 'file' not in request.files:
             print('Image not uploaded')
             return
         file = request.files['file']
         img = open_image(file)
         
         #Getting the Best class
         pred_class_1, idx_1, pred_prob = model.predict(img)
     
         #Getting all best pred
         preds_sorted, idxs = pred_prob.sort(descending=True)

         #Getting  prediction
         pred_1_prob = np.round(100*preds_sorted[0].item(),2)
         
         
         return render_template('result.html',
         label = pred_class_1, category = pred_1_prob)

if __name__ == "__main__":
    app.run(debug=True)