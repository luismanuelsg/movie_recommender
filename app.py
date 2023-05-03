from flask import Flask, render_template, request
from recommenders import Recommenders

app = Flask(__name__)
recos = Recommenders()

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/results')
def results_page():
    user_query = request.args.to_dict()
    recommender_type = user_query['recommender_type']
    user_query.pop('recommender_type')

    #The following only changes the values from string to integers
    user_query = {key:float(value) for key,value in user_query.items()}

    if recommender_type == 'NMF':
        top = recos.NMF_recommender(user_query, 5)
    else:
        top = recos.cos_sim_recommender(user_query, 5)
    
    return render_template('results.html', movies = top)

if __name__ == '__main__':
    app.run(port=5000, debug=True)