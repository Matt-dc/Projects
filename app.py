from flask import Flask, render_template, request

import tweepy_streamer as ts


app = Flask(__name__)




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_user():
    user = request.form['handle']

    tweeter = ts.Tweeter()

#####    methods to implement   #####
    latest = tweeter.latest_tweets(user, 5)
    popular = tweeter.most_popular(user, 10000)
    positivity = tweeter.positivity_rating(user, 200)    
    most_common = tweeter.most_common_words(user, 100)
    posting_frequency = tweeter.posting_frequency(user, 200)

    # to_html() ??
    return render_template('index.html', 
                            latest=latest.to_html(), 
                            popular=popular.to_html(), 
                            positivity=positivity, 
                            most_common=most_common, 
                            posting_frequency=posting_frequency
                            )




if __name__ == "__main__":
    app.run(debug=True)