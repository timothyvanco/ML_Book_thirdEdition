from flask import Flask, render_template

app = Flask(__name__)  # __name__ to let Flask know that it can find the HTML template folder (templates) in the same directory where it is located

@app.route('/')

def index():
    return render_template('first_app.html')

if __name__ == '__main__': app.run() # run function to run the application on the server only when this script was directly executed by the Python interpreter