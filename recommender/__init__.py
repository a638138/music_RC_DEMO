from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_login import LoginManager
from config import Config
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app._static_folder = './static/'
app.config.from_object(Config)
bcrypt = Bcrypt(app)
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login  = LoginManager(app)
login.login_view = 'login'

#  很重要，一定要放這邊
from recommender import view