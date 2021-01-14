import os 

class Config:
    DEBUG = True
    # pjdir = os.path.abspath(os.path.dirname(__file__))
    #  新版本的部份預設為none，會有異常，再設置True即可。
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    #  設置資料庫為sqlite3
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + \
                                            # os.path.join('static\\data\\data_register.sqlite')
    USERNAME = ''
    PASSWORD = ''
    SQLALCHEMY_DATABASE_URI = f'postgresql://{USERNAME}:{PASSWORD}@localhost/music_rec'
    
    SECRET_KEY=os.urandom(24)
    SESSION_PROTECTION = 'strong'