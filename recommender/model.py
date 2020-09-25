from recommender import db, login, bcrypt
from flask_login import UserMixin


class UserInfo(UserMixin, db.Model):
    """記錄使用者資料的資料表"""
    __tablename__ = 'Userinfo'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    is_newuser = db.Column(db.Boolean, nullable=False)

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
    def __repr__(self):
        return 'username:%s, email:%s' % (self.username, self.email)

class SessionInfo(db.Model):
    """記錄Session資訊的資料表"""
    __tablename__ = 'SessionInfo'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    u_id = db.Column(db.Integer, nullable=False)
    session_id = db.Column(db.Integer, nullable=False)
    counter = db.Column(db.Integer, nullable=False)
    song_id = db.Column(db.Integer, nullable=False)
    def __repr__(self):
        return 'session:%s' % (self.id)



# ToDO 搜尋功能
# class Song(db.Model):
#     """記錄Session資訊的資料表"""
#     __tablename__ = 'Song'
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     u_id = db.Column(db.Integer, nullable=False)
#     session_id = db.Column(db.Integer, nullable=False)
#     counter = db.Column(db.Integer, nullable=False)
#     song_id = db.Column(db.Integer, nullable=False)
#     def __repr__(self):
#         return 'session:%s' % (self.id)
    
@login.user_loader  
def load_user(user_id):  
    return UserInfo.query.get(int(user_id))