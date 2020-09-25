from flask_wtf import Form
from wtforms import StringField, SubmitField, validators, PasswordField, ValidationError, BooleanField
from wtforms.fields.html5 import EmailField
from recommender.model import UserInfo

class FormRegister(Form):
    """依照Model來建置相對應的Form
    
    password2: 用來確認兩次的密碼輸入相同
    """

    def validate_email(self, field):
        if UserInfo.query.filter_by(email=field.data).first():
            raise validators.ValidationError('Email already register by somebody')

    def validate_username(self, field):
        if UserInfo.query.filter_by(username=field.data).first():
            raise  validators.ValidationError('UserName already register by somebody')

    username = StringField('UserName', validators=[
        validators.DataRequired(),
        # validators.Length(10, 30)
    ])
    email = EmailField('Email', validators=[
        validators.DataRequired(),
        # validators.Length(1, 50),
        validators.Email()
    ])
    password = PasswordField('Password', validators=[
        validators.DataRequired(),
        # validators.Length(5, 10),
        validators.EqualTo('password2', message='PASSWORD NEED MATCH')
    ])
    password2 = PasswordField('Confirm PassWord', validators=[
        validators.DataRequired()
    ])
    submit = SubmitField('Register New Account')

class FormLogin(Form):
    """
    使用者登入使用
    以email為主要登入帳號，密碼需做解碼驗證
    記住我的部份透過flask-login來實現
    """

    email = EmailField('Email', validators=[
        validators.DataRequired(),
        validators.Length(5, 30),
        validators.Email()
    ])

    password = PasswordField('Password', validators=[
        validators.DataRequired()
    ])
    remember_me = BooleanField('Keep Logged in')
    submit = SubmitField('Log in')

