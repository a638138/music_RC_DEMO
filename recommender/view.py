#  這是view.py的import
from recommender import app, db, login, bcrypt
from recommender.model import UserInfo, SessionInfo
from recommender.form import FormRegister, FormLogin
from rec_model import rec_model
from sqlalchemy import func
from flask import render_template, flash, request, redirect, url_for, session, make_response, jsonify
from flask_bootstrap import Bootstrap
from flask_login import login_user, current_user, login_required, logout_user
import pandas as pd

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = FormRegister()
    if form.validate_on_submit():
        user = UserInfo(
            username = form.username.data,
            email = form.email.data,
            password = form.password.data,
            is_newuser = True,
        )
        user.password = bcrypt.generate_password_hash(password=user.password).decode('utf8')
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('register.html', form=form)

@app.route('/')  
@login_required 
def index():
    # if current_user.is_newuser:
    #     movie_info_list = rec_model.pop_predict()
    # else:
    #     table_df = pd.read_sql_table('SessionInfo', con=db.engine)
    #     user_data = table_df[table_df.u_id == current_user.id]
    #     # movie_info_list = rec_model.HRNN_predict(user_data)
    # # if movie_info_list == None:
    # movie_info_list = movie_info_list = [
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'}
    #         ]
    # itemnum = len(movie_info_list)


    # return render_template('index.html', movie_info_list=movie_info_list, itemnum=itemnum)
    return render_template('index.html')




@app.route('/media/<user_id>', methods=['GET', 'POST'])  
# @app.route('/media/<song_id>/<movie_id>', methods=['GET', 'POST'])  
@login_required 
# def media(song_id, movie_id):
def media(user_id):

    if request.method == 'POST':
        update_user = UserInfo.query.filter_by(id=user_id).first()
        update_user.is_newuser = False
        u_id = update_user.id
        session_song_data = SessionInfo(u_id=u_id,
                                session_id = int(session['current_session_id']),
                                counter = int(session['counter']),
                                song_id = int(request.form.get('song_id')),
            )
        session['counter'] = str(int(session['counter'])+1)
        db.session.add(session_song_data)
        db.session.add(update_user)
        db.session.commit()
        return '', 204
        
    if current_user.is_newuser:
        movie_info_list = rec_model.pop_predict()
    else:
        table_df = pd.read_sql_table('SessionInfo', con=db.engine)
        user_data = table_df[table_df.u_id == current_user.id]
        movie_info_list = rec_model.HRNN_predict(user_data)
    # if movie_info_list == None:
    # movie_info_list = movie_info_list = [
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'},
    #         {'song_id':60973, 'movie_id':'rVFR7wDZT9A', 'title':'Leroy Anderson: Ritvélin (The Tyd Symphony Orchestra Bernharður Wilkinson, conductor Steef van Oosterhout, soloist on a typewriter Ævar vísindamaður, ...', 'desc':'Provided to YouTube by Universal Music Group Peace/Dolphin Dance', 'thumbnail':'https://i.ytimg.com/vi/iaWS3CGVVJg/hqdefault.jpg'}
    #         ]
    # movie_info_list = rec_model.pop_predict()
    # print(movie_info_list)
    return jsonify(movie_info_list)
  
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = FormLogin()
    if form.validate_on_submit():
        #  當使用者按下login之後，先檢核帳號是否存在系統內。
        user = UserInfo.query.filter_by(email=form.email.data).first()
        if user:
            #  當使用者存在資料庫內再核對密碼是否正確。
            if user.check_password(form.password.data):
                session['song_event'] = ''
                session_id_max = db.session.query(func.max(SessionInfo.session_id)).all()[0][0]
                session['current_session_id'] = str(session_id_max+1 if session_id_max != None else 1)
                session['counter'] = str(1)
                #  加入參數『記得我』
                login_user(user, form.remember_me.data)
                #  使用者登入之後，將使用者導回來源url。
                #  利用request來取得參數next
                next = request.args.get('next')
                #  自定義一個驗證的function來確認使用者是否確實有該url的權限
                if not next_is_valid(next):
                    #  如果使用者沒有該url權限，那就reject掉。
                    return 'Bad Boy!!'
                return redirect(next or url_for('index'))
            else:
                #  如果密碼驗證錯誤，就顯示錯誤訊息。
                flash('Wrong Email or Password')
        else:
            #  如果資料庫無此帳號，就顯示錯誤訊息。
            flash('Wrong Email or Password')
    return render_template('login.html', form=form)
             
#  加入function
def next_is_valid(url):
    """
    為了避免被重新定向的url攻擊，必需先確認該名使用者是否有相關的權限，
    舉例來說，如果使用者調用了一個刪除所有資料的uri，那就GG了，是吧 。
    :param url: 重新定向的網址
    :return: boolean
    """
    return True


#  調整logout的route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Log Out See You.')
    session.clear()
    return redirect(url_for('login'))
  
# @app.route('/userinfo')  
# def userinfo():  
#     return 'Here is UserINFO'

