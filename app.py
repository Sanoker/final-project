from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import sqlite3
import random
app = Flask(__name__)
app.secret_key = '7373'


app.config['UPLOAD_FOLDER'] = 'D:/Final_code/static/uploads'



conn = sqlite3.connect("database.db")
conn.execute("CREATE TABLE IF NOT EXISTS customer (id INTEGER PRIMARY KEY, name TEXT, email TEXT, password TEXT)")
conn.close()
import numpy as np
import tensorflow.keras as keras
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
import cv2
import imutils
import os
import time
from glob import glob
import numpy as np
import os
import pandas as pd

df = pd.read_csv('final.csv')

csvFile=np.array(df)

x=csvFile[:,0]
targetFM=x[~np.isnan(x)]/1000
FM=targetFM
x=csvFile[:,1]
targetFMM=x[~np.isnan(x)]/1000
FMM=targetFMM
x=csvFile[:,2]
targetFS=x[~np.isnan(x)]/1000
FS=targetFS


x=csvFile[:,3]
targetRM=x[~np.isnan(x)]/1000
RM=targetRM
x=csvFile[:,4]
targetRMM=x[~np.isnan(x)]/1000
RMM=targetRMM
x=csvFile[:,5]
targetRS=x[~np.isnan(x)]/1000
RS=targetRS

x=csvFile[:,6]
targetSM=x[~np.isnan(x)]/1000
SM=targetSM
x=csvFile[:,7]
targetSMM=x[~np.isnan(x)]/1000
SMM=targetSMM
x=csvFile[:,8]
targetSS=x[~np.isnan(x)]/1000
SS=targetSS
clas1=[item[10:-1] for item in sorted(glob("./dataset/*/"))]

import os
from os import listdir
 
# get the path/directory
folder_dir1 = "./dataset/frontminor"
frontm=[]
frontmc=[]
for i in range(len(os.listdir(folder_dir1))):
    frontm.append(random.randrange(1, 17, 3))
    frontmc.append(random.randrange(5000, 10000, 3))

folder_dir2 = "./dataset/frontmoderate"
frontmo=[]
frontmoc=[]
for i in range(len(os.listdir(folder_dir2))):
    frontmo.append(random.randrange(3, 17, 3))
    frontmoc.append(random.randrange(1000, 50000, 3))

folder_dir3 = "./dataset/frontsevere"
frontms=[]
frontmsc=[]
for i in range(len(os.listdir(folder_dir3))):
    frontms.append(random.randrange(8, 17, 3))
    frontmsc.append(random.randrange(40000, 100000, 3))

folder_dir4 = "./dataset/Rearminor"
frontr=[]
frontrc=[]
for i in range(len(os.listdir(folder_dir4))):
    frontr.append(random.randrange(1, 17, 3))
    frontrc.append(random.randrange(5000, 10000, 3))
folder_dir5= "./dataset/Rearmoderate"
frontrm=[]
frontrmc=[]
for i in range(len(os.listdir(folder_dir5))):
    frontrm.append(random.randrange(1, 17, 3))
    frontrmc.append(random.randrange(10000, 50000, 3))

folder_dir6 = "./dataset/Rearsevere"
frontrs=[]
frontrsc=[]
for i in range(len(os.listdir(folder_dir6))):
    frontrs.append(random.randrange(1, 17, 3))
    frontrsc.append(random.randrange(40000, 100000, 3))

folder_dir7 = "./dataset/sideminor"
frontsm=[]
frontsmc=[]
for i in range(len(os.listdir(folder_dir7))):
    frontsm.append(random.randrange(1, 17, 3))
    frontsmc.append(random.randrange(5000, 10000, 3))


folder_dir8 = "./dataset/sidemoderate"
frontsmm=[]
frontsmmc=[]
for i in range(len(os.listdir(folder_dir8))):
    frontsmm.append(random.randrange(1, 17, 3))
    frontsmmc.append(random.randrange(10000, 50000, 3))

folder_dir9 = "./dataset/sidesevere"
frontss=[]
frontssc=[]
for i in range(len(os.listdir(folder_dir9))):
    frontss.append(random.randrange(1, 17, 3))
    frontssc.append(random.randrange(40000, 100000, 3))




from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm


from tkinter import filedialog
# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    #print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)


import random
from tensorflow.keras.models import load_model


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/damage_assessment', methods=['GET', 'POST'])
def damage_assessment():
    # Check if the user is logged in??
    if 'username' not in session:
        flash('You need to login first', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        image_data = request.files['image']
        
        if image_data.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if image_data:
            filename = secure_filename(image_data.filename)
            print(filename)
            files = glob('./static/uploads/*')
            for f in files:
                os.remove(f)
            #filename="temp.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_data.save(image_path)  # Save uploaded image
            #img = cv2.imread("./static/uploads/temp.png")



            # Render the same page with the uploaded image path
            return render_template('damage_assessment.html', image_path=filename)
#Given Car Image Predicted as   Predicted Damage cost is 
    return render_template('damage_assessment.html', image_path=None)

@app.route('/next_step/<image_filename>/<G>/<P>/<C>')
def next_step(image_filename, G,P,C): #if you want change message1,2,cost,total cost
        model2 = load_model('body.h5')
        file1="./static/uploads/temp.png"
        file=os.listdir('./static/uploads')
        print('./static/uploads/'+file[0])
        file1='./static/uploads/'+file[0]
        test_tensors = paths_to_tensor(file1)/255
        pred1=model2.predict(test_tensors)
        #print(pred1)
        pred1=np.argmax(pred1);
        print(pred1)
        if pred1==0:
            ij=0
            for images in os.listdir(folder_dir4):
                if file[0]==images:
                    old=frontr[ij]
                    cost1=frontrc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/FM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(FM[pred]*500)+random.randrange(200, 1000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "++str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
                
        if pred1==1:
            ij=0
            for images in os.listdir(folder_dir5):
                if file[0]==images:
                    old=frontrm[ij]
                    cost1=frontrmc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/FMM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(FMM[pred]*700)+random.randrange(200, 1000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
        if pred1==2:
            ij=0
            for images in os.listdir(folder_dir6):
                if file[0]==images:
                    old=frontrs[ij]
                    cost1=frontrsc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/FS.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(FS[pred]*1000)+random.randrange(2000, 10000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"

        if pred1==3:
            ij=0
            for images in os.listdir(folder_dir1):
                print(images+file1)
                if file[0]==images:
                    old=frontm[ij]
                    cost1=frontmc[ij]
                    break
                ij+=1
                
                    
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/RM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(RM[pred]*500)+random.randrange(200, 1000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
        if pred1==4:
            ij=0
            for images in os.listdir(folder_dir2):
                if file[0]==images:
                    old=frontmo[ij]
                    cost1=frontmoc[ij]
                    break
                ij+=1
                
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/RMM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(RMM[pred]*700)+random.randrange(2000, 5000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
                
        if pred1==5:
            ij=0
            for images in os.listdir(folder_dir3):
                if file[0]==images:
                    old=frontms[ij]
                    cost1=frontmsc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/RS.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(RS[pred]*1000)+random.randrange(2000, 10000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"

        if pred1==6:
            ij=0
            for images in os.listdir(folder_dir7):
                if file[0]==images:
                    old=frontsm[ij]
                    cost1=frontsmc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/SM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(SM[pred]*500)+random.randrange(200, 1000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
        if pred1==7:
            ij=0
            for images in os.listdir(folder_dir8):
                if file[0]==images:
                    old=frontsmm[ij]
                    cost1=frontsmmc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/SMM.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(SMM[pred]*700)+random.randrange(2000, 5000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
                
        if pred1==8:
            ij=0
            for images in os.listdir(folder_dir9):
                if file[0]==images:
                    old=frontss[ij]
                    cost1=frontssc[ij]
                    break
                ij+=1
            print('given  Image is Predicted as = '+clas1[pred1])
            G="iven Car Image Predicted as : "+str(clas1[pred1]+" Damage")
            model2 = load_model('Models/SS.h5')
            test_tensors = paths_to_tensor(file1)/255
            pred=model2.predict(test_tensors)
            pred=np.argmax(pred);
            #cost1=(SS[pred]*1000)+random.randrange(2000, 10000, 3)
            print('Estimated cost is: '+str(cost1) +' ₹')
            P="redicted Damage cost is : "+str(cost1) +' ₹'
            #old=random.randrange(1, 17, 3);
            if old>=15:
                C="ar is "+str(old)+" year Old you should Go To RTO"
            else:
                C="ar is "+str(old)+" year Old No Action needed"
        return render_template('next_step.html', image_filename=image_filename, message1=G, cost=P, total_cost=C)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("Username and password are required", "danger")
            return redirect(url_for('login'))

        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM customer WHERE name=? AND password=?", (username, password))
        data = cur.fetchone()
        conn.close()

        if data:
            session["username"] = data["name"]
            flash("Login Successful", "success")
            return redirect(url_for('damage_assessment'))
        else:
            flash("Incorrect username or password", "danger")
            return render_template('login.html', error="Incorrect username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']

            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            cur.execute("INSERT INTO customer (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
            flash("Record Added Successfully", "success")
        except Exception as e:
            flash("Error Insert Operation: " + str(e), "danger")
        finally:
            conn.close()
            return redirect(url_for("login"))
    return render_template('register.html')

@app.route('/ourteam')
def ourteam():
    return render_template('ourteam.html')

if __name__ == '__main__':
    app.run(debug=True)
