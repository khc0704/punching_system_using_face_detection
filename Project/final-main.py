from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.base import EventLoop
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import NumericProperty,StringProperty,ObjectProperty,ListProperty
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import numpy as np
import json
import os
import datetime
import shutil
import cv2
import face_recognition
import subprocess

#重要目錄及檔案位置
baseDirectory = r""+os.path.dirname(os.path.abspath(__file__))
baseFireStoreDirectory = r""+os.path.join(baseDirectory,'emp_firestore_data')
basePhotoDirectory = r""+os.path.join(baseDirectory,'emp_photo_data')
baseDataDirectory = r""+os.path.join(baseDirectory,'emp_data\emp_data.csv')
font1 = "msjh.ttc"

#雲端資料庫Firebase的自定義處理器，稍微用方法包裝他們提供的api
class DataHandler():
    
    if(os.path.isfile(os.path.join(baseFireStoreDirectory,"serviceAccountKey.json"))==False):
        command = 'explorer '+"\""+baseFireStoreDirectory+"\""
        print("請先將serviceAccountKey.json檔案放入文件夾，可以從FireStore上面獲得")
        subprocess.Popen(command)
        exit()
    cred = credentials.Certificate(os.path.join(baseFireStoreDirectory,("serviceAccountKey.json")))
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    def create(self,data,ref):
        ref.set(data,merge=True)

#權限設定
    auth = False
    alert_message = ""
    store_emp_list = []

    def setAlert(self,text):
        self.alert_message = text

    def logOut(self):
        self.auth = False

    def logIn(self):
        self.auth = True
'''
    #def update(self,data,ref):
    #    if ref == False:
    #        return False
    #    else:
    #        key = data.to_dict().keys()
    #        for k in key:
    #            if k not in ref[1]:
    #                return False
    #        ref[0].update(data)
    #        return True

    def delete_doc(self,ref):
        ref = self.query(ref=ref,test=True)
        if ref != False:
            ref.delete()
            return True
        else:
            return False

    def query(self,ref,conditions=[],test=False):
        ref = self.ref_extract(ref)
        if ref != False:
            if len(conditions) != 0:
                for condition in conditions:
                    ref = ref.where(condition[0],condition[1],condition[2])
            result = ref.get()
            if len(result) != 0 :
                return [True,ref]
        return [False,ref]

    def ref_extract(self,ref):
        ref_arr = ref.split('/')
        ref = self.db
        for r in range(len(ref_arr)):
            if r == 0:
                ref = ref.collection(ref_arr[r])
            elif r%2 == 0:
                ref = ref.collection(ref_arr[r])
            else:
                ref.document([ref_arr[r]])
            return ref
        else:
            return False
'''

datahandler = DataHandler()

def idFormat(empId):
    if(int(empId)>=100000000 and int(empId)<=999999999):
        return True
    else:
        return False

#呼叫設定檔的方法
def readEmpData():
    if os.path.isfile(baseDataDirectory):
        data = pd.read_csv(baseDataDirectory,delimiter=',')
        return data

#呼叫雲端員工資料的方法
def readEmpCloud():
    data = dict()
    docs = datahandler.db.collection('user_table').stream()
    i=0
    data['姓名']=[]
    data['電話']=[]
    data['group']=[]
    for doc in docs:
        d = dict(doc.to_dict())
        data['姓名'].append(d['name'])
        data['電話'].append(doc.id)
        if('group' in d.keys()):
            if(d['group']=='admin'):
                data['group'].append('admin')
            else:
                data['group'].append('user')
        else:
            data['group'].append('user')
        i=i+1
    return data

#自定義相機類別，參考Kivy原本的CameraBase做修改
class KivyCamera(Image):

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.image_frame = None

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):
        Clock.unschedule(self.update)
        self.capture = None

    def update(self, dt):
        return_value, frame = self.capture.read()
        self.image_frame = frame
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()

capture = None

#驗證id是否存在於雲端
def vertifyId(empId):
    data = readEmpCloud()
    vertifyList = [str(i) for i in data['電話'] if empId==str(i)]
    if len(vertifyList) > 0:
        return vertifyList[0]
    else:
        return False

#臉部分析，傳入一張frame，回傳False或者員工id(手機號碼)
def faceDetect(frame,forAdmin=False):
    data = readEmpCloud()
    imgList = []
    vertifyList = []
    if(forAdmin):
        vertifyList = [str(data['電話'][i]) for i in range(len(data['電話'])) if(data['group'][i])=='admin']
    else:
        vertifyList = [str(i) for i in data['電話']]
    for imgName in os.listdir(basePhotoDirectory):
        photoId = imgName.split('_')[-1].split('.')[0]
        if(imgName.split('_')[0]=='IMG' and photoId in vertifyList and 
        len(imgName.split('_')[-1]) == 13 ):
            imgList.append(os.path.join(basePhotoDirectory,imgName))

    try:
        frameEncoding = face_recognition.face_encodings(frame)[0]

        dataEncoding = []
        dataId = []
        for img in imgList:
            image = face_recognition.load_image_file(img)
            encoding = face_recognition.face_encodings(image)[0]
            dataEncoding.append(encoding)
            dataId.append(img.split('/')[-1].split('_')[-1].split('.')[0])
        
        if(len(dataEncoding) != 0):
            results = face_recognition.compare_faces(dataEncoding, frameEncoding,tolerance=0.4)
            correct = []
            for i in range(len(results)):
                if(results[i]):
                    correct.append(dataId[i])
            if(len(correct)>=1):
                return correct
        return False
    except IndexError:
        return False

#快速呼叫訊息視窗方法
def callAlertBox(text):
    datahandler.setAlert(text)
    AlertBox().open()

#自定義訊息視窗
class AlertBox(Popup):
    def __init__(self,**kw):
        super(AlertBox,self).__init__(**kw)
        self.title = "FinalProject"
        self.ids.alert_message.text = datahandler.alert_message

#自定義選擇視窗
class YesNoBox(Popup):
    message = StringProperty('')
    on_yes = ObjectProperty()
    parameters = ListProperty([])

    def __init__(self, **kw):
        super(YesNoBox,self).__init__(**kw)
        self.title = "FinalProject"
        self.auto_dismiss = False
    
    def do_action(self,action):
        if len(self.parameters)==1:
            result = action(self.parameters[0])
        elif len(self.parameters)>1:
            result = action(self.parameters)
        else:
            result = action()
        self.dismiss()

class ChooseIdBox(Popup):
    message = StringProperty('')
    on_yes = ObjectProperty()
    choice = ListProperty([])
    parameters = ListProperty([])

    def __init__(self, **kwargs):
        super(ChooseIdBox,self).__init__(**kwargs)
        self.title = "FinalProject"
        self.ids.empId.values = self.choice
        self.auto_dismiss = False

    def setID(self):
        self.ids.empId.values = self.choice

    def do_action(self,action):
        parameters = self.parameters
        if len(parameters)==1:
            result = action(parameters[0])
        elif len(parameters)>1:
            result = action(parameters)
        else:
            result = action()
        self.dismiss()

    def set_parameters(self):
        if(len(self.parameters)==0):
            self.parameters.append(self.ids.empId.text)
        elif(len(self.parameters)>=1):
            self.parameters[0] = self.ids.empId.text

#Page:上下班打卡
class ClockBox(Popup):

    def __init__(self, **kw):
        super(ClockBox,self).__init__(**kw)
        self.ids.time.text = datetime.datetime.now().strftime('%H:%M:%S')
        self.ids.date.text = datetime.datetime.today().strftime('%Y/%m/%d')
        Clock.schedule_interval(self.increment_time, .1)
        self.doStart()

    def increment_time(self, interval):
        self.ids.time.text = datetime.datetime.now().strftime('%H:%M:%S')

    def doStart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.camera.start(capture)

    def doExit(self):
        Clock.unschedule(self.increment_time)
        global capture
        if capture != None:
            capture.release()
            capture = None
        self.dismiss()

    def bindAction(self,action):
        empId = faceDetect(self.ids.camera.image_frame)
        if(empId==False):
            callAlertBox("偵測不到對應的ID，請重新試試！")
        else:
            if(len(empId)>1):
                pop = ChooseIdBox()
                pop.choice = empId
                pop.setID()
                if(action=="onDuty"):
                    pop.on_yes=self.onDuty
                    pop.open()

                elif(action=="offDuty"):
                    pop.on_yes=self.offDuty
                    pop.open()
            else:
                pop = YesNoBox()
                pop.message = "請確認你的ID為:\n"+str(empId[0])
                pop.parameters = empId
                if(action=="onDuty"):
                    pop.on_yes=self.onDuty
                    pop.open()

                elif(action=="offDuty"):
                    pop.on_yes=self.offDuty
                    pop.open()

#打卡上班功能
    def onDuty(self,empId):
        date = ''.join(str(self.ids.date.text).split('/'))
        time = ''.join(str(self.ids.time.text).split(':'))
        result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty")
        data = result.get()
        done = False
        if(data.exists):
            data = dict(data.to_dict())
            for key in data.keys():
                if key == empId:
                    done = True
                    callAlertBox("已經打過上班卡嘍！")
                    break
        if(done!=True):
            result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty")
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                for key in data.keys():
                    if key == empId:
                        done = True
                        #if int(time) < int(data[key]):
                            #datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty").set({empId:time},merge=True)
                        callAlertBox("打卡順序出錯，請聯絡管理員！")
                        break
            if(done!=True):
                datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty").set({empId:time},merge=True)
                callAlertBox("成功打卡上班！")

#打卡下班功能
    def offDuty(self,empId):
        date = ''.join(str(self.ids.date.text).split('/'))
        time = ''.join(str(self.ids.time.text).split(':'))
        result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty")
        result2 = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty")
        done = False
        data = result2.get()
        if(data.exists):
            data = dict(data.to_dict())
            for key in data.keys():
                if key == empId:
                    done = True
                    callAlertBox("已經打過下班卡嘍！")
                    break
        if(done!=True):
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                for key in data.keys():
                    if key == empId:
                        if int(time) > int(data[key]):
                            datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty").set({empId:time},merge=True)
                            callAlertBox("成功打卡下班！")

#Page:管理員登入
class AccessBox(Popup):
    def __init__(self, **kw):
        super(AccessBox,self).__init__(**kw)
        self.doStart()

    def doStart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.camera.start(capture)

    def doExit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        self.dismiss()

    def adminAccess(self):
        empId = faceDetect(self.ids.camera.image_frame,True)
        if(empId != False):
            result = datahandler.db.collection("user_table").document(empId[0])
            done = False
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                if('group' in data.keys()):
                    if(data['group']=='admin'):
                        done = True
            if(done) == True:
                datahandler.logIn()
                callAlertBox(''.join(["管理員登入成功ID為:\n",empId[0]]))
                self.doExit()
        else:
            callAlertBox("偵查不到對應的管理員ID\n請重新試試！")

#Page:補打卡
class PatchBox(Popup):
    def __init__(self, **kw):
        super(PatchBox,self).__init__(**kw)
        datas = readEmpCloud()
        datas_mix = [datas['電話'][i]+'|'+datas['姓名'][i] for i in range(len(datas['電話']))]
        if(len(datas_mix)>=1):
            self.ids.empId.values = datas_mix

    def onDuty(self):
        empId = str(self.ids.empId.text).split('|')[0]
        if(empId!=False):
            message=""
            date = ''.join(str(self.ids.date.text).split('/'))
            time = ''.join(str(self.ids.time.text).split(':'))
            result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty")
            done = False
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                for key in data.keys():
                    if key == empId:
                        if int(time) < int(data[key]):
                            datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty").set({empId:time},merge=True)
                            done = True
                            break
                        else:
                            old_time = str(data[key])[:2]+":"+str(data[key][2:4])+":"+str(data[key][4:6])
                            message="【修改失敗】\n上班時間不能晚於下班時間！\n"+"當天下班時間為:\n"+old_time
                            break
            else:
                datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty").set({empId:time},merge=True)
                done = True

            if(done):
                message = "成功修改了"+empId+"\n在"+self.ids.date.text+"的上班打卡記錄"+"\n\n[時間為:]:\n"+self.ids.time.text
                callAlertBox(message)
            else:
                if message == "":
                    callAlertBox("未知錯誤")
                else:
                    callAlertBox(message)

    def offDuty(self):
        empId = str(self.ids.empId.text).split('|')[0]
        if(empId!=False):
            message=""
            date = ''.join(str(self.ids.date.text).split('/'))
            time = ''.join(str(self.ids.time.text).split(':'))
            result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty")
            done = False
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                for key in data.keys():
                    if key == empId:
                        if int(time) > int(data[key]):
                            datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty").set({empId:time},merge=True)
                            done = True
                            break
                        else:
                            old_time = str(data[key])[:2]+":"+str(data[key][2:4])+":"+str(data[key][4:6])
                            message="【修改失敗】\n下班時間不能早於下班時間！\n"+"當天上班時間為:\n"+old_time
                            break
                if(done):
                    message = "成功修改了"+empId+"\n在"+self.ids.date.text+"的下班打卡記錄"+"\n\n[時間為:]:\n"+self.ids.time.text
                    callAlertBox(message)
                else:
                    if message == "":
                        callAlertBox("未知錯誤")
                    else:
                        callAlertBox(message)


#Page:加入員工臉部辨識照片
class PhotoBox(Popup):
    def __init__(self, **kw):
        super(PhotoBox,self).__init__(**kw)
        datas = readEmpCloud()
        datas_mix = [datas['電話'][i]+'|'+datas['姓名'][i] for i in range(len(datas['電話']))]
        if(len(datas_mix)>=1):
            self.ids.empId.values = datas_mix
        self.dostart()

    def capturePhoto(self):
        empId = str(self.ids.empId.text).split('|')[0]
        empData = readEmpCloud()
        check = False
        oneShot = []
        for i in range(len(empData['電話'])):
            if(str(empId)==str(empData['電話'][i])):
                oneShot = [empData['姓名'][i],empData['電話'][i]]
                check = True
        if check:
            frame = self.ids.camera.image_frame
            path = os.path.join(baseDirectory,"emp_photo_data/IMG_"+f"{empId}"+".png")
            cv2.imwrite(path,frame)
            message="已成功為員工"+oneShot[0]+"\nID:"+oneShot[1]+"\n更新人臉辨識照片"
            callAlertBox("更新員工圖片成功")
    
    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.camera.start(capture)

    def doexit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        self.dismiss()

#Page:匯入員工資料
class SettingBox(Popup):

    def bindAction(self):
        pop = YesNoBox()
        pop.message="上傳檔案會覆蓋雲端資料，請仔細確認。"
        pop.on_yes=self.uploadSetting
        pop.open()

    def openExplorer(self):
        path = baseDataDirectory.split('\\')[:-1]
        path = '\\'.join(path)
        command = 'explorer '+"\""+path+"\""
        subprocess.Popen(command)

    def uploadSetting(self):
        datas = readEmpData()
        check = False
        bad_data = []
        for i in range(datas.shape[0]):
            if(idFormat(datas['電話'][i])):
                datahandler.db.collection('user_table').document(str(datas['電話'][i])).set({'name':str(datas['姓名'][i]),
                'salary':str(datas['薪資'][i]),'emp_type':str(datas['聘約類型'][i])},merge=True)
            else:
                check = True
                bad_data.append(str(datas['電話'][i])+"|"+str(datas['姓名'][i]))
        if(check):
            bad_message='\n'.join(bad_data)
            message='\n'.join(["資料更新完畢！","但下列資料未能新增，因爲電話(ID)不為9位數",bad_message])

            callAlertBox(message)
        else:       
            callAlertBox("資料更新完畢！")
   # def getTemplate(self):

    #def copyCsv(self):
     #   source=r""+self.ids.empFileChooser.selection[0]
     #   if 'csv' in str(source).split('/')[-1].split('.')[-1]:
     #       dest=baseDataDirectory
     #       shutil.copy(source,dest)

class DisplayDutyBox(Popup):

    def callDisplay(self,dutyType):
        if(dutyType=="on_duty"):
            
            date = ''.join(str(self.ids.date.text).split('/'))
            result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("on_duty")
            done = False
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                display_data = data
                data_message=[]
                user_list=[]
                user_data = readEmpCloud()
                for key in display_data:
                    name =""
                    if(user_data['電話'].index(key)!=-1):
                        name = user_data['姓名'][int(list(user_data['電話']).index(key))]
                    time = str(display_data[key])[:2]+":"+str(display_data[key][2:4])+":"+str(display_data[key][4:6])
                    data_message.append(str(key+"|"+name+" : "+time))
                data_message = '\n'.join(data_message)
                display_message = self.ids.date.text + "上班記錄\n\n" + data_message
                callAlertBox(display_message)

        elif(dutyType=="off_duty"):
            
            date = ''.join(str(self.ids.date.text).split('/'))
            result = datahandler.db.collection("duty_table").document(date[:6]).collection(date[-2:]).document("off_duty")
            done = False
            data = result.get()
            if(data.exists):
                data = dict(data.to_dict())
                display_data = data
                data_message=[]
                user_data = readEmpCloud()
                for key in display_data:    
                    name =""
                    if(user_data['電話'].index(key)!=-1):
                        name = user_data['姓名'][int(list(user_data['電話']).index(key))]
                    time = str(display_data[key])[:2]+":"+str(display_data[key][2:4])+":"+str(display_data[key][4:6])
                    data_message.append(str(key+"|"+name+" : "+time))
                user_data = readEmpCloud()
                data_message = '\n'.join(data_message)
                display_message = self.ids.date.text + "下班記錄\n\n" + data_message
                callAlertBox(display_message)

class SuperAdminBox(Popup):

    def __init__(self, **kw):
        super(SuperAdminBox,self).__init__(**kw)
        self.dostart()

    def capturePhoto(self):
        empId = self.ids.phone.text
        frame = self.ids.camera.image_frame
        path = os.path.join(baseDirectory,"emp_photo_data/IMG_"+f"{empId}"+".png")
        cv2.imwrite(path,frame)
    
    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.camera.start(capture)

    def doexit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        self.dismiss()

    def uploadSuperAdmin(self):
        if(self.ids.password.text=="987654321!@#$%^&*()"):
            if(idFormat(self.ids.phone.text)):
                datahandler.db.collection('user_table').document(str(self.ids.phone.text)).set({'name':str(self.ids.username.text),
                'group':'admin'},merge=True)
                self.capturePhoto()
                callAlertBox("新增超級管理員完畢！")
            else:
                callAlertBox("id為9個號碼，建議使用手機")
        else:
            callAlertBox("密碼錯誤！")
   # def getTemplate(self):

    #def copyCsv(self):
     #   source=r""+self.ids.empFileChooser.selection[0]
     #   if 'csv' in str(source).split('/')[-1].split('.')[-1]:
     #       dest=baseDataDirectory
     #       shutil.copy(source,dest)

#Page:主選單
class MainScreen(Screen):
    def __init__(self, **kw):
        super(MainScreen,self).__init__(**kw)
        Clock.schedule_interval(self.callAccess,.1)

    def Pop(self,selection):
        if selection == "clock":
            ClockBox().open()
        elif selection == "patch":
            PatchBox().open()
        elif selection == "photo":
            PhotoBox().open()
        elif selection == "setting":
            SettingBox().open()
        elif selection == "access":
            AccessBox().open()
        elif selection == "superadmin":
            SuperAdminBox().open()
        elif selection == "displayDuty":
            DisplayDutyBox().open()

#檢查登入登出畫面改變
    def callAccess(self,interval):
        if(datahandler.auth==True):
            Clock.unschedule(self.callAccess)
            self.ids.patch.disabled = False
            self.ids.photo.disabled = False
            self.ids.setting.disabled = False
            self.ids.access_out.disabled = False
            self.ids.access.disabled = True
#           self.ids.salary.disabled = False
    
    def leaveAccess(self):
        datahandler.logOut()
        self.ids.patch.disabled = True
        self.ids.photo.disabled = True
        self.ids.setting.disabled = True
        self.ids.access_out.disabled = True
        self.ids.access.disabled = False
        Clock.schedule_interval(self.callAccess,.1)
#       self.ids.salary.disabled = True


#注冊Kivy的頁面管理器
class ScreenManagement(ScreenManager):
    pass

#載入Kivy的樣本文件（類似css的概念）
kv = Builder.load_file("main_style.kv")


#主程式物件
class Main(App):
    def build(self):
        return kv

#執行主程式
Main().run()