import os
import pywhatkit
import pyttsx3
import pyautogui
import datetime
import speech_recognition as sr
import webbrowser
import wikipedia
import pyautogui as pt
import Whatsbot
import Chat
from Chat import get_response
from englisttohindi.englisttohindi import EngtoHindi
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
print(voices)
engine.setProperty('voice',voices[1].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def wishme():
    pass
def take_command():
    r=sr.Recognizer()
    query=""
    with sr.Microphone() as source:
        print("कृप्या बोलिये सर मैं सुन रहा हूँ...")
        speak("कृप्या बोलिये सर मैं सुन रहा हूँ")
        r.pause_threshold=3
        audio=r.listen(source)
    try:
     print("पहचानने की प्रक्रिया जारी है....")
     speak("पहचानने की प्रक्रिया जारी है")
     query=r.recognize_google(audio,language='en-in')
     print(f"You said:{query}\n")
    except Exception as e:
        print("मैं आपसे क्षमा चाहता हूँ सर। क्या आप कृपया दोहरा सकते हैं?")
        speak("मैं आपसे क्षमा चाहता हूँ सर। क्या आप कृपया दोहरा सकते हैं?")
        take_command()
    return query
def op_browser():
    webbrowser.open("http://www.google.com")
def op_yt():
    webbrowser.open("http://www.youtube.com")
def pl_yt(name):
    pywhatkit.playonyt(name)
def wik_search(query):
   res= wikipedia.summary(query,sentences=5)
   wikipedia.set_lang("hi")
   res1 =wikipedia.summary(query)
   print(res1)
   print(res)
   speak(res1)
speak("हैलो! पांडे जी मैं आपकी कैसे सहायता कर सकता हूँ?")
while(True):
  query=take_command()
  if(query.lower()=="quit"):
    break
  elif("whatsapp" in query.lower().split()):
           speak("कृपया मुझे उस व्यक्ति या समूह का नाम बताएं जिसे आप संदेश भेजना चाहते हैं")
           print("कृपया मुझे उस व्यक्ति या समूह का नाम बताएं जिसे आप संदेश भेजना चाहते हैं")
           recip = take_command()
           speak("अब कृपया मुझे भेजने के लिए संदेश बताएं")
           print("अब कृपया मुझे भेजने के लिए संदेश बताएं")
           msg = take_command()
           wh = Whatsbot.Whatsapp()
           wh.send(recip, msg)
  else:
        query=EngtoHindi(query).convert
        response=get_response(query)
        if(response[0]=="open_youtube"):
           speak(response[1])
           op_yt()
        elif(response[0]=="open_browser"):
           speak(response[1])
           op_browser()
        elif (response[0] == "play_youtube"):
           speak(response[1])
           pl_yt(query)
        elif (response[0]=="retrieve_wikipedia"):
            query = query.rstrip("?")
            query = query.split()
            query = [x for x in query if x not in ["क्या","कौन","कब","कहाँ","कैसे","क्यों","किस","किसे","किसका","किसकी","किसके","किस","प्रकार","कितना","कितनी","कितने","कहाँ से","किससे","किसके" "लिए","किसके","साथ","कहाँ","तक","कैसा","कैसी","कैसे","है","हैं","था","थे","थी","हुआ","हुई","हुए"]]
            query2 = ""
            for i in query:
                query2 += i
                query2 += " "
            speak(response[1])
            try:
             wik_search(query2)
            except:
                print(f"Page ID for:{query2}not found. Please ask another Question")


        elif (response[0] =="send_whatsapp_message"):
           speak("कृपया मुझे उस व्यक्ति या समूह का नाम बताएं जिसे आप संदेश भेजना चाहते हैं")
           recip=take_command()
           speak("अब कृपया मुझे भेजने के लिए संदेश बताएं")
           msg=take_command()
           speak(response[1])
           wh = Whatsbot.Whatsapp()
           wh.send(recip,msg)
        elif(response[0]=="None"):
            print(response[1])
        else:
           speak(response[1])