import pyautogui as pt
from selenium import webdriver
import pyperclip as pc
from selenium.webdriver.support.ui import WebDriverWait
import time
class Whatsapp:
    def __init__(self,speed=.5,click_speed=.3):
        self.speed=speed
        self.click_speed=click_speed
        self.message=''

    def send(self,recipient="Papa",message="This is a default message"):
        options = webdriver.ChromeOptions()
        options.add_argument("user-data-dir=C:/Users/HP/AppData/Local/Google/Chrome/User Data")
        driver = webdriver.Chrome(options=options)
        driver.get("https://web.whatsapp.com")
        driver.maximize_window()
        WebDriverWait(driver, 200)
        time.sleep(10)
        self.message=message
        pos=pt.locateCenterOnScreen("search_box.png",confidence=0.7)
        pt.moveTo(pos[0:2],duration=self.speed)
        pt.click(interval=self.click_speed)
        pt.typewrite(recipient,interval=self.click_speed)
        pt.moveRel(0,260)
        pt.click(interval=self.click_speed)
        time.sleep(5)
        pos = pt.locateCenterOnScreen("send.png", confidence=0.4)
        pt.moveTo(pos[0:2], duration=self.speed)
        pt.click(interval=self.click_speed)
        pt.typewrite(message, interval=self.click_speed)
        pt.press("enter")
        time.sleep(5)
