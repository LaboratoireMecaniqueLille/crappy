#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import mechanize
# import re
# import unidecode
# import time
import SimpleITK as sitk
import numpy as np
import crappy
import smtplib
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email import encoders


class Alert(crappy.links.MetaCondition):
    def __init__(self):
        self.broken = False
        self.path = "/home/corentin/Bureau/"

    def evaluate(self, value):
        try:
            val = np.mean(value)
            if val < 20 and self.broken == False:
                print "light broken"
                self.broken = True
                image = sitk.GetImageFromArray(value)
                sitk.WriteImage(image, self.path + "last_img.tiff")
                mdp = "PUT YOUR PASSWORD HERE"
                fromaddr = "PUT YOUR SENDING EMAIL ADRESS HERE"
                toaddr = "PUT YOUR  RECEIVING EMAIL ADRESS HERE"
                msg = MIMEMultipart()
                msg['From'] = fromaddr
                msg['To'] = toaddr
                msg['Subject'] = "Alerte lampe !!"

                body = "Bonjour, c'est la machine qui te parle, la lampe ne fonctionne plus !! " \
                       "Je te joins la dernière image."
                msg.attach(MIMEText(body, 'plain'))

                filename = "last_img.tiff"
                attachment = open(self.path + filename, "rb")

                part = MIMEBase('application', 'octet-stream')
                part.set_payload((attachment).read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

                msg.attach(part)

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(fromaddr, mdp)
                text = msg.as_string()
                server.sendmail(fromaddr, toaddr, text)
                server.quit()
            elif val > 20 and self.broken:
                self.broken = False
                print "light changed!"
            return None

        except Exception as e:
            print e
            try:
                server.quit()
            except Exception as e:
                print e

# http://www.pythonforbeginners.com/cheatsheet/python-mechanize-cheat-sheet
# http://naelshiab.com/tutorial-send-email-python/
