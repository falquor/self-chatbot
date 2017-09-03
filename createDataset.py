# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:33:23 2017

@author: Quentin
"""

import pandas as pd
import numpy as np
#import os
#import re
from datetime import datetime
import json

def parseFbConv():
    fbFile = open('conv.txt','r',encoding = 'utf8')
    allLines = fbFile.readlines()
    name = "Quentin Bouniot"
    list_messages = []
    message = ''
    response = ''
    space = ' '
    conversation = ['','']
    n = len(allLines)
    my_message_last = False
    for k in range(len(allLines)-3,-1,-1):
#        if k%3 == 2:
            if allLines[k][:15] != name and allLines[k][-7:-4] == 'UTC':
                if not(my_message_last):
                    if allLines[k+2][-7:-4] != 'UTC':
                        message = message + space + allLines[k+2][:-1] + space
                else:
                    conversation[1] = response
                    response = ''
                    if conversation[0] != '':
                        list_messages.append(conversation)
                        conversation = ['','']
                    message = allLines[k+2][:-1]
                    my_message_last = False
            if allLines[k][:15] == name and allLines[k][-7:-4] == 'UTC':
                if my_message_last:
                    if allLines[k+2][-7:-4] != 'UTC':
                        response = response + space + allLines[k+2][:-1] + space
                else:
                    conversation[0] = message
                    message = ''
                    if conversation[1] != '':
                        list_messages.append(conversation)
                        conversation = ['','']
                    response = allLines[k+2][:-1]
                    my_message_last = True
    
    return list_messages

def cleanMessages(messages):
    cleanedMessages = []
    for message in messages:
        friendMessage = message[0]
        myMessage = message[1]
        convTot = friendMessage + ' ' + myMessage
        if not 'http' in convTot:
            if not 'UTC' in convTot:
                cleanedMessages.append([friendMessage,myMessage])
            
    return cleanedMessages
    
list_messages = parseFbConv()

cleanedMessages = cleanMessages(list_messages)

with open("messages.json",'w', encoding = 'utf8') as file:
    json.dump(cleanedMessages,file)