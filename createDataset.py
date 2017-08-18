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

def getFacebookData():
	responseDictionary = dict()
	fbFile = open('fbMessages.txt','utf8', 'r') 
	allLines = fbFile.readlines()
	myMessage, otherPersonsMessage, currentSpeaker = "","",""
	for index,lines in enumerate(allLines):
	    rightBracket = lines.find(']') + 2
	    justMessage = lines[rightBracket:]
	    colon = justMessage.find(':')
	    # Find messages that I sent
	    if (justMessage[:colon] == personName):
	        if not myMessage:
	            # Want to find the first message that I send (if I send multiple in a row)
	            startMessageIndex = index - 1
	        myMessage += justMessage[colon+2:]
	        
	    elif myMessage:
	        # Now go and see what message the other person sent by looking at previous messages
	        for counter in range(startMessageIndex, 0, -1):
	            currentLine = allLines[counter]
	            rightBracket = currentLine.find(']') + 2
	            justMessage = currentLine[rightBracket:]
	            colon = justMessage.find(':')
	            if not currentSpeaker:
	                # The first speaker not named me
	                currentSpeaker = justMessage[:colon]
	            elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
	                # A different person started speaking, so now I know that the first person's message is done
	                otherPersonsMessage = cleanMessage(otherPersonsMessage)
	                myMessage = cleanMessage(myMessage)
	                responseDictionary[otherPersonsMessage] = myMessage
	                break
	            otherPersonsMessage = justMessage[colon+2:] + otherPersonsMessage
	        myMessage, otherPersonsMessage, currentSpeaker = "","",""    
	return responseDictionary

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

list_messages = parseFbConv()