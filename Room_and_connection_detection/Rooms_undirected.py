import pandas
import spacy
import random
import en_core_web_sm

#LOADING THE DATA
data=pandas.read_json(r'final.json',lines=True)
print(data)
a = data['annotation']
d = data['document']

#TRAINING THE DATA FOR NER
def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp

#80% SPLIT OF DATA 
print(len(a)-(20*len(a))/100)

#CLEANING THE DATA FOR CONNECTIONS
Connection_Traning=[]
for n in range(len(a)-10):
    b = a[n]
    startpt=[]
    endpt=[]
    labelnm=[]
    labelindex=[]
    for i in range(len(b)-1):
        x=b[i].get("label")
        if x == "CONNECTION":
            labelindex.append(i)
    for i in labelindex:
        x=b[i].get("start")
        startpt.append(x)
        x=b[i].get("end")
        endpt.append(x)
        x=b[i].get("label")
        labelnm.append(x)
    print(startpt,endpt,labelnm)

    df = pandas.DataFrame({"start":startpt,"end":endpt,"label":labelnm})
    print(df)

    entity=[]
    for i in range(len(df)-1):
        x=(df['start'][i],df['end'][i],df['label'][i])
        entity.append(x)
    print(entity)

    train_data = (d[n],{"entities":entity})
    Connection_Traning.append(train_data)
print(Connection_Traning)

#CLEANING THE DATA FOR ROOMS
Room_Traning=[]
for n in range(len(a)-10):
    b = a[n]
    startpt=[]
    endpt=[]
    labelnm=[]
    labelindex=[]
    for i in range(len(b)-1):
        x=b[i].get("label")
        if x == "ROOM":
            labelindex.append(i)
    for i in labelindex:
        x=b[i].get("start")
        startpt.append(x)
        x=b[i].get("end")
        endpt.append(x)
        x=b[i].get("label")
        labelnm.append(x)
    print(startpt,endpt,labelnm)

    df = pandas.DataFrame({"start":startpt,"end":endpt,"label":labelnm})
    print(df)

    entity=[]
    for i in range(len(df)-1):
        x=(df['start'][i],df['end'][i],df['label'][i])
        entity.append(x)
    print(entity)

    train_data = (d[n],{"entities":entity})
    Room_Traning.append(train_data)
print(Room_Traning)

#TRAINING THE NLP MODEL FOR CONNECTIONS AND ROOMS

connection_nlp=train_spacy(Connection_Traning,20)
room_nlp=train_spacy(Room_Traning,20)

#NAMING THE MODEL AND SAVING IT SO WE CAN USE IT ON ANY PC

modelfile = input("Enter your Room Model Name: ")
room_nlp.to_disk(modelfile)
modelfile = input("Enter your Connection Model Name: ")
connection_nlp.to_disk(modelfile)

#HERE WE CAN USE ANY DOCUMENT

INPUTDATA = d[46]

trialconnection=connection_nlp(INPUTDATA)
trialconnection.ents

connected_rooms=[]
for ent in trialconnection.ents:
    connected_rooms.append(ent.text)
print("DETAILS OF THE CONNECTED ROOMS",connected_rooms)


tryroom = room_nlp(INPUTDATA)
for ent in tryroom.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)   
    

individual_rooms=[]
nlp = en_core_web_sm.load()
for i in connected_rooms:
    sentences=[]
    tryroom1=nlp(i)
    tryroom1.ents

    for ent in tryroom1.ents:
        sentences.append(ent.text)
    if len(sentences)!=0:
        individual_rooms.append(sentences)
print(individual_rooms)

#HERE THE ROMMS ARE COMBAINED WITH ITS CONNECTED ROOMS

from itertools import combinations
l = []
for w in individual_rooms:
    l1 = list(combinations(w,2))
    l.append(l1)
print(l)

room_type=[]
link=[]
for i in l:
    for j in i:
        room_type.append(j[0])
        link.append(j[1])
print(room_type)
print(link)

#CONNECTED ROOM TABLE IS PRINTED HERE

Connected_table= pandas.DataFrame({'Room_type':room_type,'Link':link})
print(Connected_table)

#PRINTING GRAPH 

Graph_data=[]
for i in l:
    for j in i:
        Graph_data.append((j[0],j[1]))
print(Graph_data)

import networkx as nx
G = nx.Graph()
G.add_edges_from(Graph_data) 
nx.draw_networkx(G, with_label = True)