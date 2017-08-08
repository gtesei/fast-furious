import nltk 
from nltk.corpus import wordnet as wn

bs = ['I had my normal periods 10 days after on my regular date however now I am one week late. Could I be pregnant?','I had sex two months ago. I had a period last month after it was 4-5 days late. This month, I am now 8 days late. What does this mean?']

gs = ['Just completed my B.Tech with an aggregate of 66%, X and XII 75, 57 respectively. The eligibilty criteria in most of companies is 60% and above in X, XII and graduation. Now that I am helpless and cannot do anything about my XII score, does this mean that I wont get a job anywhere?',"I'm not an IITian and graduated in 2016. I have CGPA around 8.4 and pretty good scores in X and XII. Can you advice considering following details?"]

bs2 =['How do I start my life?','How do I start all over again?']


tb = [nltk.pos_tag(nltk.word_tokenize(t)) for t in bs]
tb2 = [nltk.pos_tag(nltk.word_tokenize(t)) for t in bs2]

tg = [nltk.pos_tag(nltk.word_tokenize(t)) for t in gs]


def synsym(s1,s2):
    ts0 = nltk.pos_tag(nltk.word_tokenize(s1))
    ts1 = nltk.pos_tag(nltk.word_tokenize(s2))
    # adj  
    jj0 = [x for x,y in ts0 if y=='JJ' or y=='JJR' or y=='JJS']
    jj1 = [x for x,y in ts1 if y=='JJ' or y=='JJR' or y=='JJS']
    jj0w = [wn.synsets(xx,pos=wn.ADJ) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.ADJ) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    jjps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    jjps = [x for x in jjps if x != None]
    if len(jjps)==0:
      jjps = [0]
    # noum  
    jj0 = [x for x,y in ts0 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS']
    jj1 = [x for x,y in ts1 if y=='NN' or y=='NNS' or y=='NNP' or y=='NNPS']
    jj0w = [wn.synsets(xx,pos=wn.NOUN) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.NOUN) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    nps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    nps = [x for x in nps if x != None]
    if len(nps)==0:
      nps = [0]
    # verb  
    jj0 = [x for x,y in ts0 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    jj1 = [x for x,y in ts1 if y=='VB' or y=='VBD' or y=='VBG' or y=='VBN' or y=='VBP' or y=='VBZ']
    jj0w = [wn.synsets(xx,pos=wn.VERB) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.VERB) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    vps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    vps = [x for x in vps if x != None]
    if len(vps)==0:
      vps = [0]
    # adverb  
    jj0 = [x for x,y in ts0 if y=='RB' or y=='RBR' or y=='RBS' or y=='WRB']
    jj1 = [x for x,y in ts1 if y=='RB' or y=='RBR' or y=='RBS' or y=='WRB']
    jj0w = [wn.synsets(xx,pos=wn.ADV) for xx in jj0]
    jj0w = [item for sl in jj0w for item in sl]
    jj1w = [wn.synsets(xx,pos=wn.ADV) for xx in jj1]
    jj1w = [item for sl in jj1w for item in sl]
    aps = [r.path_similarity(l) for r in jj0w for l in jj1w]
    aps = [x for x in aps if x != None]
    if len(aps)==0:
      aps = [0]
    return [jjps,nps,vps,aps]


## bad 1
JJ_b0 = [x for x,y in tb[0] if y == 'JJ']
JJ_b1 = [x for x,y in tb[1] if y == 'JJ']

jjb0w = [wn.synsets(xx,pos=wn.ADJ) for xx in JJ_b0]
jjb1w = [wn.synsets(xx,pos=wn.ADJ) for xx in JJ_b1]
 
 
jjb0w = [item for sl in jjb0w for item in sl]
jjb1w = [item for sl in jjb1w for item in sl]

ps = [r.path_similarity(l) for r in jjb0w for l in jjb1w]

[x for x in ps if x != None]

## good 1
JJ_b0 = [x for x,y in tg[0] if y == 'JJ']
JJ_b1 = [x for x,y in tg[1] if y == 'JJ']

jjb0w = [wn.synsets(xx,pos=wn.ADJ) for xx in JJ_b0]
jjb1w = [wn.synsets(xx,pos=wn.ADJ) for xx in JJ_b1]
 
 
jjb0w = [item for sl in jjb0w for item in sl]
jjb1w = [item for sl in jjb1w for item in sl]

ps = [r.path_similarity(l) for r in jjb0w for l in jjb1w]

[x for x in ps if x != None]

## bad 2
JJ_b0 = [x for x,y in tb[0] if y == 'NN' or y =='NNS']
JJ_b1 = [x for x,y in tb[1] if y == 'NN' or y =='NNS']

jjb0w = [wn.synsets(xx,pos=wn.NOUM) for xx in JJ_b0]
jjb1w = [wn.synsets(xx,pos=wn.NOUM) for xx in JJ_b1]
 
 
jjb0w = [item for sl in jjb0w for item in sl]
jjb1w = [item for sl in jjb1w for item in sl]

ps = [r.path_similarity(l) for r in jjb0w for l in jjb1w]

[x for x in ps if x != None]



