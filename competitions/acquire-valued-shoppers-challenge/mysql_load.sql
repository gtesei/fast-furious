LOAD DATA LOCAL INFILE  'c:/docs/ff/gitHub/fast-furious/dataset/acquire-valued-shoppers-challenge/trainHistory'
INTO TABLE TRAIN_HISTORY
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(id,chain,offer,market,repeattrips,repeater,offerdate);




LOAD DATA LOCAL INFILE  'c:/docs/ff/gitHub/fast-furious/dataset/acquire-valued-shoppers-challenge/testHistory'
INTO TABLE TEST_HISTORY
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(id,chain,offer,market,offerdate);


LOAD DATA LOCAL INFILE  'c:/docs/ff/gitHub/fast-furious/dataset/acquire-valued-shoppers-challenge/offers'
INTO TABLE OFFER
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(offer,category,quantity,company,offervalue,brand);



