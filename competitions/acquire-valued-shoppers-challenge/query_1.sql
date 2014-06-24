--scategory - The product category (e.g. sparkling water)

select   a.id, b.category , count(*) 
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.offer = a.offer 
      and c.category = b.category 
group by a.id , b.category       
 
update train_history t1
JOIN transaction t2 ON (t1.id = t2.id  and t1.chain = t2.chain)
JOIN offer t3 ON (t3.offer=t1.offer ) 
SET t1.scategory=count(*) 
WHERE t3.category=t2.category
group by t1.id , t2.category 

create view scategory_train  as select   a.id, b.category , count(*) as scat
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.offer = a.offer 
      and c.category = b.category 
group by a.id , b.category  

create view scategory_test  as select   a.id, b.category , count(*) as scat
from test_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.offer = a.offer 
      and c.category = b.category 
group by a.id , b.category  


update train_history t1 
join scategory_train t2 on (t1.id = t2.id)
set t1.scategory = t2.scat 
where t1.id = t2.id 

update test_history t1 
join scategory_test t2 on (t1.id = t2.id)
set t1.scategory = t2.scat 
where t1.id = t2.id 



-- scompany - An id of the company that sells the item

create view scompany_train  as select   a.id, b.company , count(*) as scomp
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.offer = a.offer 
      and c.company  = b.company 
group by a.id , b.company  

 

-- sbrand - An id of the brand to which the item belongs


create view sbrand_train  as select   a.id, b.company , count(*) as sbrand
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.offer = a.offer 
      and c.brand  = b.brand
group by a.id , b.brand

-- tpurchaseamount - The dollar amousnt of the purchase

create view tot_purch_train as select a.id, sum(a.purchaseamount) as tpurchaseamount
from transaction a , train_history b 
where a.id = b.id and a.chain = b.chain 
group by id 

update train_history t1
join tot_purch_train t2 on (t1.id = t2.id)
set t1.tpurchaseamount = t2.tpurchaseamount 
where t1.id = t2.id 



445744636 - cat = 9909 
445746934 - cat = 9909 

---------scategory
445744636	9909	1
445746934	9909	2