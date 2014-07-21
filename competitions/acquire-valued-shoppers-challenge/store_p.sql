DELIMITER $$
CREATE DEFINER=`root`@`localhost` PROCEDURE `do_job`()
BEGIN

-- scategory
update test_history t1 
join scategory_test t2 on (t1.id = t2.id)
set t1.scategory = t2.scat 
where t1.id = t2.id; 

-- scompany
create view scompany_train  as select   a.id, b.company , count(*) as scomp
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.id = a.offer 
      and c.company  = b.company 
group by a.id , b.company;  

create view scompany_test  as select   a.id, b.company , count(*) as scomp
from test_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.id = a.offer 
      and c.company  = b.company 
group by a.id , b.company;  

update train_history t1 
join scompany_train t2 on (t1.id = t2.id)
set t1.scompany = t2.scomp 
where t1.id = t2.id; 

update test_history t1 
join scompany_test t2 on (t1.id = t2.id)
set t1.scompany = t2.scomp 
where t1.id = t2.id; 

-- sbrand 
create view sbrand_train  as select   a.id, b.company , count(*) as sbrand
from train_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.id = a.offer 
      and c.brand  = b.brand
group by a.id , b.brand;

create view sbrand_test  as select   a.id, b.company , count(*) as sbrand
from test_history a , transaction b , offer c 
where a.id = b.id and a.chain = b.chain and c.id = a.offer 
      and c.brand  = b.brand
group by a.id , b.brand;

update train_history t1 
join sbrand_train t2 on (t1.id = t2.id)
set t1.sbrand = t2.sbrand 
where t1.id = t2.id; 

update test_history t1 
join sbrand_test t2 on (t1.id = t2.id)
set t1.sbrand = t2.sbrand 
where t1.id = t2.id; 

-- tpurchaseamount
create view tot_purch_train as select a.id, sum(a.purchaseamount) as tpurchaseamount
from transaction a , train_history b 
where a.id = b.id and a.chain = b.chain 
group by id; 

update train_history t1
join tot_purch_train t2 on (t1.id = t2.id)
set t1.tot_amount = t2.tpurchaseamount 
where t1.id = t2.id; 

create view tot_purch_test as select a.id, sum(a.purchaseamount) as tpurchaseamount
from transaction a , test_history b 
where a.id = b.id and a.chain = b.chain 
group by id; 

update test_history t1
join tot_purch_train t2 on (t1.id = t2.id)
set t1.tot_amount = t2.tpurchaseamount 
where t1.id = t2.id; 

END$$
DELIMITER ;
