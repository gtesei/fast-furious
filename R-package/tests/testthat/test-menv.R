context("menv")

test_that('set_path and get_path in basic contest', {
  #skip_on_cran()
  ff.setBasePath(getwd())
  
  expect_equal(paste0(getwd(),.Platform$file.sep) , ff.getPath() )
  expect_equal(length(ff.getPathBindings()),1) 
})

test_that('max number of concurrent threads', {
  #skip_on_cran(
  
  expect_equal(ff.getMaxCuncurrentThreads() , 2 )
  
  ff.setMaxCuncurrentThreads(6)
  expect_equal(ff.getMaxCuncurrentThreads() , 6 )
  
  ff.setMaxCuncurrentThreads(2)
  expect_equal(ff.getMaxCuncurrentThreads() , 2 )
})

test_that('bind_path in basic contest', {
  #skip_on_cran()
  base_path = getwd()
  ff.setBasePath(base_path)
  if(! dir.exists("mydata") ) dir.create('mydata')
  ff.bindPath(type = "data",sub_path = "mydata")
  expect_equal(length(ff.getPathBindings()),2)
  expect_equal(ff.getPath('data') , paste(ff.getPath(),'mydata',.Platform$file.sep,sep = ''))
  
  expect_error(ff.bindPath(type = "data2",sub_path = "mydata2"))
  ff.bindPath(type = "data2",sub_path = "mydata2" , createDir=TRUE)
  expect_equal(ff.getPath('data2') , paste(ff.getPath(),'mydata2',.Platform$file.sep,sep = ''))
  
})

do_inside_func = function(key,val) {
  ff.bindPath(type = key,sub_path = val)
} 

test_that('bind_path inside a function', {
  #skip_on_cran()
  ff.setBasePath(getwd())
  if(! dir.exists("mydata") ) dir.create('mydata')
  ff.bindPath(type = "data",sub_path = "mydata")
  if(! dir.exists("mycode") ) dir.create('mycode')
  do_inside_func("code" , "mycode")
  expect_equal(length(ff.getPathBindings()),3)
  expect_equal(ff.getPath('data') , paste(ff.getPath(),'mydata',.Platform$file.sep,sep = ''))
})
