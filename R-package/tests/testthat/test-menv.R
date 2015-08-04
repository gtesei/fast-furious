context("menv")

test_that('set_path and get_path in basic contest', {
  #skip_on_cran()
  ff.set_base_path(getwd())
  
  expect_equal(paste0(getwd(),.Platform$file.sep) , ff.get_path() )
  expect_equal(length(ff.get_path_bindings()),1) 
})

test_that('bind_path in basic contest', {
  #skip_on_cran()
  base_path = getwd()
  ff.set_base_path(base_path)
  if(! dir.exists("mydata") ) dir.create('mydata')
  ff.bind_path(type = "data",sub_path = "mydata")
  expect_equal(length(ff.get_path_bindings()),2)
  expect_equal(ff.get_path('data') , paste(ff.get_path(),'mydata',.Platform$file.sep,sep = ''))
})

do_inside_func = function(key,val) {
  ff.bind_path(type = key,sub_path = val)
} 

test_that('bind_path inside a function', {
  #skip_on_cran()
  ff.set_base_path(getwd())
  if(! dir.exists("mydata") ) dir.create('mydata')
  ff.bind_path(type = "data",sub_path = "mydata")
  if(! dir.exists("mycode") ) dir.create('mycode')
  do_inside_func("code" , "mycode")
  expect_equal(length(ff.get_path_bindings()),3)
  expect_equal(ff.get_path('data') , paste(ff.get_path(),'mydata',.Platform$file.sep,sep = ''))
})
