context("menv")

test_that('set_path and get_path in basic contest', {
  skip_on_cran()
  base_path = getwd()
  ff.set_base_path(base_path)
  
  expect_equal(paste0(base_path,.Platform$file.sep) , ff.get_path() )
  expect_equal(length(ff.get_path_bindings()),1) 
})

test_that('bind_path in basic contest', {
  skip_on_cran()
  base_path = getwd()
  ff.set_base_path(base_path)
  ff.bind_path(type = "data",sub_path = "dataset")
  expect_equal(length(ff.get_path_bindings()),2)
  expect_equal(ff.get_path('data') , paste(ff.get_path(),'dataset',.Platform$file.sep,sep = ''))
})

do_inside_func = function(key,val) {
  ff.bind_path(type = key,sub_path = val)
} 

test_that('bind_path inside a function', {
  skip_on_cran()
  base_path = getwd()
  ff.set_base_path(base_path)
  ff.bind_path(type = "data",sub_path = "dataset")
  do_inside_func("code" , "data_process")
  expect_equal(length(ff.get_path_bindings()),3)
  expect_equal(ff.get_path('data') , paste(ff.get_path(),'dataset',.Platform$file.sep,sep = ''))
})
