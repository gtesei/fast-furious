
require(RUnit)

if (ff_test_param$verbose) cat(">>> Testing menv.R ... \n")

base_path = '/Users/gino/kaggle/fast-furious/gitHub/fast-furious'
ff.set_base_path(base_path)
checkEquals(paste0(base_path,.Platform$file.sep) , ff.get_path() )
checkEquals(length(ff.get_bindings()),1)

ff.bind_sub_path(type = "data",sub_path = "dataset")
checkEquals(length(ff.get_bindings()),2)
checkEquals(ff.get_path('data') , paste(ff.get_path(),'dataset',.Platform$file.sep,sep = '') )
checkException(ff.get_path('unkown'),silent = T)

do_inside_func = function(key,val) {
  ff.bind_sub_path(type = key,sub_path = val)
} 

do_inside_func("code" , "data_process")
checkEquals(length(ff.get_bindings()),3)
checkEquals(ff.get_path('code') , paste(ff.get_path(),'data_process',.Platform$file.sep,sep = '') )

if (ff_test_param$verbose) cat(">>> End of menv.R testing\n")