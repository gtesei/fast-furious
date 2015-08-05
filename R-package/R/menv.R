
### CONSTANTS  
assign("FAST_FURIOUS_BASE_PATH_VALUE", NULL, .GlobalEnv)
assign("FAST_FURIOUS_PTH_BINDINGS", list(), .GlobalEnv)

### FUNCS 

#' Set base path 
#' 
#' @param path the absolute path. 
#' 
#' @examples
#' ff.set_base_path('./')
#' @export
#' 
ff.set_base_path = function (path) {
  stopifnot(is.character(path), length(path) == 1 , file.exists(path) )
  
  if(!identical( substr(path, nchar(path), nchar(path) ) , .Platform$file.sep))
    path = paste0(path,.Platform$file.sep)
  
  #unlockBinding("FAST_FURIOUS_BASE_PATH_VALUE", .GlobalEnv)
  assign("FAST_FURIOUS_BASE_PATH_VALUE", path, .GlobalEnv)
  FAST_FURIOUS_PTH_BINDINGS <<- list()
  FAST_FURIOUS_PTH_BINDINGS[["base"]] <<- path
  #lockBinding("FAST_FURIOUS_BASE_PATH_VALUE", .GlobalEnv)
}

#' Get the absolute path for a kind of resources.  
#' 
#' @param type the type of resource.
#' 
#' @examples
#' ff.set_base_path('./')
#' ff.get_path() ## equivalent to ff.get_path(type="base") 
#' @export
#' 
ff.get_path = function (type="base") {
  stopifnot(is.character(type), length(type) == 1)
  stopifnot( ! is.null(FAST_FURIOUS_BASE_PATH_VALUE) )
  stopifnot( sum(unlist(lapply(names(FAST_FURIOUS_PTH_BINDINGS) , function(x) {
    identical(type,x)
  }))) > 0 )
  
  return(FAST_FURIOUS_PTH_BINDINGS[[type]])
}

#' Bind an absolute path for a kind of resources.  
#' 
#' @param type the type of resource.
#' @param sub_path the suffix to concatenate to the absolute path to get the absolute path of the kind of resource.
#' 
#' @examples
#' ff.set_base_path(getwd())
#' if(! dir.exists("mydata") ) dir.create('mydata')
#' ff.bind_path(type = "data",sub_path = "mydata")
#' @export
#' 
ff.bind_path = function (type,sub_path) {
  stopifnot(is.character(type), length(type) == 1)
  stopifnot( ! is.null(FAST_FURIOUS_BASE_PATH_VALUE) )
  stopifnot( ! identical(type,'base') )
  
  path = paste0(FAST_FURIOUS_PTH_BINDINGS[['base']],sub_path)
  if(!identical( substr(path, nchar(path), nchar(path) ) , .Platform$file.sep))
    path = paste0(path,.Platform$file.sep)
  
  stopifnot(file.exists(path)) 
  
  FAST_FURIOUS_PTH_BINDINGS[[type]] <<- path
}

#' Get the list of bindings (type of resource , absolutepath)   
#' 
#' 
#' @examples
#' ff.set_base_path(getwd())
#' if(! dir.exists("mydata") ) dir.create('mydata')
#' ff.bind_path(type = "data",sub_path = "mydata")
#' ff.get_path_bindings()
#' @export
#' 
ff.get_path_bindings = function() {
  return(FAST_FURIOUS_PTH_BINDINGS)
}


 

