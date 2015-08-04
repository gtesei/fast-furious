
### CONSTANTS  
FAST_FURIOUS_BASE_PATH_KEY = "FAST_FURIOUS_BASE_PATH"
FAST_FURIOUS_BASE_PATH_VALUE = NULL
FAST_FURIOUS_PTH_BINDINGS = list()

### FUNCS 
ff.set <- function(key , val, force = F)  {
  
  stopifnot(is.character(key), length(key) == 1)
  
  if (is.null(FAST_FURIOUS_BASE_PATH_VALUE) || force) {
    FAST_FURIOUS_BASE_PATH_VALUE <<- val 
    FAST_FURIOUS_PTH_BINDINGS <<- list()
    FAST_FURIOUS_PTH_BINDINGS[["base"]] <<- val
  } else {
    stop( "Can't set up " , key, call. = FALSE)
  }
}

#' Set base path 
#' 
#' @param path the absolute path. 
#' 
#' @examples
#' ff.set_base_path('/Users/gg/root')
#' @export
#' 
ff.set_base_path = function (path) {
  stopifnot(is.character(path), length(path) == 1 , file.exists(path) )
  
  if(!identical( substr(path, nchar(path), nchar(path) ) , .Platform$file.sep))
    path = paste0(path,.Platform$file.sep)
  
  ff.set(FAST_FURIOUS_BASE_PATH_KEY , path, force = T)
}

#' Get the absolute path for a kind of resources.  
#' 
#' @param type the type of resource.
#' 
#' @examples
#' ff.get_path() ## equivalent to ff.get_path = function (type="base") gets the base path 
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
#' ff.bind_path(type = "data",sub_path = "dataset")
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
#' ff.get_bindings()
#' @export
#' 
ff.get_path_bindings = function() {
  return(FAST_FURIOUS_PTH_BINDINGS)
}


 

