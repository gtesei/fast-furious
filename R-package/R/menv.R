
### CONSTANTS  
# assign("FAST_FURIOUS_BASE_PATH_VALUE", NULL, .GlobalEnv)
# assign("FAST_FURIOUS_MAX_THREADS", 2, .GlobalEnv)
# assign("FAST_FURIOUS_PTH_BINDINGS", list(), .GlobalEnv)

### FUNCS 

#' Set the max number of cuncurrent threads. 
#' 
#' @param nThreads max number of cuncurrent threads. 
#' 
#' @examples
#' ff.setMaxCuncurrentThreads(4)
#' @export
#' 
ff.setMaxCuncurrentThreads = function (nThreads=2) {
  stopifnot(is.numeric(nThreads), length(nThreads) == 1 )
  assign("FAST_FURIOUS_MAX_THREADS", nThreads, .GlobalEnv)
}

#' Get the max number of cuncurrent threads. 
#' 
#' @examples
#' ff.getMaxCuncurrentThreads()
#' @export
#' 
ff.getMaxCuncurrentThreads = function() {
  if (! exists(x = "FAST_FURIOUS_MAX_THREADS" , envir = .GlobalEnv ) ) assign("FAST_FURIOUS_MAX_THREADS", 2, .GlobalEnv) 
  return(get(x = "FAST_FURIOUS_MAX_THREADS" , envir = .GlobalEnv))
}


#' Set base path 
#' 
#' @param path the absolute path. 
#' 
#' @examples
#' ff.setBasePath('./')
#' @export
#' 
ff.setBasePath = function (path) {
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
#' ff.setBasePath('./')
#' ff.getPath() ## equivalent to ff.getPath(type="base") 
#' @export
#' 
ff.getPath = function (type="base") {
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
#' @param createDir set to \code{'TRUE'} to create the directory if it does not exist
#' 
#' @examples
#' ff.setBasePath(getwd())
#' if(! dir.exists("mydata") ) dir.create('mydata')
#' ff.bindPath(type = "data",sub_path = "mydata")
#' @export
#' 
ff.bindPath = function (type,sub_path,createDir=FALSE) {
  stopifnot(is.character(type), length(type) == 1)
  stopifnot( ! is.null(FAST_FURIOUS_BASE_PATH_VALUE) )
  stopifnot( ! identical(type,'base') )
  
  path = paste0(FAST_FURIOUS_PTH_BINDINGS[['base']],sub_path)
  if(!identical( substr(path, nchar(path), nchar(path) ) , .Platform$file.sep))
    path = paste0(path,.Platform$file.sep)
  
  if (! file.exists(path) && ! createDir) stop(paste0(path,' does not exist')) 
  else if (! file.exists(path) && createDir) dir.create(path , recursive = TRUE)
  
  FAST_FURIOUS_PTH_BINDINGS[[type]] <<- path
}

#' Get the list of bindings, i.e. (type resource,absolute path) pairs as a list 
#' 
#' @examples
#' ff.setBasePath(getwd())
#' if(! dir.exists("mydata") ) dir.create('mydata')
#' ff.bindPath(type = "data",sub_path = "mydata")
#' ff.getPathBindings()
#' @export
#' 
ff.getPathBindings = function() {
  return(FAST_FURIOUS_PTH_BINDINGS)
}
