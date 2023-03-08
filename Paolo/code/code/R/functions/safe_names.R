# function to create safe names (for columns and files)
# replaces all non-alphabetic and non-numeric characters with _


safe_names <- function(x){
  # replace non-alphanumeric with _
  x1 <- gsub("[^[:alnum:]]+", "_", x)
  # replace foreign accents to ASCII
  x2 <- stringi::stri_trans_general(x1, "latin-ascii")
  # return
  x2
}


safe_names_iconv <- function(x){
  # replace non-alphanumeric with _
  x1 <- gsub("[^[:alnum:]]+", "_", x)
  # replace foreign accents to ASCII
  x2 <- iconv(x1, "latin1", "ASCII//TRANSLIT", sub = "_")
  # return
  x2
}