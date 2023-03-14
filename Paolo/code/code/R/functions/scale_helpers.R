# some helpers for scaling

sc_get_coef <- function(x){
  
  mu <- mean(x)
  x_center <- x - mu
  std <- sd(x_center)
  
  c(mu = mu, std = std)
  
}


sc_apply_scaling <- function(x, sc_coef){
  
  (x - sc_coef["mu"]) / sc_coef["std"]
  
}

sc_invert_scaling <- function(y, sc_coef){
  
  (y * sc_coef["std"]) + sc_coef["mu"]
  
}