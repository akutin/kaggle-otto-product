prediction <- function(weights)
{
  m = max(weights)
  r = rep(0, length(weights))
  r[ m == weights] <- 1
  return(r)
}
