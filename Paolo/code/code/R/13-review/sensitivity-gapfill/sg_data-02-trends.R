# trends


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(nlme)
library(broom)
library(broom.mixed)

# prep data ---------------------------------------------------------------

dat_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/data-01-monthly.rds")
# dat_seasonal <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")

# for month calendar year, for seasonal hydro year, but same name
dat_month[, year0 := year - min(year)]
# dat_seasonal[, year0 := year - min(year)]

setnames(dat_month, "HS", "value")


# OLS ---------------------------------------------------------------------

f_ols <- function(dat){
  if(length(unique(dat$value)) <= 2) return(NULL)
  
  lm1 <- lm(value ~ year0, data = dat)
  dat_cf <- tidy(lm1)
  dat_summ <- glance(lm1)
  
  data.table(dat_cf, dat_summ[, c("r.squared", "sigma")], resid.sd = sd(resid(lm1)))
}

dat_month_ols <- dat_month[, f_ols(.SD), .(Name, month)]
# dat_seasonal_ols <- dat_seasonal[, f_ols(.SD), .(Name, variable)]


# GLS ---------------------------------------------------------------------

f_gls <- function(dat){
  
  if(length(unique(dat$value)) <= 2) return(NULL)
  
  gls0 <- gls(value ~ year0, data = dat)
  gls1 <- try(gls(value ~ year0, data = dat, weights = varExp(form = ~year0)),
              silent = T)
  if(inherits(gls1, "try-error")) {
  
    dat_cf <- tidy(gls0)
    dat_summ <- glance(gls0)
    rsq <- 1 - sum(resid(gls0)^2) / sum((dat$value - mean(dat$value))^2)
    
    data.table(dat_cf, 
               r.squared = rsq,
               dat_summ[, c("sigma")], 
               resid.sd = sd(resid(gls0)),
               cf.var = NA_real_,
               p.value.var = NA_real_)
    
  } else {
    
    dat_cf <- tidy(gls1)
    dat_summ <- glance(gls1)
    rsq <- 1 - sum(resid(gls1)^2) / sum((dat$value - mean(dat$value))^2)
    an <- anova(gls0, gls1)
    pval_comp <- an[2, "p-value"]
    
    data.table(dat_cf, 
               r.squared = rsq,
               dat_summ[, c("sigma")], 
               resid.sd = sd(resid(gls1)),
               cf.var = coef(gls1$modelStruct$varStruct),
               p.value.var = pval_comp)
    
  }
  
}


dat_month_gls <- dat_month[, f_gls(.SD), .(Name, month)]
# dat_seasonal_gls <- dat_seasonal[, f_gls(.SD), .(Name, variable)]



# save --------------------------------------------------------------------



save(dat_month_ols, dat_month_gls, # dat_seasonal_ols, dat_seasonal_gls,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/trends-01-1971-2019-ols-gls.rda")



