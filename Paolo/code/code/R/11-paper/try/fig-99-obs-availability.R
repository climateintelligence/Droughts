# figure on number of available stns/obs per year

library(data.table)
library(magrittr)
library(ggplot2)
library(forcats)
library(slider)
# library(mitmatmisc)
#
dat <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")

dat <- mitmatmisc::add_month_fct(dat, 10)

dat[!is.na(HS) & month %in% c(11,12,1:5), 
    .(nn = .N), 
    .(year, month_fct)] %>% 
  ggplot(aes(year, nn))+
  geom_line()+
  facet_wrap(~month_fct)+
  theme_bw()

# 1980 - 2010 best
  