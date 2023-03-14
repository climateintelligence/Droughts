# hn hs prep

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)


# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")

dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")



# dat_1[, month := month(Date)]
# dat_2[, month := month(Date)]
# 
# dat_1[, season_year := year(Date)]
# dat_2[, season_year := year(Date)]
# 
# dat_1[month <= 8, season_year := season_year - 1L]
# dat_2[month <= 8, season_year := season_year - 1L]

dat_all <- rbind(dat_2, dat_1)
setkey(dat_all, Provider, Name, Date)


dat_all[, HN := as.integer(round(HN))]
dat_all[, HS := as.integer(round(HS))]

dat_all[, HN_lag := shift(HN, type = "lag"), .(Provider, Name)]
dat_all[, HS_lag := shift(HS, type = "lag"), .(Provider, Name)]
dat_all[, HS_diff := HS - HS_lag]

dat_nonzero <- dat_all[HS != 0 & HN != 0]




# HN HS consistency -------------------------------------------------------

# dat_check <- dat_all[HS > (HN + HS_lag) | HS > (HN_lag + HS_lag)]
# dat_check <- dat_all[(HS_diff - HN_lag) > 0]
# dat_check <- dat_all[(HS_diff - HN) > 0]

# with or without tolerance of 5cm?
dat_check <- dat_all[(HS - HS_lag) > (HN + HN_lag + 5)]
dat_check <- dat_all[(HS - HS_lag) > (HN + HN_lag)]


dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% qplot()
dat_check[(HN + HN_lag) != 0, .N, .(Provider, Name)] %>% .[, N] %>% qplot()

dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% table
dat_check[, .N, .(Provider)]
dat_check[(HN + HN_lag) == 0, .N, .(Provider, Name)]
dat_check[HN + HN_lag == 0, .N, .(Provider)]


dat_check

# -> maybe just indicative, too much to check..

# EOF ---------------------------------------------------------------------


