# try out correlations plots



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)



# read data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]


dat_apgd <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/apgd-01-series.rds")
dat_eobs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/eobs-01-series.rds")

dat_chelsa <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/chelsa-01-series.rds")
dat_uerra <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/extract-stn/uerra-01-series.rds")



# obs data: apgd & eobs

dat_hs[year %in% 1971:2008 & month %in% c(11,12,1:5)] %>% 
  merge(dat_apgd) %>% 
  merge(dat_eobs) -> dat_hs_apgd_eobs

dat_hs_apgd_eobs[, 
                 n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
                 .(Name, month)]

dat_hs_apgd_eobs <- dat_hs_apgd_eobs[n_year >= 30]



# chelsa

dat_hs_chelsa <- merge(dat_hs[month %in% c(11,12,1:5)], 
                       dat_chelsa[prec < 60000])

dat_hs_chelsa[, 
              n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
              .(Name, month)]

dat_hs_chelsa <- dat_hs_chelsa[n_year >= 30]


# uerra (obs period)

dat_hs[year %in% 1971:2008 & month %in% c(11,12,1:5)] %>% 
  merge(dat_uerra) -> dat_hs_uerra

dat_hs_uerra[, 
             n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
             .(Name, month)]

dat_hs_uerra <- dat_hs_uerra[n_year >= 30]



# uerra (full period)

dat_hs[month %in% c(11,12,1:5)] %>% 
  merge(dat_uerra) -> dat_hs_uerra_full

dat_hs_uerra_full[, 
             n_year := sum(!is.na(HS) & !is.na(prec) & !is.na(tmean)),
             .(Name, month)]

dat_hs_uerra_full <- dat_hs_uerra_full[n_year >= 30]



# save --------------------------------------------------------------------

save(dat_hs_apgd_eobs, dat_hs_chelsa,
     dat_hs_uerra, dat_hs_uerra_full,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data4corr-01-merged.rda")
