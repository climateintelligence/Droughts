# trends


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(fs)

# settings ----------------------------------------------------------------

month_sub <- c(11:12, 1:5)


# monthly ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")

dat_hs <- mitmatmisc::add_hydro_year(dat_hs)


# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]


# get data (whole 1971-2019) period 

dat_hs2 <- dat_hs[year >= 1971 & year <= 2019 & month %in% month_sub & !is.na(HS)]
dat_hs2[, nn_year := .N, .(Name, month)]
dat_hs_full <- dat_hs2[nn_year == max(nn_year)]


# remove stn-months with all HS < 1cm (only Apr and May)
dat_hs_full[, all(HS < 1), .(Name, month)] %>% 
  .[!(V1 == TRUE & month %in% c(4,5)), .(Name, month)] %>% 
  merge(dat_hs_full, by = c("Name", "month")) -> dat_hs_full2

saveRDS(dat_hs_full2, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-01-monthly.rds")



# seasonal ----------------------------------------------------------------

dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/indices/") %>% 
  grep("meta_all", ., invert = T, value = T) %>% 
  lapply(function(x) {
    dat <- readRDS(x)
    dat[, frac_gapfilled := NULL]
    }) %>% 
  Reduce(function(x,y) merge(x,y,all=T), .) -> dat_ind_wide

dat_ind_wide <- dat_ind_wide[Name %in% stns_ok]
dat_ind_long <- melt(dat_ind_wide, id.vars = c("Name", "year"), variable.factor = F)



# get data (whole 1971-2019) period 

dat_ind_long2 <- dat_ind_long[year >= 1971 & year <= 2019 & !is.na(value)]
dat_ind_long2[, nn_year := .N, .(Name, variable)]
dat_ind_long2_full <- dat_ind_long2[nn_year == max(nn_year)]

saveRDS(dat_ind_long2_full, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")

