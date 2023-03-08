# make monthly data

library(data.table)
library(magrittr)
library(lubridate)
library(fs)

min_frac_avail <- 0.9




# from QC -----------------------------------------------------------------



# ** pre 1960 -------------------------------------------------------------


dat_long <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1787hn_1879hs-1960/data_long_HN_HS.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1787hn_1879hs-1960/meta_long_HN_HS.rds")

dat_month <- dat_long[, .(HN = sum(HN, na.rm = T),
                          HS = mean(HS, na.rm = T),
                          nn_HN = sum(!is.na(HN)),
                          nn_HS = sum(!is.na(HS)),
                          nn_in_month = days_in_month(Date[1])),
                      .(Name, year(Date), month(Date))]

dat_month[nn_HN < min_frac_avail * nn_in_month, HN := NA]
dat_month[nn_HS < min_frac_avail * nn_in_month, HS := NA]

out_data_long <- dat_month[!is.na(HS) | !is.na(HN), .(Name, year, month, HN, HS)]
out_meta_long <- dat_meta[Name %in% unique(out_data_long$Name)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  .[, .(month = 1:12), .(Name, year)] %>% 
  merge(out_data_long, all.x = T) -> out_data_long


data_wide_hn <- dcast(out_data_long, year + month ~ Name, value.var = "HN")
lgl_cols_to_keep <- sapply(data_wide_hn, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hn <- data_wide_hn[, chr_cols_to_keep, with = F]
out_meta_wide_hn <- out_meta_long[Name %in% chr_cols_to_keep]

data_wide_hs <- dcast(out_data_long, year + month ~ Name, value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- out_meta_long[Name %in% chr_cols_to_keep]


path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1787hn_1879hs-1960/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))

# ** post 1960 ------------------------------------------------------------


dat_long <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

dat_month <- dat_long[, .(HN = sum(HN, na.rm = T),
                          HS = mean(HS, na.rm = T),
                          nn_HN = sum(!is.na(HN)),
                          nn_HS = sum(!is.na(HS)),
                          nn_in_month = days_in_month(Date[1])),
                      .(Name, year(Date), month(Date))]

dat_month[nn_HN < min_frac_avail * nn_in_month, HN := NA]
dat_month[nn_HS < min_frac_avail * nn_in_month, HS := NA]

out_data_long <- dat_month[!is.na(HS) | !is.na(HN), .(Name, year, month, HN, HS)]
out_meta_long <- dat_meta[Name %in% unique(out_data_long$Name)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  .[, .(month = 1:12), .(Name, year)] %>% 
  merge(out_data_long, all.x = T) -> out_data_long


data_wide_hn <- dcast(out_data_long, year + month ~ Name, value.var = "HN")
lgl_cols_to_keep <- sapply(data_wide_hn, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hn <- data_wide_hn[, chr_cols_to_keep, with = F]
out_meta_wide_hn <- out_meta_long[Name %in% chr_cols_to_keep]

data_wide_hs <- dcast(out_data_long, year + month ~ Name, value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- out_meta_long[Name %in% chr_cols_to_keep]


path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))




# gapfill -----------------------------------------------------------------

# only post 1960 and HS
# add info on fraction filled


dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_hs


dat_month <- dat_hs[, .(HS = mean(HS, na.rm = T),
                        nn_HS = sum(!is.na(HS)),
                        nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                        nn_original = sum(HS_fillcode == 1, na.rm = T),
                        nn_in_month = days_in_month(Date[1])),
                    .(Name, year(Date), month(Date))]

# remove if less than min number of obs per month
dat_month[nn_HS < min_frac_avail * nn_in_month, HS := NA]



# ** all ---------------------------------------------------------------------

dat_month_all <- copy(dat_month)

out_data_long <- dat_month_all[!is.na(HS), 
                               .(Name, year, month, HS, 
                                 frac_gapfilled = nn_gapfill / nn_in_month)]
out_meta_long <- dat_meta[Name %in% unique(out_data_long$Name)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  .[, .(month = 1:12), .(Name, year)] %>% 
  merge(out_data_long, all.x = T) -> out_data_long

data_wide_hs <- dcast(out_data_long, year + month ~ Name, value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- out_meta_long[Name %in% chr_cols_to_keep]

data_wide_hs_frac <- dcast(out_data_long, year + month ~ Name, value.var = "frac_gapfilled")
out_data_wide_hs_frac <- data_wide_hs_frac[, chr_cols_to_keep, with = F]

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_all/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HS.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_data_wide_hs_frac, file = path(path_out, "data_wide_HS_frac_gapfilled.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HS.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))











# ** subset ---------------------------------------------------------------------

# subset to some meaningful for later analysis
dat_month_sub <- copy(dat_month)

# remove more than 5 years before after original period
dat_month_sub[, year_start := min(year[nn_original > 0]), .(Name)]
dat_month_sub[, year_end := max(year[nn_original > 0]), .(Name)]

dat_month_sub[year < (year_start - 5), HS := NA]
dat_month_sub[year > (year_end + 5), HS := NA]


# remove stn with more gapfilled than original data Nov-May
# and at least 20 different years of data


mitmatmisc::add_hydro_year(dat_month_sub)
dat_nn <- dat_month_sub[month %in% c(11:12, 1:5) & !is.na(HS),
                        .(nn_total = sum(nn_HS),
                          nn_gapfill = sum(nn_gapfill),
                          nn_original = sum(nn_original),
                          nn_years = length(unique(hydro_year))),
                        .(Name)]

dat_nn[nn_gapfill > nn_original]
# n_days_NovMay <- 30+31+31+28+31+30+31
# dat_nn[nn_gapfill > (0.5 * nn_years * n_days_NovMay)]
# dat_nn[nn_original > (0.5 * nn_years * n_days_NovMay)]

# dat_nn %>% 
#   ggplot(aes(nn_original, nn_gapfill, colour = nn_years, alpha = nn_years >= 20))+
#   geom_point()+
#   geom_abline()+
#   scale_color_viridis_c()

stn_gapfill_sub <- dat_nn[nn_gapfill < nn_original & nn_years >= 20, Name]
dat_meta[Name %in% stn_gapfill_sub, .N, .(Provider)]
dat_meta[Name %in% dat_nn$Name, .N, .(Provider)]



# prep for write
out_data_long <- dat_month_sub[!is.na(HS) & Name %in% stn_gapfill_sub, 
                               .(Name, year, month, HS, 
                                 frac_gapfilled = nn_gapfill / nn_in_month)]
out_meta_long <- dat_meta[Name %in% unique(out_data_long$Name)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  .[, .(month = 1:12), .(Name, year)] %>% 
  merge(out_data_long, all.x = T) -> out_data_long

data_wide_hs <- dcast(out_data_long, year + month ~ Name, value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- out_meta_long[Name %in% chr_cols_to_keep]

data_wide_hs_frac <- dcast(out_data_long, year + month ~ Name, value.var = "frac_gapfilled")
out_data_wide_hs_frac <- data_wide_hs_frac[, chr_cols_to_keep, with = F]

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HS.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_data_wide_hs_frac, file = path(path_out, "data_wide_HS_frac_gapfilled.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HS.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))











