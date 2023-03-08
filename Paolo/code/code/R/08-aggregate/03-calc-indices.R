# make monthly data

library(data.table)
library(magrittr)
library(lubridate)
library(fs)

min_frac_avail <- 0.9

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/indices/"
if(!dir_exists(path_out)) dir_create(path_out)

# from QC -----------------------------------------------------------------

# not for now, if needed copy from 01-make-monthly.R and adapt



# gapfill -----------------------------------------------------------------

# only post 1960 and HS
# add info on fraction filled


dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_hs[, year := year(Date)]
dat_hs[, month := month(Date)]
mitmatmisc::add_season_fct(dat_hs, add_n_days = T)
mitmatmisc::add_hydro_year(dat_hs)
dat_hs




f_sub_meaning <- function(dat_summ){
  
  # subset to some meaningful for later analysis
  dat_summ_sub <- copy(dat_summ)
  
  # remove if less than min number of obs per period
  dat_summ_sub[nn_value < min_frac_avail * nn_max, value := NA]
  
  # remove all missing
  dat_summ_sub[, sum(!is.na(value)), .(Name)] %>% 
    .[V1 > 1, .(Name)] %>% 
    merge(dat_summ_sub) -> dat_summ_sub
  
  # remove more than 5 years before after original period
  dat_summ_sub[, year_start := min(year[nn_original > 0]), .(Name)]
  dat_summ_sub[, year_end := max(year[nn_original > 0]), .(Name)]
  
  dat_summ_sub[year < (year_start - 5), value := NA]
  dat_summ_sub[year > (year_end + 5), value := NA]
  
  
  # remove stn with more gapfilled than original data
  # and at least 20 different years of data

  dat_nn <- dat_summ_sub[!is.na(value),
                         .(nn_total = sum(nn_value),
                           nn_gapfill = sum(nn_gapfill),
                           nn_original = sum(nn_original),
                           nn_years = length(unique(year))),
                         .(Name)]
  
  dat_nn[nn_gapfill > nn_original]
  stn_gapfill_sub <- dat_nn[nn_gapfill < nn_original & nn_years >= 20, Name]
  dat_summ_sub[Name %in% stn_gapfill_sub]
}


# ** mean DJF ----------------------------------------------------------------

dat_summ <- dat_hs[season == "DJF", 
                   .(value = mean(HS, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = unique(n_days_season)),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, meanHS_DJF = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "meanHS_DJF.rds"))


# ** mean MAM ----------------------------------------------------------------

dat_summ <- dat_hs[season == "MAM", 
                   .(value = mean(HS, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = unique(n_days_season)),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, meanHS_MAM = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "meanHS_MAM.rds"))



# ** mean NDJFMAM ----------------------------------------------------------------

dat_summ <- dat_hs[month %in% c(11:12,1:5), 
                   .(value = mean(HS, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = 30+31+31+28+31+30+31),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, meanHS_NDJFMAM = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "meanHS_NDJFMAM.rds"))




# ** max NDJFMAM ----------------------------------------------------------------

# remove missing
dat_hs[month %in% c(11:12,1:5), all(is.na(HS)), .(Name, hydro_year)] %>% 
  .[V1 == FALSE, .(Name, hydro_year)] %>% 
  merge(dat_hs) -> dat_hs2

dat_summ <- dat_hs2[month %in% c(11:12,1:5), 
                    .(value = max(HS, na.rm = T),
                      nn_value = sum(!is.na(HS)),
                      nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                      nn_original = sum(HS_fillcode == 1, na.rm = T),
                      nn_max = 30+31+31+28+31+30+31),
                    .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, maxHS_NDJFMAM = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "maxHS_NDJFMAM.rds"))



# ** SCD NDJF ----------------------------------------------------------------

dat_summ <- dat_hs[month %in% c(11:12,1:2), 
                   .(value = sum(HS > 1, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = 30+31+31+28),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, SCD_NDJF = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "SCD_NDJF.rds"))


# ** SCD MAM ----------------------------------------------------------------

dat_summ <- dat_hs[month %in% c(3:5), 
                   .(value = sum(HS > 1, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = 31+30+31),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, SCD_MAM = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "SCD_MAM.rds"))


# ** SCD NDJFMAM ----------------------------------------------------------------

dat_summ <- dat_hs[month %in% c(11:12,1:5), 
                   .(value = sum(HS > 1, na.rm = T),
                     nn_value = sum(!is.na(HS)),
                     nn_gapfill = sum(HS_fillcode == 222, na.rm = T),
                     nn_original = sum(HS_fillcode == 1, na.rm = T),
                     nn_max = 30+31+31+28+31+30+31),
                   .(Name, year = hydro_year)]

dat_summ_sub <- f_sub_meaning(dat_summ)

# prep for write
out_data_long <- dat_summ_sub[!is.na(value), 
                              .(Name, year, SCD_NDJFMAM = value, 
                                frac_gapfilled = nn_gapfill / nn_max)]

out_data_long[, .(year = min(year) : max(year)), .(Name)] %>% 
  merge(out_data_long, all.x = T, by = c("Name", "year")) -> out_data_long

saveRDS(out_data_long, file = path(path_out, "SCD_NDJFMAM.rds"))



# ** meta all -------------------------------------------------------------

file_copy("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1787hn_1879hs-1960/meta_long_HN_HS.rds",
          path(path_out, "meta_all.rds"),
          overwrite = T)







