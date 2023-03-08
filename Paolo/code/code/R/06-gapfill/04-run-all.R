# run cv of gapfill

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(fs)
library(runner)

source("R/06-gapfill/fdmg_v08.R")

# prep data ---------------------------------------------------------------



# dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
# dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")
# dat_1_w <- dcast(dat_1, Date ~ Name, value.var = "HS")
# mat_1 <- as.matrix(dat_1_w[, -c("Date"), with  = F])
# dat_meta_1 <- dat_meta_1[Name %in% colnames(mat_1)]

dat_1_w <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_wide_HS.rds")
mat_1 <- as.matrix(dat_1_w[, -c("Date"), with  = F])
dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_wide_HS.rds")


setkey(dat_meta_1, Name)
setnames(dat_meta_1, c("provider", "name", "long", "lat", "elev"))

vec_dates <- dat_1_w$Date

# out_ref_parameter <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/aux-ref-parameter/final-HS/"
out_ref_parameter <- NULL

# loop --------------------------------------------------------------------


mitmatmisc::init_parallel_ubuntu(6)

dat_out <- foreach(
  i_stn = 1:nrow(dat_meta_1),
  .final = function(x) rbindlist(x, use.names = T, fill = T)
) %dopar% {
  
  l_fill <- fill_daily_meteo_gaps(df_meta = dat_meta_1,
                                  mat_series = mat_1,
                                  vec_dates = vec_dates,
                                  stns_to_fill = i_stn,
                                  rows_to_fill = NULL,
                                  min_corr = 0.7,
                                  max_dist_horiz_km = 200,
                                  max_dist_vert_m = 500,
                                  elev_threshold = NULL,
                                  digits_round = 0,
                                  ratio_var = T,
                                  sort_by = "corr",
                                  weight_by = "dist_v",
                                  n_ref_max = 5,
                                  verbose = 0,
                                  save_ref_parameter = out_ref_parameter)
  
  
  
  data.table(Name = dat_meta_1$name[i_stn],
             Date = vec_dates,
             HS = l_fill$mat_series_filled,
             HS_fillcode = l_fill$mat_fillcodes)
  
}


saveRDS(dat_out, 
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

# EOF ---------------------------------------------------------------------


