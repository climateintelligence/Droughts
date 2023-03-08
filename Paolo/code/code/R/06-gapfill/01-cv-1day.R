# run cv of gapfill

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(fs)
library(foreach)

source("R/06-gapfill/fdmg_v08.R")

# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")

dat_1[, month := month(Date)]
dat_1[, season_year := year(Date)]
dat_1[month <= 8, season_year := season_year - 1L]

dat_1 <- dat_1[season_year >= 1981 & season_year <= 2010]

dat_1_w <- dcast(dat_1, Date ~ Name, value.var = "HS")

mat_1 <- as.matrix(dat_1_w[, -c("Date"), with  = F])

dat_meta_1 <- dat_meta_1[Name %in% colnames(mat_1)]
setkey(dat_meta_1, Name)
setnames(dat_meta_1, c("provider", "name", "long", "lat", "elev"))

vec_dates <- dat_1_w$Date


# loop --------------------------------------------------------------------

set.seed(1234)

mitmatmisc::init_parallel_ubuntu(6)

# test sub
# test_sub_stn <- sample(1:nrow(dat_meta_1), 28)

# reduce sample for DE, CH_MSWISS, and AT
# sample_stn_name <- c(
#   dat_meta_1[provider == "DE_DWD", name[sample.int(.N, 300)]],
#   dat_meta_1[provider == "CH_METEOSWISS", name[sample.int(.N, 200)]],
#   dat_meta_1[provider == "AT_HZB", name[sample.int(.N, 300)]],
#   dat_meta_1[!provider %in% c("DE_DWD", "CH_METEOSWISS", "AT_HZB"), name]
# )
# sample_stn_id <- which(dat_meta_1$name %in% sample_stn_name)

dat_out <- foreach(
  i_stn = 1:nrow(dat_meta_1),
  # i_stn = test_sub_stn,
  # i_stn = sample_stn_id,
  .final = function(x) rbindlist(x, use.names = T, fill = T)
) %dopar% {
  
  i_stn_name <- dat_meta_1$name[i_stn]
  
  # pre subset stations
  full_dist_km <- geosphere::distm(dat_meta_1[, c("long", "lat")], fun = geosphere::distCosine) / 1000

  stn_sub <- which(full_dist_km[i_stn, ] < 200)
  stn_sub <- intersect(stn_sub, 
                       which(abs(dat_meta_1$elev - dat_meta_1$elev[i_stn]) < 500))
  
  mat_i <- mat_1[, stn_sub]
  meta_i <- dat_meta_1[stn_sub]
  
  dat_out_stn <- foreach(
    i_month = c(11, 12, 1:5),
    .final = function(x) rbindlist(x, use.names = T, fill = T)
  ) %do% {
    
    i_row_possible <- which(!is.na(mat_1[, i_stn]) & month(vec_dates) == i_month)
    
    if(length(i_row_possible) == 0) return(NULL)
    
    if(length(i_row_possible) < 100){
      i_row_sample <- i_row_possible
    } else {
      i_row_sample <- sample(i_row_possible, 100)
    }
    
    
    dat_out_month <- foreach(
      i_rep = 1:length(i_row_sample),
      .final = function(x) rbindlist(x, use.names = T, fill = T)
    ) %do% {
      
      outpath_ref_parameter <- path(
        "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/aux-ref-parameter/cv-1day/",
        i_stn_name, i_month, i_rep 
      )
      dir_create(outpath_ref_parameter)
      
      i_gap <- i_row_sample[i_rep]
      mat_ii <- mat_i
      mat_ii[i_gap, i_stn_name] <- NA
      
      ii_stn <- which(colnames(mat_ii) == i_stn_name)
      
      l_fill <- fill_daily_meteo_gaps(df_meta = meta_i,
                                      mat_series = mat_ii,
                                      vec_dates = vec_dates,
                                      stns_to_fill = ii_stn,
                                      rows_to_fill = i_gap,
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
                                      save_ref_parameter = outpath_ref_parameter)
      
      
      
      data.table(i_rep = i_rep,
                 date_gap = vec_dates[i_gap],
                 value_true = mat_i[i_gap, i_stn_name],
                 value_fill = l_fill$mat_series_filled[i_gap])
      
      
    }
    
    dat_out_month[, month := i_month]
    
    }
  
  dat_out_stn[, Name := i_stn_name]
  
  
}


saveRDS(dat_out, 
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-01-1day.rds")

# EOF ---------------------------------------------------------------------


