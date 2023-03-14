# test out trends of the data



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)


# settings ----------------------------------------------------------------

month_sub <- c(11:12, 1:5)
window_length <- 20
l_windows <- lapply(1975:1995, function(x) x + seq(0, window_length - 1))
q_probs <- c(0.1, 0.3, 0.5, 0.7, 0.9)

min_year <- min(unlist(l_windows))
max_year <- max(unlist(l_windows))

# prep data ---------------------------------------------------------------

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")

dat_hs[, month := month(date)]
dat_hn[, month := month(date)]

dat_hs[, season_year := year(date)]
dat_hn[, season_year := year(date)]

dat_hs[month <= 8, season_year := season_year - 1L]
dat_hn[month <= 8, season_year := season_year - 1L]


dat_hs_nobs <- dat_hs[!is.na(hs) & 
                        season_year >= min_year & season_year <= max_year & 
                        month %in% month_sub, 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hs_nobs[, n_obs_max := days_in_month(min_date)]
dat_hs_nobs[, perc_obs := n_obs / n_obs_max]

dat_hn_nobs <- dat_hn[!is.na(hn) & 
                        season_year >= min_year & season_year <= max_year & 
                        month %in% month_sub, 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hn_nobs[, n_obs_max := days_in_month(min_date)]
dat_hn_nobs[, perc_obs := n_obs / n_obs_max]

# get trend data --------------------------------------------------------------------

mitmatmisc::init_parallel_ubuntu()

dt_out <- foreach(
  i_window = seq_along(l_windows),
  .final = rbindlist
) %dopar% {
  
  # HS ------------------ #
  # subset data
  i_dat_hs_nobs <- dat_hs_nobs[season_year %in% l_windows[[i_window]] & perc_obs > 0.9]
  i_dat_hs_nobs[, n_years := length(unique(season_year)), .(provider, stn_name, month)]
  i_dat_hs_nobs[n_years == window_length, .(provider, stn_name, season_year, month)] %>% 
    merge(dat_hs) -> i_dat_hs
  
  # agg and lm
  dat_hs_quant <- i_dat_hs[!is.na(hs),
                           .(hs_qvalue = c(mean(hs), quantile(hs, probs = q_probs)),
                             qprob = c("mean_hs", q_probs)),
                           .(provider, stn_name, season_year, month)]
  dat_hs_quant[, year0 := season_year - min(season_year)]
  dat_hs_lm <- dat_hs_quant[,
                            broom::tidy(lm(hs_qvalue ~ year0)), 
                            .(provider, stn_name, month, qprob)]
  
  # HN ------------------ #
  # subset data
  i_dat_hn_nobs <- dat_hn_nobs[season_year %in% l_windows[[i_window]] & perc_obs > 0.9]
  i_dat_hn_nobs[, n_years := length(unique(season_year)), .(provider, stn_name, month)]
  i_dat_hn_nobs[n_years == window_length, .(provider, stn_name, season_year, month)] %>% 
    merge(dat_hn) -> i_dat_hn
  
  # agg and lm
  dat_hn_quant <- i_dat_hn[!is.na(hn),
                           .(hn_qvalue = c(sum(hn), quantile(hn, probs = q_probs)),
                             qprob = c("sum_hn", q_probs)),
                           .(provider, stn_name, season_year, month)]
  dat_hn_quant[, year0 := season_year - min(season_year)]
  dat_hn_lm <- dat_hn_quant[,
                            broom::tidy(lm(hn_qvalue ~ year0)), 
                            .(provider, stn_name, month, qprob)]
  
  # return
  dat_hs_lm[, ":="(snow = "HS", mw_start_year = min(l_windows[[i_window]]))]
  dat_hn_lm[, ":="(snow = "HN", mw_start_year = min(l_windows[[i_window]]))]
  
  rbind(dat_hs_lm, dat_hn_lm)
}

saveRDS(dt_out, "data/test-trend-01.rds")

