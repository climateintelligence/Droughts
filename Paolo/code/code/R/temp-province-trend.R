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



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
dat_1[, month := month(Date)]
dat_1[, season_year := year(Date)]
dat_1[month <= 8, season_year := season_year - 1L]

dat_hs_nobs <- dat_1[!is.na(HS) & 
                        season_year >= min_year & season_year <= max_year & 
                        month %in% month_sub, 
                      .(n_obs = .N, min_date = min(Date)),
                      .(Provider, Name, season_year, month)]
dat_hs_nobs[, n_obs_max := days_in_month(min_date)]
dat_hs_nobs[, perc_obs := n_obs / n_obs_max]

dat_hn_nobs <- dat_1[!is.na(HN) & 
                        season_year >= min_year & season_year <= max_year & 
                        month %in% month_sub, 
                      .(n_obs = .N, min_date = min(Date)),
                      .(Provider, Name, season_year, month)]
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
  i_dat_hs_nobs[, n_years := length(unique(season_year)), .(Provider, Name, month)]
  i_dat_hs_nobs[n_years == window_length, .(Provider, Name, season_year, month)] %>% 
    merge(dat_1, by = c("Provider", "Name", "season_year", "month")) -> i_dat_hs
  
  # agg and lm
  dat_hs_quant <- i_dat_hs[!is.na(HS),
                           .(hs_qvalue = c(mean(HS), quantile(HS, probs = q_probs)),
                             qprob = c("mean_hs", q_probs)),
                           .(Provider, Name, season_year, month)]
  dat_hs_quant[, year0 := season_year - min(season_year)]
  dat_hs_lm <- dat_hs_quant[,
                            broom::tidy(lm(hs_qvalue ~ year0)), 
                            .(Provider, Name, month, qprob)]
  
  # HN ------------------ #
  # subset data
  i_dat_hn_nobs <- dat_hn_nobs[season_year %in% l_windows[[i_window]] & perc_obs > 0.9]
  i_dat_hn_nobs[, n_years := length(unique(season_year)), .(Provider, Name, month)]
  i_dat_hn_nobs[n_years == window_length, .(Provider, Name, season_year, month)] %>% 
    merge(dat_1, by = c("Provider", "Name", "season_year", "month")) -> i_dat_hn
  
  # agg and lm
  dat_hn_quant <- i_dat_hn[!is.na(HN),
                           .(hn_qvalue = c(sum(HN), quantile(HN, probs = q_probs)),
                             qprob = c("sum_hn", q_probs)),
                           .(Provider, Name, season_year, month)]
  dat_hn_quant[, year0 := season_year - min(season_year)]
  dat_hn_lm <- dat_hn_quant[,
                            broom::tidy(lm(hn_qvalue ~ year0)), 
                            .(Provider, Name, month, qprob)]
  
  # return
  dat_hs_lm[, ":="(snow = "HS", mw_start_year = min(l_windows[[i_window]]))]
  dat_hn_lm[, ":="(snow = "HN", mw_start_year = min(l_windows[[i_window]]))]
  
  rbind(dat_hs_lm, dat_hn_lm)
}

saveRDS(dt_out, "data/test-trend-02.rds")

