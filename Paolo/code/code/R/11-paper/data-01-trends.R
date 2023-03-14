# calculate 30 year trends based on monthly gapfilled data
# moving window starting 1960 until end



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)


# settings ----------------------------------------------------------------

month_sub <- c(11:12, 1:5)
window_length <- 30
l_windows <- lapply(1961:1990, function(x) x + seq(0, window_length - 1))
# q_probs <- c(0.1, 0.3, 0.5, 0.7, 0.9)

min_year <- min(unlist(l_windows))
max_year <- max(unlist(l_windows))

# prep data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")

dat_hs <- mitmatmisc::add_hydro_year(dat_hs)


# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]

# get trend data (moving window) hydro_year --------------------------------------------------------------------

mitmatmisc::init_parallel_ubuntu(6)

dt_out <- foreach(
  i_window = seq_along(l_windows),
  .final = rbindlist
) %dopar% {
  
  # HS ------------------ #
  # subset data
  i_dat_hs <- dat_hs[hydro_year %in% l_windows[[i_window]] & month %in% month_sub & !is.na(HS)]
  i_dat_hs[, nn_years := .N, .(Name, month)]
  i_dat_hs <- i_dat_hs[nn_years == window_length]
  
  
  # lm
  i_dat_hs[, year0 := hydro_year - min(hydro_year)]
  dat_hs_lm <- i_dat_hs[,
                        broom::tidy(lm(HS ~ year0)), 
                        .(Name, month)]
  
  
  
  # return
  dat_hs_lm[, ":="(snow = "HS", 
                   mw_year_start = min(l_windows[[i_window]]),
                   mw_year_end = max(l_windows[[i_window]]))]
  
  dat_hs_lm
}

saveRDS(dt_out, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-01-mw-hydro-year.rds")



# get trend data (moving window) calendar year --------------------------------------------------------------------

mitmatmisc::init_parallel_ubuntu(6)

dt_out <- foreach(
  i_window = seq_along(l_windows),
  .final = rbindlist
) %dopar% {
  
  # HS ------------------ #
  # subset data
  i_dat_hs <- dat_hs[year %in% l_windows[[i_window]] & month %in% month_sub & !is.na(HS)]
  i_dat_hs[, nn_years := .N, .(Name, month)]
  i_dat_hs <- i_dat_hs[nn_years == window_length]
  
  
  # lm
  i_dat_hs[, year0 := year - min(year)]
  dat_hs_lm <- i_dat_hs[,
                        broom::tidy(lm(HS ~ year0)), 
                        .(Name, month)]
  
  
  
  # return
  dat_hs_lm[, ":="(snow = "HS", 
                   mw_year_start = min(l_windows[[i_window]]),
                   mw_year_end = max(l_windows[[i_window]]))]
  
  dat_hs_lm
}

saveRDS(dt_out, "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-02-mw-calendar-year.rds")


# get trend data (whole 1971-2019) period ---------------------------------

dat_hs2 <- dat_hs[year <= 2019 & month %in% month_sub & !is.na(HS)]
dat_hs2 <- dat_hs[year >= 1971 & year <= 2019 & month %in% month_sub & !is.na(HS)]
with(dat_hs2, table(year, month))

dat_hs2[, nn_year := .N, .(Name, month)]
dat_hs_full <- dat_hs2[nn_year == max(nn_year)]

dat_meta[Name %in% dat_hs_full$Name, .N, Provider]


# lm
dat_hs_full[, year0 := year - min(year)]
dat_hs_full_lm <- dat_hs_full[,
                              broom::tidy(lm(HS ~ year0)), 
                              .(Name, month)]


saveRDS(dat_hs_full_lm,
        "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-03-full_1971-2019-calyear.rds")

