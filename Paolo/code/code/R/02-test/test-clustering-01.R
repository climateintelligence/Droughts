# test clustering

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(cluster)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
# dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")





# pam daily ---------------------------------------------------------------------

# not working so good



dat_hs_sub <- dat_hs[date >= "1971-10-01" & date <= "2000-09-30" & month(date) %in% c(11,12,1:5)]
# dat_hs_sub[, .N, .(date, provider, stn_name)] %>% 
#   .[N > 1]

dat_hs_sub_wide <- dcast(dat_hs_sub, provider + stn_name ~ date, value.var = "hs")
dat_hs_sub_wide_stn <- dat_hs_sub_wide[, .(provider, stn_name)]
dat_hs_sub_wide[, provider := NULL]
dat_hs_sub_wide[, stn_name := NULL]


pm2 <- pam(dat_hs_sub_wide, 2, stand = T)
Sys.time()



# pam monthly -------------------------------------------------------------

dat_hs[, month := month(date)]
dat_hs[, season_year := year(date)]
dat_hs[month <= 8, season_year := season_year - 1L]

dat_hs_nobs <- dat_hs[!is.na(hs) & 
                        season_year >= 1981 & season_year <= 2000 & 
                        month %in% c(11,12,1:5), 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hs_nobs[, n_obs_max := days_in_month(min_date)]
dat_hs_nobs[, perc_obs := n_obs / n_obs_max]

dat_hs_nobs_sub <- dat_hs_nobs[perc_obs > 0.9]
dat_hs_nobs_sub[, n_years := length(unique(season_year)), .(provider, stn_name, month)]
dat_hs_nobs_sub[n_years == 20, .(provider, stn_name, season_year, month)] %>% 
  merge(dat_hs) -> dat_hs_sub

dat_hs_sub_month <- dat_hs_sub[, .(mean_hs = mean(hs)), .(provider, stn_name, season_year, month)]

dat_hs_sub_month_wide <- dcast(dat_hs_sub_month, 
                               provider + stn_name ~ season_year + month,
                               value.var = "mean_hs")
dat_hs_sub_month_wide <- na.omit(dat_hs_sub_month_wide)
dat_hs_sub_month_wide_stn <- dat_hs_sub_month_wide[, .(provider, stn_name)]
dat_hs_sub_month_wide[, provider := NULL]
dat_hs_sub_month_wide[, stn_name := NULL]

dat_pam <- foreach(
  kk = 2:10,
  .final = rbindlist
) %do% {
  pm <- pam(dat_hs_sub_month_wide, kk) 
  cbind(dat_hs_sub_month_wide_stn, clust = pm$clustering, k = kk)
}

dat_pam2 <- merge(dat_pam, 
                  dat_meta[, .(provider, stn_name = Name, 
                               lon = Longitude, lat = Latitude, elev = Elevation)])

dat_pam2[k == 2] %>% 
  ggplot(aes(lon, lat, colour = factor(clust)))+
  geom_point()+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))
