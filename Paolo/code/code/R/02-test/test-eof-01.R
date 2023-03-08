# test eof

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


# eof monthly -------------------------------------------------------------

dat_hs[, month := month(date)]
dat_hs[, season_year := year(date)]
dat_hs[month <= 8, season_year := season_year - 1L]

dat_hs_nobs <- dat_hs[!is.na(hs) & 
                        season_year >= 1981 & season_year <= 2000 & 
                        month %in% c(12,1:3), 
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
# remove na: check rather in space or in time
dat_hs_sub_month_wide[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix -> mat_hs
dim(mat_hs)
apply(mat_hs, 1, function(x) sum(!is.na(x))) %>% table # stn
apply(mat_hs, 2, function(x) sum(!is.na(x))) %>% table # year-month

dat_hs_sub_month_wide[, .N, provider]
dat_hs_sub_month_wide <- na.omit(dat_hs_sub_month_wide)
dat_hs_sub_month_wide[, .N, provider]
dat_hs_sub_month_wide_stn <- dat_hs_sub_month_wide[, .(provider, stn_name)]
dat_hs_sub_month_wide[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix -> mat_hs


pc1 <- prcomp(mat_hs, scale. = T, rank. = 20, retx = T)
screeplot(pc1)
str(pc1)
pc2 <- prcomp(t(mat_hs), scale. = T, rank. = 20, retx = T)
screeplot(pc2)
str(pc2)

dat_pca <- cbind(dat_hs_sub_month_wide_stn,
                 predict(pc1))
dat_pca <- cbind(dat_hs_sub_month_wide_stn,
                 pc2$rotation)
dat_pca2 <- merge(dat_pca, 
                  dat_meta[, .(provider, stn_name = Name, 
                               lon = Longitude, lat = Latitude, elev = Elevation)])
dat_pca2[, country := substr(provider, 1, 2)]
dat_pca2 %>% 
  melt(id.vars = c("provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))

dat_pca2 %>% 
  melt(id.vars = c("country", "provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(elev, value))+
  geom_point()+
  facet_grid(variable ~ country, scales = "free_y")


# pca daily ---------------------------------------------------------------------

dat_hs_sub <- dat_hs[date >= "1981-10-01" & date <= "2000-09-30" & month(date) %in% c(12,1:3)]
# dat_hs_sub[, .N, .(date, provider, stn_name)] %>% 
#   .[N > 1]

dat_hs_sub_wide <- dcast(dat_hs_sub, provider + stn_name ~ date, value.var = "hs")
dat_hs_sub_wide[, .N, provider]
dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
dat_hs_sub_wide2[, .N, provider]

dat_hs_sub_wide_stn <- dat_hs_sub_wide2[, .(provider, stn_name)]
dat_hs_sub_wide2[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix -> mat_hs

pc1d <- prcomp(mat_hs, scale. = T, rank. = 20, retx = T)
screeplot(pc1d)
str(pc1d)
pc2d <- prcomp(t(mat_hs), scale. = T, rank. = 20, retx = T)
screeplot(pc2d)
str(pc2d)

# dat_pca <- cbind(dat_hs_sub_month_wide_stn,
                 # predict(pc1))
dat_pcad <- cbind(dat_hs_sub_wide_stn,
                 pc2d$rotation)
dat_pcad2 <- merge(dat_pcad, 
                  dat_meta[, .(provider, stn_name = Name, 
                               lon = Longitude, lat = Latitude, elev = Elevation)])
dat_pcad2[, country := substr(provider, 1, 2)]
dat_pcad2 %>% 
  melt(id.vars = c("provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))

dat_pcad2 %>% 
  melt(id.vars = c("country", "provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(elev, value))+
  geom_point()+
  facet_grid(variable ~ country, scales = "free_y")


# check PCA time component
predict(pc2d) %>% 
  data.table(date = ymd(rownames(.)), .) %>% 
  melt(id.vars = "date", variable.factor = F) %>% 
  .[, .(pca_mean = mean(value)), .(year(date), month(date), variable)] -> dt_pc_time

dt_pc_time[variable %in% paste0("PC", 1:6)] %>% 
  ggplot(aes(year, pca_mean))+
  geom_point()+
  facet_grid(variable ~ month, scales = "free_y")
# -> not conclusive

# pca daily (subsample DE+AT) ---------------------------------------------------------------------

dat_hs_sub <- dat_hs[date >= "1981-10-01" & date <= "2000-09-30" & month(date) %in% c(12,1:3)]
# dat_hs_sub[, .N, .(date, provider, stn_name)] %>% 
#   .[N > 1]

dat_hs_sub_wide <- dcast(dat_hs_sub, provider + stn_name ~ date, value.var = "hs")
dat_hs_sub_wide[, .N, provider]
dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
dat_hs_sub_wide2[, .N, provider]

set.seed(1234)
stn_sub_AT <- dat_hs_sub_wide2[provider == "AT_HZB", sample(stn_name, 150)]
stn_sub_DE <- dat_hs_sub_wide2[provider == "DE_DWD", sample(stn_name, 150)]

dat_hs_sub_wide2 <- dat_hs_sub_wide2[!(provider %in% c("AT_HZB", "DE_DWD")) | 
                   (stn_name %in% c(stn_sub_AT, stn_sub_DE))]

dat_hs_sub_wide_stn <- dat_hs_sub_wide2[, .(provider, stn_name)]
dat_hs_sub_wide2[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix -> mat_hs

pc1d <- prcomp(mat_hs, scale. = T, rank. = 20, retx = T)
screeplot(pc1d)
str(pc1d)
pc2d <- prcomp(t(mat_hs), scale. = T, rank. = 20, retx = T)
screeplot(pc2d)
str(pc2d)

# dat_pca <- cbind(dat_hs_sub_month_wide_stn,
# predict(pc1))
dat_pcad <- cbind(dat_hs_sub_wide_stn,
                  pc2d$rotation)
dat_pcad2 <- merge(dat_pcad, 
                   dat_meta[, .(provider, stn_name = Name, 
                                lon = Longitude, lat = Latitude, elev = Elevation)])
dat_pcad2[, country := substr(provider, 1, 2)]
dat_pcad2 %>% 
  melt(id.vars = c("provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))

dat_pcad2 %>% 
  melt(id.vars = c("country", "provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(elev, value))+
  geom_point()+
  facet_grid(variable ~ country, scales = "free_y")



# check influence of scaling ----------------------------------------------

dat_hs_sub <- dat_hs[date >= "1981-10-01" & date <= "2000-09-30" & month(date) %in% c(12,1:3)]
# dat_hs_sub[, .N, .(date, provider, stn_name)] %>% 
#   .[N > 1]

dat_hs_sub_wide <- dcast(dat_hs_sub, provider + stn_name ~ date, value.var = "hs")
dat_hs_sub_wide[, .N, provider]
dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
dat_hs_sub_wide2[, .N, provider]

dat_hs_sub_wide_stn <- dat_hs_sub_wide2[, .(provider, stn_name)]
dat_hs_sub_wide2[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix -> mat_hs

mat_hs_unscaled <- t(mat_hs)

# by row
mat_center <- sweep(mat_hs_unscaled, 1, rowMeans(mat_hs_unscaled), "-")
mat_hs_scaled_1time <- sweep(mat_center, 1, apply(mat_center, 1, sd), "/")

# by col
mat_center <- sweep(mat_hs_unscaled, 2, colMeans(mat_hs_unscaled), "-")
mat_hs_scaled_2stn <- sweep(mat_center, 2, apply(mat_center, 2, sd), "/")
  
pc_unscaled <- prcomp(mat_hs_unscaled, center = T, scale. = F, rank. = 20, retx = T)
pc_scaled_1time <- prcomp(mat_hs_scaled_1time, center = T, scale. = F, rank. = 20, retx = T)
pc_scaled_2stn <- prcomp(mat_hs_scaled_2stn, center = T, scale. = F, rank. = 20, retx = T)
screeplot(pc_unscaled)
screeplot(pc_scaled_1time)
screeplot(pc_scaled_2stn)

dat_pcad <- rbind(
  cbind(dat_hs_sub_wide_stn, pc_unscaled$rotation, pca = "unscaled"),
  cbind(dat_hs_sub_wide_stn, pc_scaled_1time$rotation, pca = "scale_1time"),
  cbind(dat_hs_sub_wide_stn, pc_scaled_2stn$rotation, pca = "scale_2stn"))

dat_pcad2 <- merge(dat_pcad, 
                   dat_meta[, .(provider, stn_name = Name, 
                                lon = Longitude, lat = Latitude, elev = Elevation)])
dat_pcad2[, country := substr(provider, 1, 2)]
dat_pcad2 %>% 
  melt(id.vars = c("provider", "stn_name", "lon", "lat", "elev", "pca"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_grid(pca~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))





# no rot ------------------------------------------------------------------


pc1d_norot <- prcomp(mat_hs, scale. = T, rank. = 20, retx = F)
screeplot(pc1d_norot)
str(pc1d_norot)
pc2d_norot <- prcomp(t(mat_hs), scale. = T, rank. = 20, retx = F)
screeplot(pc2d_norot)
str(pc2d_norot)

# dat_pca <- cbind(dat_hs_sub_month_wide_stn,
# predict(pc1))
dat_pcad_norot <- cbind(dat_hs_sub_wide_stn,
                        pc2d_norot$rotation)
dat_pcad_norot2 <- merge(dat_pcad_norot, 
                         dat_meta[, .(provider, stn_name = Name, 
                                      lon = Longitude, lat = Latitude, elev = Elevation)])
dat_pcad_norot2[, country := substr(provider, 1, 2)]
dat_pcad_norot2 %>% 
  melt(id.vars = c("provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))

dat_pcad_norot2 %>% 
  melt(id.vars = c("country", "provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) %>% 
  ggplot(aes(elev, value))+
  geom_point()+
  facet_grid(variable ~ country, scales = "free_y")


# cluster on pca ----------------------------------------------------------

mat_to_cluster <- pc2d$rotation

dat_pam <- foreach(
  kk = 2:10,
  .final = rbindlist
) %do% {
  pm <- pam(mat_to_cluster, kk) 
  cbind(dat_hs_sub_wide_stn, clust = pm$clustering, k = kk)
}

dat_pam2 <- merge(dat_pam, 
                  dat_meta[, .(provider, stn_name = Name, 
                               lon = Longitude, lat = Latitude, elev = Elevation)])

dat_pam2[k <= 5] %>% 
  ggplot(aes(lon, lat, colour = factor(clust)))+
  geom_point()+
  facet_wrap(~k)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))
