# try out a gam of trends

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(mgcv)

# prep data ---------------------------------------------------------------

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
# dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")

dat_hs[, month := month(date)]
# dat_hn[, month := month(date)]

dat_hs[, season_year := year(date)]
# dat_hn[, season_year := year(date)]

dat_hs[month <= 8, season_year := season_year - 1L]
# dat_hn[month <= 8, season_year := season_year - 1L]

# subset to some data --------------------------------------------------------------------


# hs 
dat_hs_nobs <- dat_hs[!is.na(hs), 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hs_nobs[, n_obs_max := days_in_month(min_date)]
dat_hs_nobs[, perc_obs := n_obs / n_obs_max]

dat_hs_nobs[perc_obs > 0.9 & month == 1, season_year] %>% qplot(binwidth = 1)
dat1 <- dat_hs_nobs[perc_obs > 0.9 & month == 1 & season_year >= 1981 & season_year <= 2000]
dat1[, n_years := length(unique(season_year)), .(provider, stn_name, month)]
table(dat1$n_years)

dat1[n_years >= 20, .(provider, stn_name)] %>% 
  unique %>% 
  .[, .N, provider]

dat1[n_years >= 20, stn_name] %>% unique %>% sort -> stn_sub

# estimate lm trends (& quick overview) ------------------------------------------------------------



dat_meanhs <- dat_hs[stn_name %in% stn_sub & month == 1 & season_year >= 1981 & season_year <= 2000,
                     .(mean_hs = mean(hs)),
                     .(provider, stn_name, season_year, month)]
dat_meanhs[, year0 := season_year - min(season_year)]
dat_lm <- dat_meanhs[, broom::tidy(lm(mean_hs ~ year0)), .(provider, stn_name, month)]

dat_lm[term == "year0" & !is.na(p.value), estimate] %>% qplot
dat_lm[term == "year0" & !is.na(p.value), p.value] %>% qplot(binwidth = 0.01)

dat_lm2 <- dat_lm[!is.na(p.value)] %>%
  merge(dat_meta[, .(provider, stn_name = Name, lon = Longitude, lat = Latitude, elev = Elevation)])

dat_lm2 %>% 
  ggplot(aes(elev, estimate, colour = provider))+
  geom_point()+
  facet_grid(term~provider, scales = "free_y")


dat_lm2[term == "year0"] %>% 
  ggplot(aes(lon, lat, colour = estimate))+
  geom_point()+
  scale_color_gradient2()+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))


dat_lm2[term == "(Intercept)"] %>% 
  ggplot(aes(lon, lat, colour = estimate))+
  geom_point()+
  scale_color_viridis_c(direction = -1)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))


dat_lm2[term == "year0"] %>% 
  ggplot(aes(estimate))+
  geom_vline(xintercept = 0)+
  geom_histogram(binwidth = 0.1)

dat_lm2[, country := substr(provider, 1, 2)]
dat_lm2[term == "year0"] %>% 
  ggplot(aes(elev, estimate))+
  geom_point()+
  facet_wrap(~country)
  

# model intercepts using gams -------------------------------------------------

dat_intercept <- dat_lm2[term == "(Intercept)" & estimate > 0] # remove the single negative value

# need to normalize data because of heteroscedascity 
library(bestNormalize)
yj_intercept <- yeojohnson(dat_intercept$estimate)
dat_intercept[, estimate_yj := predict(yj_intercept)]

# need to scale lat lon and elev for gam smooths
sc_lon <- sc_get_coef(dat_intercept$lon)
sc_lat <- sc_get_coef(dat_intercept$lat)
sc_elev <- sc_get_coef(dat_intercept$elev)

dat_intercept[, ":="(lon_scaled = sc_apply_scaling(lon, sc_lon),
                     lat_scaled = sc_apply_scaling(lat, sc_lat),
                     elev_scaled = sc_apply_scaling(elev, sc_elev))]


# ** create a predict field --------------------------------------------------
library(modelr)
ll_width <- 0.25
elev_width <- 500

dat_intercept %>% 
  data_grid(lon = seq_range(lon, by = ll_width),
            lat = seq_range(lat, by = ll_width),
            elev = seq_range(elev, by = elev_width)) %>% 
  data.table -> dat_predict

dat_predict[, ":="(lon_scaled = sc_apply_scaling(lon, sc_lon),
                   lat_scaled = sc_apply_scaling(lat, sc_lat),
                   elev_scaled = sc_apply_scaling(elev, sc_elev))]

# remove values outside of points
# sf_intercept <- st_as_sf(dat_intercept, coords = c("lon", "lat"), crs = st_crs("+proj=longlat"))
# st_intercept <- st_rasterize(sf_intercept,
#                              st_as_stars(st_bbox(sf_intercept), dx = ll_width, dy = ll_width))
# st_intercept %>% plot

dat_predict[, .(lon, lat)] %>% unique -> dat_predict_ll
for(i in 1:nrow(dat_predict_ll)){
  
  i_lon <- dat_predict_ll[i, lon]
  i_lat <- dat_predict_ll[i, lat]
  x_lon <- dat_intercept$lon
  x_lat <- dat_intercept$lat
  
  if(any({x_lon %between% (i_lon + c(-ll_width, + ll_width))} &
         {x_lat %between% (i_lat + c(-ll_width, + ll_width))})){
    out <- "yes"  
  } else {
    out <- "no"
  }
    
  dat_predict_ll[i, has_points := out]
  
}

dat_predict <-  merge(dat_predict_ll[has_points == "yes"],
                      dat_predict, 
                      by = c("lon", "lat"))

# ** gam sequence (yj) -----------------------------------------------------------------

# fix k
kk <- 8 #5-7 looks ok, not much gained by increasing

gm1 <- gam(estimate_yj ~ s(elev_scaled, k = kk), data = dat_intercept)
gm2 <- gam(estimate_yj ~ s(lon_scaled, lat_scaled, k = kk*kk), data = dat_intercept)
gm3 <- gam(estimate_yj ~ s(lon_scaled, lat_scaled, k = kk*kk) + s(elev_scaled, k = kk), data = dat_intercept)
gm4 <- gam(estimate_yj ~ s(lon_scaled, lat_scaled, elev_scaled, k = kk*kk*kk), data = dat_intercept)

AIC(gm1, gm2, gm3, gm4)

dat_predict[, gm1_fit := predict(gm1, newdata = dat_predict)]
dat_predict[, gm2_fit := predict(gm2, newdata = dat_predict)]
dat_predict[, gm3_fit := predict(gm3, newdata = dat_predict)]
dat_predict[, gm4_fit := predict(gm4, newdata = dat_predict)]

dat_plot <- melt(dat_predict, 
                 id.vars = c("lon", "lat", "elev"),
                 measure.vars = c("gm1_fit", "gm2_fit", "gm3_fit", "gm4_fit"),
                 value.name = "value_yj")
dat_plot[, value := predict(yj_intercept, newdata = value_yj, inverse = T)]

dat_plot %>% 
ggplot(aes(lon, lat, fill = value_yj))+
  geom_raster()+
  facet_grid(variable ~ elev)+
  scale_fill_viridis_c()


dat_plot[variable != "gm4_fit"] %>% 
  ggplot(aes(lon, lat, fill = value_yj))+
  geom_raster()+
  facet_grid(variable ~ elev)+
  scale_fill_viridis_c()


# ** gam sequence (raw) -----------------------------------------------------------------

# fix k
kk <- 6 #5-7 looks ok, not much gained by increasing

gm1 <- gam(estimate ~ s(elev_scaled, k = kk), data = dat_intercept)
gm2 <- gam(estimate ~ s(lon_scaled, lat_scaled, k = kk*kk), data = dat_intercept)
gm3 <- gam(estimate ~ s(lon_scaled, lat_scaled, k = kk*kk) + s(elev_scaled, k = kk), data = dat_intercept)
gm4 <- gam(estimate ~ s(lon_scaled, lat_scaled, elev_scaled, k = kk*kk*kk), data = dat_intercept)

AIC(gm1, gm2, gm3, gm4)

dat_predict[, gm1_fit := predict(gm1, newdata = dat_predict)]
dat_predict[, gm2_fit := predict(gm2, newdata = dat_predict)]
dat_predict[, gm3_fit := predict(gm3, newdata = dat_predict)]
dat_predict[, gm4_fit := predict(gm4, newdata = dat_predict)]

dat_plot <- melt(dat_predict, 
                 id.vars = c("lon", "lat", "elev"),
                 measure.vars = c("gm1_fit", "gm2_fit", "gm3_fit", "gm4_fit"),
                 value.name = "value")

dat_plot %>% 
  ggplot(aes(lon, lat, fill = value))+
  geom_raster()+
  facet_grid(variable ~ elev)+
  scale_fill_viridis_c()


dat_plot[variable != "gm4_fit"] %>% 
  ggplot(aes(lon, lat, fill = value))+
  geom_raster()+
  facet_grid(variable ~ elev)+
  scale_fill_viridis_c()



# ** remove elev, then latlon on resid (yj) ------------------------------------

# separate predict data for elev and latlon

dat_predict2_ll <- dat_predict_ll[has_points == "yes"]
dat_predict2_ll[, ":="(lon_scaled = sc_apply_scaling(lon, sc_lon),
                       lat_scaled = sc_apply_scaling(lat, sc_lat))]

dat_intercept %>% 
  data_grid(elev = seq_range(elev, by = 50)) %>% 
  data.table -> dat_predict2_elev
dat_predict2_elev[, elev_scaled := sc_apply_scaling(elev, sc_elev)]
                   
  
# model
kk <- 8 

gm1 <- gam(estimate_yj ~ s(elev_scaled, k = kk), data = dat_intercept)
dat_intercept[, gm_elev_resid := resid(gm1)]
dat_intercept[, gm_elev_fitted := fitted(gm1)]

dat_predict2_elev[, gm_fit_yj := predict(gm1, newdata = dat_predict2_elev)]
dat_predict2_elev[, gm_fit := predict(yj_intercept, newdata = gm_fit_yj, inverse = T)]

ggplot()+
  geom_line(data = dat_predict2_elev, aes(elev, gm_fit))+
  geom_point(data = dat_intercept, aes(elev, estimate))


gm2 <- gam(gm_elev_fitted ~ s(lon_scaled, lat_scaled, k = kk*kk), data = dat_intercept)

dat_predict2_ll[, gm_fit_yj := predict(gm2, newdata = dat_predict2_ll)]
dat_predict2_ll[, gm_fit := predict(yj_intercept, newdata = gm_fit_yj, inverse = T)]

ggplot()+
  geom_raster(data = dat_predict2_ll, aes(lon, lat, fill = gm_fit))+
  scale_fill_gradient2()



# ** remove elev, then latlon on resid (raw values) ------------------------------------

# separate predict data for elev and latlon

dat_predict2_ll <- dat_predict_ll[has_points == "yes"]
dat_predict2_ll[, ":="(lon_scaled = sc_apply_scaling(lon, sc_lon),
                       lat_scaled = sc_apply_scaling(lat, sc_lat))]

dat_intercept %>% 
  data_grid(elev = seq_range(elev, by = 50)) %>% 
  data.table -> dat_predict2_elev
dat_predict2_elev[, elev_scaled := sc_apply_scaling(elev, sc_elev)]



# model
kk <- 8 

gm1 <- gam(estimate ~ s(elev_scaled, k = kk), data = dat_intercept)
dat_intercept[, gm_elev_resid := resid(gm1)]
dat_intercept[, gm_elev_fitted := fitted(gm1)]

dat_predict2_elev[, gm_fit := predict(gm1, newdata = dat_predict2_elev)]


ggplot()+
  geom_line(data = dat_predict2_elev, aes(elev, gm_fit))+
  geom_point(data = dat_intercept, aes(elev, estimate))


gm2 <- gam(gm_elev_fitted ~ s(lon_scaled, lat_scaled, k = kk*kk), data = dat_intercept)

dat_predict2_ll[, gm_fit := predict(gm2, newdata = dat_predict2_ll)]

ggplot()+
  geom_raster(data = dat_predict2_ll, aes(lon, lat, fill = gm_fit))+
  scale_fill_gradient2()

