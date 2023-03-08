# figure overview 
# - map
# - altitude distribution
# - time with # (maybe divide by country)



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)
library(directlabels)
library(patchwork)

library(forcats)
library(foreach)
library(flextable)
library(officer)

library(fs)
library(stars)
library(sf)


# data --------------------------------------------------------------------

# hn paper
# load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/data-longterm-HN-HS.rda")
# stns_hn_used <- dat_meta_hn$Name
# dat_meta_hn_used <- copy(dat_meta_hn)


# hs paper
dat_meta_paper <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds") 
stns_hs_used <- dat_meta_paper$Name
dat_meta_hs_used <- copy(dat_meta_paper)

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
stns_hs_used <- stns_hs_used[stns_hs_used %in% stns_ok]
dat_meta_hs_used <- dat_meta_hs_used[Name %in% stns_ok]



# meta all
# dat_meta_hn <- unique(rbind(
#   readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HN.rds"),
#   readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HN.rds")
# ))

dat_meta_hs <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HS.rds")
))

dat_meta_all <- dat_meta_hs
# dat_meta_all <- unique(rbind(dat_meta_hn, dat_meta_hs))

# remove Giacomo's data (until no answer)
# dat_meta_all[, .N, Provider]
# dat_meta_all <- dat_meta_all[Provider != "IT_TN_GB"]
# dat_meta_hs <- dat_meta_hs[Provider != "IT_TN_GB"]
# dat_meta_hn <- dat_meta_hn[Provider != "IT_TN_GB"]


# plot --------------------------------------------------------------------

cols_2grey <- setNames(scales::grey_pal()(2),
                       c("used", "available"))



# ** DEM ------------------------------------------------------------------


sf_meta <- st_as_sf(dat_meta_all,
                    coords = c("Longitude", "Latitude"),
                    crs = 4326)

# rr1 <- raster("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/EU_DEM_v1.1/DEM_mosaic_clip_wgs84_updated.tif")
# rr1_stars <- read_stars("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/EU_DEM_v1.1/DEM_mosaic_clip_wgs84_updated.tif")

# rr1 <- raster("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/SRTM/TIF/europe_srtm30.tif")
# rr1_alps <- crop(rr1, as(sf_meta, "Spatial"))
# writeRaster(rr1_alps, file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/srtm30_alps.tif")

# rr1_stars <- read_stars("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/SRTM/TIF/europe_srtm30.tif")
# rr1_stars_alps <- st_as_stars(rr1_alps)

rr1_stars_alps <- read_stars("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/srtm30_alps.tif")

gg_dem <- 
  ggplot()+
  geom_stars(data = rr1_stars_alps, downsample = 0)+
  # coord_sf()+
  scale_fill_scico("m.a.s.l.", palette = "oleron", begin = 0.5, na.value = NA)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_all$Longitude), ylim = range(dat_meta_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.7, 0.1),
        legend.direction = "horizontal",
        legend.key.width = unit(1.3, "cm"),
        legend.background = element_rect(colour = "black"))+
  xlab(NULL)+ylab(NULL)


# ** map ------------------------------------------------------------------

dat_plot_map <- rbindlist(idcol = "ff", list(
  # "HN available" = dat_meta_hn,
  "HS available" = dat_meta_hs,
  # "HN used" = dat_meta_hn_used,
  "HS used" = dat_meta_hs_used
))

# dat_plot_map[, hnhs := substr(ff, 1, 2)]
dat_plot_map[, used := substr(ff, 4, 99)]


gg_map <-
  dat_plot_map %>% 
  ggplot(aes(Longitude, Latitude, shape = used))+
  borders()+
  # geom_point(shape = 1)+
  geom_point(colour = cols_2grey[1])+
  scale_shape_manual("", values = c(1, 19))+
  # scale_colour_manual("", values = cols_2grey)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_all$Longitude), ylim = range(dat_meta_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.8, 0.2),
        legend.direction = "horizontal",
        legend.background = element_rect(colour = "black"))









# combine and save --------------------------------------------------------

# gg_map + gg_elev + gg_ts_hn_dl + gg_ts_hs_dl + plot_layout(nrow = 2)
# (gg_map / gg_elev) | (gg_ts_hn_dl / gg_ts_hs_dl)

gg_out_1 <- gg_dem + gg_map + #gg_elev + 
  plot_annotation(tag_levels = "a", 
                  tag_suffix = ")",
                  tag_prefix = "(")

ggsave(gg_out_1,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure 1.png",
       width = 15, height = 5)


