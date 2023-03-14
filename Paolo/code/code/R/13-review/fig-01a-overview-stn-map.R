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

# all
dat_meta_hs_all <- unique(rbind(
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_wide_HS.rds"),
  readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_wide_HS.rds")
))


# region
dat_meta_hs_reg <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")

# trend
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/trends-01-full_1971-2019-calyear.rda")
dat_meta_hs_trend <- dat_meta_hs_all[Name %in% dat_hs_full_lm$Name]


# as sf
# sf_meta_hs_all <- st_as_sf(dat_meta_hs_all,
#                            coords = c("Longitude", "Latitude"),
#                            crs = 4326)
# sf_meta_hs_reg <- st_as_sf(dat_meta_hs_reg,
#                            coords = c("Longitude", "Latitude"),
#                            crs = 4326)
# sf_meta_hs_trend <- st_as_sf(dat_meta_hs_trend,
#                            coords = c("Longitude", "Latitude"),
#                            crs = 4326)
# histalp regions
sf_histalp <- read_sf("data/shp-histalp-zone")
sf_histalp2 <- dplyr::mutate(sf_histalp,
                             id_fct = fct_recode(factor(Id), 
                                                 "SW" = "0", "NW" = "1", "NE" = "2", "SE" = "3"))


# extend histalp ----------------------------------------------------------

# so only inner borders are shown
# sf_histalp2_edit <- mapedit::editFeatures(sf_histalp2)
# saveRDS(sf_histalp2_edit, file = "data/sf_histalp_extended.rds")

sf_histalp2_edit <- readRDS("data/sf_histalp_extended.rds")
sf_histalp2_edit <- sf_histalp2_edit %>% dplyr::filter(id_fct %in% c("SW","NE"))

# plot --------------------------------------------------------------------


# cols for used stations (same as for Fig2)

cols <- setNames(scales::brewer_pal(palette = "Dark2")(3),
                 c("DEM", "used (regionalization)", "used (trend analysis)"))



# ** DEM ------------------------------------------------------------------


rr1_stars_alps <- read_stars("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/srtm30_alps.tif")

gg_dem <-
  ggplot()+
  geom_stars(data = rr1_stars_alps, downsample = 0)+
  borders()+
  geom_sf(data = sf_histalp2_edit, fill = NA, colour = "black", linetype = "dashed")+
  scale_fill_scico("m.a.s.l.", palette = "oleron", rescaler = scales::rescale_mid, na.value = NA)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_hs_all$Longitude), ylim = range(dat_meta_hs_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.7, 0.1),
        legend.direction = "horizontal",
        legend.key.width = unit(1.3, "cm"),
        legend.background = element_rect(colour = "grey"),
        panel.grid.minor = element_blank())+
  xlab(NULL)+ylab(NULL)


# ** maps ------------------------------------------------------------------

gg_map1 <-
dat_meta_hs_all %>% 
  ggplot(aes(Longitude, Latitude))+
  geom_bin2d(binwidth = c(0.5, 0.25))+
  borders()+
  geom_sf(data = sf_histalp2_edit, inherit.aes = F,
          fill = NA, colour = "black", linetype = "dashed")+
  geom_point(aes(shape = "available"), size = 0.2)+
  scale_shape_manual("", values = 19)+
  scale_fill_continuous("#", low = grey(0.9), high = grey(0.4))+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_hs_all$Longitude), ylim = range(dat_meta_hs_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.8, 0.2),
        legend.direction = "horizontal",
        legend.background = element_rect(colour = "grey"),
        panel.grid.minor = element_blank())



gg_map2 <-
  dat_meta_hs_reg %>% 
  ggplot(aes(Longitude, Latitude))+
  geom_bin2d(binwidth = c(0.5, 0.25))+
  borders()+
  geom_sf(data = sf_histalp2_edit, inherit.aes = F,
          fill = NA, colour = "black", linetype = "dashed")+
  geom_point(aes(colour = "used (regionalization)"), size = 0.2)+
  scale_colour_manual("", values = cols)+
  scale_fill_continuous("#", low = alpha(cols[2], 0.1), high = alpha(cols[2], 0.6))+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_hs_all$Longitude), ylim = range(dat_meta_hs_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.8, 0.2),
        legend.direction = "horizontal",
        legend.background = element_rect(colour = "grey"),
        panel.grid.minor = element_blank())


gg_map3 <-
  dat_meta_hs_trend %>% 
  ggplot(aes(Longitude, Latitude))+
    geom_bin2d(binwidth = c(0.5, 0.25))+
  borders()+
  geom_sf(data = sf_histalp2_edit, inherit.aes = F,
          fill = NA, colour = "black", linetype = "dashed")+
  geom_point(aes(colour = "used (trend analysis)"), size = 0.2)+
  scale_colour_manual("", values = cols)+
  scale_fill_continuous("#", low = alpha(cols[3], 0.1), high = alpha(cols[3], 0.6))+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_hs_all$Longitude), ylim = range(dat_meta_hs_all$Latitude))+
  theme_bw(16)+
  theme(legend.position = c(0.8, 0.2),
        legend.direction = "horizontal",
        legend.background = element_rect(colour = "grey"),
        panel.grid.minor = element_blank())





# combine and save --------------------------------------------------------

gg_out_1 <- gg_dem + gg_map1 + gg_map2 + gg_map3 + 
  plot_annotation(tag_levels = "a", 
                  tag_suffix = ")",
                  tag_prefix = "(")

ggsave(gg_out_1,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 1.png",
       width = 15, height = 10)


