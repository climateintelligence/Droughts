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
# library(patchwork)
library(cowplot)

library(forcats)
library(foreach)
library(flextable)
library(officer)

library(fs)
library(raster)



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



# time series
dat_daily_hnhs_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")
dat_daily_hnhs_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
dat_daily_hnhs <- rbind(dat_daily_hnhs_1, dat_daily_hnhs_2)

# time series summary
# dat_ts_hn <- dat_daily_hnhs[!is.na(HN), .(n_stn = length(unique(Name))), .(Provider, year(Date))]
dat_ts_hs <- dat_daily_hnhs[!is.na(HS), .(n_stn = length(unique(Name))), .(Provider, year(Date))]




cols <- setNames(scales::brewer_pal(palette = "Dark2")(3),
                 c("DEM", "used (regionalization)", "used (trend analysis)"))

# plot elev hist ----------------------------------------------------------

cols_hist <- c(available = "#CCCCCC",
               cols[2:3])


gg_hist <-
  dat_meta_hs_all %>% 
  ggplot(aes(Elevation))+
  geom_histogram(breaks = seq(0, 3300, by = 50), aes(fill = "available"))+
  geom_histogram(breaks = seq(0, 3300, by = 50),
                 data = dat_meta_hs_reg,
                 aes(fill = "used (regionalization)"))+
    geom_histogram(breaks = seq(0, 3300, by = 50),
                   data = dat_meta_hs_trend,
                   aes(fill = "used (trend analysis)"))+
  scale_fill_manual("", values = cols_hist)+
  theme_bw(14)+
  theme(legend.position = c(0.7, 0.6),
        legend.direction = "vertical",
        legend.title = element_blank(),
        legend.background = element_rect(colour = "grey"))+
  xlab("Elevation [m]")+
  ylab("Absolute number of HS stations")


# plot relative altitude, compared to DEM ---------------------------------------------


# dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")

sf_meta_reg <- st_as_sf(dat_meta_hs_reg,
                        coords = c("Longitude", "Latitude"),
                        crs = 4326)

# mapview(sf_meta)

# sf_meta[c(107, 471, 947, 640, 237, 1685,
#           1954, 1949, 329, 1943, 818, 1085,
#           1190, 1958, 1411, 1345, 1836,
#           1488, 1708, 606, 1010, 350, 
#           1283, 1936, 55, 519), ] %>% 
#   st_combine() %>%
#   st_cast("POLYGON") -> sf_man_hull

sf_meta_reg[c(106, 467, 941, 635, 234, 1767,
          1941, 1936, 325, 1930, 812, 1080,
          1184, 1945, 1404, 1340, 1824,
          1479, 1696, 601, 1005, 346,
          1278, 1923, 54, 515), ] %>%
  st_combine() %>%
  st_cast("POLYGON") -> sf_man_hull

# sf_meta_reg %>% 
#   st_union() %>% 
#   st_convex_hull() -> sf_man_hull

# mapview::mapview(list(sf_meta_reg, sf_man_hull))


# longterm subset
sf_meta_trend <- st_as_sf(dat_meta_hs_trend,
                          coords = c("Longitude", "Latitude"),
                          crs = 4326)

# ** get DEM (EU DEM v1.1) -----------------------------------------------------------------


# rr1 <- raster("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/EU_DEM_v1.1/DEM_mosaic_clip_wgs84_updated.tif")
rr1 <- raster("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/srtm30_alps.tif")
rr1
# plot(rr1)

rr2 <- mask(rr1, as(sf_man_hull, "Spatial"))

dat_dem <- data.table(vals = values(rr2))
dat_dem2 <- dat_dem[!is.na(vals)]
# summary(dat_dem2)

breaks <- seq(-50, 4800, by = 50)

dat_dem2[, elev_fct := cut(vals, breaks = breaks, dig.lab = 5)]
dat_dem2_hist <- dat_dem2[, .N, .(elev_fct)]


# ** plot comparison of hist -------------------------------------------------


dat_meta_hs_reg[, elev_fct := cut(Elevation, breaks = breaks, dig.lab = 5)]
dat_meta_hist_reg <- dat_meta_hs_reg[, .N, .(elev_fct)]

dat_meta_hs_trend[, elev_fct := cut(Elevation, breaks = breaks, dig.lab = 5)]
dat_meta_hist_trend <- dat_meta_hs_trend[, .N, .(elev_fct)]


dat_plot <- rbindlist(list(DEM = dat_dem2_hist,
                           "used (regionalization)" = dat_meta_hist_reg,
                           "used (trend analysis)" = dat_meta_hist_trend),
                      idcol = "ff")

dat_plot[, N_tot := sum(N), ff]
dat_plot[, perc := N / N_tot]

dat_plot[, elev_xmin := as.numeric(elev_fct)*50 - 100]
dat_plot[, elev_xmax := as.numeric(elev_fct)*50 - 50]
dat_plot[, elev_xmean := (elev_xmin+elev_xmax)/2]


gg_dem_hist <- dat_plot %>% 
  ggplot(aes(x = elev_xmean, y = perc, colour = ff, linetype = ff))+
  geom_line()+
  scale_color_manual(values = cols)+
  scale_x_continuous(breaks = 0:4*1000)+
  scale_y_continuous(labels = scales::percent)+
  theme_bw(14)+
  theme(legend.title = element_blank(),
        legend.position = c(0.7, 0.5),
        legend.background = element_rect(colour = "grey"))+
  xlab("Elevation [m]")+
  ylab("Relative frequency")

# gg_hist


# small map of polygon (as inset maybe?)

gg_in <- ggplot()+
  borders()+
  geom_sf(data = sf_meta_reg, size = 0.001, colour = "grey70")+
  geom_sf(data = sf_man_hull, alpha = 0.6, colour = "grey20")+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  # theme_bw()
  theme_void()+
  theme(plot.background = element_rect(fill = "white"))



gg_dem_comp <- ggdraw()+
  draw_plot(gg_dem_hist)+
  draw_plot(gg_in, x = 0.7, y = 0.7, width = 0.25, height = 0.25)





# plot ts HN and HS -------------------------------------------------------------------

cols_country <- c(scales::brewer_pal(type = "qual")(6), "black")
cols_country <- c(scales::hue_pal()(6), "black")
lty_country <- c(rep("solid", 6), "dashed")


dat_ts_hs[, country := substr(Provider, 1, 2)]
dat_plot_ts_hs <- rbind(use.names = T,
                        dat_ts_hs[, .(n_stn = sum(n_stn)), .(year, country)],
                        dat_ts_hs[, .(n_stn = sum(n_stn), country = "Total"), .(year)])


gg_ts_hs <-
dat_plot_ts_hs %>% 
  ggplot(aes(year, n_stn, colour = country, linetype = country))+
  geom_line()+
  scale_color_manual("", values = cols_country)+
  scale_linetype_manual("", values = lty_country)+
  # xlim(xlim_common)+
  theme_bw(14)+
  theme(legend.position = c(0.2, 0.6),
        legend.background = element_rect(colour = "grey"),
        legend.title = element_blank())+
  xlab("Year")+
  ylab("Number of HS stations")




# combine and save --------------------------------------------------------

gg_out_hist <- plot_grid(gg_hist, gg_dem_comp, 
                         nrow = 1, labels = c("(b)", "(c)"), label_fontface = "plain",
                         hjust = 0)
  
  
gg_out <- plot_grid(gg_ts_hs, gg_out_hist,
                    ncol = 1, labels = c("(a)", ""), label_fontface = "plain",
                    hjust = 0)


ggsave(gg_out,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 2.png",
       width = 10, height = 8)



# numbers and other -----------------------------------------------------------------

dat_ts_hn[year == min(year)]
dat_daily_hnhs[year(Date) == 1787]

dat_ts_hs[year == min(year)]
dat_daily_hnhs[year(Date) == 1879 & !is.na(HS)]

dat_plot[ff == "DEM" & elev_xmean > 3000, sum (perc)]
dat_plot[ff == "DEM" & elev_xmean > 2000 & elev_xmean < 3000, sum(perc)]
dat_plot[ff == "DEM" & elev_xmean > 1000 & elev_xmean < 2000, sum(perc)]
dat_plot[ff == "DEM" & elev_xmean < 1000, sum(perc)]

sf_man_hull
area_m2 <- st_area(sf_man_hull)
area_km2 <- area_m2 / 1000 / 1000
area_km2

dat_plot[, sum(N), .(ff)]
area_km2 / 2162
area_km2 / 854

