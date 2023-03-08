# create convex hull of points 
# and check DEM


library(data.table)
library(dplyr)
library(magrittr)
library(sf)
library(mapview)
library(raster)
library(cowplot)



dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/meta-with-cluster-01.rds")

sf_meta <- st_as_sf(dat_meta_cluster,
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

sf_meta[c(106, 467, 941, 635, 234, 1767,
          1941, 1936, 325, 1930, 812, 1080,
          1184, 1945, 1404, 1340, 1824,
          1479, 1696, 601, 1005, 346,
          1278, 1923, 54, 515), ] %>%
  st_combine() %>%
  st_cast("POLYGON") -> sf_man_hull

# mapview(list(sf_meta, sf_man_hull))


# longterm subset
dat_trends <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-03-full_1971-2019-calyear.rds")
dat_meta_long <- dat_meta_cluster[Name %in% dat_trends$Name]
sf_meta_long <- filter(sf_meta, Name %in% dat_trends$Name)




# get DEM (EU DEM v1.1) -----------------------------------------------------------------


# rr1 <- raster("/mnt/CEPH_BASEDATA/GIS/EUROPE/ELEVATION/EU_DEM_v1.1/DEM_mosaic_clip_wgs84_updated.tif")
rr1 <- raster("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/srtm30_alps.tif")
rr1
# plot(rr1)

rr2 <- mask(rr1, as(sf_man_hull, "Spatial"))

dat_dem <- data.table(vals = values(rr2))
dat_dem2 <- dat_dem[!is.na(vals)]
# summary(dat_dem2)

breaks <- seq(-50, 4800, by = 50)

dat_dem2[, elev_fct := cut(vals, breaks = breaks, dig.lab = 5)]
dat_dem2_hist <- dat_dem2[, .N, .(elev_fct)]


# plot comparison of hist -------------------------------------------------


dat_meta_cluster[, elev_fct := cut(Elevation, breaks = breaks, dig.lab = 5)]
dat_meta_hist <- dat_meta_cluster[, .N, .(elev_fct)]

dat_meta_long[, elev_fct := cut(Elevation, breaks = breaks, dig.lab = 5)]
dat_meta_long_hist <- dat_meta_long[, .N, .(elev_fct)]


dat_plot <- rbindlist(list(DEM = dat_dem2_hist,
                           Stations = dat_meta_hist,
                           "Stations 1971-2019" = dat_meta_long_hist),
                      idcol = "ff")

dat_plot[, N_tot := sum(N), ff]
dat_plot[, perc := N / N_tot]

dat_plot[, elev_xmin := as.numeric(elev_fct)*50 - 100]
dat_plot[, elev_xmax := as.numeric(elev_fct)*50 - 50]
dat_plot[, elev_xmean := (elev_xmin+elev_xmax)/2]


gg_hist <- dat_plot %>% 
  ggplot(aes(x = elev_xmean, y = perc, colour = ff, linetype = ff))+
  geom_line()+
  scale_color_brewer(palette = "Dark2")+
  scale_x_continuous(breaks = 0:4*1000)+
  scale_y_continuous(labels = scales::percent)+
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.8, 0.2),
        legend.background = element_rect(colour = "grey"))+
  xlab("Elevation [m]")+
  ylab("Frequency")

# gg_hist


# small map of polygon (as inset maybe?)

gg_in <- ggplot()+
  borders()+
  geom_sf(data = sf_man_hull, alpha = 0.8)+
  geom_sf(data = sf_meta, size = 0.1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  # theme_bw()
  theme_void()+
  theme(plot.background = element_rect(fill = "white"))



gg_out <- ggdraw()+
  draw_plot(gg_hist)+
  draw_plot(gg_in, x = 0.6, y = 0.6, width = 0.3, height = 0.3)

  

ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/sensitivity-gapfill/Figure B5.png",
       width = 8, height = 4)





# numbers -----------------------------------------------------------------


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
