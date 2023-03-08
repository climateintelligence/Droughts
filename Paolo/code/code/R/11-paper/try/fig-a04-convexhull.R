# create convex hull of points 
# and check DEM


library(data.table)
library(dplyr)
library(magrittr)
library(sf)
library(mapview)
library(raster)
library(cowplot)




ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure B5.png",
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
