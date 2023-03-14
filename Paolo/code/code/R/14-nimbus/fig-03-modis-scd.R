# get MODIS data


# library(raster)
library(stars)
library(data.table)
library(lubridate)
library(magrittr)
library(fs)
library(ggplot2)
library(forcats)
library(scico)

all_files <- dir_ls("/mnt/CEPH_PROJECTS/SNOW_3rd_gen/CLOUDREMOVAL/v1.1/annual_SCD/")
#remove incomplete
all_files <- all_files[2:20]
all_files %>% path_file

all_files %>% 
  path_file %>% 
  substr(1,4) -> along_year

rs_all <- read_stars(all_files,
                     along = list(hydro_year = along_year))
rs_all
# plot(rs_all)

all_files %>% 
  path_file %>% 
  path_ext_remove() %>% 
  stringr::str_split("_", simplify = T) -> files_dates
new_names <- paste0(substr(files_dates[, 1], 1, 7),
                    "...",
                    substr(files_dates[, 2], 1, 7))

rs_all2 <- rs_all %>% st_set_dimensions("hydro_year", value = new_names)


# map: year ---------------------------------------------------------------



gg_all <-
  ggplot()+
  geom_stars(data = rs_all2, downsample = c(10,10,1))+
  scale_fill_scico("SCD [days]",
                   palette = "devon", na.value = NA, direction = 1)+
  facet_wrap(~hydro_year)+
  coord_sf(expand = F)+
  theme_void()+
  theme(legend.position = "bottom")

ggsave(gg_all,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_modis_year.png",
       width = 12,
       height = 8)



# map: clim --------------------------------------------------------------------

rs_clim <- st_apply(rs_all, c("x", "y"), mean)



gg_clim <-
  ggplot()+
  geom_stars(data = rs_clim, downsample = c(1))+
  scale_fill_scico("Average SCD 2000-2019 [days]",
                   palette = "devon", na.value = NA, direction = 1)+
  coord_sf(expand = F)+
  theme_void()+
  theme(legend.position = "bottom")

ggsave(gg_clim,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/map_modis_clim.png",
       width = 8,
       height = 5)



