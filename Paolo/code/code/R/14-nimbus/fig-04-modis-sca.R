# create plots of stats

library(ggplot2)
library(magrittr)
library(data.table)

source("R/14-nimbus/read_stats.R")




# v1.1 --------------------------------------------------------------------


fn_stats <- "/mnt/CEPH_PROJECTS/SNOW_3rd_gen/CLOUDREMOVAL/v1.1/stats/05_temporal_complete_max10d.csv"
fn_sample_raster <- "/mnt/CEPH_PROJECTS/SNOW_3rd_gen/CLOUDREMOVAL/v1.1/05_temporal_complete_max10d/20020716_120000.tif"


dat <- read_stats(fn_stats, fn_sample_raster)


# fill to complete MODIS period (2000-2020), hydro-year style
data.table(date = seq(ymd("1999-10-01"), ymd("2020-09-30"), by = "day")) %>% 
  merge(dat[value_fct == "snow"], 
        by = "date", all.x = T) -> dat_plot
mitmatmisc::add_hydro_year(dat_plot)
dat_plot[, hydro_year_label := paste0(hydro_year, "-", hydro_year + 1)]


gg_sca <- 
  dat_plot %>% 
  ggplot(aes(date, area_km2/1000))+
  geom_line(na.rm = T)+
  scale_x_date(date_minor_breaks = "1 month", date_labels = "%b", expand = c(0,0))+
  facet_wrap(~hydro_year_label, scales = "free_x")+
  theme_bw()+
  theme(panel.grid.minor.y = element_blank())+
  xlab(NULL)+
  ylab(expression(paste("Snow covered area [1000 ", km^2, "]")))

ggsave(gg_sca,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/fig/ts_modis_sca.png",
       width = 12,
       height = 8)




