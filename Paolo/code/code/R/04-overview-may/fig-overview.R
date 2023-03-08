# some figures for province meeting

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)


# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")

dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")



dat_1[, month := month(Date)]
dat_2[, month := month(Date)]

dat_1[, season_year := year(Date)]
dat_2[, season_year := year(Date)]

dat_1[month <= 8, season_year := season_year - 1L]
dat_2[month <= 8, season_year := season_year - 1L]

dat_all <- rbind(dat_2, dat_1)

# fig n stn by year -------------------------------------------------------

dat_hs_avail <- unique(dat_all[!is.na(HS), .(Provider, Name, season_year)])
dat_hn_avail <- unique(dat_all[!is.na(HN), .(Provider, Name, season_year)])

dat_plot <- rbindlist(list(hs = dat_hs_avail,
                           hn = dat_hn_avail),
                      idcol = "snow")
dat_plot[, country := substr(Provider, 1, 2)]

cols_country <- scales::brewer_pal(palette = "Set1")(7)[-6]

gg_n1 <- dat_plot[, 
         .(n_stn = .N),
         .(snow, country, season_year)] %>% 
  ggplot(aes(season_year, n_stn, colour = country))+
  geom_line()+
  # scale_color_brewer(palette = "Set1")+
  scale_color_manual(values = cols_country)+
  facet_wrap(~snow)+
  theme_bw()+
  ylim(0,NA)+
  xlab(NULL)+
  ylab("# stations per year")

gg_n1_dl <- direct.label(gg_n1, "extreme.grid")

ggsave(gg_n1_dl,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/number_station_by_year.png",
       width = 8, height = 4, units = "in")


# italy sub

gg_n2 <- dat_plot[country == "IT", 
         .(n_stn = .N),
         .(snow, Provider, season_year)] %>% 
  ggplot(aes(season_year, n_stn, colour = snow, linetype = snow))+
  geom_line()+
  facet_wrap(~Provider, scales = "free_y")+
  theme_bw()+
  ylim(0,NA)+
  xlab(NULL)+
  ylab("# stations per year")

# direct.label(gg_n2, "extreme.grid")

ggsave(gg_n2,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/number_station_by_year_IT.png",
       width = 8, height = 4, units = "in")



# altitude distribution ---------------------------------------------------

dat_plot_alt <- unique(rbind(dat_meta_1, dat_meta_2))
dat_plot_alt[, country := substr(Provider, 1, 2)]


dat_plot_alt %>% 
  ggplot(aes(Elevation))+
  geom_histogram(binwidth = 100)+
  facet_wrap(~country, scales = "free_y")+
  theme_bw()+
  xlab("Station altitude [m]")+
  ylab("# stations")

ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/altitude.png",
       width = 8, height = 4, units = "in")





# maps --------------------------------------------------------------------

library(sf)
library(mapview)

dat_meta_1 %>% 
  st_as_sf(coords = c("Longitude", "Latitude"),
           crs = sp::CRS("+proj=longlat +datum=WGS84")) %>% 
  mapview(zcol = "Provider") #%>% 
#mapshot(url = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/map_all.html")





