# some figures for province meeting

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)



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

dat_plot[, 
         .(n_stn = .N),
         .(snow, country, season_year)] %>% 
  ggplot(aes(season_year, n_stn, colour = snow))+
  geom_line()+
  facet_wrap(~country, scales = "free")+
  theme_bw()+
  ylim(0,NA)+
  xlab(NULL)+
  ylab("# stations per year")

ggsave(file = "fig/temp-prov-meeting/number_station_by_year.png",
       width = 8, height = 4, units = "in")




# maps --------------------------------------------------------------------

library(sf)
library(mapview)

dat_meta_1 %>% 
  st_as_sf(coords = c("Longitude", "Latitude"),
           crs = sp::CRS("+proj=longlat +datum=WGS84")) %>% 
  mapview(zcol = "Provider") #%>% 
#mapshot(url = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/map_all.html")



# eof ---------------------------------------------------------------------

library(foreach)
library(sinkr)


dat_hs_sub <- dat_1[Date >= "1981-10-01" & Date <= "2010-09-30" & month(Date) %in% c(12,1:4)]
dat_hs_sub[, .N, .(Provider, Name)] %>% .[, qplot(N)]
dat_hs_sub[, .N, .(Provider, Name)] %>% 
  .[N > 3000] %>% 
  merge(dat_hs_sub) -> dat_hs_sub

dat_hs_sub_wide <- dcast(dat_hs_sub, Provider + Name ~ Date, value.var = "HS")
dat_hs_sub_wide[, .N, Provider]
# dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
# dat_hs_sub_wide2[, .N, provider]


dat_hs_sub_wide_stn <- dat_hs_sub_wide[, .(Provider, Name)]
dat_hs_sub_wide[, -c("Provider", "Name"), with = F] %>% 
  as.matrix %>% 
  t -> mat_hs

eof1 <- eof(mat_hs, scaled = T, nu = 20, recursive = F)
plot(log(eof1$Lambda[1:50]))
# xx <- 5:50
# abline(lm(log(eof1$Lambda[xx]) ~ xx))
summary_eof <- function(xx_eof, k = 10){
  sdev <- xx_eof$Lambda
  sdev <- sdev / sum(sdev)
  data.frame(pc = paste0("PC", 1:k),
             prop_sd = sdev[1:k],
             cumsum_prop_sd = cumsum(sdev[1:k]))
}
summary_eof(eof1)

dat_eof <- cbind(dat_hs_sub_wide_stn,
                 eof1$u)
dat_eof2 <- merge(dat_eof, 
                  dat_meta_1)
dat_eof2[, country := substr(Provider, 1, 2)]
setnames(dat_eof2, paste0("V", 1:20), paste0("PC", 1:20))
dat_eof2 %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation"),
       measure.vars = paste0("PC", 1:6)) -> dat_eof3

dat_eof3[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_eof3[variable %in% paste0("PC", 1:4)] %>% 
  ggplot(aes(Longitude, Latitude, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta_1$Longitude), ylim = range(dat_meta_1$Latitude))

ggsave("fig/temp-prov-meeting/eof-regions.png",
       width = 12, height = 8, units = "in")




# trends ------------------------------------------------------------------


dt_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dt_lm <- readRDS("data/test-trend-02.rds")
dt_lm[, month_f := factor(month.abb[month], levels = month.abb[c(11,12,1:5)])]
dt_lm[, country := substr(Provider, 1, 2)]
dt_lm2 <- merge(dt_lm, 
                dt_meta,
                by = c("Provider", "Name"))

dt_year0 <- dt_lm2[term == "year0"]
# remove zugspitze because on glacier
dt_year0 <- dt_year0[Name != "Zugspitze"]
# remove all 0
dt_year0_nonan <- dt_year0[!is.na(statistic)]

# first numbers -----------------------------------------------------------

dt_year0[qprob == 0.5, table(mw_start_year, month_f, snow)]
dt_year0[mw_start_year == 1980 & is.na(statistic), table(month_f, qprob, snow)]

# hs & hn -----------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn") & month == 1] %>% 
  ggplot(aes(Elevation, estimate))+
  geom_hline(yintercept = 0)+
  geom_point()+
  facet_grid(snow ~ country, scales = "free_y")+
  theme_bw()
# -> some stations missing

dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn") & month == 1] %>% 
  dcast(Provider + Name + month_f + country + 
          Latitude + Longitude + Elevation ~ snow, value.var = "estimate") %>% 
  ggplot(aes(HN, HS))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point()+
  facet_wrap(~country)+
  theme_bw()


# HS nov - may ---------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(month_f, estimate))+
  geom_boxplot()+
  coord_cartesian(ylim = c(-5, 2.5))

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(month_f, estimate))+
  geom_boxplot()+
  coord_cartesian(ylim = c(-5, 2.5))+
  facet_wrap(~country)

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(Elevation, estimate))+
  geom_hline(yintercept = 0)+
  geom_point(size = 0.5)+
  # geom_smooth(se = F)+
  coord_cartesian(ylim = c(-5, 5))+
  facet_grid(country ~ month_f)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Linear trend (1980-2000) in mean monthly HS [cm/year]")

ggsave("fig/temp-prov-meeting/trends-month-country.png",
       width = 12, height = 8)
