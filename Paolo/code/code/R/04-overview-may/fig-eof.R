



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(sinkr)

# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")
dat_1[, month := month(Date)]
dat_1[, season_year := year(Date)]
dat_1[month <= 8, season_year := season_year - 1L]


# eof ---------------------------------------------------------------------



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

eof1 <- eof(mat_hs, scaled = T, nu = 20, recursive = T)
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
       measure.vars = paste0("PC", 1:4)) -> dat_eof3

dat_eof3[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_eof3[variable %in% paste0("PC", 1:4)] %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  theme(legend.position = "none")+
  coord_quickmap(xlim = range(dat_meta_1$Longitude), ylim = range(dat_meta_1$Latitude))

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/eof-map.png",
       width = 12, height = 8, units = "in")



# plot by lat-long-elev -------------------------------------------------------------

cols_country <- scales::brewer_pal(palette = "Set1")(7)[-6]


p1 <- dat_eof3 %>% 
  ggplot(aes(Elevation, value_sc, colour = country))+
  geom_point(size = 0.5)+
  # scale_color_brewer(palette = "Set1")+
  scale_color_manual(values = cols_country)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()
p2 <- dat_eof3 %>% 
  ggplot(aes(Latitude, value_sc, colour = country))+
  geom_point(size = 0.5)+
  # scale_color_brewer(palette = "Set1")+
  scale_color_manual(values = cols_country)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()
p3 <- dat_eof3 %>% 
  ggplot(aes(Longitude, value_sc, colour = country))+
  geom_point(size = 0.5)+
  # scale_color_brewer(palette = "Set1")+
  scale_color_manual(values = cols_country)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()

library(patchwork)
plot_pw <- p1 / p2 / p3 + plot_layout(guides = "collect")

library(directlabels)

plot_out <- direct.label(p1, "extreme.grid") / 
  direct.label(p2, "extreme.grid") / 
  direct.label(p3, "extreme.grid")

ggsave(plot_out, 
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/eof-latlongelev.png",
       width = 14, height = 8, units = "in")
