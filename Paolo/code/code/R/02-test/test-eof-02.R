# test eof with missing data

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(sinkr)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
# dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")


# pca daily ---------------------------------------------------------------------

dat_hs_sub <- dat_hs[date >= "1981-10-01" & date <= "2010-09-30" & month(date) %in% c(12,1:4)]
dat_hs_sub[, .N, .(provider, stn_name)] %>% .[, qplot(N)]
dat_hs_sub[, .N, .(provider, stn_name)] %>% 
  .[N > 3000] %>% 
  merge(dat_hs_sub) -> dat_hs_sub

dat_hs_sub_wide <- dcast(dat_hs_sub, provider + stn_name ~ date, value.var = "hs")
dat_hs_sub_wide[, .N, provider]
# dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
# dat_hs_sub_wide2[, .N, provider]

# compare to standard pca
dat_hs_sub_wide2 <- na.omit(dat_hs_sub_wide)
dat_hs_sub_wide2[, -c("provider", "stn_name"), with = F] %>% 
  as.matrix %>% 
  t -> mat_hs_na
pc_prc <- prcomp(mat_hs_na, scale. = T, rank. = 20, retx = T)
pc_eof <- eof(mat_hs_na, scaled = T, nu = 20)
plot(pc_prc$sdev[1:10]^2)
plot(pc_eof$Lambda[1:10])
plot(pc_prc$sdev[1:10]^2, pc_eof$Lambda[1:10])
# -> OK!

dat_hs_sub_wide_stn <- dat_hs_sub_wide[, .(provider, stn_name)]
dat_hs_sub_wide[, -c("provider", "stn_name"), with = F] %>% 
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
                  dat_meta[, .(provider, stn_name = Name, 
                               lon = Longitude, lat = Latitude, elev = Elevation)])
dat_eof2[, country := substr(provider, 1, 2)]
setnames(dat_eof2, paste0("V", 1:20), paste0("PC", 1:20))
dat_eof2 %>% 
  melt(id.vars = c("country", "provider", "stn_name", "lon", "lat", "elev"),
       measure.vars = paste0("PC", 1:6)) -> dat_eof3

dat_eof3[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_eof3 %>% 
  ggplot(aes(lon, lat, colour = value))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))


dat_eof3 %>% 
  ggplot(aes(elev, value))+
  geom_point()+
  facet_grid(variable ~ country, scales = "free_y")


p1 <- dat_eof3 %>% 
  ggplot(aes(elev, value_sc, colour = country))+
  geom_point(size = 0.5)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()
p2 <- dat_eof3 %>% 
  ggplot(aes(lat, value_sc, colour = country))+
  geom_point(size = 0.5)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()
p3 <- dat_eof3 %>% 
  ggplot(aes(lon, value_sc, colour = country))+
  geom_point(size = 0.5)+
  facet_grid(. ~ variable, scales = "free_y")+
  theme_bw()

library(patchwork)
p1 / p2 / p3 + plot_layout(guides = "collect")

library(directlabels)

direct.label(p1, "extreme.grid") / 
  direct.label(p2, "extreme.grid") / 
  direct.label(p3, "extreme.grid")
