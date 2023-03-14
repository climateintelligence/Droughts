# EOF analysis
# -> monthly not working!
# -> or better the sinkr with missing values not working!

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(scico)

# prep data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_wide_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_wide_HS.rds")

dat_hs <- mitmatmisc::add_hydro_year(dat_hs)

dat_hs_eof <- dat_hs[hydro_year >= 1981 & hydro_year <= 2010 &
                    month %in% c(12,1:4)]
vec_hydro_year <- dat_hs_eof$hydro_year
vec_month <- dat_hs_eof$month

dat_hs_eof[, ":="(month = NULL, year = NULL, hydro_year = NULL)]
mat_eof <- as.matrix(dat_hs_eof)


# eof ---------------------------------------------------------------------


eof1 <- eof(mat_eof, scaled = T, nu = 20, recursive = T)
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

dat_eof <- cbind(dat_meta,
                 eof1$u)

dat_eof[, country := substr(Provider, 1, 2)]
setnames(dat_eof, paste0("V", 1:20), paste0("PC", 1:20))
dat_eof %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation"),
       measure.vars = paste0("PC", 1:6)) -> dat_eof3

dat_eof3[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_eof3[variable %in% paste0("PC", 1:6)] %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  geom_point()+
  scale_color_gradient2()+
  # scale_color_viridis_c()+
  facet_wrap(~variable)+
  borders()+
  theme(legend.position = "none")+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))






# eof full ----------------------------------------------------------------

na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof2 <- mat_eof[, !na_cols]

eof_pc <- prcomp(mat_eof2, scale. = T, rank. = 20, retx = T)
plot(log(eof_pc$sdev[1:50]^2))
summary_pca <- function(xx_pca, k = 10){
  sdev <- xx_pca$sdev^2
  sdev <- sdev / sum(sdev)
  data.frame(pc = paste0("PC", 1:k),
             prop_sd = sdev[1:k],
             cumsum_prop_sd = cumsum(sdev[1:k]))
}
summary_pca(eof_pc)

data.table(Name = rownames(eof_pc$rotation), eof_pc$rotation) %>% 
  merge(dat_meta, by = "Name") -> dat_eof

dat_eof[, country := substr(Provider, 1, 2)]
# setnames(dat_eof, paste0("V", 1:20), paste0("PC", 1:20))
dat_eof %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation"),
       measure.vars = paste0("PC", 1:6)) -> dat_eof3

dat_eof3[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_eof3[variable %in% paste0("PC", 1:6)] %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  geom_point()+
  # scale_color_gradient2()+
  # scale_color_viridis_c()+
  scale_color_scico(palette = "roma")+
  facet_wrap(~variable)+
  borders()+
  theme(legend.position = "none")+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))







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
