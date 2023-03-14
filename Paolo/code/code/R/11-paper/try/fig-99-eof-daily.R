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



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

setnames(dat_hs, "Date", "date")
dat_hs <- mitmatmisc::add_hydro_year(dat_hs)



dat_hs_sub <- dat_hs[hydro_year >= 1981 & hydro_year <= 2010 &
                       month(date) %in% c(12, 1:2)]
dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, qplot(N)]
dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, max(N)] -> max_n
dat_hs_sub[!is.na(HS), .N, .(Name)] %>% 
  .[N > 0.7*max_n] %>% 
  merge(dat_hs_sub) -> dat_hs_sub2

dat_hs_eof2 <- dcast(dat_hs_sub2, date ~ Name, value.var = "HS")


dat_hs_eof2[, ":="(date = NULL)]
mat_eof <- as.matrix(dat_hs_eof2)


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


data.table(Name = names(eof1$F1_center), eof1$u) %>% 
  merge(dat_meta, by = "Name") -> dat_eof


dat_eof[, country := substr(Provider, 1, 2)]
setnames(dat_eof, paste0("V", 1:20), paste0("PC", 1:20))
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
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()

ggsave("fig/temp-eof/eof-daily-sinkr.png",
       width = 12, height = 6)




# eof full ----------------------------------------------------------------

na_cols <- sapply(dat_hs_eof2, function(x) any(is.na(x)))
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
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()

ggsave("fig/temp-eof/eof-daily-prcomp.png",
       width = 12, height = 6)



# conclusion --------------------------------------------------------------

# -> not many difference between "normal" eof and sinkr::eof with missing values
# PC1 needs to be checked!



# check PC1 ---------------------------------------------------------------


na_cols <- sapply(dat_hs_eof2, function(x) any(is.na(x)))
mat_eof2 <- mat_eof[, !na_cols]
eof_pc <- prcomp(mat_eof2, scale. = T, rank. = 20, retx = T)
data.table(Name = rownames(eof_pc$rotation), eof_pc$rotation) %>% 
  merge(dat_meta, by = "Name") -> dat_eof

dat_eof %>% summary
dat_eof$PC1 %>% qplot
dat_eof[PC1 < 0.01]

with(dat_eof, plot(Elevation, PC1)) # high alT! > 1000
with(dat_eof, plot(Elevation, PC2)) # low alt < 1500

dat_check_pc1 <- copy(dat_eof)
dat_check_pc1[, PC1_sc := scales::rescale(PC1, c(-1,1))]
dat_check_pc1 %>% 
  ggplot(aes(Longitude, Latitude, colour = PC1))+
  geom_point()+
  # scale_color_gradient2()+
  # scale_color_viridis_c()+
  # scale_color_scico(palette = "roma")+
  scale_color_binned(type = "viridis", n.breaks = 8)+
  borders()+
  theme(legend.position = "none")+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()






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
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()

