# try to cluster the stations?
# based on daily series or on first 5 PCs



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
# library(foreach)
library(scico)
library(lemon)
library(cluster)
# library(sinkr)
library(foreach)


# prep data ---------------------------------------------------------------


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")


# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]


setnames(dat_hs, "Date", "date")
dat_hs <- mitmatmisc::add_hydro_year(dat_hs)

dat_hs_sub <- dat_hs[hydro_year >= 1981 & hydro_year <= 2010 &
                       month(date) %in% c(12, 1:4)]

# subset based on # of available daily obs (better since based on daily data!)

dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, qplot(N)]
dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, max(N)] -> max_n
dat_hs_sub[!is.na(HS), .N, .(Name)] %>%
  .[N > 0.7*max_n] %>%
  merge(dat_hs_sub) -> dat_hs_sub2

# or subset based on gapfill data

# dat_meta_gp <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
# dat_hs_sub2 <- dat_hs_sub[Name %in% dat_meta_gp$Name]

dat_hs_eof <- dcast(dat_hs_sub2, date ~ Name, value.var = "HS")


dat_hs_eof[, ":="(date = NULL)]
mat_eof <- as.matrix(dat_hs_eof)




na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]
mat_eof_sub_sc <- scale(mat_eof_sub)
mat_clust1_obs <- t(mat_eof_sub_sc)


pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)
pca_prcomp$rotation %>% str
# mat_clust2 <- pca_prcomp$rotation[, 1:5]
mat_clust2_pca <- pca_prcomp$rotation


# extensive clustering - obs and pca --------------------------------------

dat_clust1_obs <- foreach(
  kk = 2:8,
  .final = rbindlist
) %do% {
  
  km_fit <- kmeans(mat_clust1_obs, kk)
  stn_clust <- km_fit$cluster
  data.table(kk = kk, cluster = stn_clust, Name = names(stn_clust))
  
}




dat_loop <- expand.grid(kk = 2:8, pca_k = 2:8)
setDT(dat_loop)
dat_loop <- dat_loop[kk <= pca_k]

dat_clust2_pca <- foreach(
  ii = 1:nrow(dat_loop),
  .final = rbindlist
) %do% {
  kk <- dat_loop[ii, kk]
  pca_k <- dat_loop[ii, pca_k]
  
  km_fit <- kmeans(mat_clust2_pca[, 1:pca_k], kk)
  stn_clust <- km_fit$cluster
  data.table(pca_k = pca_k, kk = kk, cluster = stn_clust, Name = names(stn_clust))
  
}



# plot --------------------------------------------------------------------

dat_clust1_obs[, pca_k := "obs"]
dat_plot <- rbindlist(list(dat_clust1_obs, dat_clust2_pca), use.names = T)

dat_plot2 <- merge(dat_plot, dat_meta, by = "Name")

dat_plot2 %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.2)+
  facet_grid(pca_k ~ kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(palette = "Set1")





# plot single -------------------------------------------------------------

# maybe 6 & 6, but not so straightforward as 5


dat_plot2[kk == 6 & pca_k == 6] %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 1)+
  # facet_grid(pca_k ~ kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_grey()+
  scale_color_brewer(palette = "Set1")



