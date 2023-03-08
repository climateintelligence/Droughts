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
library(sinkr)


# prep data ---------------------------------------------------------------


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

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


# cluster on stns with full daily data -----------------------------------------------------------

na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]
mat_eof_sub_sc <- scale(mat_eof_sub)
mat_clust1 <- t(mat_eof_sub_sc)
# mat_clust_sc <- scale(mat_clust)


dat_clust1 <- foreach(
  kk = 2:7,
  .final = rbindlist
) %do% {
  
  km_fit <- kmeans(mat_clust1, kk)
  stn_clust <- km_fit$cluster
  data.table(kk = kk, cluster = stn_clust, Name = names(stn_clust))
  
}


dat_plot1 <- merge(dat_clust1, dat_meta, by = "Name")

dat_plot1 %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(type = "qual")


# !nice



# cluster on PCA for stns with full daily data -----------------------------------------------------------

pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)
pca_prcomp$rotation %>% str
# mat_clust2 <- pca_prcomp$rotation[, 1:5]
mat_clust2 <- pca_prcomp$rotation

dat_clust2 <- foreach(
  kk = 2:7,
  .final = rbindlist
) %do% {
  
  km_fit <- kmeans(mat_clust2[, 1:kk], kk)
  # km_fit <- kmeans(mat_clust2, kk)
  stn_clust <- km_fit$cluster
  data.table(kk = kk, cluster = stn_clust, Name = names(stn_clust))
  
}


dat_plot2 <- merge(dat_clust2, dat_meta, by = "Name")

dat_plot2 %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(type = "qual")



# cluster on sinkr::eof data -----------------------------------------------------------

sinkr_eof <- eof(mat_eof, centered = T, scaled = T, nu = 20, recursive = T)

mat_clust3 <- sinkr_eof$u
# mat_clust3 <- sinkr_eof$u[, 1:5]

rownames(mat_clust3) <- colnames(mat_eof)


dat_clust3 <- foreach(
  kk = 2:7,
  .final = rbindlist
) %do% {
  
  km_fit <- kmeans(mat_clust3[, 1:kk], kk)
  # km_fit <- kmeans(mat_clust3, kk)
  stn_clust <- km_fit$cluster
  data.table(kk = kk, cluster = stn_clust, Name = names(stn_clust))
  
}


dat_plot3 <- merge(dat_clust3, dat_meta, by = "Name")

dat_plot3 %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.5)+
  facet_wrap(~kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(type = "qual")



