# try to cluster the stations?
# based on daily series or on first 5 PCs



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(scico)
library(lemon)
library(cluster)
library(sinkr)
library(foreach)



# prep data ---------------------------------------------------------------


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/spatcons_stns_ok.rds")
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


# pca's
sinkr_eof <- eof(mat_eof, centered = T, scaled = T, nu = 20, recursive = T)

na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]
pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)

mat_eof_sub_sc <- scale(mat_eof_sub)
mat_clust1_obs <- t(mat_eof_sub_sc)
mat_clust2_pca <- pca_prcomp$rotation
mat_clust3_eof <- sinkr_eof$u

dd1 <- dist(mat_clust1_obs)
# dd2 <- dist(mat_clust2_pca)
# dd3 <- dist(mat_clust3_eof)

# cluster and calc metrics ------------------------------------------------

dat_clust1_obs <- foreach(
  kk = 2:8,
  .final = rbindlist
) %do% {
  
  km_fit <- kmeans(mat_clust1_obs, kk)
  km_sil <- silhouette(km_fit$cluster, dd1)
  stn_clust <- km_fit$cluster
  data.table(kk = kk, 
             cluster = stn_clust, 
             Name = names(stn_clust),
             sil_width = km_sil[, 3],
             ss_tot = km_fit$totss,
             ss_withintot = km_fit$tot.withinss
             )
  
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
  km_sil <- silhouette(km_fit$cluster, dist(mat_clust2_pca[, 1:pca_k]))
  stn_clust <- km_fit$cluster
  data.table(pca_k = pca_k,
             kk = kk, 
             cluster = stn_clust,
             Name = names(stn_clust),
             sil_width = km_sil[, 3],
             ss_tot = km_fit$totss,
             ss_withintot = km_fit$tot.withinss)
  
}

dat_clust3_eof <- foreach(
  ii = 1:nrow(dat_loop),
  .final = rbindlist
) %do% {
  kk <- dat_loop[ii, kk]
  pca_k <- dat_loop[ii, pca_k]
  
  km_fit <- kmeans(mat_clust3_eof[, 1:pca_k], kk)
  km_sil <- silhouette(km_fit$cluster, dist(mat_clust3_eof[, 1:pca_k]))
  stn_clust <- km_fit$cluster
  data.table(pca_k = pca_k,
             kk = kk, 
             cluster = stn_clust,
             Name = colnames(mat_eof),
             sil_width = km_sil[, 3],
             ss_tot = km_fit$totss,
             ss_withintot = km_fit$tot.withinss)
  
}


# plot --------------------------------------------------------------------

dat_clust1_obs[, pca_k := "obs"]
dat_clust1_obs[, pca := "obs"]
dat_clust2_pca[, pca := "prcomp"]
dat_clust3_eof[, pca := "sinkr"]
dat_plot <- rbindlist(list(dat_clust1_obs, dat_clust2_pca, dat_clust3_eof), use.names = T)
dat_plot2 <- merge(dat_plot, dat_meta, by = "Name")

# ** elbow method ------------------------------------------------------------

dat_plot[, .(pca, pca_k, kk, ss_tot, ss_withintot, var_exp = 1 - ss_withintot/ss_tot)] %>% 
  unique %>% 
  ggplot(aes(kk, var_exp))+
  geom_point()+
  geom_line()+
  facet_wrap(~pca+pca_k)

# -> for pca/sinkr k = 5, for obs k = 4-5



# ** silhoutte ------------------------------------------------------------

dat_plot %>% 
  ggplot(aes(as.factor(kk), sil_width))+
  geom_boxplot()+
  facet_wrap(~pca+pca_k)

dat_plot[, 
         .(avg_sil = mean(sil_width)), 
         .(pca, pca_k, kk)] %>% 
  ggplot(aes(kk, avg_sil))+
  geom_point()+
  geom_line()+
  facet_wrap(~pca+pca_k)


# mixed, obs=4, prcomp 2, sinkr 2,5

dat_plot2[(pca_k == "obs" & kk == 4) | 
            (pca == "prcomp" & pca_k == 8 & kk == 2) | 
            (pca == "sinkr" & pca_k == 8 & kk %in% c(2,5))] %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.2)+
  facet_wrap(~pca + pca_k + kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(palette = "Set1")



dat_plot2[(pca_k == "obs") | 
            (pca == "prcomp" & pca_k == kk) | 
            (pca == "sinkr" & pca_k == kk )] %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.2)+
  facet_grid(pca ~ kk)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  scale_color_brewer(palette = "Set1")



# save a plot for SM ------------------------------------------------------

dat_plot_sm <- dat_plot2[(pca_k == "obs") | 
                           (pca == "prcomp" & pca_k == kk) | 
                           (pca == "sinkr" & pca_k == kk )] 

dat_plot_sm[, fct_col := fct_recode(factor(pca),
                                    "Obs" = "obs",
                                    "PCA (no NA)" = "prcomp",
                                    "PCA (with NA)" = "sinkr")]

gg_out <- 
  dat_plot_sm %>% 
  ggplot(aes(Longitude, Latitude, colour = as.factor(cluster)))+
  borders()+
  geom_point(size = 0.2)+
  facet_grid(kk ~ fct_col)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "°E"),
                     breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "°N"),
                     breaks = c(44,46,48))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  scale_color_brewer("# clusters", palette = "Set1")


ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/SM-clustering-maps.png",
       width = 10,
       height = 12)


# another for avg sil and var_exp
dat_plot[pca == "obs" | (pca != "obs" & kk == pca_k),
         .(avg_sil = mean(sil_width),
           var_exp = unique(1 - ss_withintot/ss_tot)),
         .(pca, kk)] %>% 
  melt(id.vars = c("pca", "kk")) %>% 
  ggplot(aes(kk, value, colour = pca))+
  geom_point()+
  geom_line(aes(group = pca))+
  facet_grid(variable ~ pca)+
  theme_bw()

dat_plot[pca == "obs" | (pca != "obs" & pca_k == 8),
         .(avg_sil = mean(sil_width),
           var_exp = unique(1 - ss_withintot/ss_tot)),
         .(pca, kk)] %>% 
  melt(id.vars = c("pca", "kk")) %>% 
  ggplot(aes(kk, value, colour = pca))+
  geom_point()+
  geom_line(aes(group = pca))+
  facet_grid(variable ~ pca, scales = "free_y")+
  theme_bw()

# not so good...



