# plot of clusters



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
# library(foreach)
library(scico)
library(lemon)
library(patchwork)
library(directlabels)
library(sf)
library(cluster)


# eof ---------------------------------------------------------------------


# snow cluster
dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")
sf_meta <- st_as_sf(dat_meta_cluster,
                    coords = c("Longitude", "Latitude"),
                    crs = 4326)


# histalp regions
sf_histalp <- read_sf("data/shp-histalp-zone")
sf_histalp2 <- dplyr::mutate(sf_histalp,
                             id_fct = fct_recode(factor(Id), 
                                                 "SW" = "0", "NW" = "1", "NE" = "2", "SE" = "3"))

gg_clust <- 
ggplot()+
  borders()+
  geom_sf(data = sf_histalp2, aes(fill = id_fct), linetype = "blank", alpha = 0.5)+
  geom_sf(data = sf_meta, aes(colour = cluster_fct, shape = cluster_fct))+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  theme(legend.position = c(0.65, 0.15),
        # legend.position = "bottom",
        # legend.title = element_blank(),
        legend.background = element_rect(colour = "grey"),
        legend.direction = "horizontal",
        legend.box.just = "right")+
  scale_shape("Snow depth clusters")+
  scale_color_brewer("Snow depth clusters", palette = "Set1")+
  guides(color = guide_legend(override.aes = list(size = 4)))+
  scale_fill_brewer("HISTALP regions", palette = "Pastel2")



ggsave(gg_clust,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 5.png",
       width = 7, height = 4, scale = 1.3)



# pca full ----------------------------------------------------------------




dat_meta_cluster2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-02-pca-full.rds")

gg_clust2 <-
  dat_meta_cluster2 %>% 
  ggplot(aes(Longitude, Latitude, colour = cluster_fct, shape = cluster_fct))+
  borders()+
  geom_point(size = 1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_cluster2$Longitude), ylim = range(dat_meta_cluster2$Latitude))+
  theme_bw()+
  theme(legend.position = c(0.65, 0.1),
        legend.title = element_blank(),
        legend.background = element_rect(colour = "grey"),
        legend.direction = "horizontal",
        legend.box.just = "right")+
  scale_color_brewer(palette = "Set1")+
  guides(color = guide_legend(override.aes = list(size = 4)))

ggsave(gg_clust2,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/Figure S4.png",
       width = 7, height = 4)


# test --------------------------------------------------------------------

dat_meta_cluster %>% 
  ggplot(aes(Longitude, Elevation, colour = cluster_fct))+
  geom_point()+
  scale_color_brewer(palette = "Set1")+
  theme_bw()

dat_meta_cluster %>% 
  with(table(Provider, cluster_fct))
  


# compare clustering eof vs pca -------------------------------------------

dat_comp_clust <- merge(
  dat_meta_cluster[, .(Name, cluster_eof = cluster_fct)],
  dat_meta_cluster2[, .(Name, cluster_pca = cluster_fct)]
)

dat_comp_clust %>% with(table(cluster_eof, cluster_pca))
dat_comp_clust[, sum(cluster_eof == cluster_pca) / .N]
dat_comp_clust[, sum(cluster_eof == cluster_pca)]



# check misclustered stations ---------------------------------------------

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/regions-01-sinkr-eof.rda")
mat_clust3 <- sinkr_eof$u[, 1:5]
rownames(mat_clust3) <- colnames(mat_eof)

km_fit <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/regions-03-km-obj.rds")

km_sil <- silhouette(km_fit$cluster, dist(mat_clust3))

dat_sil <- data.table(Name = names(km_fit$cluster), 
                       cluster = km_sil[, 1],
                       cluster_neigh = km_sil[, 2],
                       sil_width = km_sil[, 3])
dat_sil[, cluster_fct := fct_recode(factor(cluster),
                                    "NW" = "1", 
                                    "NE" = "2", 
                                    "South & high Alpine" = "3",
                                    "SE" = "4", 
                                    "North & high Alpine" = "5")]
dat_sil[, cluster_neigh_fct := fct_recode(factor(cluster_neigh),
                                          "NW" = "1",
                                          "NE" = "2",  
                                          "South & high Alpine" = "3",
                                          "SE" = "4", 
                                          "North & high Alpine" = "5")]


mapview::mapview(sf_meta, zcol = "cluster_fct")

dat_sil[Name %in% dat_meta_cluster[substr(Provider, 1, 2) == "CH" & cluster_fct %in% c("SE"), Name]]
check_se <- dat_meta_cluster[!Provider %in% c("SI_ARSO", "AT_HZB") & cluster_fct %in% c("SE")]
dat_sil[Name %in% check_se$Name]
dat_sil[Name %in% c("Grono", "Magadino_Cadenazzo")]


dat_sil$sil_width %>% summary
dat_plot_sil <- sf_meta %>%
  merge(dat_sil) %>% 
  dplyr::mutate(sil_width_fct := cut(sil_width, breaks = c(-0.1, .1, .3, .8)))

gg_sil <- dat_plot_sil %>%
  ggplot()+
  borders()+
  geom_sf(aes(colour = cluster_fct, alpha = fct_rev(sil_width_fct)))+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_sf(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()+
  scale_color_brewer("Snow depth clusters", palette = "Set1")+
  scale_alpha_discrete("Silhouette")
  

ggsave(gg_sil,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure B1.png",
       width = 8, height = 4)


# numbers -----------------------------------------------------------------

dat_meta_cluster[, .(min = min(Elevation), 
                     max = max(Elevation),
                     mean = mean(Elevation),
                     median = median(Elevation)),
                 cluster_fct]


# EOF ---------------------------------------------------------------------


