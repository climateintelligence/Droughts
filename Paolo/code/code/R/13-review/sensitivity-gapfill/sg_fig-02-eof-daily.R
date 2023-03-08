# EOF analysis


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



load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/regions-02-pca-full.rda")
# no new clusters
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-02-pca-full.rds")

data.table(Name = colnames(mat_eof_sub), pca_prcomp$rotation) %>% 
  merge(dat_meta, by = "Name") -> dat_pca_prcomp

dat_pca_prcomp[, country := substr(Provider, 1, 2)]
# setnames(dat_pca_prcomp, paste0("V", 1:20), paste0("PC", 1:20))
dat_pca_prcomp %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation", "cluster_fct"),
       measure.vars = paste0("PC", 1:5)) -> dat_pca_plot

dat_pca_plot[, value_sc := scales::rescale(value, c(-1,1)), variable]
pca_prcomp_summary[, prop_sd_format := scales::percent(prop_sd, .1)]
pca_prcomp_summary[, facet_lbl := paste0(pc, " (", prop_sd_format, ")")]
pc_rename <- setNames(pca_prcomp_summary$facet_lbl, pca_prcomp_summary$pc)
dat_pca_plot[, facet_lbl := pc_rename[variable]]


gg_pca <- dat_pca_plot %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  borders()+
  geom_point(size = 0.5)+
  scale_color_scico("Scaled loadings", palette = "roma",
                    guide = guide_colorbar(title.position = "top", direction = "horizontal"))+
  facet_wrap(~facet_lbl)+
  
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()

gg_pca2 <- reposition_legend(gg_pca, "center", panel = "panel-3-2")

ggsave(gg_pca2,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/Figure S12.png",
       width = 8, height = 4)



# numbers -----------------------------------------------------------------


sinkr_eof_summary


