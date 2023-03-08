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


# prep data ---------------------------------------------------------------


load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/regions-01-sinkr-eof.rda")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")

data.table(Name = colnames(mat_eof), sinkr_eof$u) %>% 
  merge(dat_meta, by = "Name") -> dat_sinkr_eof

dat_sinkr_eof[, country := substr(Provider, 1, 2)]
setnames(dat_sinkr_eof, paste0("V", 1:20), paste0("PC", 1:20))
dat_sinkr_eof %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation", "cluster_fct"),
       measure.vars = paste0("PC", 1:5)) -> dat_sinkr_plot

dat_sinkr_plot[, value_sc := scales::rescale(value, c(-1,1)), variable]
sinkr_eof_summary[, prop_sd_format := scales::percent(prop_sd, .1)]
sinkr_eof_summary[, facet_lbl := paste0(pc, " (", prop_sd_format, ")")]
pc_rename <- setNames(sinkr_eof_summary$facet_lbl, sinkr_eof_summary$pc)
dat_sinkr_plot[, facet_lbl := pc_rename[variable]]


gg_eof <- dat_sinkr_plot %>% 
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

gg_eof2 <- reposition_legend(gg_eof, "center", panel = "panel-3-2")

ggsave(gg_eof2,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 3.png",
       width = 8, height = 4)



# pc by elev and region ---------------------------------------------------------------

# dat_sinkr_plot[variable %in% c("PC1", "PC2")] %>% 
#   ggplot(aes(Elevation, value_sc))+
#   geom_point(size = 0.5)+
#   # geom_smooth()+
#   facet_wrap(~variable)+
#   theme_bw()+
#   xlab("Elevation [m]")+
#   ylab("Scaled loading")

gg_pc_elev <- dat_sinkr_plot %>% 
  ggplot(aes(value_sc, Elevation, colour = cluster_fct))+
  geom_point(size = 0.5)+
  # geom_smooth()+
  scale_color_brewer("Region", palette = "Set1")+
  facet_wrap(~variable)+
  theme_bw()+
  ylab("Elevation [m]")+
  xlab("Scaled loading")


gg_pc_elev2 <- reposition_legend(gg_pc_elev, "center", panel = "panel-3-2")

ggsave(gg_pc_elev2,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 4.png",
       width = 6, height = 4)

# 
# dat_pca_prcomp_plot[variable %in% c("PC1", "PC2")] %>% 
#   ggplot(aes(Elevation, value))+
#   geom_point(size = 0.5)+
#   geom_smooth()+
#   facet_wrap(~variable)+
#   theme_bw()


# supplement pc loading by data provider ----------------------------------

dat_plot_country <- copy(dat_sinkr_plot)
dat_plot_country[, fct_row := ifelse(country %in% c("CH", "DE", "AT"), "row1", "row2")]
dat_plot_country[, country2 := factor(country, levels = c("CH", "DE", "AT", "FR", "IT", "SI"))]

l_gg <- lapply(paste0("PC", 1:5), function(i_pc){
  
  gg1 <-
  dat_plot_country[variable == i_pc & fct_row == "row1"] %>% 
    ggplot(aes(Longitude, Latitude, colour = value_sc))+
    # borders()+
    geom_point(size = 0.5)+
    scale_color_scico("Scaled loadings", palette = "roma",
                      guide = guide_colorbar(title.position = "top", direction = "horizontal"))+
    facet_grid(. ~ country2, scales = "free", space = "free")+
    xlab(NULL)+ylab(NULL)+
    theme_bw()+
    theme(axis.ticks = element_blank(),
          axis.text = element_blank(),
          legend.position = "none")+
    ggtitle(i_pc)
  
  gg2 <- dat_plot_country[variable == i_pc & fct_row == "row2"] %>% 
    ggplot(aes(Longitude, Latitude, colour = value_sc))+
    # borders()+
    geom_point(size = 0.5)+
    scale_color_scico("Scaled loadings", palette = "roma",
                      guide = guide_colorbar(title.position = "top", direction = "horizontal"))+
    facet_grid(. ~ country2, scales = "free", space = "free")+
    xlab(NULL)+ylab(NULL)+
    theme_bw()+
    theme(axis.ticks = element_blank(),
          axis.text = element_blank(),
          legend.position = "none")
  
  gg1 / gg2
})

gg_out <- wrap_plots(l_gg, nrow = 2)

ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/SM-PCload-country.png",
       width = 12, height = 6)



# appendix PCA full ----------------------------------------------------------------


load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/regions-02-pca-full.rda")
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
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/Figure S3.png",
       width = 8, height = 4)



# numbers -----------------------------------------------------------------


sinkr_eof_summary


